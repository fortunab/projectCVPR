"""
Vision Transformer Training Engine for PolypGen.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_models import get_vit_parameter_groups
from evaluate import compute_all_metrics


class ViTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        output_dir: Path,
        active_seed: int,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.1,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_seed = active_seed

        self.use_amp = use_amp and (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        param_groups = get_vit_parameter_groups(model=self.model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups)

        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss, total = 0.0, 0
        all_preds, all_targets, all_probas = [], [], []

        for images, labels in loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            total += images.size(0)

        return total_loss / total, np.array(all_targets), np.array(all_preds), np.array(all_probas)

    def fit(self, epochs: int = 15, warmup_epochs: int = 3) -> Dict[str, Any]:
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        best_val_f1 = 0.0
        best_ckpt = self.output_dir / "best_vit_model.pth"

        print(f"[ViT-Trainer] Training for {epochs} epochs on {self.device} (Seed: {self.active_seed})...")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, y_true_val, y_pred_val, y_proba_val = self.evaluate_loader(self.val_loader)
            val_metrics = compute_all_metrics(y_true_val, y_pred_val, y_proba_val, self.class_names, seed=self.active_seed)
            val_acc, val_f1 = val_metrics["Accuracy"], val_metrics["Macro_F1"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            scheduler.step()

            saved = ""
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(), "seed": self.active_seed}, best_ckpt)
                saved = " -> [SAVED BEST]"

            print(f" Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%{saved}")

        checkpoint = torch.load(best_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, y_true_test, y_pred_test, y_proba_test = self.evaluate_loader(self.test_loader)
        test_metrics = compute_all_metrics(y_true_test, y_pred_test, y_proba_test, self.class_names, seed=self.active_seed)

        self.plot_curves(self.output_dir / "training_curves_vit.png")
        return {"test_metrics": test_metrics, "y_true_test": y_true_test, "y_pred_test": y_pred_test, "y_proba_test": y_proba_test}

    def plot_curves(self, path: Path):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(10, 4), dpi=300)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss vs. Epochs", weight='bold')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [a*100 for a in self.history["train_acc"]], label="Train Acc")
        plt.plot(epochs, [a*100 for a in self.history["val_acc"]], label="Val Acc")
        plt.title("Accuracy (%) vs. Epochs", weight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
