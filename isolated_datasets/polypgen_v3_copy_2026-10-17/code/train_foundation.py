"""
Foundation Model Embedding Extraction & Linear Probe Engine for PolypGen (70/15/15 Split).
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from foundation_models import FoundationEncoder
from polyp_dataset import setup_random_seed


def extract_foundation_embeddings(
    encoder: FoundationEncoder,
    loader: DataLoader,
    device: torch.device,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            print(f"[Foundation-Trainer] Loading cached embeddings from {cache_path}...")
            data = np.load(cache_path)
            return data["X"], data["y"]

    encoder = encoder.to(device)
    encoder.eval()
    all_emb, all_lbl = [], []
    use_amp = (device.type == 'cuda')

    print(f"[Foundation-Trainer] Extracting foundation embeddings on {device}...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting Foundation Representations"):
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    feats = encoder(images)
            else:
                feats = encoder(images)
            feats = nn.functional.normalize(feats, dim=-1)
            all_emb.append(feats.cpu().numpy())
            all_lbl.append(labels.numpy())

    X = np.vstack(all_emb).astype(np.float32)
    y = np.concatenate(all_lbl).astype(np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


def train_foundation_linear_probe_split(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    C: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, int]:
    active_seed = setup_random_seed(seed)
    
    # 70% Train / 30% Holdout
    holdout_ratio = val_split + test_split
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=holdout_ratio, stratify=y, random_state=active_seed
    )
    
    # 15% Val / 15% Test
    test_rel_ratio = test_split / holdout_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=test_rel_ratio, stratify=y_holdout, random_state=active_seed
    )

    print(f"[Foundation-Train] Split: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%) | Seed: {active_seed}")

    probe_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('probe', LogisticRegression(C=C, max_iter=1000, class_weight='balanced', random_state=active_seed))
    ])

    probe_pipeline.fit(X_train, y_train)
    val_acc = probe_pipeline.score(X_val, y_val)
    print(f"[Foundation-Train] Validation Accuracy (15% split): {val_acc*100:.2f}%")

    y_test_proba = probe_pipeline.predict_proba(X_test)
    y_test_pred = np.argmax(y_test_proba, axis=1)

    return y_test, y_test_pred, y_test_proba, probe_pipeline, active_seed
