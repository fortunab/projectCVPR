"""Feature extraction for additional backbones on audited PolypGen cohort.

Extracts normalized CLS/pooled representations for:
- convnext_tiny (768 dim)
- vit_b_16 (768 dim)
- owkin/phikon (768 dim)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, vit_b_16, ViT_B_16_Weights

ROOT = Path("C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines").resolve()
EXT_DIR = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20"
MANIFEST_PATH = ROOT / "isolated_datasets/polypgen_v3_copy_2026-10-17/audit/manifest.csv"
OUT_DIR = EXT_DIR / "multi_backbone/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded manifest with {len(rows)} samples.")
    return rows


def extract_convnext(rows):
    out_path = OUT_DIR / "convnext_tiny.npz"
    if out_path.exists():
        print(f"ConvNeXt-Tiny already extracted: {out_path}")
        return out_path

    print("\n--- Extracting ConvNeXt-Tiny Features ---")
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    transform = weights.transforms()
    model = convnext_tiny(weights=weights).eval()

    all_features = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i:i + BATCH_SIZE]
        tensors = []
        for r in batch_rows:
            img = Image.open(r["image_path"]).convert("RGB")
            tensors.append(transform(img))
        batch_tensor = torch.stack(tensors)

        with torch.inference_mode():
            feats = model.features(batch_tensor)
            pooled = model.avgpool(feats)
            flat = torch.flatten(pooled, 1)
            normed = flat / flat.norm(dim=1, keepdim=True)
            all_features.append(normed.cpu().numpy().astype(np.float32))

        if (i // BATCH_SIZE) % 15 == 0 or i + BATCH_SIZE >= len(rows):
            done = min(i + BATCH_SIZE, len(rows))
            elapsed = time.time() - t0
            fps = done / (elapsed + 1e-6)
            print(f"  Processed {done}/{len(rows)} frames ({fps:.1f} fps)")

    matrix = np.concatenate(all_features, axis=0)
    sample_ids = np.array([r["sample_id"] for r in rows])
    metadata = {
        "model": "torchvision.models.convnext_tiny",
        "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "dimension": 768,
        "n_samples": len(rows),
        "normalized": True
    }
    np.savez_compressed(out_path, features=matrix, sample_ids=sample_ids, metadata=json.dumps(metadata))
    print(f"Saved ConvNeXt-Tiny features ({matrix.shape}) -> {out_path}")
    return out_path


def extract_vit(rows):
    out_path = OUT_DIR / "vit_b_16.npz"
    if out_path.exists():
        print(f"ViT-B/16 already extracted: {out_path}")
        return out_path

    print("\n--- Extracting ViT-B/16 Features ---")
    weights = ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    model = vit_b_16(weights=weights).eval()

    all_features = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i:i + BATCH_SIZE]
        tensors = []
        for r in batch_rows:
            img = Image.open(r["image_path"]).convert("RGB")
            tensors.append(transform(img))
        batch_tensor = torch.stack(tensors)

        with torch.inference_mode():
            tokens = model._process_input(batch_tensor)
            n = tokens.shape[0]
            b_cls = model.class_token.expand(n, -1, -1)
            tokens = torch.cat([b_cls, tokens], dim=1)
            enc = model.encoder(tokens)
            cls_token = enc[:, 0]
            normed = cls_token / cls_token.norm(dim=1, keepdim=True)
            all_features.append(normed.cpu().numpy().astype(np.float32))

        if (i // BATCH_SIZE) % 15 == 0 or i + BATCH_SIZE >= len(rows):
            done = min(i + BATCH_SIZE, len(rows))
            elapsed = time.time() - t0
            fps = done / (elapsed + 1e-6)
            print(f"  Processed {done}/{len(rows)} frames ({fps:.1f} fps)")

    matrix = np.concatenate(all_features, axis=0)
    sample_ids = np.array([r["sample_id"] for r in rows])
    metadata = {
        "model": "torchvision.models.vit_b_16",
        "weights": "ViT_B_16_Weights.IMAGENET1K_V1",
        "dimension": 768,
        "n_samples": len(rows),
        "normalized": True
    }
    np.savez_compressed(out_path, features=matrix, sample_ids=sample_ids, metadata=json.dumps(metadata))
    print(f"Saved ViT-B/16 features ({matrix.shape}) -> {out_path}")
    return out_path


def extract_phikon(rows):
    out_path = OUT_DIR / "phikon.npz"
    if out_path.exists():
        print(f"Phikon already extracted: {out_path}")
        return out_path

    print("\n--- Extracting Owkin Phikon Features ---")
    from transformers import AutoModel
    phk_dir = Path.home() / ".cache/huggingface/hub/models--owkin--phikon/snapshots/057cc0295895c2df3dd7681a89680da6015cbefe"
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = AutoModel.from_pretrained(phk_dir, local_files_only=True).eval()

    all_features = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i:i + BATCH_SIZE]
        tensors = []
        for r in batch_rows:
            img = Image.open(r["image_path"]).convert("RGB")
            tensors.append(transform(img))
        batch_tensor = torch.stack(tensors)

        with torch.inference_mode():
            out = model(batch_tensor)
            cls_token = out.last_hidden_state[:, 0]
            normed = cls_token / cls_token.norm(dim=1, keepdim=True)
            all_features.append(normed.cpu().numpy().astype(np.float32))

        if (i // BATCH_SIZE) % 15 == 0 or i + BATCH_SIZE >= len(rows):
            done = min(i + BATCH_SIZE, len(rows))
            elapsed = time.time() - t0
            fps = done / (elapsed + 1e-6)
            print(f"  Processed {done}/{len(rows)} frames ({fps:.1f} fps)")

    matrix = np.concatenate(all_features, axis=0)
    sample_ids = np.array([r["sample_id"] for r in rows])
    metadata = {
        "model": "owkin/phikon",
        "revision": "057cc0295895c2df3dd7681a89680da6015cbefe",
        "dimension": 768,
        "n_samples": len(rows),
        "normalized": True
    }
    np.savez_compressed(out_path, features=matrix, sample_ids=sample_ids, metadata=json.dumps(metadata))
    print(f"Saved Phikon features ({matrix.shape}) -> {out_path}")
    return out_path


def main():
    rows = load_manifest()
    extract_convnext(rows)
    extract_vit(rows)
    extract_phikon(rows)
    print("\nAll extractions completed successfully!")


if __name__ == "__main__":
    main()
