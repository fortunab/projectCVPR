"""
Main Execution Script for Foundation Model Baseline on PolypGen.
Evaluates on the 70% Train / 15% Val / 15% Test split.
"""

import os
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from polyp_dataset import load_polyp_dataset_paths_and_labels, PolypDataset, get_endoscopy_transforms, CLASS_DESCRIPTIONS, setup_random_seed
from foundation_models import create_foundation_model
from train_foundation import extract_foundation_embeddings, train_foundation_linear_probe_split
from evaluate import compute_all_metrics, print_evaluation_report, plot_confusion_matrix, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Pathology/Vision Foundation Model on PolypGen")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--cache-file", type=str, default="./cache/features_polyp_foundation.npz")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--model-name", type=str, default="dinov2_base")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    cache_file = Path(args.cache_file) if Path(args.cache_file).is_absolute() else project_root / args.cache_file
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    active_seed = setup_random_seed(args.seed)

    print("="*80)
    print(f"  POLYPGEN BASELINE 4: FOUNDATION MODEL ({args.model_name}) + LINEAR PROBE (70/15/15 SPLIT)")
    print("="*80)

    image_paths, labels, class_names = load_polyp_dataset_paths_and_labels(data_dir)

    if args.subsample is not None and args.subsample < len(image_paths):
        np.random.seed(active_seed)
        idx = np.random.choice(len(image_paths), size=args.subsample, replace=False)
        image_paths = [image_paths[i] for i in idx]
        labels = labels[idx]

    if cache_file.exists() and not args.force_recompute:
        print(f"[Main] Loading cached embeddings from {cache_file}...")
        data = np.load(cache_file)
        X, y = data["X"], data["y"]
    else:
        encoder, cfg = create_foundation_model(model_name=args.model_name, hf_token=args.hf_token)
        tf = get_endoscopy_transforms(img_size=cfg["img_size"], is_training=False, mean=cfg["mean"], std=cfg["std"])
        ds = PolypDataset(image_paths, labels, transform=tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        X, y = extract_foundation_embeddings(encoder=encoder, loader=loader, device=device, cache_path=cache_file, force_recompute=args.force_recompute)

    y_test_true, y_test_pred, y_test_proba, probe, seed = train_foundation_linear_probe_split(
        X=X, y=y, val_split=args.val_split, test_split=args.test_split, C=args.C, seed=active_seed
    )

    joblib.dump(probe, results_dir / "foundation_linear_probe.joblib")

    readable_names = [CLASS_DESCRIPTIONS.get(c, c) for c in class_names]
    metrics = compute_all_metrics(
        y_true=y_test_true, y_pred=y_test_pred, y_proba=y_test_proba, class_names=readable_names, seed=seed
    )

    print_evaluation_report(metrics, readable_names)

    json_metrics = {k: v for k, v in metrics.items() if k != "Confusion_Matrix"}
    with open(results_dir / "metrics_summary_foundation.json", "w") as f:
        json.dump(json_metrics, f, indent=4)

    plot_confusion_matrix(metrics["Confusion_Matrix"], class_names=["Normal", "Polyp"], output_path=results_dir / "confusion_matrix_foundation.png", title=f"Foundation ({args.model_name}) - 15% Test Confusion Matrix")
    plot_roc_curves(y_true=y_test_true, y_proba=y_test_proba, output_path=results_dir / "roc_curves_foundation.png", title=f"Foundation ({args.model_name}) - 15% Test ROC Curve")
    print(f"[Main] Foundation Model results exported to: {results_dir}")


if __name__ == "__main__":
    main()
