"""
Main Execution Script for Classical SVM Baseline on PolypGen.
Evaluates on the 70% Train / 15% Val / 15% Test holdout split.
"""

import os
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

from polyp_dataset import load_polyp_dataset_paths_and_labels, CLASS_NAMES, CLASS_DESCRIPTIONS, setup_random_seed
from feature_extractor import extract_polyp_features_parallel
from train_svm import train_polyp_svm_split
from evaluate import compute_all_metrics, print_evaluation_report, plot_confusion_matrix, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Classical Handcrafted + SVM Baseline on PolypGen")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--cache-file", type=str, default="./cache/features_polyp_svm.npz", help="Cache path")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation ratio (default: 0.15 / 15%)")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test ratio (default: 0.15 / 15%)")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C regularization")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (randomized if omitted)")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample images")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing features")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    cache_file = Path(args.cache_file) if Path(args.cache_file).is_absolute() else project_root / args.cache_file
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("  POLYPGEN BASELINE 1: HANDCRAFTED TEXTURE & COLOR + RBF-SVM (70/15/15 SPLIT)")
    print("="*80)

    image_paths, labels, class_names = load_polyp_dataset_paths_and_labels(data_dir)
    active_seed = setup_random_seed(args.seed)

    if args.subsample is not None and args.subsample < len(image_paths):
        np.random.seed(active_seed)
        idx = np.random.choice(len(image_paths), size=args.subsample, replace=False)
        image_paths = [image_paths[i] for i in idx]
        labels = labels[idx]

    X, y = extract_polyp_features_parallel(
        image_paths=image_paths, labels=labels, cache_path=cache_file, force_recompute=args.force_recompute
    )

    y_test_true, y_test_pred, y_test_proba, model, seed = train_polyp_svm_split(
        X, y, val_split=args.val_split, test_split=args.test_split, C=args.C, seed=active_seed
    )

    readable_names = [CLASS_DESCRIPTIONS.get(c, c) for c in class_names]
    metrics = compute_all_metrics(
        y_true=y_test_true, y_pred=y_test_pred, y_proba=y_test_proba, class_names=readable_names, seed=seed
    )

    print_evaluation_report(metrics, readable_names)

    json_metrics = {k: v for k, v in metrics.items() if k != "Confusion_Matrix"}
    json_path = results_dir / "metrics_summary_svm.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=4)

    plot_confusion_matrix(metrics["Confusion_Matrix"], class_names=["Normal", "Polyp"], output_path=results_dir / "confusion_matrix_svm.png", title="SVM - 15% Test Confusion Matrix")
    plot_roc_curves(y_true=y_test_true, y_proba=y_test_proba, output_path=results_dir / "roc_curves_svm.png", title="SVM - 15% Test ROC Curve")
    print(f"[Main] Results and metrics exported to: {results_dir}")


if __name__ == "__main__":
    main()
