"""
Main Execution Script for ConvNeXt Baseline on PolypGen.
"""

import os
import argparse
import json
from pathlib import Path
import torch

from polyp_dataset import create_polyp_dataloaders, CLASS_DESCRIPTIONS
from convnext_models import create_convnext_model
from train_convnext import ConvNeXtTrainer
from evaluate import print_evaluation_report, plot_confusion_matrix, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description="ConvNeXt Fine-Tuning on PolypGen")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--model-name", type=str, default="convnext_tiny")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    print("="*80)
    print(f"  POLYPGEN BASELINE 2: CONVNEXT ({args.model_name.upper()}) FINE-TUNING")
    print("="*80)

    model, cfg = create_convnext_model(model_name=args.model_name, num_classes=2, pretrained=True)
    img_size = args.img_size

    train_l, val_l, test_l, class_names, seed = create_polyp_dataloaders(
        data_dir=data_dir, img_size=img_size, batch_size=args.batch_size, seed=args.seed, subsample=args.subsample
    )

    trainer = ConvNeXtTrainer(
        model=model, train_loader=train_l, val_loader=val_l, test_loader=test_l,
        class_names=class_names, device=device, output_dir=results_dir,
        active_seed=seed, backbone_lr=args.backbone_lr, head_lr=args.head_lr,
        use_amp=not args.no_amp
    )

    res = trainer.fit(epochs=args.epochs)
    test_metrics = res["test_metrics"]

    readable_names = [CLASS_DESCRIPTIONS.get(c, c) for c in class_names]
    print_evaluation_report(test_metrics, readable_names)

    json_metrics = {k: v for k, v in test_metrics.items() if k != "Confusion_Matrix"}
    with open(results_dir / "metrics_summary_convnext.json", "w") as f:
        json.dump(json_metrics, f, indent=4)

    plot_confusion_matrix(test_metrics["Confusion_Matrix"], class_names=["Normal", "Polyp"], output_path=results_dir / "confusion_matrix_convnext.png", title=f"ConvNeXt ({args.model_name}) - Confusion Matrix")
    plot_roc_curves(y_true=res["y_true_test"], y_proba=res["y_proba_test"], output_path=results_dir / "roc_curves_convnext.png", title=f"ConvNeXt ({args.model_name}) - ROC Curve")
    print(f"[Main] ConvNeXt results exported to: {results_dir}")


if __name__ == "__main__":
    main()
