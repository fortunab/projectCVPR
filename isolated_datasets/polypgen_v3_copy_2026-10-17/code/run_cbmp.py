"""Audited frozen-encoder leave-one-center-out CBMP evaluation.

Run an audit first. The verified-sequence option is an explicit preliminary
subset policy; it cannot be presented as a complete-release experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cbmp_data import MANIFEST_SCHEMA_VERSION, build_manifest, load_manifest, make_outer_splits
from cbmp_evaluation import candidate_methods, evaluate_loco

FEATURE_SCHEMA = 2


class ManifestImages(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["image_path"]) as image:
            result = self.transform(image.convert("RGB"))
        return result, index


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(rows):
    columns = ("sample_id", "sha256", "pixel_sha256", "center", "group_id", "label",
               "label_source", "mask_sha256", "bbox_sha256")
    payload = [{key: row[key] for key in columns} for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_inputs(rows):
    for row in rows:
        for path_key, hash_key in (("image_path", "sha256"), ("mask_path", "mask_sha256"), ("bbox_path", "bbox_sha256")):
            if row[path_key] and file_sha256(row[path_key]) != row[hash_key]:
                raise ValueError(f"Input changed since annotation audit: {row[path_key]}; rebuild manifest")


def extract_features(rows, model_name, batch_size, workers, device, cache_path, force=False):
    verify_inputs(rows)
    import torchvision

    metadata = {"feature_schema": FEATURE_SCHEMA, "manifest_digest": _manifest_digest(rows),
                "model_name": model_name, "torch": torch.__version__, "torchvision": torchvision.__version__,
                "normalized_features": True}
    encoder = None
    if model_name == "resnet18_imagenet1k_v1":
        from torchvision.models import ResNet18_Weights, resnet18
        weights_path = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
        if not weights_path.is_file():
            raise FileNotFoundError("The official ResNet18 checkpoint is not cached; provide it before this offline run")
        checkpoint_hash = file_sha256(weights_path)
        if not checkpoint_hash.startswith("f37072fd"):
            raise ValueError("ResNet18 checkpoint checksum does not match official ImageNet1K V1 weights")
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        metadata.update(resolved_model="torchvision.resnet18.IMAGENET1K_V1",
                        checkpoint_sha256=checkpoint_hash, preprocessing=repr(transform))
    elif model_name.startswith("dinov2"):
        # Explicit architecture resolution; never accept the legacy ViT fallback.
        from foundation_models import create_foundation_model
        from polyp_dataset import get_endoscopy_transforms
        encoder, cfg = create_foundation_model(model_name=model_name)
        if "dinov2" not in str(cfg["model_name"]).lower():
            raise ValueError("Requested DINOv2 but factory resolved another architecture")
        transform = get_endoscopy_transforms(img_size=cfg["img_size"], is_training=False,
                                             mean=cfg["mean"], std=cfg["std"])
        metadata.update(resolved_model=str(cfg["model_name"]), preprocessing=repr(transform))
    else:
        raise ValueError("Supported explicit encoders: resnet18_imagenet1k_v1 or DINOv2")

    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cache:
            if "metadata" not in cache or json.loads(str(cache["metadata"].item())) != metadata:
                raise ValueError("Feature cache differs from audited inputs/weights/preprocessing; use a new cache path")
            features = cache["features"].copy()
        if (features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all()
                or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid normalized feature cache")
        print(f"Reused feature cache: {cache_path}", flush=True)
        return features, metadata

    if encoder is None:
        from torchvision.models import resnet18
        encoder = resnet18(weights=None)
        encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
        encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).to(device).eval()
    loader = DataLoader(ManifestImages(rows, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device == "cuda")
    features = None
    with torch.inference_mode():
        for images, positions in tqdm(loader, desc=f"Frozen {model_name}", mininterval=10):
            output = encoder(images.to(device)).float().cpu().numpy()
            if output.ndim != 2 or not np.isfinite(output).all():
                raise ValueError("Encoder output must be a finite image-by-feature matrix")
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise ValueError("Zero encoder embedding")
            output = output / norms
            if features is None:
                features = np.empty((len(rows), output.shape[1]), dtype=np.float32)
            features[positions.numpy()] = output
    if features is None:
        raise ValueError("No features extracted")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as stream:
        np.savez_compressed(stream, features=features, metadata=json.dumps(metadata, sort_keys=True))
    return features, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/cbmp/manifest.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cbmp"))
    parser.add_argument("--feature-cache", type=Path, default=Path("cache/cbmp_normalized_features_v2.npz"))
    parser.add_argument("--model-name", default="resnet18_imagenet1k_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--sequence-metadata", type=Path)
    parser.add_argument("--sequence-only", action="store_true")
    parser.add_argument("--verified-sequences-only", action="store_true",
                        help="Preliminary subset restricted to reviewed sequence metadata")
    parser.add_argument("--allow-image-only-groups", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Save eligibility and selection plan without fitting")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        parser.error("Invalid batch-size/workers/threads")
    torch.set_num_threads(args.threads)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and any((args.results_dir / name).exists()
                               for name in ("metrics_by_center.json", "predictions.csv")):
        parser.error("Completed/partial results exist; preserve them and choose a new results directory")
    if not args.manifest.exists():
        build_manifest(args.data_root, args.manifest, sequence_metadata=args.sequence_metadata,
                       sequence_only=args.sequence_only, verified_sequences_only=args.verified_sequences_only)
    audit_path = args.manifest.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Legacy manifest; rebuild the annotation and decoded-pixel audit")
    if Path(audit["data_root"]).resolve() != args.data_root.resolve():
        raise ValueError("Manifest data root differs from --data-root")
    if audit.get("blocking_rejections"):
        raise ValueError("Blocking annotation/center errors remain; inspect rejected CSV")
    if bool(audit.get("sequence_only")) != args.sequence_only:
        raise ValueError("Manifest sequence-only policy differs from requested run")
    if bool(audit.get("verified_sequences_only")) != args.verified_sequences_only:
        raise ValueError("Manifest verified-subset policy differs from requested run")
    if args.sequence_metadata:
        if (audit.get("sequence_metadata") != str(args.sequence_metadata.resolve())
                or audit.get("sequence_metadata_sha256") != file_sha256(args.sequence_metadata)):
            raise ValueError("Reviewed metadata differs from the audited version; rebuild manifest")
    rows = load_manifest(args.manifest)
    if any(r["group_source"] == "image_only_no_sequence_id" for r in rows) and not args.allow_image_only_groups:
        raise ValueError("Ungrouped still images require explicit diagnostic opt-in")
    splits = make_outer_splits(rows, seed=args.seed)
    ready = [s["heldout_center"] for s in splits if s["status"] == "ready"]
    (args.results_dir / "outer_splits.json").write_text(
        json.dumps({"manifest_digest": _manifest_digest(rows), "splits": splits}, indent=2), encoding="utf-8")
    methods = candidate_methods()
    config = {"seed": args.seed, "scope": audit.get("scope"), "manifest_digest": _manifest_digest(rows),
              "data_audit": audit, "eligible_centers": ready, "candidate_methods": methods,
              "selection": "equal-source-center validation balanced accuracy; deterministic candidate-order ties",
              "calibration": "bounded scalar minimization of equal-source-center validation NLL",
              "fitting": "selected head reconstructed from all outer source training images",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_sha256": {name: file_sha256(Path(__file__).parent / name)
                                for name in ("run_cbmp.py", "cbmp_evaluation.py", "cbmp.py", "polypgen_manifest.py", "cbmp_data.py")}}
    print(f"Scope: {config['scope']}", flush=True)
    print("Eligible held-out centers:", ready, flush=True)
    (args.results_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not ready:
        raise ValueError("No eligible center-held-out folds; no experiment started")
    if args.dry_run:
        print("Dry run complete; no encoder extraction or classifier fitting.", flush=True)
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    features, metadata = extract_features(rows, args.model_name, args.batch_size, args.workers, device,
                                          args.feature_cache, force=args.force_features)
    config["encoder"] = metadata
    config["device"] = device
    (args.results_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with threadpool_limits(limits=args.threads):
        evaluate_loco(features, rows, splits, args.results_dir, seed=args.seed, methods=methods)
    (args.results_dir / "COMPLETE").write_text("All eligible folds and predeclared methods completed.\n", encoding="utf-8")
    print(f"Saved complete held-out-center results: {args.results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
