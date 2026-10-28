"""Refit unchanged main heads from audited features; never open raw images.

This level verifies provided feature/manifest provenance, not raw image bytes.
Use reproduce.py for an independent reconstruction from the archive instead.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
from reproduce import HERE, verify_bundle, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with its unchanged adjacent .audit.json")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-version-drift", action="store_true")
    args = parser.parse_args()
    bundle = verify_bundle()
    if sys.version_info[:2] != (3, 12) or args.threads < 1:
        raise ValueError("Python 3.12 and positive thread count required")
    versions = {name: importlib.metadata.version(name) for name in bundle["reference_packages"]}
    drift = {name: {"reference": expected, "actual": versions[name]}
             for name, expected in bundle["reference_packages"].items()
             if versions[name].split("+")[0] != expected.split("+")[0]}
    if drift and not args.allow_version_drift:
        raise ValueError(f"Resolve reference package versions or explicitly allow portability drift: {drift}")
    sys.path.insert(0, str(HERE / "sources/baseline/code"))
    import numpy as np
    from threadpoolctl import threadpool_limits
    from cbmp_data import load_manifest, make_outer_splits
    from cbmp_evaluation import candidate_methods, evaluate_loco
    from run_cbmp import _manifest_digest
    rows = load_manifest(args.manifest)
    digest = _manifest_digest(rows)
    reference = json.loads((HERE / "cohort_order.json").read_text())
    if digest != reference["manifest_digest"] or [r["sample_id"] for r in rows] != reference["sample_ids"]:
        raise ValueError("Provided manifest differs from the exact reviewed PolypGen cohort/order")
    with np.load(args.features, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        features = cache["features"].copy()
        dino = metadata.get("model_id") == "facebook/dinov2-base"
        dimension = 768 if dino else 512
        expected_sha = bundle["assets"]["model.safetensors" if dino else "resnet18-f37072fd.pth"]["sha256"]
        if metadata.get("manifest_digest") != digest or metadata.get("checkpoint_sha256") != expected_sha:
            raise ValueError("Features differ from recorded manifest/checkpoint")
        if dino:
            if (metadata.get("revision") != bundle["dinov2_revision"] or metadata.get("processor_config_sha256") != bundle["assets"]["preprocessor_config.json"]["sha256"]
                    or "sample_ids" not in cache or cache["sample_ids"].tolist() != reference["sample_ids"]):
                raise ValueError("DINOv2 revision, processor or row identity differs")
        elif metadata.get("resolved_model") != "torchvision.resnet18.IMAGENET1K_V1":
            raise ValueError("Unsupported frozen feature encoder")
    if (features.shape != (len(rows), dimension) or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid normalized feature matrix")
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError("Use a new absent output directory")
    splits = make_outer_splits(rows, seed=args.seed)
    if [s["heldout_center"] for s in splits if s["status"] == "ready"] != ["C2", "C4", "C5", "C6"]:
        raise ValueError("Unexpected target-center feasibility")
    write_json(out / "outer_splits.json", {"manifest_digest": digest, "splits": splits})
    write_json(out / "cached_refit_provenance.json", {"seed": args.seed, "threads": args.threads,
               "feature_metadata": metadata, "candidate_methods": candidate_methods(),
               "feature_file_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "bundle_manifest_sha256": hashlib.sha256((HERE / "sources_manifest.json").read_bytes()).hexdigest(),
               "raw_images_opened": False, "raw_image_hashes_reverified": False,
               "scope": "Provided-feature main-head reproduction; not an independent raw-data audit",
               "packages": versions, "version_drift": drift})
    with threadpool_limits(limits=args.threads):
        records = evaluate_loco(features, rows, splits, out, seed=args.seed, methods=candidate_methods())
    if len(records) != 36:
        raise ValueError("Incomplete source-only cached-feature evaluation")
    (out / "COMPLETE").write_text("36 unchanged main-head evaluations on provided features; no raw-data verification.\n")


if __name__ == "__main__":
    main()
