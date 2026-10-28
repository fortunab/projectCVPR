"""Export a small, hash-addressed source bundle; never copy research images."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
ISOLATED = EXT.parent
BASE = ISOLATED / "polypgen_v3_copy_2026-10-17"
FOLLOW = ISOLATED / "polypgen_followups_2026-10-17"
WORKSPACE = ISOLATED.parent
CORE = ("cbmp.py", "cbmp_data.py", "cbmp_evaluation.py", "polypgen_manifest.py")
PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "torch", "torchvision",
            "Pillow", "matplotlib", "tqdm", "pandas", "pyarrow", "transformers", "huggingface-hub", "safetensors")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh only this bundle's source snapshots")
    args = parser.parse_args()
    destination = HERE / "sources"
    if destination.exists() and not args.refresh:
        raise FileExistsError("Source bundle exists; use --refresh only to refresh this reproduction package")
    entries = []
    def copy(source, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, target)
        if sha256(target) != before or sha256(source) != before:
            raise ValueError(f"Source changed while snapshotting: {source.name}")
        entries.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                        "sha256": before, "original_workspace_path": source.relative_to(WORKSPACE).as_posix()})
    for name in ("process_audit.py", "run_experiments.py", "make_report.py", "PROTOCOL.md"):
        copy(BASE / name, Path("baseline") / name)
    for name in (*CORE, "run_cbmp.py", "audit_cbmp_data.py"):
        copy(BASE / "code" / name, Path("baseline/code") / name)
    for name in ("polypgen_sequence_center_mapping_review_v2.json", "polypgen_sequence_class_audit.json",
                 "reviewed_sequence_centers.csv"):
        copy(BASE / "audit" / name, Path("baseline/audit") / name)
    for scope, root in (("followups", FOLLOW), ("extensions", EXT)):
        for source in sorted(root.glob("*.py")):
            copy(source, Path(scope) / source.name)
        copy(root / "PROTOCOL.md", Path(scope) / "PROTOCOL.md")
        for name in CORE:
            if sha256(root / "code" / name) != sha256(BASE / "code" / name):
                raise ValueError("Original classifier snapshots differ; do not silently choose one")
    pilot = WORKSPACE / "work/family_baselines/histology_cbmp"
    for name in ("__init__.py", "data.py", "run.py", "report.py", "README.md", "requirements.txt"):
        copy(pilot / name, Path("histology/histology_cbmp") / name)
    copy(BASE / "code/cbmp.py", Path("histology/polypgen/cbmp.py"))
    with (BASE / "audit/manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    config = json.loads((BASE / "results/seed17/experiment_config.json").read_text())
    order = {"sample_ids": [r["sample_id"] for r in rows], "n_frames": len(rows),
             "n_groups": len({r["group_id"] for r in rows}), "manifest_digest": config["manifest_digest"],
             "why": "Original portable sample identities/order only; no images or per-sample labels. Rebuild labels and hashes from the user's local archive."}
    (HERE / "cohort_order.json").write_text(json.dumps(order, indent=2), encoding="utf-8")
    encoder = json.loads((EXT / "exp4_dinov2/experiment_config.json").read_text())["encoder"]
    revision = encoder["revision"]
    assets = {"resnet18-f37072fd.pth": {"url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
              "sha256": config["encoder"]["checkpoint_sha256"], "location": "assets/torch/hub/checkpoints/resnet18-f37072fd.pth"}}
    for filename, key in (("model.safetensors", "checkpoint_sha256"), ("config.json", "config_sha256"),
                          ("preprocessor_config.json", "processor_config_sha256")):
        assets[filename] = {"url": f"https://huggingface.co/facebook/dinov2-base/resolve/{revision}/{filename}",
                            "sha256": encoder[key], "location": f"assets/dinov2/{revision}/{filename}"}
    extraction = json.loads((BASE / "extraction_complete.json").read_text())
    manifest = {"files": entries, "archive_sha256": extraction["archive_sha256"],
                "archive_bytes": extraction["archive_bytes"], "assets": assets,
                "dinov2_revision": revision, "reference_packages": {p: importlib.metadata.version(p) for p in PACKAGES},
                "cohort_order_sha256": sha256(HERE / "cohort_order.json"),
                "raw_data_included": False,
                "notes": "This captures code, protocols, reviewed mapping metadata, checkpoint identities, and ordering. Running data/results are never copied or changed."}
    (HERE / "sources_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {len(entries)} source/metadata files under {destination}; no raw dataset images copied.")


if __name__ == "__main__":
    main()
