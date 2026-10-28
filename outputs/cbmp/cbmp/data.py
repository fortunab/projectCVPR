"""Explicit manifests: labels and group IDs are supplied, never guessed from folders."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

FIELDS = ("sample_id", "image_path", "label", "center", "group_id")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    group_centers = {}
    for row in rows:
        if any(not row[k].strip() for k in FIELDS):
            raise ValueError("Blank IDs, labels, or paths are not permitted")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample_id")
        seen.add(row["sample_id"])
        row["label"] = int(row["label"])
        g, c = row["group_id"], row["center"]
        if g in group_centers and group_centers[g] != c:
            raise ValueError("A group spans centers; reconcile grouping before center splits")
        group_centers[g] = c
        image = Path(row["image_path"])
        row["resolved_path"] = str(image if image.is_absolute() else path.parent / image)
    labels = sorted({r["label"] for r in rows})
    if labels != list(range(len(labels))) or len(labels) < 2:
        raise ValueError("Labels must be contiguous integers 0..C-1, with at least two classes")
    return rows


def audit_manifest(path, output):
    rows = load_manifest(path)
    hashes, decoded = {}, {}
    errors = []
    for r in rows:
        image_path = Path(r["resolved_path"])
        if not image_path.is_file():
            errors.append(f"Missing image: {r['sample_id']}")
            continue
        try:
            file_hash = digest(image_path)
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            for index, value, kind in ((hashes, file_hash, "file"), (decoded, pixel_hash, "decoded pixels")):
                if value in index:
                    errors.append(f"Duplicate {kind}: {index[value]} / {r['sample_id']}")
                index[value] = r["sample_id"]
            r["sha256"] = file_hash
        except (OSError, ValueError) as exc:
            errors.append(f"Unreadable image {r['sample_id']}: {exc}")
    counts = Counter((r["center"], r["label"]) for r in rows)
    report = {
        "manifest_sha256": digest(path), "samples": len(rows), "errors": errors,
        "center_class_counts": [{"center": c, "label": y, "images": n,
            "groups": len({r['group_id'] for r in rows if r['center'] == c and r['label'] == y})}
            for (c, y), n in sorted(counts.items())],
        "limitations": ["Exact file/pixel checks do not detect all near duplicates.",
                        "Labels and patient/sequence identities require verification against original metadata.",
                        "No synthetic fallback and no directory-based label inference are used."],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError(f"Audit failed with {len(errors)} errors; see {output / 'audit.json'}")
    (output / "verified_images.json").write_text(json.dumps([
        {"sample_id": r["sample_id"], "sha256": r["sha256"]} for r in rows], indent=2), encoding="utf-8")
    return report


def extract(manifest, audit, output, model_name, checkpoint, device="cpu", batch_size=16):
    """Load a local checkpoint only; timm defines its deterministic input transform."""
    import torch
    import timm
    from timm.data import resolve_model_data_config, create_transform
    rows = load_manifest(manifest)
    report = json.loads((Path(audit) / "audit.json").read_text(encoding="utf-8"))
    if report["errors"] or report["manifest_sha256"] != digest(manifest):
        raise ValueError("Successful audit of this exact manifest required")
    expected = json.loads((Path(audit) / "verified_images.json").read_text(encoding="utf-8"))
    if [r["sample_id"] for r in expected] != [r["sample_id"] for r in rows]:
        raise ValueError("Audit image order mismatch")
    for row, item in zip(rows, expected):
        if digest(row["resolved_path"]) != item["sha256"]:
            raise ValueError("Image changed after audit")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = timm.create_model(model_name, pretrained=False, checkpoint_path=str(checkpoint))
    model.reset_classifier(0)
    model.eval().to(device)
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    features = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                with Image.open(row["resolved_path"]) as im:
                    images.append(transform(im.convert("RGB")))
            batch = model(torch.stack(images).to(device))
            if not isinstance(batch, torch.Tensor) or batch.ndim != 2:
                raise ValueError("Encoder must emit one N x d embedding matrix")
            features.append(batch.cpu().numpy())
    from .models import normalize
    z = normalize(np.concatenate(features))
    provenance = {"manifest_sha256": digest(manifest), "checkpoint_sha256": digest(checkpoint),
        "checkpoint": str(Path(checkpoint).resolve()), "model": model_name,
        "transform": str(transform), "data_config": config, "torch": torch.__version__,
        "timm": timm.__version__, "device": device, "audit": report,
        "data_kind": "real_manifest", "pretraining_overlap": "not established; investigate before publication"}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=z, sample_id=np.array([r['sample_id'] for r in rows]),
        y=np.array([r['label'] for r in rows]), centers=np.array([r['center'] for r in rows]),
        groups=np.array([r['group_id'] for r in rows]), provenance=json.dumps(provenance))


def load_features(path, allow_synthetic=False):
    with np.load(path, allow_pickle=False) as f:
        required = {"x", "y", "centers", "groups", "sample_id", "provenance"}
        if not required.issubset(f.files):
            raise ValueError("Feature cache lacks metadata/provenance")
        data = {key: f[key].copy() for key in required if key != "provenance"}
        provenance = json.loads(str(f["provenance"]))
    if provenance.get("data_kind") != "real_manifest" and not allow_synthetic:
        raise ValueError("Synthetic/unverified cache requires --allow-synthetic")
    from .models import normalize
    data["x"] = normalize(data["x"])
    n = len(data["x"])
    if any(data[k].ndim != 1 or len(data[k]) != n for k in ("y", "centers", "groups", "sample_id")):
        raise ValueError("Feature metadata shape mismatch")
    if len(set(data["sample_id"].tolist())) != n:
        raise ValueError("Duplicate feature sample IDs")
    if any(not str(v).strip() for k in ("centers", "groups", "sample_id") for v in data[k]):
        raise ValueError("Blank metadata IDs")
    labels = np.unique(data["y"])
    if not np.array_equal(labels, np.arange(len(labels))) or len(labels) < 2:
        raise ValueError("Labels must be 0..C-1")
    for g in np.unique(data["groups"]):
        if len(np.unique(data["centers"][data["groups"] == g])) != 1:
            raise ValueError("A group spans centers")
    return data, provenance
