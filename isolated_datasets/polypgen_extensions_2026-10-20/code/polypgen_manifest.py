"""Annotation-based indexing of the official PolypGen v3 archive.

Source: Synapse syn26376615/wiki/613312, compiled entity syn45200214.
Positive sequence membership is NOT a frame-level presence label.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_SCHEMA_VERSION = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CENTER_RE = re.compile(r"^data[_-]c([1-6])$", re.I)
SEQUENCE_RE = re.compile(r"^seq[_-]?(\d+)(_neg)?$", re.I)
ORIGINAL_RE = re.compile(r"^images[_-](c[1-6]|seq[_-]?\d+)$", re.I)
FIELDS = ["sample_id", "image_path", "relative_path", "center", "group_id",
          "group_id_original", "group_source", "sequence_id", "center_source",
          "label", "label_source", "mask_path", "bbox_path", "sha256",
          "mask_sha256", "bbox_sha256", "pixel_sha256"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _sequence_id(value: str) -> str:
    match = SEQUENCE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid sequence ID: {value}")
    return f"seq{int(match.group(1))}" + ("_neg" if match.group(2) else "")


def _load_sequence_metadata(path: Path | None) -> dict:
    if path is None:
        return {}
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"sequence_id", "center", "source"}.issubset(reader.fieldnames or []):
            raise ValueError("Sequence metadata needs sequence_id,center,source columns; patient_id is optional")
        for row in reader:
            seq = _sequence_id(row["sequence_id"])
            center = row["center"].strip().upper()
            if not re.fullmatch(r"C[1-6]", center) or not row["source"].strip():
                raise ValueError(f"Sequence {seq} requires a verified C1..C6 center and provenance source")
            if seq in result:
                raise ValueError(f"Duplicate sequence mapping: {seq}")
            result[seq] = {"center": center, "source": row["source"].strip(),
                           "patient_id": row.get("patient_id", "").strip()}
    return result


def _context(path: Path, root: Path) -> tuple[str | None, str, bool]:
    parts = (root.name,) + path.relative_to(root).parts[:-1]
    centers = {f"C{m.group(1)}" for part in parts if (m := CENTER_RE.fullmatch(part))}
    original_match = ORIGINAL_RE.fullmatch(path.parent.name)
    if original_match and re.fullmatch(r"c[1-6]", original_match.group(1), re.I):
        centers.add(original_match.group(1).upper())
    if len(centers) > 1:
        raise ValueError("Conflicting center markers in original-image path")
    sequences = {_sequence_id(part) for part in parts if SEQUENCE_RE.fullmatch(part)}
    if original_match and original_match.group(1).lower().startswith("seq"):
        sequences.add(_sequence_id(original_match.group(1)))
    if len(sequences) > 1:
        raise ValueError("Conflicting sequence markers in original-image path")
    seq = next(iter(sequences), "")
    negative = bool(seq.endswith("_neg")) or any(
        _key(part) in {"negative", "negativeonly", "negativeframes", "negativesamples"} for part in parts
    )
    return next(iter(centers), None), seq, negative


def _is_original(path: Path) -> bool:
    # Whitelist image leaves. In particular masks_C*, bbox_image_C*, and the
    # pooled imagesAll_positive copy are not input directories.
    if ORIGINAL_RE.fullmatch(path.parent.name) or _key(path.parent.name) in {"originalimages", "images"}:
        return True
    match = SEQUENCE_RE.fullmatch(path.parent.name)
    return bool(match and match.group(2))  # unannotated negative frames directly in seqN_neg


def _annotation_file(image: Path, kind: str, folder_cache: dict) -> Path | None:
    container = image.parent.parent
    match = ORIGINAL_RE.fullmatch(image.parent.name)
    suffix = match.group(1) if match else ""
    allowed = {"masks", "mask", "gt", "groundtruth"} if kind == "mask" else {"bbox", "bboxes", "boundingboxes"}
    if suffix:
        allowed |= {_key(prefix + suffix) for prefix in (("masks", "mask") if kind == "mask" else ("bbox", "bboxes"))}
    if container not in folder_cache:
        folder_cache[container] = {p: list(p.iterdir()) for p in container.iterdir() if p.is_dir()}
    stems = {image.stem.lower()}
    if kind == "mask":
        stems |= {image.stem.lower() + "_mask", "mask_" + image.stem.lower()}
    matches = [p for folder, files in folder_cache[container].items() if _key(folder.name) in allowed
               for p in files if p.is_file() and p.stem.lower() in stems
               and ((p.suffix.lower() in IMAGE_EXTENSIONS) if kind == "mask" else p.suffix.lower() == ".txt")]
    if len(matches) > 1:
        raise ValueError(f"Multiple matching {kind} annotations")
    return matches[0] if matches else None


def _bbox_presence(path: Path) -> int:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for line in lines:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Expected VOC ground truth: class x1 y1 x2 y2")
        x1, y1, x2, y2 = map(float, fields[-4:])
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError("Malformed/nonpositive bounding box")
    return int(bool(lines))


def _label(image: Path, negative: bool, threshold: int, min_pixels: int, folder_cache: dict) -> dict:
    mask = _annotation_file(image, "mask", folder_cache)
    bbox = _annotation_file(image, "bbox", folder_cache)
    observed = {}
    if mask:
        with Image.open(mask) as mask_image, Image.open(image) as original:
            if mask_image.size != original.size:
                raise ValueError("Mask/image dimensions disagree")
            occupancy = np.count_nonzero(np.asarray(mask_image.convert("L")) > threshold)
            observed["mask"] = int(occupancy >= min_pixels)
    if bbox:
        observed["bbox"] = _bbox_presence(bbox)
    if len(set(observed.values())) > 1:
        raise ValueError("Mask/bounding-box presence disagreement")
    if negative and any(observed.values()):
        raise ValueError("Negative-only sequence has a positive annotation")
    if not observed and not negative:
        raise ValueError("Missing annotation; absence of a file is not a negative label")
    return {"label": next(iter(observed.values()), 0),
            "label_source": "+".join(sorted(observed)) if observed else "documented_negative_only_sequence",
            "mask_path": str(mask.resolve()) if mask else "", "bbox_path": str(bbox.resolve()) if bbox else "",
            "mask_sha256": _sha256(mask) if mask else "", "bbox_sha256": _sha256(bbox) if bbox else ""}


def _merge_duplicate_groups(rows: list[dict]) -> list[list[str]]:
    # Union entire sequences/patients connected by duplicate images. Replacing
    # only duplicated rows would split the rest of a sequence across partitions.
    parent = {row["group_id"]: row["group_id"] for row in rows}

    def find(group):
        while group != parent[group]:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    by_hash = defaultdict(list)
    for row in rows:
        by_hash[("bytes", row["sha256"])].append(row)
        by_hash[("pixels", row["pixel_sha256"])].append(row)
    for matches in by_hash.values():
        if len({row["label"] for row in matches}) > 1:
            raise ValueError("Identical image content has contradictory labels; review annotations before training")
        representative = find(matches[0]["group_id"])
        for row in matches[1:]:
            other = find(row["group_id"])
            if representative != other:
                low, high = sorted((representative, other))
                parent[high] = low
                representative = low
    for row in rows:
        row["group_id_original"] = row["group_id"]
        row["group_id"] = find(row["group_id"])
    duplicate_sets = {tuple(sorted(row["relative_path"] for row in matches))
                      for matches in by_hash.values() if len(matches) > 1}
    return [list(paths) for paths in sorted(duplicate_sets)]


def build_manifest(data_root: Path, output_csv: Path, sequence_metadata: Path | None = None,
                   sequence_only: bool = False, mask_threshold: int = 127,
                   min_foreground_pixels: int = 1, verified_sequences_only: bool = False) -> dict:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PolypGen directory does not exist: {root}")
    if not 0 <= mask_threshold <= 254 or min_foreground_pixels < 1:
        raise ValueError("Invalid mask occupancy settings")
    mappings = _load_sequence_metadata(sequence_metadata)
    if verified_sequences_only and (not sequence_only or not mappings):
        raise ValueError("Verified-subset auditing requires sequence_only and nonempty reviewed sequence metadata")
    candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not candidates:
        raise FileNotFoundError(f"No images under {root}; synthetic fallback is disabled")
    rows, rejected, unresolved, folder_cache = [], [], set(), {}
    for image in candidates:
        relative = image.relative_to(root).as_posix()
        if "__MACOSX" in image.relative_to(root).parts or image.name.startswith("._"):
            rejected.append({"path": relative, "reason": "system_metadata"})
            continue
        if not _is_original(image):
            rejected.append({"path": relative, "reason": "not_original_image_leaf"})
            continue
        try:
            center, seq, negative = _context(image, root)
            if sequence_only and not seq:
                rejected.append({"path": relative, "reason": "still_frame_excluded_by_sequence_only"})
                continue
            if verified_sequences_only and seq not in mappings:
                rejected.append({"path": relative, "reason": "sequence_excluded_by_verified_metadata_policy"})
                continue
            mapping = mappings.get(seq)
            if mapping:
                if center and center != mapping["center"]:
                    raise ValueError("Sequence mapping conflicts with folder center")
                center = mapping["center"]
            if not center:
                if seq:
                    unresolved.add(seq)
                raise ValueError("unresolved_sequence_center: provide reviewed sequence_metadata CSV")
            if seq:
                patient = mapping.get("patient_id", "") if mapping else ""
                group = f"patient:{patient}" if patient else f"sequence:{seq}"
                group_source = "reviewed_patient_metadata" if patient else "sequence_folder"
            else:
                group, group_source = f"image:{relative}", "image_only_no_sequence_id"
            label = _label(image, negative, mask_threshold, min_foreground_pixels, folder_cache)
            with Image.open(image) as original:
                rgb = original.convert("RGB")
                pixel_digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            rows.append({"sample_id": hashlib.sha1(relative.encode()).hexdigest(),
                         "image_path": str(image.resolve()), "relative_path": relative,
                         "center": center, "group_id": group, "group_id_original": group,
                         "group_source": group_source, "sequence_id": seq,
                         "center_source": mapping["source"] if mapping else "official_center_directory",
                         **label, "sha256": _sha256(image), "pixel_sha256": pixel_digest})
        except (ValueError, OSError) as exc:
            rejected.append({"path": relative, "reason": str(exc)})
    duplicates = _merge_duplicate_groups(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    rejected_path = output.with_suffix(".rejected.csv")
    with rejected_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(rejected)
    audit = {"schema_version": MANIFEST_SCHEMA_VERSION, "data_root": str(root),
             "manifest": str(output.resolve()), "manifest_sha256": _sha256(output), "samples_indexed": len(rows),
             "samples_rejected": len(rejected), "rejection_counts": dict(Counter(r["reason"] for r in rejected)),
             "unresolved_sequences": sorted(unresolved), "sequence_only": sequence_only,
             "verified_sequences_only": verified_sequences_only,
             "included_sequence_ids": sorted(mappings) if verified_sequences_only else None,
             "scope": ("Preliminary subset restricted to independently verified sequence-center mappings; "
                       "not a complete-release evaluation" if verified_sequences_only else "Full requested image/sequence scope"),
             "sequence_metadata": str(Path(sequence_metadata).resolve()) if sequence_metadata else None,
             "sequence_metadata_sha256": _sha256(Path(sequence_metadata)) if sequence_metadata else None,
             "mask_threshold": mask_threshold, "min_foreground_pixels": min_foreground_pixels,
             "class_counts": {str(c): sum(r["label"] == c for r in rows) for c in (0, 1)},
             "center_class_counts": {center: {str(c): sum(r["center"] == center and r["label"] == c for r in rows)
                                                   for c in (0, 1)} for center in sorted({r["center"] for r in rows})},
             "groups_without_sequence_metadata": sum(r["group_source"] == "image_only_no_sequence_id" for r in rows),
             "duplicate_content_groups": duplicates,
             "warning": "Frame labels describe annotated polyp presence. Sequence grouping does not establish independence between different sequences from the same patient."}
    normal_exclusions = {"not_original_image_leaf", "still_frame_excluded_by_sequence_only", "system_metadata",
                         "sequence_excluded_by_verified_metadata_policy"}
    audit["blocking_rejections"] = sum(r["reason"] not in normal_exclusions for r in rejected)
    if unresolved:
        mapping_template = output.with_suffix(".sequence_metadata_template.csv")
        with mapping_template.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sequence_id", "center", "source", "patient_id"])
            writer.writeheader()
            writer.writerows({"sequence_id": seq, "center": "", "source": "", "patient_id": ""}
                             for seq in sorted(unresolved))
        audit["sequence_metadata_template"] = str(mapping_template.resolve())
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def load_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Legacy/incomplete manifest: rebuild with the annotation-based audit")
        rows = list(reader)
    audit_path = Path(path).with_suffix(".audit.json")
    if not audit_path.is_file():
        raise ValueError("Missing manifest audit; rebuild with audit_cbmp_data.py")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != MANIFEST_SCHEMA_VERSION or audit.get("manifest_sha256") != _sha256(Path(path)):
        raise ValueError("Manifest changed or uses a legacy audit; rebuild the annotation audit")
    if not rows:
        raise ValueError("Manifest is empty; inspect audit JSON and rejected CSV")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in manifest")
    for row in rows:
        row["label"] = int(row["label"])
    if {r["label"] for r in rows} != {0, 1}:
        raise ValueError("Binary presence classification requires both labels 0 and 1; inspect annotation/center coverage")
    return rows
