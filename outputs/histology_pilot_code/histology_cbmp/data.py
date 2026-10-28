"""Audited local preparation of the real Kather 2016 / TFDS histology tiles.

There is no synthetic-data fallback.  This dataset does not supply patient or
center identifiers through its folder layout or TFDS feature schema; an image
manifest therefore cannot establish patient-disjoint or cross-center evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any
import urllib.request
import zipfile

from PIL import Image

CLASS_NAMES = (
    "01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO",
    "05_DEBRIS", "06_MUCOSA", "07_ADIPOSE", "08_EMPTY",
)
OFFICIAL_URL = (
    "https://zenodo.org/records/53169/files/"
    "Kather_texture_2016_image_tiles_5000.zip?download=1"
)
OFFICIAL_MD5 = "0ddbebfc56344752028fda72602aaade"
SOURCE_URLS = (
    "https://zenodo.org/records/53169",
    "https://www.tensorflow.org/datasets/catalog/colorectal_histology",
    "https://doi.org/10.1038/srep27988",
)
SCHEMA_VERSION = 1
TFDS_SHARD_LENGTHS = (2500, 2500)
TFDS_PER_CLASS = 625
TFDS_LABEL_NAMES = ("tumor", "stroma", "complex", "lympho", "debris", "mucosa", "adipose", "empty")
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
FIELDS = (
    "sample_id", "image_path", "label", "class_name", "sha256",
    "pixel_sha256", "width", "height", "mode",
)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_md5: str = OFFICIAL_MD5) -> str:
    """Verify the entire archive before trusting its contents."""
    actual = file_hash(Path(path), "md5")
    if actual.lower() != expected_md5.lower():
        raise ValueError(f"Archive MD5 mismatch: expected {expected_md5}, got {actual}")
    return actual


def download_archive(
    archive_path: Path, *, url: str = OFFICIAL_URL,
    expected_md5: str = OFFICIAL_MD5,
) -> Path:
    """Download atomically, requiring the published checksum; reuse verified ZIPs."""
    archive_path = Path(archive_path).resolve()
    if archive_path.exists():
        verify_archive(archive_path, expected_md5)
        return archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".part")
    # Exclusive creation prevents overwriting a previous interrupted download.
    with partial.open("xb") as destination:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "histology-cbmp/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        except BaseException:
            destination.close()
            partial.unlink(missing_ok=True)
            raise
    try:
        verify_archive(partial, expected_md5)
        if archive_path.exists():
            raise FileExistsError(f"Archive appeared during download: {archive_path}")
        partial.rename(archive_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return archive_path


def safe_extract_zip(archive_path: Path, destination: Path) -> Path:
    """Preflight every ZIP path, then extract without overwriting existing files.

    Call ``verify_archive`` separately when extracting the official release.
    Existing directory containers are allowed, existing files are never replaced.
    """
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        plans: list[tuple[zipfile.ZipInfo, Path]] = []
        seen: set[str] = set()
        for info in archive.infolist():
            name = info.orig_filename
            parts = PurePosixPath(name).parts
            if (not parts or "\x00" in name or name.startswith(("/", "\\")) or "\\" in name
                    or any(part in {"..", "."} or ":" in part for part in parts)
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            key = "/".join(parts).casefold()
            if key in seen:
                raise ValueError(f"Duplicate ZIP path: {name!r}")
            seen.add(key)
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"ZIP member escapes destination: {name!r}")
            if target.exists() and (not info.is_dir() or not target.is_dir()):
                raise FileExistsError(f"Extraction would overwrite: {target}")
            for ancestor in target.parents:
                if ancestor == destination.parent:
                    break
                if ancestor.exists() and not ancestor.is_dir():
                    raise FileExistsError(f"ZIP ancestor is a file: {ancestor}")
            plans.append((info, target))
        if not plans:
            raise ValueError("ZIP archive is empty")
        # Reject a file used as another member's parent before writing anything.
        file_keys = {
            "/".join(PurePosixPath(info.filename).parts).casefold()
            for info, _ in plans if not info.is_dir()
        }
        for info, _ in plans:
            parts = PurePosixPath(info.filename).parts
            if any("/".join(parts[:i]).casefold() in file_keys for i in range(1, len(parts))):
                raise ValueError(f"ZIP file/directory conflict: {info.filename!r}")
        destination.mkdir(parents=True, exist_ok=True)
        for info, target in plans:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return destination


def _locate_image_root(data_root: Path) -> Path:
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_root}")
    if any((data_root / name).is_dir() for name in CLASS_NAMES):
        return data_root
    candidates = sorted({
        p.parent for p in data_root.rglob(CLASS_NAMES[0])
        if p.is_dir() and "__MACOSX" not in p.relative_to(data_root).parts
    })
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one Kather class-folder root, found {len(candidates)} in {data_root}. "
            "A TFRecord shard is not an extracted image dataset."
        )
    return candidates[0]


def _tfds_metadata(tfds_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    info = json.loads((tfds_dir / "dataset_info.json").read_text(encoding="utf-8"))
    if (info.get("name") != "colorectal_histology" or info.get("version") != "2.0.0"
            or info.get("fileFormat") != "tfrecord"):
        raise ValueError("Expected the colorectal_histology/2.0.0 TFRecord dataset")
    splits = info.get("splits", [])
    if (len(splits) != 1 or splits[0].get("name") != "train"
            or tuple(map(int, splits[0].get("shardLengths", []))) != TFDS_SHARD_LENGTHS):
        raise ValueError("TFDS metadata must describe both complete 2500-record train shards")
    features = json.loads((tfds_dir / "features.json").read_text(encoding="utf-8"))
    definition = features.get("featuresDict", {}).get("features", {})
    image = definition.get("image", {}).get("image", {})
    if (set(definition) != {"filename", "image", "label"}
            or tuple(map(int, image.get("shape", {}).get("dimensions", []))) != (150, 150, 3)
            or image.get("dtype") != "uint8"
            or int(definition.get("label", {}).get("classLabel", {}).get("numClasses", 0)) != 8):
        raise ValueError("TFDS feature metadata does not match RGB150x150/eight-class histology")
    labels = tuple((tfds_dir / "label.labels.txt").read_text(encoding="utf-8").splitlines())
    if labels != TFDS_LABEL_NAMES:
        raise ValueError("TFDS label order differs from the expected eight histology classes")
    shards = [tfds_dir / f"colorectal_histology-train.tfrecord-{i:05d}-of-00002" for i in range(2)]
    present = set(tfds_dir.glob("colorectal_histology-*.tfrecord-*"))
    if set(shards) != present or any(not p.is_file() for p in shards):
        raise ValueError("Both complete TFDS shard files are required; a single downloaded shard is incomplete")
    return info, shards


def import_tfds_records(tfds_dir: Path, destination: Path) -> Path:
    """Import the complete local TFDS v2 cache without tensorflow_datasets.

    TensorFlow's TFRecord reader validates record CRCs. Every Example, class
    count, image and shard length is checked before the staging directory is
    moved into place. Source filenames are provenance, never output paths.
    The resulting PNG tiles preserve decoded RGB pixels losslessly.
    """
    tfds_dir, destination = Path(tfds_dir).resolve(), Path(destination).resolve()
    _, shards = _tfds_metadata(tfds_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"TFDS import destination must be new or empty: {destination}")
    # TensorFlow is optional for users who ingest the original extracted ZIP.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required only to import existing TFRecord files") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in CLASS_NAMES}
    source_records: list[dict[str, Any]] = []
    shard_hashes = {p.name: file_hash(p) for p in shards}
    metadata_hashes = {name: file_hash(tfds_dir / name) for name in ("dataset_info.json", "features.json", "label.labels.txt")}
    with tempfile.TemporaryDirectory(prefix="_tfds-import-", dir=destination.parent) as scratch:
        scratch_path = Path(scratch).resolve()
        if not scratch_path.is_relative_to(destination.parent):
            raise ValueError("Unexpected import staging directory")
        stage = scratch_path / "tiles"
        stage.mkdir()
        for name in CLASS_NAMES:
            (stage / name).mkdir()
        index = 0
        for shard, expected in zip(shards, TFDS_SHARD_LENGTHS):
            records = 0
            try:
                for raw in tf.data.TFRecordDataset(str(shard), num_parallel_reads=1):
                    example = tf.train.Example.FromString(bytes(raw.numpy()))
                    values = example.features.feature
                    if set(values) != {"filename", "image", "label"}:
                        raise ValueError("Unexpected TFRecord Example features")
                    filename_values = values["filename"].bytes_list.value
                    image_values = values["image"].bytes_list.value
                    label_values = values["label"].int64_list.value
                    if len(filename_values) != 1 or len(image_values) != 1 or len(label_values) != 1:
                        raise ValueError("Expected exactly one filename, image and label per Example")
                    filename = filename_values[0].decode("utf-8")
                    if (not filename or filename in {".", ".."}
                            or any(character in filename for character in ("/", "\\", ":", "\x00"))):
                        raise ValueError("Unsafe source filename in TFRecord")
                    label = int(label_values[0])
                    if not 0 <= label < len(CLASS_NAMES):
                        raise ValueError(f"TFRecord label out of range: {label}")
                    encoded = bytes(image_values[0])
                    relative = Path(CLASS_NAMES[label]) / f"{index:05d}.png"
                    with Image.open(io.BytesIO(encoded)) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (150, 150) or getattr(image, "n_frames", 1) != 1:
                            raise ValueError(f"TFRecord image must be single RGB150x150: {filename}")
                        image.save(stage / relative, format="PNG")
                    counts[CLASS_NAMES[label]] += 1
                    source_records.append({
                        "sample_id": relative.as_posix(), "source_filename": filename,
                        "source_shard": shard.name, "source_record_index": records,
                        "source_encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                    })
                    records += 1
                    index += 1
                    if records > expected:
                        raise ValueError(f"Too many records in {shard.name}")
            except (tf.errors.OpError, OSError) as exc:
                raise ValueError(f"Corrupt/unreadable TFRecord shard {shard.name}: {exc}") from exc
            if records != expected:
                raise ValueError(f"Incomplete shard {shard.name}: expected {expected}, got {records}")
        if any(count != TFDS_PER_CLASS for count in counts.values()):
            raise ValueError(f"TFDS import requires {TFDS_PER_CLASS} images per class; found {counts}")
        # Detect concurrent source modification rather than merely recording an
        # initial hash for files that may have changed while they were read.
        if any(file_hash(p) != shard_hashes[p.name] for p in shards):
            raise ValueError("A TFRecord shard changed during import")
        mapping_path = stage / "tfds_source_records.csv"
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(source_records[0]))
            writer.writeheader()
            writer.writerows(source_records)
        provenance = {
            "import_format": "TFDS_TFRecord", "dataset": "colorectal_histology", "version": "2.0.0",
            "tfds_dir": str(tfds_dir), "source_urls": list(SOURCE_URLS),
            "shard_sha256": shard_hashes, "metadata_sha256": metadata_hashes,
            "source_records_sha256": file_hash(mapping_path), "class_counts": counts,
            "record_count": index, "pixel_conversion": "Lossless RGB PNG encoding",
            "checksum_scope": "Local file SHA256 provenance; not an official archive MD5 verification",
        }
        (stage / "tfds_import_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            # Only an empty directory was accepted above; rmdir cannot remove files.
            destination.rmdir()
        stage.rename(destination)
    return destination


def build_manifest(
    data_root: Path, manifest_path: Path, *, require_complete: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a deterministic manifest and audit; reject invalid images/labels.

    Full-release mode requires all 5,000 real RGB 150x150 tiles, 625 per class.
    Incomplete mode is explicit and never changes the underlying images.
    ``sha256`` hashes file bytes, ``pixel_sha256`` hashes decoded RGB pixels.
    Both duplicate groups must be kept together when constructing image splits.
    """
    root = _locate_image_root(Path(data_root))
    if provenance is None and (root / "tfds_import_provenance.json").is_file():
        provenance = json.loads((root / "tfds_import_provenance.json").read_text(encoding="utf-8"))
    manifest_path = Path(manifest_path).resolve()
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    errors: list[str] = []
    counts = {name: 0 for name in CLASS_NAMES}
    byte_labels: dict[str, int] = {}
    pixel_labels: dict[str, int] = {}
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        sample_id = relative.as_posix()
        if "__MACOSX" in relative.parts or path.name.startswith("._"):
            excluded.append({"sample_id": sample_id, "reason": "system_metadata"})
            continue
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root))):
            errors.append(f"Symlink image is unsupported: {sample_id}")
            continue
        if len(relative.parts) != 2 or relative.parts[0] not in CLASS_NAMES:
            errors.append(f"Image outside an expected class folder: {sample_id}")
            continue
        class_name = relative.parts[0]
        label = CLASS_NAMES.index(class_name)
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (150, 150):
                    raise ValueError(f"expected RGB 150x150, got {image.mode} {image.size}")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("expected a single-frame image")
                pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            sha = file_hash(path)
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid image {sample_id}: {exc}")
            continue
        conflicting = False
        for digest, known, kind in ((sha, byte_labels, "byte"), (pixel_sha, pixel_labels, "pixel")):
            if digest in known and known[digest] != label:
                errors.append(f"Conflicting class labels for {kind}-identical image: {sample_id}")
                conflicting = True
            known[digest] = label
        if conflicting:
            continue
        counts[class_name] += 1
        rows.append({
            "sample_id": sample_id, "image_path": str(path.resolve()), "label": label,
            "class_name": class_name, "sha256": sha, "pixel_sha256": pixel_sha,
            "width": 150, "height": 150, "mode": "RGB",
        })
    complete = all(count == 625 for count in counts.values())
    if require_complete and not complete:
        errors.append(f"Official release requires 625 images per class (5000 total); found {counts}")
    if not rows:
        errors.append("No valid real dataset images found")
    audit = {
        "schema_version": SCHEMA_VERSION, "dataset": "Kather2016_colorectal_histology",
        "data_root": str(root), "source_urls": list(SOURCE_URLS),
        "provenance": provenance or {}, "class_names": list(CLASS_NAMES),
        "class_counts": counts, "sample_count": len(rows), "complete_release": complete,
        "require_complete": require_complete, "valid": not errors,
        "patient_metadata_available": False, "center_metadata_available": False,
        "evaluation_limit": (
            "Image-level pilot only. No verified patient, slide, or center IDs. "
            "Image-disjoint splits do not establish patient-disjoint or cross-center generalization."
        ),
        "byte_duplicate_count": len(rows) - len({r["sha256"] for r in rows}),
        "pixel_duplicate_count": len(rows) - len({r["pixel_sha256"] for r in rows}),
        "excluded": excluded, "errors": errors,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit["manifest_sha256"] = file_hash(manifest_path)
    audit_path = manifest_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"Dataset audit failed ({len(errors)} issue(s)); inspect {audit_path}")
    return audit


def load_manifest(manifest_path: Path, *, require_complete: bool = True) -> list[dict[str, Any]]:
    """Load a previously audited manifest, requiring unchanged manifest bytes."""
    path = Path(manifest_path)
    audit = json.loads(path.with_suffix(".audit.json").read_text(encoding="utf-8"))
    if audit.get("schema_version") != SCHEMA_VERSION or not audit.get("valid"):
        raise ValueError("Manifest audit is invalid or unsupported; rebuild the manifest")
    if require_complete and not audit.get("complete_release"):
        raise ValueError("Manifest is incomplete; explicitly allow an incomplete pilot if intended")
    if file_hash(path) != audit.get("manifest_sha256"):
        raise ValueError("Manifest changed after auditing; rebuild the manifest")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected manifest columns")
        rows = list(reader)
    for row in rows:
        row["label"] = int(row["label"])
        row["width"], row["height"] = int(row["width"]), int(row["height"])
    if len(rows) != audit["sample_count"]:
        raise ValueError("Manifest row count differs from audited count")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted class folders or their parent")
    parser.add_argument("--tfds-dir", type=Path, help="Complete local colorectal_histology/2.0.0 TFRecord cache")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicit image-level subset pilot")
    parser.add_argument("--download", action="store_true", help="Download the official checksum-verified ZIP")
    parser.add_argument("--archive", type=Path, help="Local official ZIP; default beside manifest in data/")
    parser.add_argument("--extract-dir", type=Path, help="New extraction folder, default beside archive")
    args = parser.parse_args()
    if args.tfds_dir and (args.download or args.archive or args.data_root):
        parser.error("--tfds-dir cannot be combined with --download, --archive, or --data-root")
    if args.data_root and (args.download or args.archive):
        parser.error("--data-root cannot be combined with --download or --archive")
    provenance: dict[str, Any] | None = None
    try:
        if args.tfds_dir:
            root = import_tfds_records(args.tfds_dir, args.extract_dir or args.manifest.parent / "data" / "from_tfds")
            provenance = json.loads((root / "tfds_import_provenance.json").read_text(encoding="utf-8"))
        elif args.download or args.archive:
            archive = args.archive or args.manifest.parent / "data" / "Kather_texture_2016_image_tiles_5000.zip"
            if args.download:
                download_archive(archive)
            checksum = verify_archive(archive)
            destination = args.extract_dir or archive.parent / "extracted"
            root = safe_extract_zip(archive, destination)
            provenance = {"archive": str(archive.resolve()), "archive_md5": checksum, "archive_url": OFFICIAL_URL}
        elif args.data_root:
            root = args.data_root
        else:
            parser.error("provide --data-root, --tfds-dir, --archive, or --download")
        audit = build_manifest(root, args.manifest, require_complete=not args.allow_incomplete, provenance=provenance)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({key: audit[key] for key in ("sample_count", "class_counts", "complete_release", "evaluation_limit")}, indent=2))


if __name__ == "__main__":
    main()
