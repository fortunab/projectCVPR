"""Prespecified independent CAMELYON17-WILDS frozen-transfer experiment.

Writes only exp5_camelyon in this extension folder. The immutable core head
implementation in code is shared with the previous experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exp5_camelyon"
PIN = "d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f"
REPO = "wltjr1007/Camelyon17-WILDS"
PREFIX = "camelyon17-patient-cap400-seed2027|"
SEEDS = (17, 29, 43)
SOURCES = (0, 3, 4)
TARGETS = (2, 1)
CAP = 400
MAX_TRANSFER = 12_000_000_000
OFFICIAL = "https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/"
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

# Initialize PyTorch native libraries before PyArrow on Windows. Loading them
# after a complete Arrow image audit can fail c10.dll initialization.
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def guarded(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError("Experiment 5 writes must stay in its new isolated directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path, value):
    guarded(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty experiment table")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with guarded(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def verify_protocol():
    setup = json.loads((ROOT / "setup_complete.json").read_text())
    if sha256(ROOT / "PROTOCOL.md") != setup["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    for filename, checksum in setup["copied_code_sha256"].items():
        if sha256(ROOT / "code" / filename) != checksum:
            raise ValueError(f"Frozen core changed: {filename}")
    return setup


def public_request(url, *, headers=None, timeout=60):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"huggingface.co", "worksheets.codalab.org"}:
        raise ValueError("Unexpected public acquisition host")
    req = urllib.request.Request(url, headers={"User-Agent": "CBMP-Research/1.0", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def acquire(workers=2):
    """Pinned full-shard acquisition: streamed, checksummed, resumable by files."""
    verify_protocol()
    status = {"author_url": OFFICIAL, "mirror_repository": REPO, "mirror_revision": PIN,
              "community_mirror_not_author_distribution": True, "credentials_supplied": False}
    try:
        with public_request(OFFICIAL, headers={"Range": "bytes=0-7"}, timeout=12) as response:
            status["author_probe"] = {"http_status": response.status, "first_eight_bytes_hex": response.read(8).hex()}
        # The frozen protocol permits the mirror if the author source is
        # unavailable. A responsive original source requires separate handling.
        if status["author_probe"]["http_status"] in (200, 206):
            dump_json(OUT / "acquisition/author_probe.json", status)
            raise RuntimeError("Author bundle became accessible; review original-source acquisition before mirror fallback")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status["author_probe"] = {"unavailable": True, "exception": type(exc).__name__, "message": str(exc)}
    dump_json(OUT / "acquisition/author_probe.json", status)
    tree_path = OUT / "acquisition/pinned_tree.json"
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
    else:
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{PIN}/data?expand=false"
        with public_request(url) as response:
            tree = json.load(response)
        dump_json(tree_path, tree)
    shards = [r for r in tree if r["path"].endswith(".parquet")]
    shards.sort(key=lambda r: r["path"])
    if len(shards) != 21 or sum(int(r["size"]) for r in shards) > MAX_TRANSFER:
        raise ValueError("Pinned redistribution shard count/size differs from the frozen access plan")
    if any(not r.get("lfs", {}).get("oid") for r in shards):
        raise ValueError("Every full shard must have a pinned LFS SHA256")
    plan = {"repository": REPO, "revision": PIN, "shards": shards,
            "expected_bytes": sum(int(r["size"]) for r in shards), "download_workers": workers,
            "strategy": "full pinned shards, complete SHA256 verification; metadata columns read before subset image extraction",
            "selection_rule": PREFIX + "original_patch_identity", "cap": CAP,
            "cap_unit": "patient per eligible original partition", "source_hospitals": SOURCES,
            "primary_hospital": 2, "secondary_hospital": 1, "max_download_bytes": MAX_TRANSFER}
    dump_json(OUT / "acquisition/plan.json", plan)
    first = shards[0]
    benchmark_path = OUT / "acquisition/throughput_probe.json"
    probe_new_bytes = 0
    if not benchmark_path.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{PIN}/{first['path']}"
        tick = time.perf_counter()
        with public_request(url, headers={"Range": "bytes=0-8388607"}) as response:
            block = response.read(8_388_608)
            probe = {"bytes": len(block), "seconds": time.perf_counter() - tick,
                     "range_status": response.status, "content_range": response.headers.get("Content-Range"),
                     "path": first["path"], "probe_sha256": hashlib.sha256(block).hexdigest()}
        probe["megabytes_per_second"] = len(block) / 1e6 / probe["seconds"]
        probe_new_bytes = len(block)
        probe["estimated_full_transfer_seconds"] = plan["expected_bytes"] / len(block) * probe["seconds"]
        dump_json(benchmark_path, probe)
        print("[ACQUIRE] throughput " + json.dumps(probe), flush=True)

    def fetch(record):
        dest = guarded(OUT / "download" / Path(record["path"]).name)
        expected = record["lfs"]["oid"]
        if dest.exists():
            if dest.stat().st_size != record["size"] or sha256(dest) != expected:
                raise ValueError(f"Existing shard mismatch: {dest.name}")
            print(f"[ACQUIRE] verified existing {dest.name}", flush=True)
            return {**json.loads(dest.with_suffix(".source.json").read_text()), "newly_transferred_bytes": 0}
        part = dest.with_suffix(".parquet.part")
        # A previous partial is retained as its own artifact; no append to a
        # response without a verified byte-range contract.
        if part.exists():
            part = dest.with_suffix(f".parquet.retry{time.time_ns()}.part")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{PIN}/{record['path']}"
        tick = previous = time.perf_counter()
        received, digest = 0, hashlib.sha256()
        with public_request(url) as response, part.open("xb") as stream:
            if int(response.headers.get("Content-Length", record["size"])) != record["size"]:
                raise ValueError("Pinned shard Content-Length mismatch")
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
                digest.update(block)
                received += len(block)
                if received > record["size"]:
                    raise ValueError("Pinned shard exceeded expected size")
                if time.perf_counter() - previous > 15:
                    print(f"[ACQUIRE] {dest.name} {received/1e6:.1f}/{record['size']/1e6:.1f}MB", flush=True)
                    previous = time.perf_counter()
            final = urllib.parse.urlparse(response.geturl())
        if received != record["size"] or digest.hexdigest() != expected:
            raise ValueError("Incomplete or corrupt pinned shard")
        part.replace(dest)
        result = {"repository_revision": PIN, "source_url": url, "expected_sha256": expected,
                  "sha256": digest.hexdigest(), "bytes": received, "seconds": time.perf_counter()-tick,
                  "newly_transferred_bytes": received,
                  "final_host_path": final.scheme + "://" + final.netloc + final.path,
                  "credentials_supplied": False}
        dump_json(dest.with_suffix(".source.json"), result)
        print(f"[ACQUIRE] COMPLETE {dest.name} {received/1e6:.1f}MB {result['seconds']:.1f}s", flush=True)
        return result

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(fetch, row): row for row in shards}
        for completed in as_completed(tasks):
            records.append(completed.result())
            dump_json(OUT / "acquisition/downloaded.json", records)
    dump_json(OUT / "acquisition/complete.json", {**status, "files": records,
        "new_shard_and_probe_payload_bytes_this_invocation": sum(r["newly_transferred_bytes"] for r in records) + probe_new_bytes,
        "transfer_accounting_scope": "downloaded shard and throughput-probe bodies; excludes tiny API/author probes, HTTP headers and failed partial requests",
        "verified_resident_shard_bytes": sum(r["bytes"] for r in records),
        "total_pinned_shard_bytes": plan["expected_bytes"], "complete": True})
    print("[ACQUIRE] ALL21 PINNED SHARDS COMPLETE", flush=True)


HOSPITAL_NAMES = {"train-center1": 0, "validation-center": 1, "test-center": 2,
                  "train-center2": 3, "train-center3": 4}


def decode_hospitals(values, feature):
    names = feature.get("names")
    if names is None:
        raise ValueError("Parquet must retain per-file ClassLabel hospital names")
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]
    if any(name not in HOSPITAL_NAMES for name in names):
        raise ValueError(f"Unknown hospital ClassLabel names: {names}")
    raw = np.asarray(values, dtype=int)
    if np.any(raw < 0) or np.any(raw >= len(names)):
        raise ValueError("Hospital encoding outside file ClassLabel support")
    return np.array([HOSPITAL_NAMES[names[v]] for v in raw], dtype=int), names


def patch_identity(patient, node, x, y):
    return f"patch_patient_{int(patient):03d}_node_{int(node)}_x_{int(x)}_y_{int(y)}.png"


def sampling_rank(identity):
    return hashlib.sha256((PREFIX + identity).encode("utf-8")).hexdigest()


def select_cohort(frame):
    """No label column is used to choose any patient or patch."""
    eligible = frame[frame["partition"].isin(["source_train", "primary_test", "secondary_test"])].copy()
    eligible["sampling_sha256"] = eligible["sample_id"].map(sampling_rank)
    eligible = eligible.sort_values(["partition", "patient", "sampling_sha256", "sample_id"])
    selected = eligible.groupby(["partition", "patient"], sort=False).head(CAP)
    return selected.sort_values(["partition", "patient", "sampling_sha256"]).reset_index(drop=True)


def validate_cohort(frame):
    if len(frame) == 0 or frame["sample_id"].duplicated().any():
        raise ValueError("Empty cohort or duplicate patch coordinates")
    if np.any(frame["center"].to_numpy() != frame["patient"].to_numpy() // 20):
        raise ValueError("Hospital/patient mapping differs from author definition")
    if np.any(frame["center"].to_numpy() != frame["slide"].to_numpy() // 10):
        raise ValueError("Hospital/slide mapping differs from author definition")
    if frame.groupby("patient")["center"].nunique().max() != 1:
        raise ValueError("Patient crosses hospitals")
    allowed = {"source_train": set(SOURCES), "primary_test": {2}, "secondary_test": {1}}
    for partition, centers in allowed.items():
        part = frame[frame["partition"] == partition]
        if set(part["center"]) != centers:
            raise ValueError("Missing or unexpected hospitals in required partition")
        for hospital in centers:
            if set(part.loc[part["center"] == hospital, "label"]) != {0, 1}:
                raise ValueError(f"Label-blind subset lacks a class in required hospital {hospital}; stop, do not resample")
    grouped = frame.groupby(["partition", "patient"]).size()
    if grouped.max() > CAP:
        raise ValueError("Patient sampling cap exceeded")
    sources = frame[frame["partition"] == "source_train"]
    for hospital in SOURCES:
        train = sources[sources["center"] != hospital]
        val = sources[sources["center"] == hospital]
        if set(train["label"]) != {0, 1} or set(val["label"]) != {0, 1}:
            raise ValueError("Required source-only fold lacks a class")
        if set(train["patient"]) & set(val["patient"]):
            raise ValueError("Patient crosses inner folds")
    if set(sources["patient"]) & set(frame.loc[frame["partition"] != "source_train", "patient"]):
        raise ValueError("Source and target patient overlap")


def audit_metadata():
    import pyarrow.parquet as pq
    verify_protocol()
    if not (OUT / "acquisition/complete.json").exists():
        raise FileNotFoundError("Complete pinned shards are required before metadata audit")
    output_path = OUT / "audit/selected_metadata.csv"
    if (OUT / "audit/metadata_complete.json").exists():
        completed = json.loads((OUT / "audit/metadata_complete.json").read_text())
        if sha256(output_path) != completed["selected_metadata_sha256"]:
            raise ValueError("Saved metadata selection changed after audit")
        df = pd.read_csv(output_path)
        validate_cohort(df)
        return df
    frames, schemas = [], []
    for path in sorted((OUT / "download").glob("*.parquet")):
        file = pq.ParquetFile(path)
        hf = json.loads(file.schema_arrow.metadata[b"huggingface"])
        fields = hf["info"]["features"]
        label_names = fields.get("label", {}).get("names")
        if isinstance(label_names, dict):
            label_names = [label_names[str(i)] for i in range(len(label_names))]
        if label_names != ["non-tumor", "tumor"]:
            raise ValueError("Label ClassLabel encoding must be 0 non-tumor and1 tumor")
        cols = ["label", "center", "image_id", "patient", "node", "x_coord", "y_coord", "slide"]
        if any(c not in file.schema_arrow.names for c in cols):
            raise ValueError("Redistributed metadata fields missing")
        df = file.read(columns=cols, use_threads=False).to_pandas()
        decoded, names = decode_hospitals(df["center"], fields["center"])
        df["redistributed_center_encoded"] = df["center"]
        df["center"] = decoded
        df["shard"] = path.name
        df["shard_row"] = np.arange(len(df))
        split = path.name.split("-")[0]
        df["redistributed_split"] = split
        if split == "train":
            if set(decoded) - set(SOURCES):
                raise ValueError("Target hospital present in original train shard")
            df["partition"] = "source_train"
        elif split == "validation":
            if set(decoded) - set(SOURCES) - {1}:
                raise ValueError("Unexpected validation hospital")
            df["partition"] = np.where(decoded == 1, "secondary_test", "unused_source_id_validation")
        elif split == "test":
            if set(decoded) != {2}:
                raise ValueError("Original test shard must be hospital2 only")
            df["partition"] = "primary_test"
        else:
            raise ValueError("Unknown redistributed split")
        df["sample_id"] = [patch_identity(*r) for r in df[["patient", "node", "x_coord", "y_coord"]].itertuples(index=False, name=None)]
        df["group_id"] = [f"patient:{v:03d}" for v in df["patient"]]
        offset = 0
        rowgroups = []
        for ri in range(file.num_row_groups):
            nrows = file.metadata.row_group(ri).num_rows
            rowgroups.extend([ri] * nrows)
            offset += nrows
        df["row_group"] = rowgroups
        schemas.append({"file": path.name, "rows": len(df), "row_groups": file.num_row_groups,
                        "center_names": names, "arrow_schema": str(file.schema_arrow), "features": fields})
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    if len(all_rows) != 455954 or all_rows["sample_id"].duplicated().any():
        raise ValueError("Redistribution total or unique coordinate identities differ")
    id_duplicates = int(all_rows["image_id"].duplicated().sum())
    split_id_duplicates = int(all_rows.duplicated(["redistributed_split", "image_id"]).sum())
    if split_id_duplicates:
        raise ValueError("Redistribution image_id must at least be unique within each original split")
    if set(all_rows["label"]) != {0, 1}:
        raise ValueError("Labels differ from binary tumor annotation")
    selected = select_cohort(all_rows)
    validate_cohort(selected)
    eligible_patients = all_rows.loc[all_rows["partition"] != "unused_source_id_validation", ["partition", "patient"]].drop_duplicates()
    if set(map(tuple, selected[["partition", "patient"]].to_numpy())) != set(map(tuple, eligible_patients.to_numpy())):
        raise ValueError("Not all eligible patients retained")
    write_csv(OUT / "audit/all_metadata.csv", all_rows.to_dict("records"))
    write_csv(output_path, selected.to_dict("records"))
    dump_json(OUT / "audit/file_schemas.json", schemas)
    summary = selected.groupby(["partition", "center", "label"]).agg(patches=("sample_id", "size"), patients=("patient", "nunique"), slides=("slide", "nunique")).reset_index().to_dict("records")
    full_counts = all_rows.groupby(["partition", "center", "label"]).size().reset_index(name="patches").to_dict("records")
    result = {"selected_patches": len(selected), "selected_patients": int(selected["patient"].nunique()),
              "metadata_rows": len(all_rows), "summary": summary, "full_counts": full_counts,
              "image_id_globally_unique": id_duplicates == 0,
              "image_id_duplicate_count_across_splits": id_duplicates,
              "image_id_unique_within_redistributed_split": True,
              "selection_uses_labels": False, "selection_uses_scores": False,
              "all_eligible_patients_retained": True, "patient_cap": CAP,
              "selected_metadata_sha256": sha256(output_path), "protocol_sha256": sha256(ROOT/"PROTOCOL.md"),
              "patient_boundaries_valid": True, "all_required_classes_present": True,
              "original_ID_validation_rows_excluded": int((all_rows["partition"] == "unused_source_id_validation").sum())}
    dump_json(OUT / "audit/metadata_complete.json", result)
    print("[METADATA AUDIT] " + json.dumps(result), flush=True)
    return selected


def extract_images(frame):
    import pyarrow.parquet as pq
    manifest_path = OUT / "audit/image_manifest.csv"
    if (OUT / "audit/images_complete.json").exists():
        completed = json.loads((OUT / "audit/images_complete.json").read_text())
        if sha256(manifest_path) != completed["manifest_sha256"]:
            raise ValueError("Saved image manifest changed after content audit")
        existing = pd.read_csv(manifest_path)
        if existing["sample_id"].tolist() != frame["sample_id"].tolist():
            raise ValueError("Existing selected image order differs")
        return existing
    records = {}
    for shard_name, rows in frame.groupby("shard", sort=True):
        file = pq.ParquetFile(OUT / "download" / shard_name)
        start_by_rg = np.cumsum([0] + [file.metadata.row_group(ri).num_rows for ri in range(file.num_row_groups)])
        for row_group, selected_rows in rows.groupby("row_group", sort=True):
            content = file.read_row_group(int(row_group), columns=["image"], use_threads=False)["image"].to_pylist()
            offset = start_by_rg[int(row_group)]
            for row in selected_rows.to_dict("records"):
                value = content[int(row["shard_row"])-offset]
                image_bytes = value.get("bytes")
                if image_bytes is None:
                    raise ValueError("Redistributed image bytes missing; never substitute another image")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != (96, 96):
                        raise ValueError("Expected original 96x96 histology patch")
                    pixels = rgb.tobytes()
                path = guarded(OUT / "images" / f"patient_{int(row['patient']):03d}" / row["sample_id"])
                checksum = hashlib.sha256(image_bytes).hexdigest()
                if path.exists():
                    if sha256(path) != checksum:
                        raise ValueError("Existing extracted image bytes differ")
                else:
                    path.write_bytes(image_bytes)
                row.update(image_path=str(path), sha256=checksum, pixel_sha256=hashlib.sha256(pixels).hexdigest(),
                           label_source="CAMELYON17-WILDS central32x32 tumor annotation; pinned redistribution label")
                records[row["sample_id"]] = row
        print(f"[IMAGES] {shard_name}: extracted {len(rows)} selected patches", flush=True)
    ordered = pd.DataFrame([records[s] for s in frame["sample_id"]])
    duplicate_groups = []
    for pixel_hash, duplicates in ordered.groupby("pixel_sha256"):
        if len(duplicates) < 2:
            continue
        if duplicates["label"].nunique() > 1 or duplicates["center"].nunique() > 1:
            raise ValueError("Contradictory label or cross-hospital exact content duplicate: stop")
        if duplicates["patient"].nunique() > 1:
            raise ValueError("Duplicate-linked patients require renewed independent grouping review: stop")
        duplicate_groups.append({"pixel_sha256": pixel_hash, "samples": duplicates["sample_id"].tolist(),
                                 "patient": int(duplicates["patient"].iloc[0])})
    validate_cohort(ordered)
    write_csv(manifest_path, ordered.to_dict("records"))
    dump_json(OUT / "audit/images_complete.json", {"images": len(ordered), "manifest_sha256": sha256(manifest_path),
        "cross_patient_exact_duplicates": 0, "cross_center_exact_duplicates": 0,
        "within_patient_duplicate_groups": duplicate_groups, "patient_groups_unchanged": True,
        "all_images_96x96_RGB": True})
    return ordered


def extract_features(frame, threads=2, batch_size=64):
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18
    torch.set_num_threads(threads)
    weights_path = Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if not weights_path.is_file():
        raise FileNotFoundError("Frozen official cached ImageNet1K_V1 ResNet18 checkpoint required")
    checkpoint_sha = sha256(weights_path)
    if not checkpoint_sha.startswith("f37072fd"):
        raise ValueError("Official ResNet18 checkpoint hash prefix mismatch")
    cache_path = OUT / "features/resnet18_imagenet1k_v1.npz"
    metadata = {"encoder": "torchvision.resnet18", "weights": "IMAGENET1K_V1", "checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": sha256(OUT/"audit/image_manifest.csv"), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "representation": "global_average_pool_before_fc;512dim;L2normalized",
                "preprocessing": str(ResNet18_Weights.IMAGENET1K_V1.transforms()), "eval": True, "frozen": True}
    expected_ids = frame["sample_id"].to_numpy(dtype=str)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if (json.loads(str(saved["metadata"].item())) != metadata or not np.array_equal(saved["sample_ids"], expected_ids)):
                raise ValueError("Feature cache provenance changed")
            features = saved["features"].copy()
        validate_features(features, len(frame))
        return features
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    values = np.empty((len(frame), 512), dtype=np.float32)
    tick = time.perf_counter()
    paths = frame["image_path"].tolist()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            images = []
            for row in frame.iloc[start:start+batch_size].to_dict("records"):
                if sha256(row["image_path"]) != row["sha256"]:
                    raise ValueError("Extracted image changed since audit")
                with Image.open(row["image_path"]) as image:
                    images.append(transform(image.convert("RGB")))
            inputs = torch.stack(images)
            batch = encoder(inputs).cpu().numpy()
            norm = np.linalg.norm(batch, axis=1, keepdims=True)
            if np.any(norm <= 1e-12) or not np.isfinite(batch).all():
                raise ValueError("Invalid frozen embedding")
            values[start:start+len(batch)] = batch/norm
            if start == 0:
                repeated = encoder(inputs).cpu().numpy()
                repeated /= np.linalg.norm(repeated, axis=1, keepdims=True)
                if not np.array_equal(values[:len(batch)], repeated):
                    raise ValueError("Repeated frozen feature extraction differs")
            if start % (batch_size*10) == 0:
                elapsed = time.perf_counter()-tick
                count = start+len(batch)
                print(f"[RESNET18] {count}/{len(frame)} patches;{elapsed:.1f}s;ETA{elapsed/count*(len(frame)-count):.1f}s", flush=True)
    validate_features(values, len(frame))
    with guarded(cache_path).open("xb") as stream:
        np.savez_compressed(stream, features=values, sample_ids=expected_ids, metadata=json.dumps(metadata, sort_keys=True))
    dump_json(OUT/"features/extraction.json", {**metadata, "patches": len(frame), "threads": threads,
        "workers": 0, "batch_size": batch_size, "seconds": time.perf_counter()-tick,
        "deterministic_first_batch": True, "feature_sha256": sha256(cache_path)})
    return values


def validate_features(features, n):
    if (features.shape != (n,512) or features.dtype != np.float32 or not np.isfinite(features).all()
        or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid normalized ResNet18 feature matrix")


def source_folds(frame):
    source = np.flatnonzero(frame["partition"].to_numpy() == "source_train")
    hospital = frame["center"].to_numpy()
    inner = [(source[hospital[source] != c], source[hospital[source] == c]) for c in SOURCES]
    return source, inner


def fit_all_source_heads(features, frame):
    from cbmp import select_temperature_folds
    from cbmp_evaluation import candidate_methods, fit_head, head_scores
    validate_cohort(frame)
    source, inner = source_folds(frame)
    y = frame["label"].to_numpy(dtype=int)
    centers = frame["center"].to_numpy().astype(str)
    groups = frame["group_id"].to_numpy().astype(str)
    plan_path = OUT/"cohort_plan.json"
    if not plan_path.exists():
        raise ValueError("Cohort plan must be saved before first fit")
    if (OUT/"fits/source_heads_frozen.json").exists():
        frozen = json.loads((OUT/"fits/source_heads_frozen.json").read_text())
        if len(frozen["models"]) != 27:
            raise ValueError("Frozen source bank incomplete")
        for model in frozen["models"]:
            if sha256(OUT/model["path"]) != model["sha256"]:
                raise ValueError("Frozen head changed")
        return frozen
    methods = candidate_methods()
    bank, searches = [], []
    for seed in SEEDS:
        fold_cache = {}
        for method, candidates in methods.items():
            final_path = OUT/f"fits/models/seed{seed}/{method}.joblib"
            search_path = OUT/f"fits/search/seed{seed}/{method}.json"
            if final_path.exists() and search_path.exists():
                search = json.loads(search_path.read_text())
                if search["candidate_grid"] != candidates or search["cohort_plan_sha256"] != sha256(plan_path):
                    raise ValueError("Partial source bank incompatible with current frozen experiment")
                bank.append({"seed": seed, "method": method, "path": str(final_path.relative_to(OUT)), "sha256": sha256(final_path)})
                searches.append(search)
                continue
            if final_path.exists() or search_path.exists():
                raise ValueError("Incomplete per-head save: preserve and review")
            tick = time.perf_counter()
            best, records = None, []
            for spec in candidates:
                scores_by_fold, labels_by_fold, bas = [], [], []
                for fi, (tr, va) in enumerate(inner):
                    key = (json.dumps(spec, sort_keys=True), fi, seed+fi)
                    if key not in fold_cache:
                        model = fit_head(spec, features[tr], y[tr], centers[tr], groups[tr], seed+fi)
                        scores = head_scores(model, features[va])
                        fold_cache[key] = scores
                    scores = fold_cache[key]
                    from sklearn.metrics import balanced_accuracy_score
                    bas.append(float(balanced_accuracy_score(y[va], scores.argmax(axis=1))))
                    scores_by_fold.append(scores)
                    labels_by_fold.append(y[va])
                mean = float(np.mean(bas))
                records.append({"configuration": spec, "mean_source_validation_balanced_accuracy": mean,
                                "per_source_validation_fold": bas})
                if best is None or mean > best[0]:
                    best = (mean, spec, scores_by_fold, labels_by_fold)
            tau = select_temperature_folds(best[2], best[3])
            model = fit_head(best[1], features[source], y[source], centers[source], groups[source], seed)
            guarded(final_path)
            joblib.dump({"model": model, "temperature": tau, "configuration": best[1], "seed": seed,
                         "method": method, "source_indices": source.tolist(), "source_hospitals": SOURCES,
                         "protocol_sha256": sha256(ROOT/"PROTOCOL.md"), "cohort_plan_sha256": sha256(plan_path)}, final_path)
            search = {"seed": seed, "method": method, "candidate_grid": candidates, "candidates": records,
                      "selected_configuration": best[1], "temperature": tau, "source_selection_seconds": time.perf_counter()-tick,
                      "cohort_plan_sha256": sha256(plan_path), "source_hospitals": SOURCES,
                      "source_inner_folds": [{"validation_hospital": c, "train": tr.tolist(), "validation": va.tolist()} for c,(tr,va) in zip(SOURCES,inner)],
                      "cache_policy": "exact identical spec, seed and source-fold arrays reused within seed; head math unchanged"}
            dump_json(search_path, search)
            bank.append({"seed": seed, "method": method, "path": str(final_path.relative_to(OUT)), "sha256": sha256(final_path)})
            searches.append(search)
            print(f"[SOURCE FIT] seed{seed} {method} selected={best[1]} sourceBA={best[0]:.4f} {time.perf_counter()-tick:.1f}s", flush=True)
    if len(bank) != 27:
        raise ValueError("Exactly27 final heads must be frozen before target scoring")
    freeze = {"models": bank, "cohort_plan_sha256": sha256(plan_path), "protocol_sha256": sha256(ROOT/"PROTOCOL.md"),
              "target_scores_evaluated": False, "all_models_saved_before_target_evaluation": True,
              "source_training_rows": len(source), "source_inner_hospitals": SOURCES,
              "created_unix_seconds": time.time()}
    dump_json(OUT/"fits/all_searches.json", searches)
    dump_json(OUT/"fits/source_heads_frozen.json", freeze)
    return freeze


def all_metrics(labels, prob):
    from cbmp_evaluation import classification_metrics
    pred = np.argmax(prob, axis=1)
    result = classification_metrics(labels, prob)
    result["sensitivity"] = float(np.mean(pred[labels == 1] == 1))
    result["specificity"] = float(np.mean(pred[labels == 0] == 0))
    return result


def evaluate_frozen_bank(features, frame):
    from cbmp_evaluation import head_scores, probabilities
    frozen_path = OUT/"fits/source_heads_frozen.json"
    if not frozen_path.exists():
        raise ValueError("Target scoring forbidden until full27-head bank saved")
    frozen = json.loads(frozen_path.read_text())
    if len(frozen["models"]) != 27 or frozen["target_scores_evaluated"] is not False:
        raise ValueError("Invalid source freeze gate")
    if (OUT/"evaluation/complete.json").exists():
        return
    results, predictions = [], []
    for entry in frozen["models"]:
        path = OUT/entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Frozen source-trained model changed before evaluation")
        saved = joblib.load(path)
        for hospital in TARGETS:
            indices = np.flatnonzero(frame["center"].to_numpy() == hospital)
            rows = frame.iloc[indices]
            y = rows["label"].to_numpy(dtype=int)
            scores = head_scores(saved["model"], features[indices])
            probs = probabilities(scores, saved["temperature"])
            metrics = all_metrics(y, probs)
            results.append({"seed": entry["seed"], "method": entry["method"], "hospital": hospital,
                "role": "primary_test" if hospital == 2 else "secondary_test", "n_patches": len(indices),
                "n_patients": rows["patient"].nunique(), "configuration": saved["configuration"],
                "temperature": saved["temperature"], **metrics})
            for row, prob, score in zip(rows.to_dict("records"), probs, scores):
                predictions.append({"sample_id": row["sample_id"], "patient": int(row["patient"]), "node": int(row["node"]),
                    "slide": int(row["slide"]), "group_id": row["group_id"], "hospital": hospital,
                    "role": row["partition"], "label": int(row["label"]), "seed": entry["seed"],
                    "method": entry["method"], "prediction": int(prob.argmax()),
                    "probability_class_1": float(prob[1]), "score_class_0": float(score[0]),
                    "score_class_1": float(score[1]), "temperature": float(saved["temperature"])})
            print(f"[TARGET] seed{entry['seed']} {entry['method']} hospital{hospital} BA={metrics['balanced_accuracy']:.4f}", flush=True)
    if len(results) != 54:
        raise ValueError("Expected54 target metric records")
    write_csv(OUT/"evaluation/metrics_by_seed_hospital.csv", results)
    write_csv(OUT/"evaluation/predictions.csv", predictions)
    dump_json(OUT/"evaluation/complete.json", {"records": 54, "predictions": len(predictions),
        "source_bank_sha256_before_scoring": sha256(frozen_path), "all27_heads_frozen_before_any_target_scoring": True,
        "primary_hospital": 2, "secondary_hospital": 1, "no_target_selection": True})


def run(threads=2, batch_size=64):
    verify_protocol()
    frame = audit_metadata()
    frame = extract_images(frame)
    plan = {"selected_manifest_sha256": sha256(OUT/"audit/image_manifest.csv"),
            "metadata_audit_sha256": sha256(OUT/"audit/metadata_complete.json"),
            "protocol_sha256": sha256(ROOT/"PROTOCOL.md"), "selected_rows": len(frame),
            "patient_cap": CAP, "rank_prefix": PREFIX, "original_identity_format": "patch_patient_{patient:03d}_node_{node}_x_{x}_y_{y}.png",
            "source_train_only": True, "source_hospitals": SOURCES, "source_inner_LOCO": SOURCES,
            "primary_target_hospital": 2, "secondary_target_hospital": 1, "seeds": SEEDS,
            "nine_unchanged_head_grids": True, "target_selected_parameters": False}
    plan_path = OUT/"cohort_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("Cohort plan changed after freeze")
    dump_json(plan_path, plan)
    features = extract_features(frame, threads=threads, batch_size=batch_size)
    with threadpool_limits(limits=threads):
        fit_all_source_heads(features, frame)
        evaluate_frozen_bank(features, frame)
    print("[EXP5] ALL54 TARGET RECORDS COMPLETE; report separately", flush=True)


def main():
    import pyarrow
    pyarrow.set_cpu_count(2)
    pyarrow.set_io_thread_count(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("acquire", "audit", "run"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "acquire":
        acquire(args.workers)
    elif args.mode == "audit":
        audit_metadata()
    else:
        run(args.threads, args.batch_size)


if __name__ == "__main__":
    main()
