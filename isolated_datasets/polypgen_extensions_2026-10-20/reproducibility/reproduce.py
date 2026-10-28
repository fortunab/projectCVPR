"""Portable staging and logged serial reproduction of the frozen experiments."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_NAME = "polypgen_v3_copy_2026-10-17"
FOLLOW_NAME = "polypgen_followups_2026-10-17"
EXT_NAME = "polypgen_extensions_2026-10-20"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def verify_bundle():
    manifest = json.loads((HERE / "sources_manifest.json").read_text())
    for record in manifest["files"]:
        path = HERE / "sources" / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing/changed source snapshot: {record['path']}")
    if sha256(HERE / "cohort_order.json") != manifest["cohort_order_sha256"]:
        raise ValueError("Original cohort-order reference changed")
    return manifest


def checked_zip_members(archive, destination):
    destination = Path(destination).resolve()
    members = archive.infolist()
    for member in members:
        # Reject Windows traversal syntax even when this helper runs on Linux.
        name = member.filename
        if "\\" in name or ":" in name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"Unsafe ZIP member syntax: {name}")
        target = (destination / name).resolve()
        if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"Unsafe ZIP member: {name}")
    return members


def patch_checkpoint_paths(extension):
    patches = []
    replacements = {
        "exp4_dinov2.py": ('MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--facebook--dinov2-base/snapshots" / SNAPSHOT_ID',
                           'MODEL_DIR = Path(os.environ["CBMP_DINO_MODEL_DIR"]).expanduser().resolve()'),
        "exp5_camelyon.py": ('    weights_path = Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"',
                            '    weights_path = Path(os.environ["CBMP_RESNET_CHECKPOINT"]).expanduser().resolve()')}
    for filename, (old, new) in replacements.items():
        path = extension / filename
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            raise ValueError(f"Checkpoint relocation patch no longer matches exactly: {filename}")
        before = sha256(path)
        path.write_text(original.replace(old, new), encoding="utf-8")
        patches.append({"file": filename, "original_sha256": before, "staged_sha256": sha256(path),
                        "old": old, "new": new, "scope": "Checkpoint location only; no model, preprocessing, fitting or evaluation math changed"})
    return patches


def stage(run_dir, archive_path):
    bundle = verify_bundle()
    run_dir, archive_path = run_dir.expanduser().resolve(), archive_path.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError("Use a NEW absent run directory; existing data/output is never merged or replaced")
    if not archive_path.is_file() or archive_path.stat().st_size != bundle["archive_bytes"] or sha256(archive_path) != bundle["archive_sha256"]:
        raise ValueError("Local PolypGen v3 ZIP bytes/SHA256 differ from the recorded release")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = checked_zip_members(archive, run_dir / "isolated_datasets" / BASE_NAME / "extracted")
        required_bytes = sum(m.file_size for m in members) + 25 * 1024**3
    if shutil.disk_usage(run_dir.parent).free < required_bytes:
        raise OSError(f"Need approximately {required_bytes / 1024**3:.1f} GiB free for extraction, public shards, images, environments and outputs")
    isolated = run_dir / "isolated_datasets"
    for scope, name in (("baseline", BASE_NAME), ("followups", FOLLOW_NAME), ("extensions", EXT_NAME)):
        shutil.copytree(HERE / "sources" / scope, isolated / name)
    shutil.copytree(HERE / "sources/histology", run_dir / "histology_pilot")
    patches = patch_checkpoint_paths(isolated / EXT_NAME)
    write_json(run_dir / "reproduction/stage.json", {
        "bundle_manifest_sha256": sha256(HERE / "sources_manifest.json"), "run_directory": str(run_dir),
        "polypgen_archive": str(archive_path), "archive_sha256": bundle["archive_sha256"],
        "patches": patches, "core_classifier_changes": [], "created_unix_seconds": time.time()})
    print(f"Staged new reproduction workspace: {run_dir}")


def load_stage(run_dir):
    run_dir = run_dir.expanduser().resolve()
    bundle = verify_bundle()
    stage_info = json.loads((run_dir / "reproduction/stage.json").read_text())
    if stage_info["bundle_manifest_sha256"] != sha256(HERE / "sources_manifest.json"):
        raise ValueError("Source bundle changed since staging; preserve this run and stage a new one")
    core = ("cbmp.py", "cbmp_data.py", "cbmp_evaluation.py", "polypgen_manifest.py")
    for name in core:
        original = HERE / "sources/baseline/code" / name
        staged = run_dir / "isolated_datasets" / BASE_NAME / "code" / name
        if sha256(original) != sha256(staged):
            raise ValueError(f"Staged core changed: {name}")
    return run_dir, bundle, stage_info


def environment(run_dir, bundle):
    env = os.environ.copy()
    dino = run_dir / "assets/dinov2" / bundle["dinov2_revision"]
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
               TORCH_HOME=str(run_dir / "assets/torch"), HF_HOME=str(run_dir / "runtime/huggingface"),
               HF_HUB_CACHE=str(run_dir / "runtime/huggingface/hub"), MPLCONFIGDIR=str(run_dir / "runtime/matplotlib"),
               XDG_CACHE_HOME=str(run_dir / "runtime/cache"), CBMP_DINO_MODEL_DIR=str(dino),
               CBMP_RESNET_CHECKPOINT=str(run_dir / "assets/torch/hub/checkpoints/resnet18-f37072fd.pth"))
    return env


def assets(run_dir, download, resnet=None, dino_dir=None):
    run_dir, bundle, _ = load_stage(run_dir)
    records = []
    for name, asset in bundle["assets"].items():
        target = run_dir / asset["location"]
        supplied = resnet if name.endswith(".pth") else (dino_dir / name if dino_dir else None)
        if target.exists():
            if sha256(target) != asset["sha256"]:
                raise ValueError(f"Existing checkpoint asset checksum mismatch: {target}")
        elif supplied is not None and supplied.is_file():
            if sha256(supplied) != asset["sha256"]:
                raise ValueError(f"Supplied checkpoint asset checksum mismatch: {supplied}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(supplied, target)
        elif download:
            parsed = urllib.parse.urlparse(asset["url"])
            if parsed.scheme != "https" or parsed.hostname not in {"download.pytorch.org", "huggingface.co"}:
                raise ValueError("Unexpected checkpoint source URL")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".{time.time_ns()}.part")
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "CBMP-reproduction/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
                shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
            if sha256(temporary) != asset["sha256"]:
                raise ValueError(f"Downloaded asset failed pinned SHA256: {name}; partial is preserved")
            temporary.rename(target)
        else:
            raise FileNotFoundError(f"Provide {name} locally, or explicitly pass assets --download")
        records.append({"path": str(target.relative_to(run_dir)), "bytes": target.stat().st_size,
                        "sha256": sha256(target), "pinned_source_url": asset["url"]})
    checkpoint = run_dir / "isolated_datasets" / EXT_NAME / "exp4_dinov2/checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    processor = run_dir / bundle["assets"]["preprocessor_config.json"]["location"]
    destination = checkpoint / processor.name
    if destination.exists() and sha256(destination) != sha256(processor):
        raise ValueError("Staged official processor changed")
    if not destination.exists():
        shutil.copyfile(processor, destination)
    write_json(run_dir / "reproduction/checkpoint_assets.json", records)
    print("All official checkpoint assets verified against pinned full SHA256 values.")


def preflight(run_dir, allow_version_drift=False):
    run_dir, bundle, stage_info = load_stage(run_dir)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use Python 3.12; these source files use Python 3.12 syntax/APIs")
    versions, mismatches = {}, []
    for name, expected in bundle["reference_packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"Missing package: {name}; install the environment requirements first") from None
        versions[name] = actual
        if actual.split("+")[0] != expected.split("+")[0]:
            mismatches.append({"package": name, "reference": expected, "actual": actual})
    if mismatches and not allow_version_drift:
        raise RuntimeError(f"Package versions differ from recorded environment: {mismatches}. An explicitly documented portability run may use --allow-version-drift; exact-version reproduction must resolve the environment first.")
    for name, asset in bundle["assets"].items():
        path = run_dir / asset["location"]
        if not path.is_file() or sha256(path) != asset["sha256"]:
            raise ValueError(f"Missing/changed official checkpoint asset: {name}")
    archive = Path(stage_info["polypgen_archive"])
    if not archive.is_file() or sha256(archive) != stage_info["archive_sha256"]:
        raise ValueError("Original local archive is missing or changed")
    record = {"python": sys.version, "executable": sys.executable, "platform": sys.platform,
              "packages": versions, "version_drift": mismatches, "version_drift_explicitly_allowed": allow_version_drift,
              "bundle_manifest_sha256": sha256(HERE / "sources_manifest.json"),
              "raw_archive_verified": True, "checkpoint_hashes_verified": True,
              "notes": "This preflight does not claim numerical equality across OS/CPU/library builds."}
    write_json(run_dir / "reproduction/preflight.json", record)
    print(json.dumps(record, indent=2))


def logged(run_dir, command, cwd, env, label):
    directory = run_dir / "reproduction/logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.time_ns()}_{label}.log"
    with path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"argv": list(map(str, command)), "cwd": str(cwd)}) + "\n")
        log.flush()
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Stage {label} failed; preserve outputs and inspect {path}")


def normalize_manifest(base):
    """Rebuild absolute paths locally; reproduce the original OS-neutral row order."""
    reference = json.loads((HERE / "cohort_order.json").read_text())
    path = base / "audit/manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    mapping = {row["sample_id"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(reference["sample_ids"]):
        raise ValueError("Rebuilt subset differs from the original reviewed cohort; no fitting allowed")
    rows = [mapping[sample] for sample in reference["sample_ids"]]
    columns = ("sample_id", "sha256", "pixel_sha256", "center", "group_id", "label", "label_source", "mask_sha256", "bbox_sha256")
    payload = [{k: int(row[k]) if k == "label" else row[k] for k in columns} for row in rows]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != reference["manifest_digest"]:
        raise ValueError("Audited image/pixel/annotation/label/group identities differ; do not reuse original results or silently alter the cohort")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audit_path = path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    audit["manifest_sha256"] = sha256(path)
    write_json(audit_path, audit)
    write_json(base / "audit/portable_order_validation.json", {"manifest_digest": digest,
               "same_original_sample_identity_order": True, "n_frames": len(rows),
               "absolute_paths_rebuilt_from_local_archive": True,
               "scope": "Filesystem enumeration order normalized only; no IDs, labels, groups, annotations, image bytes or classifier code changed"})


def run_steps(run_dir, steps, allow_version_drift=False, encoder_threads=4):
    preflight(run_dir, allow_version_drift)
    run_dir, bundle, info = load_stage(run_dir)
    env = environment(run_dir, bundle)
    isolated = run_dir / "isolated_datasets"
    base, follow, ext = (isolated / n for n in (BASE_NAME, FOLLOW_NAME, EXT_NAME))
    def run(root, filename, *args):
        logged(run_dir, [sys.executable, "-u", root / filename, *args], root, env, Path(filename).stem)
    for step in steps:
        if step == "audit":
            if (base / "audit/portable_order_validation.json").exists():
                print("Completed portable audit preserved; no audit files rewritten.")
                continue
            extraction = base / "extraction_complete.json"
            if not extraction.exists():
                destination = base / "extracted"
                if destination.exists():
                    raise FileExistsError("Partial extraction exists; preserve it and use a new run directory")
                with zipfile.ZipFile(info["polypgen_archive"]) as archive:
                    members = checked_zip_members(archive, destination)
                    for number, member in enumerate(members, 1):
                        archive.extract(member, destination)
                        if number % 2000 == 0:
                            print(f"Extracted {number}/{len(members)} members; ZIP CRC checked", flush=True)
                write_json(extraction, {"archive_sha256": bundle["archive_sha256"], "archive": info["polypgen_archive"],
                           "zip_entries": len(members), "extracted": str(destination),
                           "nested_archives": "Retained; not recursively expanded"})
            if not (base / "audit/AUDIT_COMPLETE").exists():
                run(base, "process_audit.py")
            normalize_manifest(base)
        elif step == "baseline":
            if not (base / "audit/portable_order_validation.json").exists():
                raise RuntimeError("Run the audit step first")
            if not (base / "experiments_complete.json").exists():
                run(base, "run_experiments.py")
            if not (base / "report/summary_metrics.json").exists():
                run(base, "make_report.py")
        elif step == "followups":
            if not (base / "report/summary_metrics.json").exists():
                raise RuntimeError("Finish baseline and its report first")
            if not (follow / "setup_complete.json").exists():
                run(follow, "setup_followups.py")
            if not (follow / "exp1_failure/COMPLETE.json").exists():
                run(follow, "exp1_failure.py")
            if not (follow / "exp2_factorial/COMPLETE").exists():
                run(follow, "exp2_factorial.py")
            if not (follow / "exp3_stress/COMPLETE.json").exists():
                run(follow, "exp3_stress.py")
            run(follow, "setup_followups.py", "--check")
        elif step in ("exp4", "exp5", "finalize"):
            if not (follow / "exp2_factorial/COMPLETE").exists():
                raise RuntimeError("Complete followups first: paired encoder reports require the ResNet18 factorial")
            if not (ext / "setup_complete.json").exists():
                run(ext, "setup_extensions.py")
            if step == "exp4":
                if not (ext / "exp4_dinov2/FITS_COMPLETE").exists():
                    run(ext, "exp4_dinov2.py", "inspect", "--batch-size", "16", "--threads", str(encoder_threads))
                    run(ext, "exp4_dinov2.py", "run", "--batch-size", "16", "--threads", "4", "--encoder-threads", str(encoder_threads))
                if not (ext / "exp4_dinov2/COMPLETE").exists():
                    run(ext, "exp4_report.py")
            elif step == "exp5":
                if not (ext / "exp5_camelyon/acquisition/complete.json").exists():
                    run(ext, "exp5_camelyon.py", "acquire", "--workers", "2")
                if not (ext / "exp5_camelyon/evaluation/complete.json").exists():
                    run(ext, "exp5_camelyon.py", "run", "--threads", "2", "--batch-size", "64")
                if not (ext / "exp5_camelyon/COMPLETE").exists():
                    run(ext, "exp5_report.py")
            else:
                run(ext, "setup_extensions.py", "--check")
                run(ext, "finalize_extensions.py")
        else:
            raise ValueError(f"Unknown step: {step}")
    print("Requested stages finished; inspect their scoped reports and completion markers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--polypgen-zip", type=Path, required=True)
    p = sub.add_parser("assets")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--resnet-checkpoint", type=Path)
    p.add_argument("--dino-model-dir", type=Path)
    for command in ("preflight", "run"):
        p = sub.add_parser(command)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--allow-version-drift", action="store_true")
        if command == "run":
            p.add_argument("--steps", nargs="+", choices=("audit", "baseline", "followups", "exp4", "exp5", "finalize"), required=True)
            p.add_argument("--encoder-threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.run_dir, args.polypgen_zip)
    elif args.command == "assets":
        assets(args.run_dir, args.download, args.resnet_checkpoint, args.dino_model_dir)
    elif args.command == "preflight":
        preflight(args.run_dir, args.allow_version_drift)
    else:
        if args.encoder_threads < 1:
            parser.error("Positive encoder threads required")
        run_steps(args.run_dir, args.steps, args.allow_version_drift, args.encoder_threads)


if __name__ == "__main__":
    main()
