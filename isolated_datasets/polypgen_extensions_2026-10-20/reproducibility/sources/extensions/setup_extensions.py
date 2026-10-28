"""Snapshot classifier code and record all protected previous experiment files."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
BASE = ROOT.parent / "polypgen_v3_copy_2026-10-17"
PREVIOUS = [BASE, ROOT.parent / "polypgen_followups_2026-10-17", WORKSPACE / "outputs"] + [
    WORKSPACE / "work" / "family_baselines" / "polypgen" / name for name in ("data", "cache", "results")]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    result = {}
    for folder in PREVIOUS:
        for path in folder.rglob("*"):
            if path.is_file():
                status = path.stat()
                result[path.relative_to(WORKSPACE).as_posix()] = [status.st_size, status.st_mtime_ns]
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    marker = ROOT / "setup_complete.json"
    if args.check:
        setup = json.loads(marker.read_text())
        before = json.loads((ROOT / "preservation_baseline.json").read_text())
        after = inventory()
        changes = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
        mismatches = [p for p, h in setup["protected_artifact_sha256"].items()
                      if not (WORKSPACE / p).is_file() or sha256(WORKSPACE / p) != h]
        report = {"files_checked_size_mtime": len(before), "changed_paths": changes,
                  "baseline_artifacts_hashed": len(setup["protected_artifact_sha256"]),
                  "hash_mismatches": mismatches, "unchanged": not changes and not mismatches}
        (ROOT / "preservation_check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if not report["unchanged"]:
            raise ValueError("Previous data/output changed")
        return
    if marker.exists() or (ROOT / "code").exists():
        raise FileExistsError("Setup exists; refusing overwrite")
    (ROOT / "code").mkdir()
    copied = {}
    for name in ("cbmp.py", "cbmp_data.py", "cbmp_evaluation.py", "polypgen_manifest.py"):
        shutil.copy2(BASE / "code" / name, ROOT / "code" / name)
        copied[name] = sha256(ROOT / "code" / name)
    (ROOT / "preservation_baseline.json").write_text(json.dumps(inventory()))
    hashes = {}
    for folder in PREVIOUS[:2]:
        for path in folder.rglob("*"):
            if path.is_file() and "extracted" not in path.parts and path.suffix != ".zip" and "__pycache__" not in path.parts:
                hashes[path.relative_to(WORKSPACE).as_posix()] = sha256(path)
    marker.write_text(json.dumps({"protocol_sha256": sha256(ROOT / "PROTOCOL.md"),
                                  "copied_code_sha256": copied, "protected_artifact_sha256": hashes,
                                  "protected_roots": [str(p) for p in PREVIOUS]}, indent=2))
    print(f"Extension setup complete; {len(hashes)} protected artifact hashes", flush=True)


if __name__ == "__main__":
    main()
