"""Snapshot core code and preservation evidence without changing baseline inputs."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "polypgen_v3_copy_2026-10-17"
WORKSPACE = ROOT.parents[1]
PROTECTED = [BASE, WORKSPACE / "outputs"] + [WORKSPACE / "work" / "family_baselines" / "polypgen" / name
                                             for name in ("data", "cache", "results")]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    result = {}
    for root in PROTECTED:
        for path in root.rglob("*"):
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
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        hashes_changed = [name for name, value in setup["baseline_artifact_sha256"].items()
                          if not (BASE / name).is_file() or sha256(BASE / name) != value]
        report = {"files_checked_size_mtime": len(before), "changed_paths": changed,
                  "baseline_artifacts_hashed": len(setup["baseline_artifact_sha256"]),
                  "hash_mismatches": hashes_changed, "unchanged": not changed and not hashes_changed}
        (ROOT / "preservation_check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if not report["unchanged"]:
            raise ValueError("Protected files changed; inspect preservation_check.json")
        return
    if marker.exists() or (ROOT / "code").exists():
        raise FileExistsError("Setup already exists; refusing to overwrite it")
    (ROOT / "code").mkdir()
    copied = {}
    for name in ("cbmp.py", "cbmp_data.py", "cbmp_evaluation.py", "polypgen_manifest.py"):
        shutil.copy2(BASE / "code" / name, ROOT / "code" / name)
        copied[name] = sha256(ROOT / "code" / name)
    state = inventory()
    (ROOT / "preservation_baseline.json").write_text(json.dumps(state))
    hashes = {path.relative_to(BASE).as_posix(): sha256(path) for path in BASE.rglob("*")
              if path.is_file() and "extracted" not in path.relative_to(BASE).parts and path.suffix.lower() != ".zip"}
    setup = {"baseline": str(BASE), "copied_code_sha256": copied,
             "baseline_artifact_sha256": hashes,
             "protected_roots": [str(path) for path in PROTECTED], "files_recorded_size_mtime": len(state),
             "protocol_sha256": sha256(ROOT / "PROTOCOL.md"),
             "scope": "Exploratory post-test followups1-3; no feature extraction or baseline mutation"}
    marker.write_text(json.dumps(setup, indent=2))
    print(f"Setup complete; recorded {len(state)} protected files and hashed {len(hashes)} baseline artifacts", flush=True)


if __name__ == "__main__":
    main()
