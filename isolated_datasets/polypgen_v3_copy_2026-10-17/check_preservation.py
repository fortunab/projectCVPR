"""Check file size/mtime preservation of existing project data and outputs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROTECTED = [WORKSPACE / "outputs"] + [WORKSPACE / "work" / "family_baselines" / "polypgen" / name
                                      for name in ("data", "results", "cache")]


def inventory():
    result = {}
    for folder in PROTECTED:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                status = path.stat()
                result[path.relative_to(WORKSPACE).as_posix()] = [status.st_size, status.st_mtime_ns]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    baseline_path = ROOT / "preservation_baseline.json"
    current = inventory()
    if not args.check:
        if baseline_path.exists():
            raise FileExistsError("Preservation baseline already exists")
        baseline_path.write_text(json.dumps(current), encoding="utf-8")
        print(f"Recorded size/mtime preservation baseline for {len(current)} existing files")
        return
    baseline = json.loads(baseline_path.read_text())
    changes = sorted(key for key in set(baseline) | set(current) if baseline.get(key) != current.get(key))
    result = {"files_checked": len(baseline), "changed_paths": changes, "unchanged": not changes,
              "check": "File size and last-modification timestamp; not a content-hash check",
              "roots": [str(path) for path in PROTECTED]}
    (ROOT / "preservation_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if changes:
        raise RuntimeError("Existing data/output changed; inspect preservation_check.json")


if __name__ == "__main__":
    main()
