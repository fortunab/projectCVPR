"""Check the recorded scientific package versions; do not silently permit drift."""
import argparse
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-version-drift", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = {}
    for filename in ("requirements-common.txt", "requirements-torch-cpu.txt"):
        for line in (ENV / filename).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                name, version = line.split("==")
                expected[name] = version
    actual, missing = {}, []
    for name in expected:
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    drift = {name: {"expected": version, "actual": actual[name]} for name, version in expected.items()
             if name in actual and version.split("+")[0] != actual[name].split("+")[0]}
    result = {"python": platform.python_version(), "python_executable": sys.executable,
              "platform": platform.platform(), "packages": actual, "missing": missing, "version_drift": drift,
              "exact_version_condition": not missing and not drift and sys.version_info[:2] == (3, 12),
              "version_drift_explicitly_allowed": args.allow_version_drift,
              "scope": "Package metadata only; run pip check and numerical verification separately."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if missing or sys.version_info[:2] != (3, 12) or (drift and not args.allow_version_drift):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
