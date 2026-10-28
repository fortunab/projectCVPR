"""Execute the predeclared source-only experiment in this isolated folder."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def logged_command(command, log_path):
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       NUMEXPR_NUM_THREADS="4", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + subprocess.list2cmdline(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=CODE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", env=environment, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError(f"Processing command failed; inspect {log_path}")


def main():
    if not (ROOT / "audit" / "AUDIT_COMPLETE").exists():
        raise FileNotFoundError("Finish process_audit.py successfully first")
    summary = json.loads((ROOT / "audit" / "eligibility_summary.json").read_text())
    if summary["eligible_heldout_centers"] != ["C2", "C4", "C5", "C6"]:
        raise ValueError("Audited eligibility differs from the predeclared plan; review before fitting")
    common = [sys.executable, "-u", str(CODE / "run_cbmp.py"),
              "--data-root", str(ROOT / "extracted" / "PolypGen2021_MultiCenterData_v3"),
              "--manifest", str(ROOT / "audit" / "manifest.csv"),
              "--sequence-metadata", str(ROOT / "audit" / "reviewed_sequence_centers.csv"),
              "--sequence-only", "--verified-sequences-only",
              "--model-name", "resnet18_imagenet1k_v1", "--device", "cpu",
              "--batch-size", "32", "--workers", "0", "--threads", "4",
              "--feature-cache", str(ROOT / "features" / "resnet18_imagenet1k_v1.npz")]
    started = time.perf_counter()
    logged_command(common + ["--results-dir", str(ROOT / "plan"), "--seed", "17", "--dry-run"],
                   ROOT / "logs" / "dry_run.log")
    for seed in (17, 29, 43):
        print(f"Starting predeclared head seed {seed}", flush=True)
        logged_command(common + ["--results-dir", str(ROOT / "results" / f"seed{seed}"), "--seed", str(seed)],
                       ROOT / "logs" / f"seed{seed}.log")
    (ROOT / "experiments_complete.json").write_text(json.dumps({
        "seeds": [17, 29, 43], "eligible_centers": summary["eligible_heldout_centers"],
        "elapsed_seconds": time.perf_counter() - started,
        "scope": summary["scope"], "complete": True
    }, indent=2), encoding="utf-8")
    print("All three predeclared seed runs completed", flush=True)


if __name__ == "__main__":
    main()
