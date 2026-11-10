"""Fit the same heads using separately audited pathology/official task specs."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 61, 89, 101, 131, 173, 211, 257])
    parser.add_argument("--protocols", nargs="+", choices=("historical", "matched4", "factorial"), default=["matched4"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.threads < 1 or args.workers < 1 or min(args.seeds) < 0
            or len(set(args.seeds)) != len(args.seeds) or len(set(args.protocols)) != len(args.protocols)):
        parser.error("Use positive resources and distinct non-negative seeds and protocols")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    from external_review_data import load_external_task
    from review_protocol import ROOT, grids, sha256, write_json
    from run_review_revision import fit_worker
    inputs, specs = {}, {}
    for path in args.task_spec:
        features, rows, splits, provenance = load_external_task(path)
        task = provenance["task"]
        if task in inputs:
            parser.error("Task identifiers must be unique")
        if provenance["protocol"] == "official_ood_validation" and len(args.seeds) != 10:
            parser.error("The official-split CAMELYON protocol requires ten distinct fitting seeds")
        inputs[task], specs[task] = provenance, str(path.resolve())
        del features, rows, splits
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f"Resolve environment conflicts before fitting:\n{check.stdout}\n{check.stderr}")
    plan = {
        "tasks": list(inputs), "seeds": args.seeds, "protocols": args.protocols,
        "threads": args.threads, "workers": args.workers, "inputs": inputs, "task_specs": specs,
        "candidate_grids": {name: grids(name) for name in args.protocols},
        "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in
                        ("run_external_review.py", "external_review_data.py", "run_review_revision.py", "review_protocol.py")},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "Pillow")},
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(), "numpy_build": np.__config__.CONFIG,
        "scope": "external task-spec replication; externally pretrained, not standard-pretraining leaderboard",
        "source_cache_policy": "identical source candidate fits shared within task/seed/partition",
        "target_data_previously_inspected": True,
    }
    output = args.run_dir.resolve()
    manifest = output / "run_manifest.json"
    if output.exists():
        if not args.resume or not manifest.exists() or json.loads(manifest.read_text()) != plan:
            raise FileExistsError("Use a new run directory or resume identical inputs, code, and settings")
    else:
        output.mkdir(parents=True)
        write_json(manifest, plan)
        (output / "pip_check.txt").write_text(check.stdout, encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                                capture_output=True, text=True, check=True)
        (output / "installed_packages.txt").write_text(freeze.stdout, encoding="utf-8")
    if args.prepare_only:
        print(f"Validated external plan: {output}")
        return
    lock = output / "RUNNING"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        pending = []
        for task in inputs:
            for seed in args.seeds:
                folder = output / task / f"seed{seed}"
                if (folder / "COMPLETE.json").exists():
                    done = json.loads((folder / "COMPLETE.json").read_text())
                    if (sha256(folder / "predictions.csv") != done["predictions_sha256"]
                            or sha256(folder / "source_heads_frozen.json") != done["source_choices_sha256"]):
                        raise ValueError("Completed external outputs changed")
                else:
                    pending.append((task, seed))
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_worker, output, task, seed, args.protocols, args.threads, specs[task])
                       for task, seed in pending]
            for future in concurrent.futures.as_completed(futures):
                print("COMPLETE", json.dumps(future.result()), flush=True)
        write_json(output / "COMPLETE.json", {"tasks": list(inputs), "seeds": args.seeds,
                                            "protocols": args.protocols, "completed_jobs": len(inputs) * len(args.seeds)})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
