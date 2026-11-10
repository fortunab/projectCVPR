"""Run the approved ten-seed protocols on existing audited feature caches."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def source_sequence_threshold(scores, labels, groups, centers):
    """Select one maximum-margin threshold using eligible source centers only."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    grouped = {}
    for score, label, group, center in zip(scores, labels, groups, centers):
        key = (str(center), str(group))
        margin = float(score[1] - score[0])
        record = grouped.setdefault(key, {"label": 0, "margin": -np.inf})
        record["label"] = max(record["label"], int(label))
        record["margin"] = max(record["margin"], margin)
    eligible = {}
    for (center, group), record in grouped.items():
        eligible.setdefault(center, []).append(record)
    eligible = {center: rows for center, rows in eligible.items() if {r["label"] for r in rows} == {0, 1}}
    if not eligible:
        return None
    values = np.unique([r["margin"] for rows in eligible.values() for r in rows])
    candidates = np.unique(np.concatenate(([0.0, np.nextafter(values[0], -np.inf), values[-1]],
                                          values[:-1] + (values[1:] - values[:-1]) / 2)))
    best = None
    for threshold in candidates:
        score = float(np.mean([
            balanced_accuracy_score([r["label"] for r in rows],
                                    [r["margin"] > threshold for r in rows])
            for rows in eligible.values()
        ]))
        preference = (abs(float(threshold)), float(threshold))
        if (best is None or score > best[2] + 1e-12
                or (abs(score - best[2]) <= 1e-12 and preference < best[0])):
            best = (preference, float(threshold), score)
    return {"threshold": best[1], "source_balanced_accuracy": best[2],
            "eligible_source_centers": sorted(eligible), "aggregation": "maximum raw score margin",
            "label": "any audited positive frame", "decision": "margin > threshold"}


def fit_worker(run_dir, task, seed, protocols, threads, task_spec=None):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from review_protocol import load_cached_task, grids, sha256, write_json, write_csv
    from cbmp import select_temperature_folds
    from cbmp_evaluation import fit_head, head_scores, probabilities, classification_metrics
    from sklearn.metrics import balanced_accuracy_score

    root = Path(run_dir) / task / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log):
        with threadpool_limits(limits=threads):
            if task_spec is None:
                features, rows, splits, provenance = load_cached_task(task)
            else:
                from external_review_data import load_external_task
                features, rows, splits, provenance = load_external_task(task_spec)
                if provenance["task"] != task:
                    raise ValueError("External task identity differs from the run plan")
            plan = json.loads((Path(run_dir) / "run_manifest.json").read_text())
            if provenance != plan["inputs"][task]:
                raise ValueError("Worker inputs changed after the run was planned")
            labels = np.array([r["label"] for r in rows], dtype=int)
            centers = np.array([r["center"] for r in rows])
            groups = np.array([r["group_id"] for r in rows])
            choices = []
            frozen = root / "source_heads_frozen.json"
            if frozen.exists():
                choices = json.loads(frozen.read_text())["choices"]
                for choice in choices:
                    if sha256(root / choice["model"]) != choice["model_sha256"]:
                        raise ValueError("A frozen source head changed")
                print("Verified previously frozen source heads", flush=True)
            else:
                for outer in splits:
                    train = np.array(outer["outer_train"], dtype=int)
                    fold_indices = [(np.array(f["train"], dtype=int), np.array(f["validation"], dtype=int))
                                    for f in outer["inner_folds"]]
                    training_key = hashlib.sha256(json.dumps({
                        "train": train.tolist(),
                        "folds": [(a.tolist(), b.tolist()) for a, b in fold_indices],
                    }, sort_keys=True).encode()).hexdigest()[:20]
                    cache_dir = root / "source_cache" / training_key
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    specifications = {}
                    for protocol in protocols:
                        for candidates in grids(protocol).values():
                            for spec in candidates:
                                specifications[json.dumps(spec, sort_keys=True)] = spec
                    search = {}
                    for spec_key, spec in specifications.items():
                        key = hashlib.sha256(spec_key.encode()).hexdigest()[:20]
                        cache_path = cache_dir / f"validation_{key}.joblib"
                        if cache_path.exists():
                            entry = joblib.load(cache_path)
                            if (entry["configuration"] != spec
                                    or any(not np.array_equal(saved, expected[1]) for saved, expected in
                                           zip(entry["validation_rows"], fold_indices))
                                    or len(entry["validation_rows"]) != len(fold_indices)):
                                raise ValueError("Source-score cache configuration mismatch")
                        else:
                            started = time.perf_counter()
                            score_folds, label_folds, validation_rows, accuracies = [], [], [], []
                            for fold_index, (fit, validation) in enumerate(fold_indices):
                                head = fit_head(spec, features[fit], labels[fit], centers[fit], groups[fit],
                                                seed + fold_index)
                                scores = head_scores(head, features[validation])
                                if not np.isfinite(scores).all():
                                    raise ValueError("Non-finite source-validation scores")
                                score_folds.append(scores)
                                label_folds.append(labels[validation])
                                validation_rows.append(validation)
                                accuracies.append(float(balanced_accuracy_score(labels[validation], scores.argmax(1))))
                            entry = {"configuration": spec, "score_folds": score_folds, "label_folds": label_folds,
                                     "validation_rows": validation_rows, "per_source_validation_fold": accuracies,
                                     "mean_source_validation_balanced_accuracy": float(np.mean(accuracies)),
                                     "fit_seconds": time.perf_counter() - started}
                            temporary = cache_path.with_suffix(".tmp")
                            joblib.dump(entry, temporary)
                            temporary.replace(cache_path)
                        search[spec_key] = entry
                        print(f"SOURCE {task} seed={seed} target={outer['heldout_center']} {spec}", flush=True)
                    for protocol in protocols:
                        for method, candidates in grids(protocol).items():
                            records = [search[json.dumps(spec, sort_keys=True)] for spec in candidates]
                            winner = max(range(len(records)),
                                         key=lambda i: records[i]["mean_source_validation_balanced_accuracy"])
                            selected = records[winner]
                            spec = selected["configuration"]
                            temperature = select_temperature_folds(selected["score_folds"], selected["label_folds"])
                            threshold = None
                            if task.startswith("polypgen"):
                                indices = np.concatenate(selected["validation_rows"])
                                threshold = source_sequence_threshold(
                                    np.concatenate(selected["score_folds"]), labels[indices],
                                    groups[indices], centers[indices])
                            model_key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
                            model_path = cache_dir / f"model_{model_key}.joblib"
                            if not model_path.exists():
                                started = time.perf_counter()
                                model = fit_head(spec, features[train], labels[train], centers[train], groups[train], seed)
                                temporary = model_path.with_suffix(".tmp")
                                joblib.dump({"model": model, "configuration": spec, "fit_seconds":
                                             time.perf_counter() - started}, temporary)
                                temporary.replace(model_path)
                            elif joblib.load(model_path)["configuration"] != spec:
                                raise ValueError("Final model cache configuration mismatch")
                            choice = {
                                "protocol": protocol, "center": outer["heldout_center"], "method": method,
                                "configuration": spec, "temperature": temperature, "sequence_rule": threshold,
                                "model": model_path.relative_to(root).as_posix(), "model_sha256": sha256(model_path),
                                "candidate_count": len(candidates),
                                "validation_fit_count": len(candidates) * len(fold_indices),
                                "shared_source_cache": True,
                                "search": [{"configuration": r["configuration"],
                                            "mean_source_validation_balanced_accuracy":
                                            r["mean_source_validation_balanced_accuracy"],
                                            "per_source_validation_fold": r["per_source_validation_fold"],
                                            "measured_unique_fit_seconds": r["fit_seconds"]} for r in records],
                            }
                            choices.append(choice)
                write_json(frozen, {"inputs": provenance, "seed": seed, "choices": choices,
                                    "target_predictions_produced": False})
                print(f"FROZEN {len(choices)} source-selected heads before target evaluation", flush=True)
            predictions, metrics = [], []
            by_center = {s["heldout_center"]: np.asarray(s["outer_test"], dtype=int) for s in splits}
            for choice in choices:
                model = joblib.load(root / choice["model"])["model"]
                indices = by_center[choice["center"]]
                scores = head_scores(model, features[indices])
                prob = probabilities(scores, choice["temperature"])
                metrics.append({"task": task, "seed": seed, "protocol": choice["protocol"],
                                "center": choice["center"], "method": choice["method"],
                                **classification_metrics(labels[indices], prob)})
                for index, score, probability in zip(indices, scores, prob):
                    row = rows[index]
                    threshold = choice["sequence_rule"]
                    predictions.append({
                        "task": task, "seed": seed, "protocol": choice["protocol"], "center": choice["center"],
                        "sample_id": row["sample_id"], "group_id": row["group_id"], "label": row["label"],
                        "dataset_index": row.get("dataset_index", ""),
                        "method": choice["method"], "prediction": int(probability.argmax()),
                        "probability_class_1": float(probability[1]), "margin": float(score[1] - score[0]),
                        "source_sequence_threshold": threshold["threshold"] if threshold else "",
                    })
            write_csv(root / "predictions.csv", predictions)
            write_json(root / "metrics.json", metrics)
            write_json(root / "COMPLETE.json", {"task": task, "seed": seed, "models": len(choices),
                                               "predictions_sha256": sha256(root / "predictions.csv"),
                                               "source_choices_sha256": sha256(frozen)})
    return {"task": task, "seed": seed, "evaluations": len(metrics)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["polypgen_resnet18", "polypgen_dinov2", "camelyon_resnet18"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43, 61, 89, 101, 131, 173, 211, 257])
    parser.add_argument("--protocols", nargs="+", choices=("historical", "matched4", "factorial"),
                        default=["historical", "matched4", "factorial"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.threads < 1 or args.workers < 1 or not args.seeds or min(args.seeds) < 0
            or len(set(args.seeds)) != len(args.seeds) or len(set(args.tasks)) != len(args.tasks)
            or len(set(args.protocols)) != len(args.protocols)):
        parser.error("Positive resources and unique non-negative seeds/tasks/protocols required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from review_protocol import ROOT, TASKS, load_cached_task, grids, sha256, write_json
    import numpy as np

    if not set(args.tasks) <= set(TASKS):
        parser.error(f"Tasks must be selected from {TASKS}")
    inputs = {}
    for task in args.tasks:
        features, rows, splits, provenance = load_cached_task(task)
        inputs[task] = provenance
        del features
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "Pillow")}
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f"Resolve environment conflicts before fitting:\n{check.stdout}\n{check.stderr}")
    plan = {
        "tasks": args.tasks, "seeds": args.seeds, "protocols": args.protocols,
        "threads": args.threads, "workers": args.workers,
        "inputs": inputs, "candidate_grids": {p: grids(p) for p in args.protocols},
        "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in
                        ("run_review_revision.py", "review_protocol.py")},
        "packages": packages, "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "numpy_build": np.__config__.CONFIG,
        "scope": "approved exploratory cached-feature replication and candidate-budget sensitivity",
        "feature_extraction": "existing audited frozen caches",
        "target_data_previously_inspected": True,
        "source_cache_policy": "share identical candidate scores and model fits within a task/seed/source partition",
    }
    out = args.run_dir.resolve()
    manifest = out / "run_manifest.json"
    if out.exists():
        if not args.resume or not manifest.exists() or json.loads(manifest.read_text()) != plan:
            raise FileExistsError("Use a new run directory or resume with identical code, inputs, and configuration")
    else:
        out.mkdir(parents=True)
        write_json(manifest, plan)
        (out / "pip_check.txt").write_text(check.stdout, encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
        (out / "installed_packages.txt").write_text(freeze.stdout, encoding="utf-8")
    if args.prepare_only:
        print(f"Validated plan: {out}", flush=True)
        return
    lock = out / "RUNNING"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        pending = []
        for task in args.tasks:
            for seed in args.seeds:
                complete = out / task / f"seed{seed}/COMPLETE.json"
                if complete.exists():
                    record = json.loads(complete.read_text())
                    folder = complete.parent
                    if (sha256(folder / "predictions.csv") != record["predictions_sha256"]
                            or sha256(folder / "source_heads_frozen.json") != record["source_choices_sha256"]):
                        raise ValueError("Completed outputs changed")
                else:
                    pending.append((task, seed))
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_worker, out, task, seed, args.protocols, args.threads) for task, seed in pending]
            for future in concurrent.futures.as_completed(futures):
                print("COMPLETE", json.dumps(future.result()), flush=True)
        write_json(out / "COMPLETE.json", {"tasks": args.tasks, "seeds": args.seeds,
                                          "protocols": args.protocols, "completed_jobs": len(args.tasks) * len(args.seeds)})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
