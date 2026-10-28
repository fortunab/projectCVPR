"""Matched frozen-feature heads with nested source-only selection.

All candidate fits and calibration scores use inner source folds. Outer test
features/labels are consulted only after the final source-trained model is fixed.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, roc_auc_score)

from cbmp import (center_group_weights, fit_cosine_head, fit_prototypes,
                  select_temperature_folds)


def candidate_methods():
    ks, betas = (1, 2, 4, 8), (0.0, 0.5, 1.0)
    cs, rates = (0.01, 0.1, 1.0, 10.0, 100.0), (0.01, 0.03, 0.1)
    return {
        "cbmp": [{"kind": "prototype", "k": k, "beta": b} for k in ks for b in betas],
        "multi_prototype_unbalanced": [{"kind": "prototype", "k": k, "beta": 0.0} for k in ks],
        "linear_logistic": [{"kind": "linear", "C": c} for c in cs],
        "trainable_cosine": [{"kind": "cosine", "learning_rate": lr} for lr in rates],
        "linear_group_center_weighted": [{"kind": "linear", "C": c, "beta": b}
                                           for b in betas for c in cs],
        "cosine_group_center_weighted": [{"kind": "cosine", "learning_rate": lr, "beta": b}
                                           for b in betas for lr in rates],
        "single_centroid_group_weighted": [{"kind": "prototype", "k": 1, "beta": 0.0}],
        "single_centroid_center_balanced": [{"kind": "prototype", "k": 1, "beta": 1.0}],
        "single_centroid_image_weighted": [{"kind": "prototype", "k": 1, "beta": 0.0, "image_weighted": True}],
    }


def fit_head(spec, features, labels, centers, groups, seed):
    if spec["kind"] == "prototype":
        fitting_groups = np.arange(len(labels)).astype(str) if spec.get("image_weighted") else groups
        return fit_prototypes(features, labels, centers, fitting_groups, n_classes=2,
                              k=spec["k"], beta=spec["beta"], restarts=5, seed=seed)
    weights = None
    if "beta" in spec:
        weights = center_group_weights(labels, centers, groups, spec["beta"])
        weights = weights * len(weights) / weights.sum()
    if spec["kind"] == "linear":
        model = LogisticRegression(C=spec["C"], max_iter=3000, random_state=seed,
                                   class_weight="balanced" if weights is None else None)
        model.fit(features, labels, sample_weight=weights)
        if np.any(model.n_iter_ >= model.max_iter):
            raise RuntimeError("Logistic regression did not converge; no target result was used")
        return model
    if spec["kind"] == "cosine":
        return fit_cosine_head(features, labels, n_classes=2, seed=seed, steps=300,
                               learning_rate=spec["learning_rate"], sample_weights=weights)
    raise ValueError(f"Unknown classifier kind: {spec['kind']}")


def head_scores(model, features):
    if hasattr(model, "scores"):
        return model.scores(features)
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected binary class order 0,1")
    decision = model.decision_function(features)
    return np.column_stack([np.zeros(len(decision)), decision])


def probabilities(scores, temperature):
    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def select_head(candidates, features, labels, centers, groups, inner_folds, seed):
    if not inner_folds:
        raise ValueError("Source-only validation folds are required")
    records, best = [], None
    for spec in candidates:
        score_folds, label_folds, accuracies = [], [], []
        for fold_id, (train, validation) in enumerate(inner_folds):
            model = fit_head(spec, features[train], labels[train], centers[train], groups[train], seed + fold_id)
            scores = head_scores(model, features[validation])
            accuracies.append(float(balanced_accuracy_score(labels[validation], scores.argmax(axis=1))))
            score_folds.append(scores)
            label_folds.append(labels[validation])
        mean_score = float(np.mean(accuracies))
        records.append({"configuration": spec, "mean_source_validation_balanced_accuracy": mean_score,
                        "per_source_validation_fold": accuracies})
        # Candidate order is fixed before evaluation; strict comparison preserves
        # deterministic ties (smaller K, beta, C, and learning rate first).
        if best is None or mean_score > best[0]:
            best = (mean_score, spec, score_folds, label_folds)
    tau = select_temperature_folds(best[2], best[3])
    return best[1], tau, records


def classification_metrics(labels, probabilities_):
    predictions = probabilities_.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(labels, probabilities_[:, 1])),
            "negative_log_likelihood": float(log_loss(labels, probabilities_, labels=[0, 1])),
            "brier_score": float(np.mean((probabilities_[:, 1] - labels) ** 2))}


def evaluate_loco(features, rows, splits, results_dir, seed=17, methods=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    labels = np.array([r["label"] for r in rows], dtype=int)
    centers = np.array([r["center"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    content = np.array([r.get("pixel_sha256", r["sha256"]) for r in rows])
    methods = candidate_methods() if methods is None else methods
    fold_records, predictions, search_records = [], [], []
    for outer in splits:
        if outer["status"] != "ready":
            print(f"[SKIP] {outer['heldout_center']}: {outer['status']}", flush=True)
            continue
        train = np.array(outer["outer_train"], dtype=int)
        test = np.array(outer["outer_test"], dtype=int)
        inner = [(np.array(f["train"], dtype=int), np.array(f["validation"], dtype=int))
                 for f in outer["inner_folds"]]
        if set(train) & set(test) or any((set(a) | set(b)) - set(train) or set(a) & set(b) for a, b in inner):
            raise ValueError("Invalid source/target partition boundaries")
        if (not np.all(centers[test] == outer["heldout_center"])
                or np.any(centers[train] == outer["heldout_center"])
                or set(groups[train]) & set(groups[test]) or set(content[train]) & set(content[test])):
            raise ValueError("Invalid center/group/content partition boundaries")
        if any(set(groups[a]) & set(groups[b]) or set(content[a]) & set(content[b]) for a, b in inner):
            raise ValueError("Invalid inner source group/content boundaries")
        model_dir = results_dir / "models" / outer["heldout_center"]
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, candidates in methods.items():
            started = time.perf_counter()
            spec, tau, search = select_head(candidates, features, labels, centers, groups, inner, seed)
            selection_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model = fit_head(spec, features[train], labels[train], centers[train], groups[train], seed)
            fitting_seconds = time.perf_counter() - started
            joblib.dump({"model": model, "temperature": tau, "configuration": spec, "seed": seed,
                         "heldout_center": outer["heldout_center"]}, model_dir / f"{name}.joblib")
            # First and only consultation of this method's outer target scores.
            test_prob = probabilities(head_scores(model, features[test]), tau)
            record = {"heldout_center": outer["heldout_center"], "method": name,
                      "configuration": spec, "temperature": tau,
                      "selection_seconds": selection_seconds, "fitting_seconds": fitting_seconds,
                      "selection_protocol": outer["selection_protocol"],
                      "metrics": classification_metrics(labels[test], test_prob)}
            fold_records.append(record)
            search_records.append({"heldout_center": outer["heldout_center"], "method": name,
                                   "selected_configuration": spec, "candidates": search})
            for index, prob in zip(test, test_prob):
                predictions.append({"sample_id": rows[index]["sample_id"], "center": rows[index]["center"],
                                    "group_id": rows[index]["group_id"], "label": int(labels[index]),
                                    "method": name, "prediction": int(prob.argmax()),
                                    "probability_class_1": float(prob[1])})
            print(f"[DONE] heldout={outer['heldout_center']} method={name} "
                  f"source-selected={spec} test_bAcc={record['metrics']['balanced_accuracy']:.4f}", flush=True)
        # Preserve completed folds if a later independent fit fails.
        (results_dir / "metrics_by_center.json").write_text(json.dumps(fold_records, indent=2), encoding="utf-8")
        (results_dir / "selection_search.json").write_text(json.dumps(search_records, indent=2), encoding="utf-8")
        with (results_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    if not fold_records:
        raise ValueError("No eligible center-held-out evaluation folds")
    return fold_records
