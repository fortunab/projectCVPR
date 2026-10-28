"""Nested source-only selection, saved splits, predictions and uncertainty."""
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import sklearn
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from .models import CBMP, FrozenHead
from .data import digest, load_features

TEMPERATURES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, probability):
    c = probability.shape[1]
    prediction = probability.argmax(axis=1)
    present = np.unique(y)
    result = {"balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=np.arange(c), average="macro", zero_division=0)),
        "nll": float(log_loss(y, probability, labels=np.arange(c))), "auroc": None,
        "samples": len(y), "present_classes": present.tolist()}
    if len(present) == c:
        result["auroc"] = float(roc_auc_score(y, probability[:, 1]) if c == 2 else
            roc_auc_score(y, probability, multi_class="ovr", average="macro"))
    return result


def inner_splits(data, source, seed):
    """Exclude unsupported folds consistently; group fallback never sees the target."""
    all_classes = np.unique(data["y"][source])
    valid, excluded = [], []
    for center in sorted(np.unique(data["centers"][source])):
        train = source[data["centers"][source] != center]
        val = source[data["centers"][source] == center]
        if np.array_equal(np.unique(data["y"][train]), all_classes) and np.array_equal(np.unique(data["y"][val]), all_classes):
            valid.append((train, val))
        else:
            excluded.append({"center": str(center), "reason": "Missing class in inner training or validation"})
    if valid:
        return valid, {"strategy": "leave_one_source_center_out", "excluded": excluded}
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=seed)
    for tr, va in splitter.split(source, groups=data["groups"][source]):
        tr, va = source[tr], source[va]
        if np.array_equal(np.unique(data["y"][tr]), all_classes) and np.array_equal(np.unique(data["y"][va]), all_classes):
            return [(tr, va)], {"strategy": "source_group_holdout", "excluded": excluded}
    raise ValueError("No valid source-only validation split with complete class coverage")


def candidates(method):
    if method in ("linear", "cosine"):
        return [{"regularization": a} for a in (0.001, 0.01, 0.1)]
    ks = [1] if method in ("centroid", "balanced_centroid", "image_centroid") else [1, 2, 4, 8]
    betas = [0.0, 0.5, 1.0] if method in ("cbmp", "balanced_centroid") else [0.0]
    return [{"k": k, "beta": b} for k in ks for b in betas]


def make_model(method, params, seed):
    if method in ("linear", "cosine"):
        return FrozenHead(cosine=method == "cosine", seed=seed, **params)
    return CBMP(seed=seed, group_weighting=method != "image_centroid", **params)


def fit_model(method, params, seed, data, indices):
    return make_model(method, params, seed).fit(*[data[key][indices] for key in ("x", "y", "centers", "groups")])


def select(method, data, splits, seed):
    records, best = [], None
    for params in candidates(method):
        fold_scores, outputs = [], []
        for train, val in splits:
            model = fit_model(method, params, seed, data, train)
            logits = model.decision_function(data["x"][val])
            fold_scores.append(float(balanced_accuracy_score(data["y"][val], logits.argmax(axis=1))))
            outputs.append((data["y"][val], logits))
        score = float(np.mean(fold_scores))
        records.append({"parameters": params, "mean_balanced_accuracy": score, "fold_scores": fold_scores})
        # Iteration order implements smaller K, smaller beta tie-breaking.
        if best is None or score > best[0] + 1e-12:
            best = (score, params, outputs)
    nclass = len(np.unique(data["y"]))
    calibration = [{"temperature": t, "mean_nll": float(np.mean([
        log_loss(y, softmax(logits / t, axis=1), labels=np.arange(nclass)) for y, logits in best[2]]))}
        for t in TEMPERATURES]
    temperature = min(calibration, key=lambda x: x["mean_nll"])["temperature"]
    return best[1], temperature, {"candidates": records, "calibration": calibration}


def bootstrap(y, p, groups, seed, count):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = []
    for _ in range(count):
        sample = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sample])
        if len(np.unique(y[idx])) == p.shape[1]:
            values.append(balanced_accuracy_score(y[idx], p[idx].argmax(axis=1)))
    return {"balanced_accuracy_95_percentile": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
            "valid_replicates": len(values), "requested_replicates": count, "groups": len(unique),
            "scope": "Conditional on fitted model and observed center; not unseen-hospital uncertainty"}


def run(cache, output, methods, seeds, allow_synthetic=False, bootstraps=500):
    data, provenance = load_features(cache, allow_synthetic)
    centers = sorted(np.unique(data["centers"]).tolist())
    if len(centers) < 3:
        raise ValueError("At least three centers required")
    classes = np.unique(data["y"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"status": "running", "cache_sha256": digest(cache),
        "provenance": provenance, "methods": methods, "seeds": seeds, "bootstraps": bootstraps,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "implementation_sha256": {
            name: digest(Path(__file__).parent / name) for name in ("models.py", "data.py", "experiment.py")}})
    all_results = []
    for center_index, center in enumerate(centers):
        source = np.flatnonzero(data["centers"] != center)
        target = np.flatnonzero(data["centers"] == center)
        if not np.array_equal(np.unique(data["y"][source]), classes):
            raise ValueError(f"Outer source missing classes for held-out center {center}")
        for seed in seeds:
            splits, split_info = inner_splits(data, source, seed)
            fold = output / f"center_{center_index:02d}" / f"seed_{seed}"
            fold.mkdir(parents=True)
            ids = lambda idx: data["sample_id"][idx].tolist()
            write_json(fold / "splits.json", {"target_center": center, "source_ids": ids(source),
                "target_ids": ids(target), "inner": [{"train_ids": ids(a), "validation_ids": ids(b)} for a,b in splits],
                "selection_protocol": split_info})
            for method in methods:
                params, temperature, selection = select(method, data, splits, seed)
                model = fit_model(method, params, seed, data, source)
                probability = softmax(model.decision_function(data["x"][target]) / temperature, axis=1)
                result = {"center": center, "seed": seed, "method": method, "parameters": params,
                    "temperature": temperature, "diagnostics": model.diagnostics_,
                    "metrics": metrics(data["y"][target], probability),
                    "bootstrap": bootstrap(data["y"][target], probability, data["groups"][target], seed, bootstraps)}
                folder = fold / method
                folder.mkdir()
                model.save(folder / "model.npz")
                write_json(folder / "selection.json", selection)
                write_json(folder / "result.json", result)
                np.savez_compressed(folder / "predictions.npz", sample_id=data["sample_id"][target],
                    y=data["y"][target], groups=data["groups"][target], probabilities=probability)
                with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["sample_id", "label", "group_id", "prediction"] + [f"p_{c}" for c in classes])
                    for j,i in enumerate(target):
                        writer.writerow([data['sample_id'][i], int(data['y'][i]), data['groups'][i], int(probability[j].argmax()), *probability[j]])
                all_results.append(result)
                print(f"center={center} seed={seed} method={method} BA={result['metrics']['balanced_accuracy']:.4f}", flush=True)
    write_json(output / "results.json", all_results)
    meta = json.loads((output / "run.json").read_text())
    meta["status"] = "complete"
    write_json(output / "run.json", meta)
    return all_results
