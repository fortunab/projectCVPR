"""Run a frozen-encoder pilot, with validation-only selection and a fixed test set.

This is patch-level, within-dataset classification. The dataset provides no
patient/center IDs, so this experiment does not estimate cross-center transfer.
Run from the repository root: python -m histology_cbmp.run --help
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from histology_cbmp.data import CLASS_NAMES, load_manifest
from polypgen.cbmp import fit_cosine_head, fit_prototypes, select_temperature

FEATURE_VERSION = 1
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
SCOPE = ("Within-dataset patch classification pilot; patient/slide/center identities "
         "are unavailable. Content-disjoint splits do not establish patient independence "
         "or unseen-center generalization. Center balancing is not evaluated.")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(rows):
    payload = [{key: row[key] for key in ("sample_id", "sha256", "pixel_sha256", "label", "class_name")}
               for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_manifest_files(rows):
    """Reject changed inputs before computing or reusing any cached features."""
    for row in rows:
        if sha256_file(row["image_path"]) != row["sha256"]:
            raise ValueError(f"Image changed since audit: {row['image_path']}; rebuild the manifest")


def make_splits(rows, seed=17):
    """Stratify unique content groups, then expand to image indices (60/20/20)."""
    by_hash = {}
    for index, row in enumerate(rows):
        key = row["pixel_sha256"]
        if key in by_hash and by_hash[key]["label"] != int(row["label"]):
            raise ValueError("Identical decoded image pixels have conflicting labels")
        by_hash.setdefault(key, {"label": int(row["label"]), "indices": []})["indices"].append(index)
    hashes = sorted(by_hash)
    labels = [by_hash[key]["label"] for key in hashes]
    development, test = train_test_split(hashes, test_size=0.2, stratify=labels, random_state=seed)
    train, validation = train_test_split(
        development, test_size=0.25, stratify=[by_hash[key]["label"] for key in development],
        random_state=seed + 1)
    split = {name: np.array(sorted(i for key in keys for i in by_hash[key]["indices"]), dtype=int)
             for name, keys in (("train", train), ("validation", validation), ("test", test))}
    expected = set(range(len(CLASS_NAMES)))
    for name, indices in split.items():
        if {int(rows[i]["label"]) for i in indices} != expected:
            raise ValueError(f"Not all classes occur in {name}; dataset is inadequate for this pilot")
    return split


def save_splits(rows, split, results_dir):
    membership = {int(index): name for name, indices in split.items() for index in indices}
    with (results_dir / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "sha256", "pixel_sha256", "label", "class_name", "split"])
        writer.writeheader()
        writer.writerows({**{key: row[key] for key in ("sample_id", "sha256", "pixel_sha256", "label", "class_name")},
                             "split": membership[index]} for index, row in enumerate(rows))


def extract_features(rows, cache_path, weights_path, batch_size=32, workers=0, threads=4):
    verify_manifest_files(rows)
    import torch
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18
    from tqdm import tqdm

    torch.set_num_threads(threads)
    if not weights_path.is_file():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(weights_path), hash_prefix="f37072fd")
    weight_digest = sha256_file(weights_path)
    if not weight_digest.startswith("f37072fd"):
        raise ValueError("Checkpoint does not match the official ResNet18 ImageNet1K V1 checksum")
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    metadata = {"feature_version": FEATURE_VERSION, "manifest_digest": manifest_digest(rows),
                "encoder": "torchvision.resnet18", "weights": "IMAGENET1K_V1",
                "weights_sha256": weight_digest, "weights_url": WEIGHTS_URL,
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "preprocessing": repr(transform), "device": "cpu", "normalized": True}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if json.loads(str(cache["metadata"].item())) != metadata:
                raise ValueError("Feature cache metadata mismatch; choose a new --feature-cache path")
            features = cache["features"].copy()
        if features.shape != (len(rows), 512) or not np.isfinite(features).all():
            raise ValueError("Invalid cached features")
        if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5):
            raise ValueError("Cached features are not normalized")
        print(f"Reusing audited feature cache: {cache_path}", flush=True)
        return features, metadata

    # Class must live in module scope for Windows workers; workers=0 is the
    # portable default and required by this local dataset wrapper.
    if workers:
        raise ValueError("This pilot supports --workers 0 (portable Windows loading)")

    class Images(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(rows[index]["image_path"]) as image:
                tensor = transform(image.convert("RGB"))
            return tensor

    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval()
    features = []
    loader = DataLoader(Images(), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images in tqdm(loader, desc="Frozen ResNet18 features"):
            output = encoder(images).float().numpy()
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            if not np.isfinite(output).all() or np.any(norms <= 1e-12):
                raise ValueError("Encoder returned invalid or zero features")
            features.append(output / norms)
    features = np.concatenate(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as stream:
        np.savez_compressed(stream, features=features, metadata=json.dumps(metadata, sort_keys=True))
    return features, metadata


def softmax(scores, temperature=1.0):
    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "macro_auroc_ovr": float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
            "negative_log_likelihood": float(log_loss(labels, probabilities, labels=range(len(CLASS_NAMES))))}


def run_experiment(rows, features, results_dir, seed=17, cosine_steps=300):
    """Fit train only, select/calibrate on validation, evaluate test once."""
    split = make_splits(rows, seed)
    save_splits(rows, split, results_dir)
    train, val, test = (split[key] for key in ("train", "validation", "test"))
    labels = np.array([row["label"] for row in rows], dtype=int)
    # This constant identifies one dataset to the shared weighting function.
    # With beta=0 it provides no domain signal; duplicates share one group.
    centers = np.full(len(rows), "single_dataset")
    groups = np.array([row["pixel_sha256"] for row in rows])
    n_classes = len(CLASS_NAMES)
    search = []
    prototype_candidates = []
    for k in (1, 2, 4, 8):
        model = fit_prototypes(features[train], labels[train], centers[train], groups[train],
                               n_classes, k=k, beta=0.0, restarts=5, seed=seed)
        score = balanced_accuracy_score(labels[val], model.predict(features[val]))
        search.append({"family": "prototype", "K": k, "validation_balanced_accuracy": float(score)})
        prototype_candidates.append((float(score), -k, model))
    selected = max(prototype_candidates, key=lambda item: (item[0], item[1]))
    selected_k = -selected[1]
    models = {"single_prototype": prototype_candidates[0][2],
              "multiple_prototypes_selected": selected[2]}
    # The selected grid includes K=1; validation may correctly favor it.
    linear_candidates = []
    for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=seed)
        model.fit(features[train], labels[train])
        if np.any(model.n_iter_ >= model.max_iter):
            raise RuntimeError(f"Linear classifier failed to converge for C={c_value}")
        score = balanced_accuracy_score(labels[val], model.predict(features[val]))
        search.append({"family": "linear", "C": c_value, "validation_balanced_accuracy": float(score)})
        linear_candidates.append((float(score), -c_value, model))
    linear_selected = max(linear_candidates, key=lambda item: (item[0], item[1]))
    models["linear_classifier"] = linear_selected[2]
    models["cosine_classifier"] = fit_cosine_head(features[train], labels[train], n_classes,
                                                 seed=seed, steps=cosine_steps)
    summary = {"scope": SCOPE, "seed": seed, "manifest_digest": manifest_digest(rows),
               "split_counts": {name: len(indices) for name, indices in split.items()},
               "split_class_counts": {name: np.bincount(labels[idx], minlength=n_classes).tolist()
                                       for name, idx in split.items()},
               "selection": {"criterion": "validation balanced accuracy; ties prefer smaller K/C",
                             "K_candidates": [1, 2, 4, 8], "selected_K": selected_k,
                             "C_candidates": [0.01, 0.1, 1.0, 10.0, 100.0], "selected_C": -linear_selected[1],
                             "cosine_steps_fixed": cosine_steps, "cosine_learning_rate_fixed": 0.03,
                             "prototype_restarts": 5, "beta": 0.0}, "methods": {}}
    predictions_rows = []
    for name, model in models.items():
        is_linear = name == "linear_classifier"
        val_scores = model.decision_function(features[val]) if is_linear else model.scores(features[val])
        tau = select_temperature(val_scores, labels[val])
        test_scores = model.decision_function(features[test]) if is_linear else model.scores(features[test])
        test_probabilities = softmax(test_scores, tau)
        val_probabilities = softmax(val_scores, tau)
        if not is_linear:
            model = replace(model, temperature=tau)
        joblib.dump({"model": model, "temperature": tau, "class_names": list(CLASS_NAMES),
                     "manifest_digest": manifest_digest(rows)}, results_dir / f"{name}.joblib")
        summary["methods"][name] = {"temperature": tau, "validation": metrics(labels[val], val_probabilities),
                                    "test": metrics(labels[test], test_probabilities),
                                    "confusion_matrix": confusion_matrix(labels[test], test_probabilities.argmax(axis=1),
                                                                         labels=range(n_classes)).tolist()}
        for i, probabilities in zip(test, test_probabilities):
            predictions_rows.append({"sample_id": rows[i]["sample_id"], "method": name, "split": "test",
                                     "label": int(labels[i]), "prediction": int(probabilities.argmax()),
                                     **{f"probability_{c}": float(p) for c, p in enumerate(probabilities)}})
        print(name, json.dumps(summary["methods"][name]["test"]), flush=True)
    (results_dir / "selection.json").write_text(json.dumps(search, indent=2), encoding="utf-8")
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (results_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions_rows[0]))
        writer.writeheader()
        writer.writerows(predictions_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=base / "results/pilot/manifest.csv")
    parser.add_argument("--results-dir", type=Path, default=base / "results/pilot")
    parser.add_argument("--feature-cache", type=Path, default=base / "cache/resnet18_imagenet1k_v1.npz")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if any((args.results_dir / name).exists() for name in ("metrics.json", "predictions.csv", "splits.csv")):
        parser.error("Results already exist; use a new --results-dir to preserve the fixed test record")
    rows = load_manifest(args.manifest, require_complete=True)
    if args.weights is None:
        import torch
        existing = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
        args.weights = existing if existing.is_file() else base / "cache/resnet18-f37072fd.pth"
    print(SCOPE, flush=True)
    features, encoder_metadata = extract_features(rows, args.feature_cache, args.weights,
                                                  batch_size=args.batch_size, threads=args.threads)
    with threadpool_limits(limits=args.threads):
        summary = run_experiment(rows, features, args.results_dir, seed=args.seed)
    metadata = {"encoder": encoder_metadata, "python": platform.python_version(),
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": {"runner": sha256_file(__file__),
                                  "prototype_core": sha256_file(inspect.getfile(fit_prototypes)),
                                  "data_loader": sha256_file(inspect.getfile(load_manifest))},
                "command_arguments": {key: str(value) if isinstance(value, Path) else value
                                      for key, value in vars(args).items()}}
    (args.results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved pilot results to {args.results_dir}; selected K={summary['selection']['selected_K']}", flush=True)


if __name__ == "__main__":
    main()
