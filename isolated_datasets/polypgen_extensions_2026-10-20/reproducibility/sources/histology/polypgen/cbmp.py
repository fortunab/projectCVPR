"""Center-balanced multi-prototype classification on frozen image features.

The module is deliberately independent of the dataset downloader and does not
create synthetic data. Features must be extracted using a frozen encoder and
the supplied manifest/split protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score


@dataclass(frozen=True)
class PrototypeModel:
    prototypes: np.ndarray  # (classes, max_prototypes, dimensions)
    valid: np.ndarray       # (classes, max_prototypes)
    temperature: float = 1.0

    def scores(self, features: np.ndarray) -> np.ndarray:
        z = _unit(features)
        values = np.einsum("nd,ckd->nck", z, self.prototypes)
        values = np.where(self.valid[None, :, :], values, -np.inf)
        return np.max(values, axis=2)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(features), axis=1)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.scores(features) / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class CosineHead:
    weights: np.ndarray
    temperature: float = 0.1

    def scores(self, features: np.ndarray) -> np.ndarray:
        return _unit(features) @ self.weights.T

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(features), axis=1)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.scores(features) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def fit_cosine_head(features: np.ndarray, labels: np.ndarray, n_classes: int,
                    seed: int = 0, steps: int = 300, learning_rate: float = 0.03,
                    sample_weights: np.ndarray | None = None) -> CosineHead:
    """Fit a normalized cosine head with weighted cross-entropy.

    Without explicit weights, retain inverse-class-frequency weighting. Supplied
    weights replace that weighting entirely and are normalized to mean one;
    callers can therefore apply the same class/group/center weights as the
    prototype method without introducing a second class-balancing factor.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or not all(features.shape):
        raise ValueError("features must be a nonempty (samples, dimensions) array")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must be one-dimensional with one label per feature row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    z = _unit(features)
    y = labels.astype(np.int64, copy=False)
    if set(np.unique(y).tolist()) != set(range(n_classes)):
        raise ValueError("Every class must occur when fitting the cosine head")
    if sample_weights is None:
        counts = np.bincount(y, minlength=n_classes)
        fit_weights = 1.0 / counts[y]
        fit_weights *= len(y) / fit_weights.sum()
    else:
        fit_weights = np.asarray(sample_weights, dtype=np.float64)
        if fit_weights.ndim != 1 or len(fit_weights) != len(y):
            raise ValueError("sample_weights must be one-dimensional with one weight per sample")
        if np.any(~np.isfinite(fit_weights)) or np.any(fit_weights < 0):
            raise ValueError("sample_weights must be finite and nonnegative")
        if any(not np.any(fit_weights[y == cls] > 0) for cls in range(n_classes)):
            raise ValueError("sample_weights must give every class positive total weight")
        # Rescale before summing to avoid overflow for large finite weights.
        fit_weights = fit_weights / fit_weights.max()
        fit_weights *= len(y) / fit_weights.sum()
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0, (n_classes, z.shape[1]))
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    one_hot = np.eye(n_classes)[y]
    tau = 0.1
    for step in range(1, steps + 1):
        logits = (z @ weights.T) / tau
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        error = (probs - one_hot) * fit_weights[:, None] / len(y)
        gradient = error.T @ z / tau
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        weights -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        weights /= np.maximum(norms, 1e-12)
    return CosineHead(weights)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(x)) or np.any(norms <= 1e-12):
        raise ValueError("Features must be finite and have nonzero norm")
    return x / norms


def center_group_weights(
    labels: np.ndarray, centers: np.ndarray, groups: np.ndarray, beta: float
) -> np.ndarray:
    """Return within-class weights, equalizing groups then centers by beta."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=str)
    groups = np.asarray(groups, dtype=str)
    if not (len(labels) == len(centers) == len(groups)):
        raise ValueError("labels, centers, and groups must have equal length")
    weights = np.zeros(len(labels), dtype=np.float64)
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        group_sample_counts = {}
        center_groups = {}
        for i in idx:
            group_key = (centers[i], groups[i])
            group_sample_counts[group_key] = group_sample_counts.get(group_key, 0) + 1
            center_groups.setdefault(centers[i], set()).add(groups[i])
        center_group_counts = {c: len(gs) for c, gs in center_groups.items()}
        raw = np.empty(len(idx), dtype=np.float64)
        for j, i in enumerate(idx):
            group_key = (centers[i], groups[i])
            raw[j] = center_group_counts[centers[i]] ** (-beta) / group_sample_counts[group_key]
        weights[idx] = raw / raw.sum()
    return weights


def _cluster_one(
    z: np.ndarray, weights: np.ndarray, k: int, seed: int, max_iter: int = 100
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(z)
    k = min(k, n)
    if k < 1:
        raise ValueError("Each class needs at least one training example")
    centers = z[rng.choice(n, size=k, replace=False, p=weights / weights.sum())].copy()
    for _ in range(max_iter):
        assignment = np.argmax(z @ centers.T, axis=1)
        updated = centers.copy()
        nearest = np.max(z @ centers.T, axis=1)
        reseeded = set()
        for j in range(k):
            members = assignment == j
            if np.any(members):
                mean = np.sum(z[members] * weights[members, None], axis=0)
                norm = np.linalg.norm(mean)
                if norm > 1e-12:
                    updated[j] = mean / norm
                else:
                    members = np.zeros(n, dtype=bool)
            if not np.any(members):
                residual = weights * (1.0 - nearest)
                if reseeded:
                    residual[list(reseeded)] = -np.inf
                candidate = int(np.argmax(residual))
                reseeded.add(candidate)
                updated[j] = z[candidate]
        new_assignment = np.argmax(z @ updated.T, axis=1)
        if np.array_equal(new_assignment, assignment):
            centers = updated
            break
        centers = updated
    return centers


def fit_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    k: int = 1,
    beta: float = 0.0,
    restarts: int = 5,
    seed: int = 0,
) -> PrototypeModel:
    z = _unit(features)
    labels = np.asarray(labels, dtype=np.int64)
    if len(z) != len(labels):
        raise ValueError("features and labels must have equal length")
    d = z.shape[1]
    bank = np.zeros((n_classes, k, d), dtype=np.float64)
    valid = np.zeros((n_classes, k), dtype=bool)
    all_weights = center_group_weights(labels, centers, groups, beta)
    for cls in range(n_classes):
        idx = np.flatnonzero(labels == cls)
        if not len(idx):
            raise ValueError(f"Class {cls} is absent from a training fold")
        local_w = all_weights[idx]
        local_w = local_w / local_w.sum()
        class_z = z[idx]
        class_k = min(k, len(idx), len(np.unique(np.asarray(groups)[idx])))
        best, best_obj = None, np.inf
        for restart in range(restarts):
            candidate = _cluster_one(class_z, local_w, class_k, seed + cls * 1009 + restart)
            similarity = class_z @ candidate.T
            objective = float(np.sum(local_w * (1.0 - np.max(similarity, axis=1))))
            if objective < best_obj:
                best, best_obj = candidate, objective
        bank[cls, :class_k] = best
        valid[cls, :class_k] = True
    return PrototypeModel(bank, valid)


def select_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on source validation scores only."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        logits = scores / t
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def select_temperature_folds(score_folds: Iterable[np.ndarray], label_folds: Iterable[np.ndarray]) -> float:
    """Optimize the mean per-fold NLL so large centers do not dominate tau."""
    pairs = [(np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.int64))
             for s, y in zip(score_folds, label_folds)]
    if not pairs:
        raise ValueError("No validation folds supplied for temperature selection")

    def objective(log_t: float) -> float:
        temperature = float(np.exp(log_t))
        losses = []
        for scores, labels in pairs:
            logits = scores / temperature
            logits -= logits.max(axis=1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            losses.append(float(-log_probs[np.arange(len(labels)), labels].mean()))
        return float(np.mean(losses))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def select_hyperparameters(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    groups: np.ndarray,
    validation_folds: Iterable[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    k_values=(1, 2, 4, 8),
    beta_values=(0.0, 0.5, 1.0),
    seed: int = 0,
) -> tuple[int, float]:
    """Select K,beta by mean center-fold balanced accuracy; deterministic ties."""
    results = []
    folds = list(validation_folds)
    if not folds:
        raise ValueError("No valid source-only validation folds")
    for k in k_values:
        for beta in beta_values:
            fold_scores = []
            for fold_id, (train_idx, val_idx) in enumerate(folds):
                model = fit_prototypes(
                    features[train_idx], labels[train_idx], centers[train_idx],
                    groups[train_idx], n_classes, k=k, beta=beta, seed=seed + fold_id
                )
                fold_scores.append(balanced_accuracy_score(labels[val_idx], model.predict(features[val_idx])))
            results.append((float(np.mean(fold_scores)), -k, -beta, k, beta))
    best = max(results)
    return best[3], best[4]


def calibrate_model(model: PrototypeModel, features: np.ndarray, labels: np.ndarray) -> PrototypeModel:
    return PrototypeModel(model.prototypes, model.valid, select_temperature(model.scores(features), labels))

