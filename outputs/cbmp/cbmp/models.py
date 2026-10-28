"""Class-conditional weighted spherical clustering and matched frozen-feature heads."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite nonempty matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-norm embeddings are invalid")
    return x / norms


def class_weights(y, centers, groups, beta=0.0, group_weighting=True):
    """Within each class, each center has mass G_sc ** (1-beta)."""
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0,1]")
    y, centers, groups = map(np.asarray, (y, centers, groups))
    if not (len(y) == len(centers) == len(groups)):
        raise ValueError("Metadata length mismatch")
    w = np.zeros(len(y), dtype=float)
    for c in np.unique(y):
        cmask = y == c
        for s in np.unique(centers[cmask]):
            idx = np.flatnonzero(cmask & (centers == s))
            if group_weighting:
                _, inv, counts = np.unique(groups[idx], return_inverse=True, return_counts=True)
                w[idx] = len(counts) ** (-beta) / counts[inv]
            else:
                w[idx] = len(idx) ** (-beta)
        w[cmask] /= w[cmask].sum()
    return w


class CBMP:
    def __init__(self, k=1, beta=0.0, seed=0, restarts=5, max_iter=100, group_weighting=True):
        if int(k) != k or k < 1 or restarts < 1 or max_iter < 1:
            raise ValueError("Positive integer k, restarts and max_iter required")
        self.k, self.beta, self.seed = int(k), beta, seed
        self.restarts, self.max_iter = restarts, max_iter
        self.group_weighting = group_weighting

    def fit(self, x, y, centers, groups):
        x = normalize(x)
        y, centers, groups = map(np.asarray, (y, centers, groups))
        if len(x) != len(y):
            raise ValueError("Embedding/label length mismatch")
        w = class_weights(y, centers, groups, self.beta, self.group_weighting)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        self.prototypes_, self.diagnostics_ = [], []
        for c in self.classes_:
            mask = y == c
            z, wc = x[mask], w[mask]
            # Draw distinct embedding directions, preserving their aggregate mass.
            unique, inverse = np.unique(z, axis=0, return_inverse=True)
            mass = np.bincount(inverse, weights=wc)
            count_groups = len(set(zip(centers[mask].tolist(), groups[mask].tolist())))
            k = min(self.k, len(z), count_groups, len(unique))
            best = None
            for _ in range(self.restarts):
                p = unique[rng.choice(len(unique), k, replace=False, p=mass / mass.sum())].copy()
                previous = None
                for iteration in range(self.max_iter):
                    scores = z @ p.T
                    assignment = scores.argmax(axis=1)
                    if previous is not None and np.array_equal(assignment, previous):
                        break
                    previous = assignment.copy()
                    valid = []
                    invalid = []
                    for j in range(k):
                        v = (z[assignment == j] * wc[assignment == j, None]).sum(axis=0)
                        norm = np.linalg.norm(v)
                        if norm > 1e-12:
                            p[j] = v / norm
                            valid.append(j)
                        else:
                            invalid.append(j)
                    if invalid:
                        # No well-defined single centroid for exact cancellation.
                        # Resolve deterministically by residual reseeding.
                        for j in invalid:
                            similarity = unique @ p[valid].T if valid else np.zeros((len(unique), 1))
                            residual = mass * (1 - similarity.max(axis=1))
                            if valid:
                                residual[(similarity > 1 - 1e-12).any(axis=1)] = -np.inf
                            pick = int(np.argmax(residual))
                            p[j] = unique[pick]
                            valid.append(j)
                        previous = None
                objective = float(np.sum(wc * (1 - (z @ p.T).max(axis=1))))
                if best is None or objective < best[0]:
                    best = (objective, p.copy(), iteration + 1)
            self.prototypes_.append(best[1])
            self.diagnostics_.append({"class": int(c), "k": k, "objective": best[0], "iterations": best[2]})
        return self

    def decision_function(self, x):
        z = normalize(x)
        return np.column_stack([(z @ p.T).max(axis=1) for p in self.prototypes_])

    def predict(self, x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]

    def save(self, path):
        np.savez_compressed(path, classes=self.classes_, **{f"prototype_{i}": p for i, p in enumerate(self.prototypes_)})


class FrozenHead:
    """Linear/cosine softmax heads fitted with the same weighted NLL and L-BFGS budget."""
    def __init__(self, cosine=False, regularization=0.01, seed=0, max_iter=300):
        self.cosine, self.regularization = cosine, regularization
        self.seed, self.max_iter = seed, max_iter

    def fit(self, x, y, centers, groups):
        x = normalize(x)
        self.classes_, yi = np.unique(y, return_inverse=True)
        nclass, d = len(self.classes_), x.shape[1]
        if nclass < 2:
            raise ValueError("Classifier requires at least two classes")
        weights = class_weights(y, centers, groups) / nclass
        rng = np.random.default_rng(self.seed)
        init = np.concatenate([rng.normal(0, 0.05, (nclass, d)).ravel(), np.zeros(nclass)])

        def objective(v):
            raw, bias = v[:nclass*d].reshape(nclass, d), v[nclass*d:]
            norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
            kernel = raw / norms if self.cosine else raw
            scale = 10.0 if self.cosine else 1.0
            logits = scale * x @ kernel.T + (0 if self.cosine else bias)
            loss = np.sum(weights * (logsumexp(logits, axis=1) - logits[np.arange(len(x)), yi]))
            delta = softmax(logits, axis=1)
            delta[np.arange(len(x)), yi] -= 1
            delta *= weights[:, None]
            grad = scale * delta.T @ x
            if self.cosine:
                grad = (grad - (grad * kernel).sum(axis=1, keepdims=True) * kernel) / norms
            loss += 0.5 * self.regularization * np.sum(raw**2)
            grad += self.regularization * raw
            gb = np.zeros(nclass) if self.cosine else delta.sum(axis=0)
            return float(loss), np.concatenate([grad.ravel(), gb])

        result = minimize(objective, init, method="L-BFGS-B", jac=True, options={"maxiter": self.max_iter})
        self.raw_ = result.x[:nclass*d].reshape(nclass, d)
        self.bias_ = result.x[nclass*d:]
        self.diagnostics_ = {"converged": bool(result.success), "message": str(result.message), "iterations": int(result.nit)}
        return self

    def decision_function(self, x):
        kernel = normalize(self.raw_) if self.cosine else self.raw_
        return (10.0 if self.cosine else 1.0) * normalize(x) @ kernel.T + (0 if self.cosine else self.bias_)

    def save(self, path):
        np.savez_compressed(path, classes=self.classes_, raw=self.raw_, bias=self.bias_, cosine=self.cosine)
