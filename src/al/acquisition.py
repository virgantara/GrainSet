"""True pool-based acquisition (does NOT receive hidden labels)."""
from __future__ import annotations

import numpy as np

METHODS = ("random", "least_confidence", "entropy", "margin", "hybrid")


def scores(prob: np.ndarray, method: str) -> np.ndarray:
    """Higher = more uncertain; probabilities shape [N, C]."""
    if prob.ndim != 2 or prob.shape[1] < 2:
        raise ValueError("Expected probability matrix [N, >=2 classes]")
    if method == "least_confidence":
        return 1.0 - prob.max(axis=1)
    if method in {"entropy", "hybrid"}:
        return -(prob * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1)
    if method == "margin":
        largest2 = np.partition(prob, -2, axis=1)[:, -2:]
        margin = largest2.max(axis=1) - largest2.min(axis=1)
        return -margin
    raise ValueError(f"Unsupported uncertainty method: {method}")


def select_indices(method: str, pool_indices: np.ndarray, count: int,
                   prob: np.ndarray | None, embeddings: np.ndarray | None,
                   seed: int, candidate_multiplier: int = 3) -> np.ndarray:
    """Select unique ABSOLUTE dataset indices, no ground-truth access.

    Hybrid: top 3B high-entropy candidates + embedding cosine farthest-first.
    To make diversity selection efficient, project ONLY candidates to 64-D
    with a deterministic Gaussian random matrix (no labels or fit).
    """
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    if count < 0 or count > len(pool_indices):
        raise ValueError(f"Requested {count} of {len(pool_indices)} samples")
    if len(set(pool_indices.tolist())) != len(pool_indices):
        raise ValueError("Pool indices must be unique")
    if not count:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if method == 'random':
        return rng.choice(pool_indices, size=count, replace=False)
    if prob is None or len(prob) != len(pool_indices):
        raise ValueError("Probability rows must match pool indices")
    ranking = scores(prob, method)
    if method != 'hybrid':
        # Stable to guarantee reproducibility when entropy/margins are tied.
        return pool_indices[np.argsort(-ranking, kind='stable')[:count]]
    if embeddings is None or len(embeddings) != len(pool_indices):
        raise ValueError("Hybrid requires embeddings for the entire pool")
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be >= 1")
    candidate_count = min(len(pool_indices), candidate_multiplier * count)
    candidate_rows = np.argsort(-ranking, kind='stable')[:candidate_count]
    candidate_ids = pool_indices[candidate_rows]
    if candidate_count == count:
        return candidate_ids
    x = np.asarray(embeddings[candidate_rows], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected 2D feature matrix")
    if x.shape[1] > 64:
        projection = rng.standard_normal((x.shape[1], 64)).astype(np.float32)
        projection /= np.sqrt(64.0)
        x = x @ projection
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    # Farthest first via squared Euclidean on unit normalized embeddings.
    selected = np.zeros(candidate_count, dtype=bool)
    chosen = np.empty(count, dtype=np.int64)
    first = 0  # highest-entropy candidate
    min_d2 = np.full(candidate_count, np.inf, dtype=np.float32)
    for j in range(count):
        pick = first if j == 0 else int(np.argmax(np.where(selected, -1, min_d2)))
        chosen[j] = pick
        selected[pick] = True
        d2 = np.sum((x - x[pick]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return candidate_ids[chosen]
