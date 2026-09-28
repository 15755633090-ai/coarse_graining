"""Fixed spherical k-means partitions; clustering never receives topology."""
from __future__ import annotations

import math
import numpy as np


def spherical_regions(embeddings, seed=100_000, atoms_per_center=8, max_centers=8,
                      n_init=10, max_iter=100):
    """Return representative atom indices and disjoint, exhaustive membership.

    Zero vectors remain zero. Empty clusters take the worst represented atom
    from a non-singleton cluster. Ties use atom order; renumbering invariance
    is not claimed. Representatives maximize similarity to their own centroid.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("embeddings must be a finite nonempty matrix")
    if min(atoms_per_center, max_centers, n_init, max_iter) < 1:
        raise ValueError("clustering parameters must be positive")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    nodes = len(x)
    count = min(nodes, max_centers, max(2, math.ceil(nodes / atoms_per_center)))
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    for _ in range(n_init):
        centers = x[rng.choice(nodes, count, replace=False)].copy()
        previous = None
        for _ in range(max_iter):
            similarities = x @ centers.T
            labels = similarities.argmax(axis=1)
            sizes = np.bincount(labels, minlength=count)
            for empty in np.flatnonzero(sizes == 0):
                candidates = np.flatnonzero(sizes[labels] > 1)
                moved = candidates[np.argmin(similarities[candidates, labels[candidates]])]
                sizes[labels[moved]] -= 1
                labels[moved] = empty
                sizes[empty] += 1
            centers = np.stack([x[labels == k].sum(axis=0) for k in range(count)])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
        score = float((x * centers[labels]).sum())
        if score > best_score:
            best_score = score
            best = labels.copy(), centers.copy()
    labels, centers = best
    members = labels[None, :] == np.arange(count)[:, None]
    representatives = []
    for k in range(count):
        indices = np.flatnonzero(members[k])
        representatives.append(indices[np.argmax(x[indices] @ centers[k])])
    return np.asarray(representatives), members
