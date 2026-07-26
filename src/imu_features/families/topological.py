"""Topological feature family: descriptors of the shape of the signal's
reconstructed phase-space trajectory, grounded in computational topology.

Two complementary techniques, both built from a time-delay embedding of the
1-D channel (no extra dependencies beyond numpy/scipy):

- **0-dimensional persistent homology** via the minimum spanning tree of the
  embedded point cloud. The MST edge weights are exactly the birth-death
  "lifetimes" of the H0 persistence diagram (connected-component merges
  under a growing distance threshold) — a standard, efficient equivalence
  for 0-dim persistence.
- **Recurrence quantification analysis (RQA)**: recurrence rate,
  determinism (fraction of recurrences forming diagonal lines, i.e.
  deterministic/periodic structure), and laminarity (fraction forming
  vertical lines, i.e. laminar/stationary states).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from ..core.registry import register_feature
from ..utils.embedding import time_delay_embedding


def _embed(x, dim=3, tau=1):
    return time_delay_embedding(np.asarray(x, dtype=float), dim, tau)


def _mst_edge_weights(points):
    dist = squareform(pdist(points))
    mst = minimum_spanning_tree(dist)
    return mst.data


@register_feature(
    "h0_total_persistence",
    family="topological",
    min_samples=16,
    description="Sum of 0-dim persistent homology lifetimes (MST edge weights) of the time-delay embedded trajectory.",
)
def h0_total_persistence(x, sample_rate=None, dim=3, tau=1):
    points = _embed(x, dim, tau)
    if len(points) < 3:
        return 0.0
    return float(_mst_edge_weights(points).sum())


@register_feature(
    "h0_max_persistence",
    family="topological",
    min_samples=16,
    description="Largest single 0-dim persistence lifetime (final MST merge) — the biggest topological 'gap' between clusters.",
)
def h0_max_persistence(x, sample_rate=None, dim=3, tau=1):
    points = _embed(x, dim, tau)
    if len(points) < 3:
        return 0.0
    weights = _mst_edge_weights(points)
    return float(weights.max()) if weights.size else 0.0


@register_feature(
    "h0_persistence_entropy",
    family="topological",
    min_samples=16,
    description="Shannon entropy of the normalized H0 persistence lifetimes — low if one merge dominates, high if lifetimes are uniform.",
)
def h0_persistence_entropy(x, sample_rate=None, dim=3, tau=1):
    points = _embed(x, dim, tau)
    if len(points) < 3:
        return 0.0
    weights = _mst_edge_weights(points)
    total = weights.sum()
    if total <= 0:
        return 0.0
    p = weights / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _recurrence_matrix(points, threshold_ratio):
    dist = squareform(pdist(points))
    max_dist = dist.max()
    threshold = threshold_ratio * max_dist if max_dist > 0 else 0.0
    rec = (dist <= threshold).astype(int)
    np.fill_diagonal(rec, 0)
    return rec


def _diagonal_line_points(rec, min_len):
    n = rec.shape[0]
    total = 0
    for offset in range(-(n - 1), n):
        run = 0
        for v in np.diagonal(rec, offset=offset):
            if v:
                run += 1
            else:
                if run >= min_len:
                    total += run
                run = 0
        if run >= min_len:
            total += run
    return total


def _vertical_line_points(rec, min_len):
    n = rec.shape[0]
    total = 0
    for col in range(n):
        run = 0
        for v in rec[:, col]:
            if v:
                run += 1
            else:
                if run >= min_len:
                    total += run
                run = 0
        if run >= min_len:
            total += run
    return total


@register_feature(
    "recurrence_rate",
    family="topological",
    min_samples=16,
    description="Fraction of embedded state pairs that recur within a distance threshold (RQA recurrence rate).",
)
def recurrence_rate(x, sample_rate=None, dim=3, tau=1, threshold_ratio=0.1):
    points = _embed(x, dim, tau)
    n = len(points)
    if n < 3:
        return 0.0
    rec = _recurrence_matrix(points, threshold_ratio)
    return float(rec.sum() / (n * n - n))


@register_feature(
    "determinism",
    family="topological",
    min_samples=16,
    description="RQA determinism: fraction of recurrence points forming diagonal lines (>=2 long), indicating deterministic/periodic structure.",
)
def determinism(x, sample_rate=None, dim=3, tau=1, threshold_ratio=0.1, min_diag=2):
    points = _embed(x, dim, tau)
    if len(points) < 3:
        return 0.0
    rec = _recurrence_matrix(points, threshold_ratio)
    total_points = rec.sum()
    if total_points == 0:
        return 0.0
    return float(_diagonal_line_points(rec, min_diag) / total_points)


@register_feature(
    "laminarity",
    family="topological",
    min_samples=16,
    description="RQA laminarity: fraction of recurrence points forming vertical lines (>=2 long), indicating laminar/stationary states.",
)
def laminarity(x, sample_rate=None, dim=3, tau=1, threshold_ratio=0.1, min_vert=2):
    points = _embed(x, dim, tau)
    if len(points) < 3:
        return 0.0
    rec = _recurrence_matrix(points, threshold_ratio)
    total_points = rec.sum()
    if total_points == 0:
        return 0.0
    return float(_vertical_line_points(rec, min_vert) / total_points)
