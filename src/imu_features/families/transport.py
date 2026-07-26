"""Optimal-transport (Wasserstein) feature family.

The `statistical` family summarizes a channel's distribution with moments
(mean, variance, skewness, kurtosis). Moments are cheap but lossy: two very
differently-shaped distributions can share the same first four moments.
This family instead measures a *distance between whole distributions*,
using the 1-Wasserstein (earth mover's / Kantorovich-Rubinstein) metric —
the minimum "cost" of reshaping one distribution into another, where cost
is probability mass times the distance it has to move. In one dimension it
has a simple closed form: the mean absolute difference between the two
distributions' quantile functions, which for two empirical samples of the
same size is just the mean absolute difference of their sorted values.

Three reference comparisons:

- **Distance to a Gaussian** with the same mean/std: an interpretable,
  original-units measure of non-normality (unlike skewness/kurtosis, which
  are unitless and can be zero for non-Gaussian shapes that happen to
  balance out).
- **Distance to a Uniform distribution** over the channel's own observed
  range: how far the channel is from being evenly spread across its range,
  vs. concentrated (peaked) or bimodal.
- **Distance between the first and second half of the window**: a
  distributional "drift" / non-stationarity measure. Every other family
  implicitly assumes the window is one coherent regime; this one directly
  quantifies how much that assumption is violated, e.g. a window that
  transitions from standing still to walking partway through.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..core.registry import register_feature


@register_feature(
    "wasserstein_gaussian_distance",
    family="transport",
    min_samples=8,
    description=(
        "1-Wasserstein distance between the channel's empirical distribution and a Gaussian "
        "with the same mean/std -- an interpretable, original-units measure of non-normality."
    ),
)
def wasserstein_gaussian_distance(x, sample_rate=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    sorted_x = np.sort(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    ref_quantiles = np.mean(x) + std * norm.ppf(probs)
    return float(np.mean(np.abs(sorted_x - ref_quantiles)))


@register_feature(
    "wasserstein_uniform_distance",
    family="transport",
    min_samples=8,
    description=(
        "1-Wasserstein distance between the channel's empirical distribution and a Uniform "
        "distribution over its own [min, max] range."
    ),
)
def wasserstein_uniform_distance(x, sample_rate=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    lo, hi = np.min(x), np.max(x)
    if hi == lo:
        return 0.0
    sorted_x = np.sort(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    ref_quantiles = lo + (hi - lo) * probs
    return float(np.mean(np.abs(sorted_x - ref_quantiles)))


@register_feature(
    "wasserstein_split_half_distance",
    family="transport",
    min_samples=8,
    description=(
        "1-Wasserstein distance between the first-half and second-half empirical distributions "
        "of the channel -- a distributional drift / non-stationarity measure within the window."
    ),
)
def wasserstein_split_half_distance(x, sample_rate=None):
    x = np.asarray(x, dtype=float)
    n = len(x) // 2
    if n < 2:
        return 0.0
    first = np.sort(x[:n])
    second = np.sort(x[-n:])
    return float(np.mean(np.abs(first - second)))
