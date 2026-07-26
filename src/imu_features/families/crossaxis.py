"""Cross-axis feature family: pairwise relationships between the x/y/z
channels of a single sensor — Pearson correlation, covariance, and
best-lag cross-correlation. Useful for detecting coordinated multi-axis
motion (e.g. rotational coupling) that per-axis features can't see.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _max_cross_correlation(a, b, max_lag):
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0:
        return 0.0
    n = len(a)
    max_lag = min(max_lag, n - 1)
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            num = np.sum(a[lag:] * b[: n - lag])
        else:
            num = np.sum(a[: n + lag] * b[-lag:])
        val = num / denom
        if abs(val) > abs(best):
            best = val
    return float(best)


@register_feature(
    "corr_xy", family="crossaxis", scope="triaxial", min_samples=2, description="Pearson correlation between the x and y axes."
)
def corr_xy(arr, sample_rate=None):
    return _corr(arr[:, 0], arr[:, 1])


@register_feature(
    "corr_xz", family="crossaxis", scope="triaxial", min_samples=2, description="Pearson correlation between the x and z axes."
)
def corr_xz(arr, sample_rate=None):
    return _corr(arr[:, 0], arr[:, 2])


@register_feature(
    "corr_yz", family="crossaxis", scope="triaxial", min_samples=2, description="Pearson correlation between the y and z axes."
)
def corr_yz(arr, sample_rate=None):
    return _corr(arr[:, 1], arr[:, 2])


@register_feature(
    "cov_xy", family="crossaxis", scope="triaxial", min_samples=2, description="Covariance between the x and y axes."
)
def cov_xy(arr, sample_rate=None):
    return float(np.cov(arr[:, 0], arr[:, 1])[0, 1])


@register_feature(
    "cov_xz", family="crossaxis", scope="triaxial", min_samples=2, description="Covariance between the x and z axes."
)
def cov_xz(arr, sample_rate=None):
    return float(np.cov(arr[:, 0], arr[:, 2])[0, 1])


@register_feature(
    "cov_yz", family="crossaxis", scope="triaxial", min_samples=2, description="Covariance between the y and z axes."
)
def cov_yz(arr, sample_rate=None):
    return float(np.cov(arr[:, 1], arr[:, 2])[0, 1])


@register_feature(
    "max_cross_correlation_xy",
    family="crossaxis",
    scope="triaxial",
    min_samples=4,
    description="Peak normalized cross-correlation between x and y over lags up to 10 samples.",
)
def max_cross_correlation_xy(arr, sample_rate=None, max_lag=10):
    return _max_cross_correlation(arr[:, 0], arr[:, 1], max_lag)


@register_feature(
    "max_cross_correlation_xz",
    family="crossaxis",
    scope="triaxial",
    min_samples=4,
    description="Peak normalized cross-correlation between x and z over lags up to 10 samples.",
)
def max_cross_correlation_xz(arr, sample_rate=None, max_lag=10):
    return _max_cross_correlation(arr[:, 0], arr[:, 2], max_lag)


@register_feature(
    "max_cross_correlation_yz",
    family="crossaxis",
    scope="triaxial",
    min_samples=4,
    description="Peak normalized cross-correlation between y and z over lags up to 10 samples.",
)
def max_cross_correlation_yz(arr, sample_rate=None, max_lag=10):
    return _max_cross_correlation(arr[:, 1], arr[:, 2], max_lag)
