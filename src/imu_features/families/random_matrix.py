"""Random-matrix (cross-channel spectral) feature family.

Every other family analyzes one sensor, or a specific fixed pair, at a
time. This family instead stacks *every available sensor's* x/y/z channels
into a single data matrix and asks a question straight out of random matrix
theory (RMT): of the eigenvalues of the resulting cross-channel correlation
matrix, how many are larger than pure noise could ever produce?

Concretely: for a window of `n` samples and `p = 3 * (number of sensors
present)` standardized channels, the **Marchenko-Pastur law** (Marcenko &
Pastur, 1967) predicts the eigenvalue spectrum of the correlation matrix of
`p` *independent, uncorrelated* channels — it is supported on
`[(1 - sqrt(q))^2, (1 + sqrt(q))^2]` with `q = p / n`. Real IMU channels are
never independent: gravity, gait rhythm, and shared rigid-body motion
couple every axis of every sensor together. Genuine motion structure shows
up as eigenvalues escaping *above* that noise bulk. This is the exact
technique used to separate signal from noise in financial correlation
matrices (Laloux et al.'s "cleaning" of portfolio correlation matrices) and
in neuroscience functional-connectivity analysis. It answers a question no
other family here does: how many *independent, statistically significant
modes of coordinated motion* are present — as opposed to counting axes or
sensors, which only tells you how much you happened to record, not how
much independent structure is actually in it.

A companion feature, the inverse participation ratio (IPR) of the leading
eigenvector, asks whether that dominant coordinated-motion mode is spread
broadly across most channels (e.g. gravity, felt by every accelerometer
axis and, through gait-linked rotation, every gyroscope axis) or
concentrated on just a couple.

Because this genuinely needs more than one sensor's worth of channels to
say anything interesting, it is registered with `scope="fusion"`.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature

_SENSOR_ORDER = ("accel", "gyro", "mag")


def _build_channel_matrix(window):
    sensors = window.sensors()
    return np.concatenate([sensors[name] for name in _SENSOR_ORDER if name in sensors], axis=1)


def _standardize_columns(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    return (X - mean) / std_safe


def _correlation_eigen(window):
    X = _standardize_columns(_build_channel_matrix(window))
    n, p = X.shape
    corr = (X.T @ X) / n
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending order
    order = np.argsort(eigvals)[::-1]
    return np.clip(eigvals[order], 0, None), eigvecs[:, order], n, p


def _marchenko_pastur_upper_bound(n, p):
    q = p / n
    return (1.0 + np.sqrt(q)) ** 2


@register_feature(
    "rmt_spectrum",
    family="random_matrix",
    scope="fusion",
    requires=("accel", "gyro"),
    min_samples=20,
    description=(
        "Cross-channel correlation-matrix eigenvalue spectrum vs. the Marchenko-Pastur "
        "random-matrix noise bound: leading-eigenvalue strength, count/fraction of "
        "statistically significant coordinated modes, spectral entropy, and effective rank."
    ),
)
def rmt_spectrum(window):
    eigvals, _, n, p = _correlation_eigen(window)
    upper = _marchenko_pastur_upper_bound(n, p)
    total = eigvals.sum()
    probs = eigvals[eigvals > 0] / total if total > 0 else np.array([])
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    significant = eigvals > upper
    return {
        "largest_eigenvalue_ratio": float(eigvals[0] / p) if p > 0 else 0.0,
        "mp_upper_bound": float(upper),
        "significant_mode_count": float(np.sum(significant)),
        "significant_mode_fraction": float(np.sum(significant) / p) if p > 0 else 0.0,
        "spectral_entropy": entropy,
        "effective_rank": float(np.exp(entropy)),
    }


@register_feature(
    "rmt_leading_mode",
    family="random_matrix",
    scope="fusion",
    requires=("accel", "gyro"),
    min_samples=20,
    description=(
        "Inverse participation ratio of the leading correlation eigenvector: how many "
        "channels the dominant coordinated-motion mode is spread across (low IPR / high "
        "effective channel count = broadly shared, high IPR = concentrated on a few channels)."
    ),
)
def rmt_leading_mode(window):
    _, eigvecs, _, _ = _correlation_eigen(window)
    v1 = eigvecs[:, 0]
    norm = np.linalg.norm(v1)
    v1 = v1 / norm if norm > 0 else v1
    ipr = float(np.sum(v1**4))
    return {
        "participation_ratio": ipr,
        "effective_channel_count": float(1.0 / ipr) if ipr > 0 else 0.0,
    }
