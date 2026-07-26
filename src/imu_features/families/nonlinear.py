"""Nonlinear-dynamics and information-theoretic feature family.

Signals from human/machine motion are rarely simple periodic or random
processes — these features characterize complexity, regularity, and
self-similarity, borrowed from dynamical-systems analysis and widely used in
gait and tremor analysis: entropy measures (Shannon, sample, approximate,
permutation), fractal dimension (Higuchi, Petrosian), and long-range
correlation (Hurst exponent, detrended fluctuation analysis).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.registry import register_feature
from ..utils.embedding import time_delay_embedding


@register_feature(
    "shannon_entropy",
    family="nonlinear",
    min_samples=8,
    description="Shannon entropy (bits) of the channel's amplitude histogram.",
)
def shannon_entropy(x, sample_rate=None, bins=16):
    hist, _ = np.histogram(x, bins=bins)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


@register_feature(
    "sample_entropy",
    family="nonlinear",
    min_samples=20,
    description="Sample entropy (SampEn, m=2): negative log ratio of m+1 to m template matches. Lower = more regular/predictable.",
)
def sample_entropy(x, sample_rate=None, m=2, r_factor=0.2):
    x = np.asarray(x, dtype=float)
    r = r_factor * np.std(x)
    if r == 0:
        return 0.0

    def _count_matches(order):
        templates = time_delay_embedding(x, order, 1)
        k = len(templates)
        count = 0
        for i in range(k - 1):
            d = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(d <= r))
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


@register_feature(
    "approximate_entropy",
    family="nonlinear",
    min_samples=20,
    description="Approximate entropy (ApEn, m=2): includes self-matches, unlike sample entropy.",
)
def approximate_entropy(x, sample_rate=None, m=2, r_factor=0.2):
    x = np.asarray(x, dtype=float)
    r = r_factor * np.std(x)
    if r == 0:
        return 0.0

    def _phi(order):
        templates = time_delay_embedding(x, order, 1)
        k = len(templates)
        total = 0.0
        for i in range(k):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            c_i = np.sum(d <= r) / k
            total += np.log(c_i)
        return total / k

    return float(_phi(m) - _phi(m + 1))


@register_feature(
    "permutation_entropy",
    family="nonlinear",
    min_samples=12,
    description="Normalized permutation entropy (order=3): complexity of the ordinal patterns of consecutive samples.",
)
def permutation_entropy(x, sample_rate=None, order=3, delay=1):
    x = np.asarray(x, dtype=float)
    if len(x) < order * delay + 1:
        return 0.0
    embedded = time_delay_embedding(x, order, delay)
    perms = np.argsort(embedded, axis=1)
    _, counts = np.unique(perms, axis=0, return_counts=True)
    p = counts / counts.sum()
    pe = -np.sum(p * np.log2(p))
    max_entropy = math.log2(math.factorial(order))
    return float(pe / max_entropy) if max_entropy > 0 else 0.0


@register_feature(
    "hurst_exponent",
    family="nonlinear",
    min_samples=32,
    description="Hurst exponent via rescaled-range (R/S) analysis: >0.5 persistent, <0.5 anti-persistent, 0.5 random walk.",
)
def hurst_exponent(x, sample_rate=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_lag = max(4, n // 2)
    lags = sorted(set(np.floor(np.logspace(math.log10(4), math.log10(max_lag), num=15)).astype(int)))
    valid_lags, rs_values = [], []
    for lag in lags:
        if lag < 4:
            continue
        n_chunks = n // lag
        if n_chunks < 1:
            continue
        chunk_rs = []
        for c in range(n_chunks):
            seg = x[c * lag : (c + 1) * lag]
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            s = seg.std()
            if s > 0:
                chunk_rs.append(r / s)
        if chunk_rs:
            valid_lags.append(lag)
            rs_values.append(np.mean(chunk_rs))
    if len(valid_lags) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
    return float(slope)


@register_feature(
    "dfa_alpha",
    family="nonlinear",
    min_samples=32,
    description="Detrended fluctuation analysis scaling exponent: ~0.5 uncorrelated, ~1.0 1/f noise, >1 strong long-range correlation.",
)
def dfa_alpha(x, sample_rate=None, order=1):
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.cumsum(x - x.mean())
    max_scale = max(4, n // 4)
    scales = sorted(set(np.floor(np.logspace(math.log10(4), math.log10(max_scale), num=12)).astype(int)))
    valid_scales, flucts = [], []
    for s in scales:
        if s < 4:
            continue
        n_seg = n // s
        if n_seg < 2:
            continue
        rms_vals = []
        t = np.arange(s)
        for v in range(n_seg):
            seg = y[v * s : (v + 1) * s]
            coeffs = np.polyfit(t, seg, order)
            trend = np.polyval(coeffs, t)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        valid_scales.append(s)
        flucts.append(np.mean(rms_vals))
    if len(valid_scales) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(valid_scales), np.log(flucts), 1)
    return float(slope)


@register_feature(
    "higuchi_fractal_dimension",
    family="nonlinear",
    min_samples=16,
    description="Higuchi fractal dimension: curve-length scaling estimate of signal complexity (~1 smooth, ~2 noise-like).",
)
def higuchi_fractal_dimension(x, sample_rate=None, k_max=8):
    x = np.asarray(x, dtype=float)
    n = len(x)
    k_max = max(2, min(k_max, n // 2))
    lk = []
    for k in range(1, k_max + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            diff_sum = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (len(idx) * k)
            lm.append(diff_sum * norm / k)
        if lm:
            lk.append(np.mean(lm))
    if len(lk) < 2:
        return 1.0
    ks = np.arange(1, len(lk) + 1)
    coeffs = np.polyfit(np.log(1.0 / ks), np.log(np.maximum(lk, 1e-12)), 1)
    return float(coeffs[0])


@register_feature(
    "petrosian_fractal_dimension",
    family="nonlinear",
    min_samples=8,
    description="Petrosian fractal dimension: fast complexity estimate from the number of sign changes in the first difference.",
)
def petrosian_fractal_dimension(x, sample_rate=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    diff = np.diff(x)
    n_sign_changes = int(np.sum(np.diff(np.sign(diff)) != 0))
    if n_sign_changes == 0:
        return 1.0
    log_n = math.log10(n)
    return float(log_n / (log_n + math.log10(n / (n + 0.4 * n_sign_changes))))
