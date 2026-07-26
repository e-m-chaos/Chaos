"""Gait / step-detection feature family.

Peak-based step detection on an acceleration channel (most meaningful on the
vertical axis or the resultant magnitude, though — like every channel-scope
family — the engine applies it to x/y/z/mag alike for consistency). Derives
cadence and step-interval regularity, standard gait-quality indicators in
locomotion and fall-risk research.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..core.registry import register_feature


def _detect_step_peaks(x, sample_rate, min_step_period_s=0.25):
    x = np.asarray(x, dtype=float)
    height = np.mean(x) + 0.5 * np.std(x)
    distance = max(1, int(round(min_step_period_s * sample_rate)))
    peaks, _ = find_peaks(x, height=height, distance=distance)
    return peaks


@register_feature(
    "step_count",
    family="gait",
    min_samples=8,
    description="Number of detected peaks (candidate steps) in the window.",
)
def step_count(x, sample_rate):
    return float(len(_detect_step_peaks(x, sample_rate)))


@register_feature(
    "cadence",
    family="gait",
    min_samples=8,
    description="Detected steps per minute over the window's duration.",
)
def cadence(x, sample_rate):
    n_peaks = len(_detect_step_peaks(x, sample_rate))
    duration_min = len(x) / sample_rate / 60.0
    return float(n_peaks / duration_min) if duration_min > 0 else 0.0


@register_feature(
    "step_interval_mean",
    family="gait",
    min_samples=8,
    description="Mean time (s) between consecutive detected steps.",
)
def step_interval_mean(x, sample_rate):
    peaks = _detect_step_peaks(x, sample_rate)
    if len(peaks) < 2:
        return 0.0
    return float(np.mean(np.diff(peaks) / sample_rate))


@register_feature(
    "step_interval_cv",
    family="gait",
    min_samples=8,
    description="Coefficient of variation of step intervals — a gait-regularity index (lower = more regular).",
)
def step_interval_cv(x, sample_rate):
    peaks = _detect_step_peaks(x, sample_rate)
    if len(peaks) < 2:
        return 0.0
    intervals = np.diff(peaks) / sample_rate
    m = np.mean(intervals)
    return float(np.std(intervals) / m) if m > 0 else 0.0
