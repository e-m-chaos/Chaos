"""Mechanical/physical feature family: quantities with a direct physical
interpretation when applied to accelerometer channels — jerk (rate of change
of acceleration), snap, energy-like proxies, and deviation from gravity.
Applied generically across sensors by the engine; the physical
interpretation is strongest for the accelerometer.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature

STANDARD_GRAVITY = 9.80665


@register_feature(
    "jerk_rms",
    family="mechanical",
    min_samples=2,
    description="RMS of the first time-derivative (jerk) of the channel.",
)
def jerk_rms(x, sample_rate):
    dx = np.diff(x) * sample_rate
    return float(np.sqrt(np.mean(dx**2))) if dx.size else 0.0


@register_feature(
    "jerk_mean_abs",
    family="mechanical",
    min_samples=2,
    description="Mean absolute jerk.",
)
def jerk_mean_abs(x, sample_rate):
    dx = np.diff(x) * sample_rate
    return float(np.mean(np.abs(dx))) if dx.size else 0.0


@register_feature(
    "jerk_max_abs",
    family="mechanical",
    min_samples=2,
    description="Peak absolute jerk.",
)
def jerk_max_abs(x, sample_rate):
    dx = np.diff(x) * sample_rate
    return float(np.max(np.abs(dx))) if dx.size else 0.0


@register_feature(
    "snap_rms",
    family="mechanical",
    min_samples=3,
    description="RMS of the second time-derivative (snap/jounce) of the channel.",
)
def snap_rms(x, sample_rate):
    d2x = np.diff(x, n=2) * (sample_rate**2)
    return float(np.sqrt(np.mean(d2x**2))) if d2x.size else 0.0


@register_feature(
    "velocity_change_proxy",
    family="mechanical",
    min_samples=2,
    description="Range of the running integral of the channel (proxy for velocity change if the channel is an acceleration).",
)
def velocity_change_proxy(x, sample_rate):
    dt = 1.0 / sample_rate
    v = np.cumsum(x) * dt
    return float(v.max() - v.min())


@register_feature(
    "kinetic_energy_proxy",
    family="mechanical",
    description="0.5 * mean(x^2), a kinetic-energy-shaped proxy.",
)
def kinetic_energy_proxy(x, sample_rate=None):
    return float(0.5 * np.mean(np.square(x)))


@register_feature(
    "dynamic_static_ratio",
    family="mechanical",
    min_samples=2,
    description="Fraction of mean-square power attributable to the AC (dynamic) component vs. the DC (static) component.",
)
def dynamic_static_ratio(x, sample_rate=None):
    dc = np.mean(x) ** 2
    ac = np.var(x)
    total = dc + ac
    return float(ac / total) if total > 0 else 0.0


@register_feature(
    "gravity_deviation",
    family="mechanical",
    scope="triaxial",
    description="Difference between the mean resultant magnitude and standard gravity (9.80665 m/s^2); meaningful for accelerometer channels.",
)
def gravity_deviation(arr, sample_rate=None):
    mags = np.linalg.norm(arr, axis=1)
    return float(np.mean(mags) - STANDARD_GRAVITY)
