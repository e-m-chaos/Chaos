"""Magnitude feature family.

Per-axis magnitude statistics (mean, std, ...) fall out for free because the
engine already treats the vector-norm signal as a fourth "mag" channel and
runs every channel-scope family over it. This module holds descriptors that
genuinely need the three raw axes at once — signal magnitude area, resultant
energy/intensity — rather than the norm alone.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature


@register_feature(
    "signal_magnitude_area",
    family="magnitude",
    scope="triaxial",
    description="Mean of |x| + |y| + |z| across the window (SMA, a standard HAR intensity feature).",
)
def signal_magnitude_area(arr, sample_rate=None):
    return float(np.mean(np.sum(np.abs(arr), axis=1)))


@register_feature(
    "resultant_energy",
    family="magnitude",
    scope="triaxial",
    description="Sum of squared resultant (vector-norm) magnitude over the window.",
)
def resultant_energy(arr, sample_rate=None):
    mag = np.linalg.norm(arr, axis=1)
    return float(np.sum(mag**2))


@register_feature(
    "movement_intensity",
    family="magnitude",
    scope="triaxial",
    description="Mean resultant magnitude over the window.",
)
def movement_intensity(arr, sample_rate=None):
    mag = np.linalg.norm(arr, axis=1)
    return float(np.mean(mag))


@register_feature(
    "magnitude_variability",
    family="magnitude",
    scope="triaxial",
    description="Standard deviation of the resultant magnitude over the window.",
)
def magnitude_variability(arr, sample_rate=None):
    mag = np.linalg.norm(arr, axis=1)
    return float(np.std(mag))


@register_feature(
    "peak_resultant",
    family="magnitude",
    scope="triaxial",
    description="Maximum resultant magnitude observed in the window.",
)
def peak_resultant(arr, sample_rate=None):
    mag = np.linalg.norm(arr, axis=1)
    return float(np.max(mag))
