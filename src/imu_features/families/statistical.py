"""Statistical (time-domain) feature family.

Classic distributional/shape descriptors of a single channel over a window:
central tendency, dispersion, higher moments, and simple morphological counts.
These are the workhorses of activity-recognition and gesture-recognition
feature sets (see Bulling et al.'s HAR survey for the canonical list).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import iqr as _iqr
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from ..core.registry import register_feature


@register_feature("mean", family="statistical", description="Arithmetic mean of the channel.")
def mean(x, sample_rate=None):
    return float(np.mean(x))


@register_feature("std", family="statistical", description="Standard deviation of the channel.")
def std(x, sample_rate=None):
    return float(np.std(x))


@register_feature("variance", family="statistical", description="Variance of the channel.")
def variance(x, sample_rate=None):
    return float(np.var(x))


@register_feature("minimum", family="statistical", description="Minimum value.")
def minimum(x, sample_rate=None):
    return float(np.min(x))


@register_feature("maximum", family="statistical", description="Maximum value.")
def maximum(x, sample_rate=None):
    return float(np.max(x))


@register_feature("range", family="statistical", description="Peak-to-peak range (max - min).")
def range_(x, sample_rate=None):
    return float(np.ptp(x))


@register_feature("median", family="statistical", description="Median value.")
def median(x, sample_rate=None):
    return float(np.median(x))


@register_feature("rms", family="statistical", description="Root-mean-square value.")
def rms(x, sample_rate=None):
    return float(np.sqrt(np.mean(np.square(x))))


@register_feature(
    "mean_absolute_deviation",
    family="statistical",
    description="Mean absolute deviation from the channel mean.",
)
def mean_absolute_deviation(x, sample_rate=None):
    return float(np.mean(np.abs(x - np.mean(x))))


@register_feature(
    "interquartile_range", family="statistical", description="Interquartile range (Q3 - Q1)."
)
def interquartile_range(x, sample_rate=None):
    return float(_iqr(x))


@register_feature(
    "skewness", family="statistical", min_samples=3, description="Distribution skewness."
)
def skewness(x, sample_rate=None):
    return float(_skew(x)) if np.std(x) > 0 else 0.0


@register_feature(
    "kurtosis", family="statistical", min_samples=4, description="Distribution excess kurtosis."
)
def kurtosis(x, sample_rate=None):
    return float(_kurtosis(x)) if np.std(x) > 0 else 0.0


@register_feature("percentile_25", family="statistical", description="25th percentile.")
def percentile_25(x, sample_rate=None):
    return float(np.percentile(x, 25))


@register_feature("percentile_75", family="statistical", description="75th percentile.")
def percentile_75(x, sample_rate=None):
    return float(np.percentile(x, 75))


@register_feature(
    "zero_crossing_rate",
    family="statistical",
    min_samples=2,
    description="Fraction of consecutive samples where the sign flips.",
)
def zero_crossing_rate(x, sample_rate=None):
    signs = np.sign(x)
    signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs)) > 0) / (len(x) - 1))


@register_feature(
    "mean_crossing_rate",
    family="statistical",
    min_samples=2,
    description="Fraction of consecutive samples where the signal crosses its own mean.",
)
def mean_crossing_rate(x, sample_rate=None):
    centered = np.asarray(x) - np.mean(x)
    signs = np.sign(centered)
    signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs)) > 0) / (len(x) - 1))


@register_feature("signal_energy", family="statistical", description="Sum of squared samples.")
def signal_energy(x, sample_rate=None):
    return float(np.sum(np.square(x)))


@register_feature(
    "waveform_length",
    family="statistical",
    min_samples=2,
    description="Cumulative length of the waveform (sum of |first differences|).",
)
def waveform_length(x, sample_rate=None):
    return float(np.sum(np.abs(np.diff(x))))


@register_feature(
    "slope_sign_changes",
    family="statistical",
    min_samples=3,
    description="Number of times the sign of the first difference changes.",
)
def slope_sign_changes(x, sample_rate=None):
    d = np.diff(x)
    return float(np.sum(np.diff(np.sign(d)) != 0))


@register_feature(
    "coefficient_of_variation",
    family="statistical",
    description="Standard deviation normalized by the mean.",
)
def coefficient_of_variation(x, sample_rate=None):
    m = np.mean(x)
    return float(np.std(x) / m) if m != 0 else 0.0


@register_feature(
    "peak_count",
    family="statistical",
    min_samples=3,
    description="Number of local maxima (simple 3-point comparison).",
)
def peak_count(x, sample_rate=None):
    x = np.asarray(x)
    greater_prev = x[1:-1] > x[:-2]
    greater_next = x[1:-1] > x[2:]
    return float(np.sum(greater_prev & greater_next))
