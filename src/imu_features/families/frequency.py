"""Frequency-domain feature family, built on the real FFT of the (mean-removed)
channel. Covers spectral shape (centroid/spread/skew/kurtosis), spectral
information content (entropy, flatness), and coarse band-power ratios tuned
to typical human-motion bandwidths.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from ..core.registry import register_feature


def _spectrum(x, sample_rate):
    x = np.asarray(x, dtype=float)
    n = len(x)
    windowed = x - x.mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mags = np.abs(np.fft.rfft(windowed))
    power = mags**2
    return freqs, mags, power


@register_feature(
    "dominant_frequency",
    family="frequency",
    min_samples=4,
    description="Frequency of the largest spectral peak, excluding DC.",
)
def dominant_frequency(x, sample_rate):
    freqs, mags, _ = _spectrum(x, sample_rate)
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(mags[1:])) + 1
    return float(freqs[idx])


@register_feature(
    "dominant_frequency_amplitude",
    family="frequency",
    min_samples=4,
    description="FFT magnitude at the dominant (non-DC) frequency.",
)
def dominant_frequency_amplitude(x, sample_rate):
    _, mags, _ = _spectrum(x, sample_rate)
    if len(mags) <= 1:
        return 0.0
    return float(np.max(mags[1:]))


@register_feature(
    "spectral_energy", family="frequency", min_samples=4, description="Total power spectral energy."
)
def spectral_energy(x, sample_rate):
    _, _, power = _spectrum(x, sample_rate)
    return float(np.sum(power))


@register_feature(
    "spectral_entropy",
    family="frequency",
    min_samples=4,
    description="Shannon entropy of the normalized power spectrum (0 = tonal, 1 = flat/noisy).",
)
def spectral_entropy(x, sample_rate):
    _, _, power = _spectrum(x, sample_rate)
    total = power.sum()
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


@register_feature(
    "spectral_centroid",
    family="frequency",
    min_samples=4,
    description="Amplitude-weighted mean frequency ('center of mass' of the spectrum).",
)
def spectral_centroid(x, sample_rate):
    freqs, mags, _ = _spectrum(x, sample_rate)
    total = mags.sum()
    return float(np.sum(freqs * mags) / total) if total > 0 else 0.0


@register_feature(
    "spectral_spread",
    family="frequency",
    min_samples=4,
    description="Amplitude-weighted standard deviation of frequency around the spectral centroid.",
)
def spectral_spread(x, sample_rate):
    freqs, mags, _ = _spectrum(x, sample_rate)
    total = mags.sum()
    if total == 0:
        return 0.0
    centroid = np.sum(freqs * mags) / total
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mags) / total))


@register_feature(
    "spectral_skewness",
    family="frequency",
    min_samples=4,
    description="Skewness of the spectral magnitude distribution.",
)
def spectral_skewness(x, sample_rate):
    _, mags, _ = _spectrum(x, sample_rate)
    return float(_skew(mags)) if np.std(mags) > 0 else 0.0


@register_feature(
    "spectral_kurtosis",
    family="frequency",
    min_samples=4,
    description="Excess kurtosis of the spectral magnitude distribution.",
)
def spectral_kurtosis(x, sample_rate):
    _, mags, _ = _spectrum(x, sample_rate)
    return float(_kurtosis(mags)) if np.std(mags) > 0 else 0.0


@register_feature(
    "spectral_flatness",
    family="frequency",
    min_samples=4,
    description="Geometric-mean / arithmetic-mean of the power spectrum (Wiener entropy; 1 = white noise, 0 = tonal).",
)
def spectral_flatness(x, sample_rate):
    _, _, power = _spectrum(x, sample_rate)
    power = power[power > 0]
    if power.size == 0:
        return 0.0
    gmean = np.exp(np.mean(np.log(power)))
    amean = np.mean(power)
    return float(gmean / amean) if amean > 0 else 0.0


@register_feature(
    "spectral_rolloff",
    family="frequency",
    min_samples=4,
    description="Frequency below which 85% of the spectral power is contained.",
)
def spectral_rolloff(x, sample_rate, rolloff=0.85):
    freqs, _, power = _spectrum(x, sample_rate)
    total = power.sum()
    if total == 0:
        return 0.0
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, rolloff * total))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _band_power_ratio(x, sample_rate, low, high):
    freqs, _, power = _spectrum(x, sample_rate)
    mask = (freqs >= low) & (freqs < high)
    total = power.sum()
    return float(power[mask].sum() / total) if total > 0 else 0.0


@register_feature(
    "band_power_low",
    family="frequency",
    min_samples=4,
    description="Fraction of spectral power in 0-3 Hz (postural/slow-motion band).",
)
def band_power_low(x, sample_rate):
    return _band_power_ratio(x, sample_rate, 0.0, 3.0)


@register_feature(
    "band_power_mid",
    family="frequency",
    min_samples=4,
    description="Fraction of spectral power in 3-6 Hz (locomotion band, e.g. walking cadence).",
)
def band_power_mid(x, sample_rate):
    return _band_power_ratio(x, sample_rate, 3.0, 6.0)


@register_feature(
    "band_power_high",
    family="frequency",
    min_samples=4,
    description="Fraction of spectral power above 6 Hz (fast/vibratory motion band).",
)
def band_power_high(x, sample_rate):
    return _band_power_ratio(x, sample_rate, 6.0, np.inf)
