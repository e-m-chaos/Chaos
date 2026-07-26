"""Wavelet feature family: multi-resolution energy distribution via a
Haar discrete wavelet transform, implemented directly on numpy (no extra
dependency such as PyWavelets required). Each decomposition level splits the
signal into a coarser approximation and a detail (high-frequency residual);
the relative energy across levels summarizes where, in scale, the signal's
power is concentrated.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature

_SQRT2 = np.sqrt(2.0)
_DEFAULT_LEVELS = 4


def _haar_dwt_levels(x, levels):
    approx = np.asarray(x, dtype=float)
    details = []
    for _ in range(levels):
        n = len(approx)
        if n < 2:
            break
        if n % 2 == 1:
            approx = approx[:-1]
            n -= 1
        even, odd = approx[0::2], approx[1::2]
        details.append((even - odd) / _SQRT2)
        approx = (even + odd) / _SQRT2
    return approx, details


def _level_energies(x, levels):
    approx, details = _haar_dwt_levels(x, levels)
    return [float(np.sum(d**2)) for d in details] + [float(np.sum(approx**2))]


@register_feature(
    "wavelet_energy_entropy",
    family="wavelet",
    min_samples=8,
    description="Shannon entropy of the relative energy across Haar wavelet decomposition levels.",
)
def wavelet_energy_entropy(x, sample_rate=None, levels=_DEFAULT_LEVELS):
    energies = np.array(_level_energies(x, levels))
    total = energies.sum()
    if total <= 0:
        return 0.0
    p = energies / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _make_level_energy_ratio(level_idx, label):
    def feature(x, sample_rate=None, levels=_DEFAULT_LEVELS):
        energies = _level_energies(x, levels)
        total = sum(energies)
        if level_idx >= len(energies) or total <= 0:
            return 0.0
        return float(energies[level_idx] / total)

    feature.__name__ = f"wavelet_energy_ratio_{label}"
    return feature


for _i in range(_DEFAULT_LEVELS):
    register_feature(
        f"wavelet_energy_ratio_d{_i + 1}",
        family="wavelet",
        min_samples=8,
        description=f"Relative energy of Haar detail coefficients at decomposition level {_i + 1}.",
    )(_make_level_energy_ratio(_i, f"d{_i + 1}"))

register_feature(
    "wavelet_energy_ratio_approx",
    family="wavelet",
    min_samples=8,
    description="Relative energy of the final Haar approximation coefficients (coarsest scale).",
)(_make_level_energy_ratio(_DEFAULT_LEVELS, "approx"))
