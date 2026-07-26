"""Geometrical feature family: shape and orientation descriptors of the
triaxial trajectory traced by a sensor over the window.

PCA descriptors (linearity/planarity/sphericity) classify whether the motion
traces out a line, a plane, or fills 3-D space evenly — a standard shape
signature from point-cloud/structure-tensor analysis. The angle features
give a quasi-static orientation estimate, valid when the sensor axis is
dominated by gravity (accelerometer) or a stable reference field
(magnetometer).
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature


def _pca_eigs(arr):
    centered = arr - arr.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return np.sort(eigvals)[::-1]


@register_feature(
    "pca_linearity",
    family="geometrical",
    scope="triaxial",
    min_samples=4,
    description="(lambda1 - lambda2) / lambda1 of the covariance eigenvalues; 1 = motion confined to a line.",
)
def pca_linearity(arr, sample_rate=None):
    l1, l2, _ = _pca_eigs(arr)
    return float((l1 - l2) / l1) if l1 > 0 else 0.0


@register_feature(
    "pca_planarity",
    family="geometrical",
    scope="triaxial",
    min_samples=4,
    description="(lambda2 - lambda3) / lambda1; 1 = motion confined to a plane.",
)
def pca_planarity(arr, sample_rate=None):
    l1, l2, l3 = _pca_eigs(arr)
    return float((l2 - l3) / l1) if l1 > 0 else 0.0


@register_feature(
    "pca_sphericity",
    family="geometrical",
    scope="triaxial",
    min_samples=4,
    description="lambda3 / lambda1; 1 = motion fills 3-D space isotropically.",
)
def pca_sphericity(arr, sample_rate=None):
    l1, _, l3 = _pca_eigs(arr)
    return float(l3 / l1) if l1 > 0 else 0.0


@register_feature(
    "pca_dominant_variance_ratio",
    family="geometrical",
    scope="triaxial",
    min_samples=4,
    description="Fraction of total variance explained by the first principal axis.",
)
def pca_dominant_variance_ratio(arr, sample_rate=None):
    eigs = _pca_eigs(arr)
    total = eigs.sum()
    return float(eigs[0] / total) if total > 0 else 0.0


@register_feature(
    "inclination_angle",
    family="geometrical",
    scope="triaxial",
    description="Angle (degrees) between the mean vector and the z-axis.",
)
def inclination_angle(arr, sample_rate=None):
    mean_vec = arr.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(mean_vec[2] / norm, -1.0, 1.0))))


@register_feature(
    "roll_angle",
    family="geometrical",
    scope="triaxial",
    description="Quasi-static roll estimate (degrees) assuming the z-axis tracks a stable reference (e.g. gravity).",
)
def roll_angle(arr, sample_rate=None):
    mean_vec = arr.mean(axis=0)
    return float(np.degrees(np.arctan2(mean_vec[1], mean_vec[2])))


@register_feature(
    "pitch_angle",
    family="geometrical",
    scope="triaxial",
    description="Quasi-static pitch estimate (degrees) assuming the z-axis tracks a stable reference (e.g. gravity).",
)
def pitch_angle(arr, sample_rate=None):
    mean_vec = arr.mean(axis=0)
    return float(np.degrees(np.arctan2(-mean_vec[0], np.sqrt(mean_vec[1] ** 2 + mean_vec[2] ** 2))))


@register_feature(
    "azimuth_angle",
    family="geometrical",
    scope="triaxial",
    description="Heading estimate (degrees) in the x-y plane.",
)
def azimuth_angle(arr, sample_rate=None):
    mean_vec = arr.mean(axis=0)
    return float(np.degrees(np.arctan2(mean_vec[1], mean_vec[0])))


@register_feature(
    "bounding_box_volume",
    family="geometrical",
    scope="triaxial",
    description="Volume of the axis-aligned bounding box spanned by the trajectory.",
)
def bounding_box_volume(arr, sample_rate=None):
    ranges = arr.max(axis=0) - arr.min(axis=0)
    return float(np.prod(ranges))
