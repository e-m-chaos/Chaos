"""Coupling feature family: exterior-algebra invariants of the paired
accelerometer + gyroscope vectors, exposing how strongly — and in what
sense — linear and angular motion are coupled.

At each sample, accel(t) and gyro(t) are two vectors in R^3. Their **dot
product** a(t).g(t) is a pseudoscalar directly analogous to *kinetic
helicity* in fluid mechanics (v.omega) and to the linear/angular momentum
coupling term in rigid-body dynamics: large and positive during
"corkscrew"/drilling motion where translation and rotation are aligned,
large and negative for the mirror-image (opposite-handed) motion, and near
zero when the two vectors are orthogonal or either is negligible. Their
**cross product** a(t) x g(t) is the corresponding bivector (the Clifford/
geometric-algebra wedge product collapses to the familiar cross product in
3D); its magnitude is the *unsigned* coupling strength, independent of
sign convention. Normalizing the dot product by both vector norms gives a
unitless *alignment index* — the cosine of the angle between them, in
[-1, 1] — comparable across sensors, subjects, or devices regardless of
amplitude.

This is a genuinely different feature family from what's already here:
unlike `crossaxis` (which relates the x/y/z channels *within* one sensor)
or `orientation` (which fuses accel+gyro into an attitude estimate),
`coupling` relates the two *sensors* to each other via vector algebra,
sample-by-sample, without attempting to estimate orientation at all.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature


def _dot_per_sample(a, g):
    return np.sum(a * g, axis=1)


@register_feature(
    "kinematic_helicity",
    family="coupling",
    scope="fusion",
    requires=("accel", "gyro"),
    min_samples=2,
    description=(
        "Sample-wise dot product of accel and gyro vectors — a fluid-helicity-like "
        "invariant of linear/angular motion coupling, plus a net-handedness (chirality) index."
    ),
)
def kinematic_helicity(window):
    h = _dot_per_sample(window.accel, window.gyro)
    n = len(h)
    n_pos = int(np.sum(h > 0))
    n_neg = int(np.sum(h < 0))
    return {
        "mean": float(np.mean(h)),
        "std": float(np.std(h)),
        "abs_mean": float(np.mean(np.abs(h))),
        "chirality_index": float((n_pos - n_neg) / n) if n > 0 else 0.0,
    }


@register_feature(
    "kinematic_alignment",
    family="coupling",
    scope="fusion",
    requires=("accel", "gyro"),
    min_samples=2,
    description=(
        "Unitless cosine-of-angle alignment between accel and gyro vectors, and the "
        "magnitude of their cross product (the coupling bivector norm)."
    ),
)
def kinematic_alignment(window):
    a, g = window.accel, window.gyro
    a_norm = np.linalg.norm(a, axis=1)
    g_norm = np.linalg.norm(g, axis=1)
    denom = a_norm * g_norm

    cos_theta = np.zeros(len(a))
    valid = denom > 1e-12
    cos_theta[valid] = _dot_per_sample(a, g)[valid] / denom[valid]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    cross_mag = np.linalg.norm(np.cross(a, g), axis=1)

    return {
        "alignment_index_mean": float(np.mean(cos_theta)),
        "alignment_index_std": float(np.std(cos_theta)),
        "cross_magnitude_mean": float(np.mean(cross_mag)),
        "cross_magnitude_std": float(np.std(cross_mag)),
    }
