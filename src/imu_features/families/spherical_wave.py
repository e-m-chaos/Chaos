"""Spherical-wave feature family: wave analysis on a non-Euclidean manifold.

Every other family treats a channel as a signal on the real line (flat) or
a raw vector in R^3. This family instead normalizes the triaxial vector to
a *direction* u(t) = v(t)/|v(t)| — a point on the unit 2-sphere S^2, a
curved Riemannian manifold with constant positive curvature, not a flat
Euclidean space. Two pieces of mathematics native to that curved setting:

1. **Directional (Fisher) statistics.** On a flat line, dispersion is
   variance; on the sphere, the intrinsically curvature-correct analog is
   `1 - R`, where `R = |mean(u_i)|` is the mean resultant length — Fisher's
   spherical variance (Fisher, 1953, "Dispersion on a sphere"). Its
   companion, the concentration parameter `kappa` of the fitted von
   Mises-Fisher distribution (Mardia & Jupp's `p=3` approximation
   `kappa ~= R(3 - R^2) / (1 - R^2)`), measures how tightly the direction
   samples cluster around a single point on the sphere.

2. **Spherical harmonics.** On the real line, "waves" are sines and
   cosines — the eigenfunctions of the flat Laplacian — and the Fourier
   transform expands a signal in that basis. On the sphere, the
   eigenfunctions of the Laplace-Beltrami operator are the *spherical
   harmonics* Y_lm, and expanding the direction samples in that basis gives
   an angular power spectrum C_l, the sphere's exact analog of a Fourier
   power spectrum for waves that live on curved, not flat, space. We
   compute the dipole (l=1) and quadrupole (l=2) power — the two lowest
   nontrivial "wave modes" of the sphere — from the empirical harmonic
   moments of the direction samples, the same estimator used to build
   angular power spectra from discrete/scattered directional data (e.g.
   CMB temperature maps, crystallographic pole figures).

A third piece treats the **geodesic** (great-circle) distance between
consecutive direction samples, `arccos(u(t) . u(t+1))`, as this family's
literal propagating wave: that distance sequence is the sphere's intrinsic
angular-speed signal, and its FFT gives a dominant frequency and spectral
energy that measure oscillation *of the manifold trajectory itself* —
distinct from the `frequency` family's features, which never leave flat
amplitude space.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature

_EPS = 1e-12


def _unit_directions(arr):
    norms = np.linalg.norm(arr, axis=1)
    nonzero = norms > _EPS
    unit = np.zeros_like(arr)
    unit[nonzero] = arr[nonzero] / norms[nonzero, None]
    unit[~nonzero] = np.array([0.0, 0.0, 1.0])  # degenerate zero-vector fallback
    return unit


def _resultant_length(u):
    return float(np.linalg.norm(np.mean(u, axis=0)))


@register_feature(
    "spherical_dispersion",
    family="spherical_wave",
    scope="triaxial",
    min_samples=3,
    description="Fisher's spherical variance (1 - R) of the direction samples on S^2: 0 = perfectly concentrated direction, 1 = maximally dispersed.",
)
def spherical_dispersion(arr, sample_rate=None):
    R = _resultant_length(_unit_directions(arr))
    return float(1.0 - R)


@register_feature(
    "spherical_concentration",
    family="spherical_wave",
    scope="triaxial",
    min_samples=3,
    description="Von Mises-Fisher concentration kappa estimated from the mean resultant length; large = directions tightly clustered, small = widely dispersed.",
)
def spherical_concentration(arr, sample_rate=None):
    R = min(_resultant_length(_unit_directions(arr)), 0.999999)
    return float(R * (3.0 - R**2) / (1.0 - R**2))


@register_feature(
    "geodesic_path_length",
    family="spherical_wave",
    scope="triaxial",
    min_samples=3,
    description="Total great-circle (geodesic) distance traveled by the direction samples on S^2 -- the manifold-intrinsic analog of waveform length.",
)
def geodesic_path_length(arr, sample_rate=None):
    u = _unit_directions(arr)
    cos_d = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(cos_d)))


def _geodesic_angular_speed(arr, sample_rate):
    u = _unit_directions(arr)
    cos_d = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return np.arccos(cos_d) * sample_rate


@register_feature(
    "geodesic_dominant_frequency",
    family="spherical_wave",
    scope="triaxial",
    min_samples=8,
    description="Dominant frequency of the sphere's intrinsic angular-speed signal (FFT of consecutive geodesic distances) -- the frequency of the 'wave' the direction traces on the curved manifold.",
)
def geodesic_dominant_frequency(arr, sample_rate):
    speed = _geodesic_angular_speed(arr, sample_rate)
    n = len(speed)
    if n < 4:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mags = np.abs(np.fft.rfft(speed - speed.mean()))
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(mags[1:])) + 1
    return float(freqs[idx])


@register_feature(
    "geodesic_spectral_energy",
    family="spherical_wave",
    scope="triaxial",
    min_samples=8,
    description="Total spectral energy of the sphere's intrinsic angular-speed signal.",
)
def geodesic_spectral_energy(arr, sample_rate):
    speed = _geodesic_angular_speed(arr, sample_rate)
    if len(speed) < 4:
        return 0.0
    mags = np.abs(np.fft.rfft(speed - speed.mean()))
    return float(np.sum(mags**2))


def _real_sph_harmonics_l1(u):
    """Real spherical harmonics of degree 1 (the dipole basis)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    c = np.sqrt(3.0 / (4 * np.pi))
    return np.stack([c * y, c * z, c * x], axis=1)  # m = -1, 0, +1


def _real_sph_harmonics_l2(u):
    """Real spherical harmonics of degree 2 (the quadrupole basis; the same
    functional forms as the atomic d-orbitals: dxy, dyz, dz^2, dxz, dx^2-y^2)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    c1 = 0.5 * np.sqrt(15.0 / np.pi)
    c0 = 0.25 * np.sqrt(5.0 / np.pi)
    c2 = 0.25 * np.sqrt(15.0 / np.pi)
    return np.stack(
        [c1 * x * y, c1 * y * z, c0 * (3 * z**2 - 1), c1 * x * z, c2 * (x**2 - y**2)],
        axis=1,
    )


def _angular_power(u, harmonics_fn, n_modes):
    a_lm = np.mean(harmonics_fn(u), axis=0)
    return float(np.sum(a_lm**2) / n_modes)


@register_feature(
    "spherical_dipole_power",
    family="spherical_wave",
    scope="triaxial",
    min_samples=3,
    description="Angular power spectrum coefficient C_1: energy of the direction samples' l=1 spherical-harmonic (dipole) mode, the sphere's lowest wave mode.",
)
def spherical_dipole_power(arr, sample_rate=None):
    return _angular_power(_unit_directions(arr), _real_sph_harmonics_l1, 3)


@register_feature(
    "spherical_quadrupole_power",
    family="spherical_wave",
    scope="triaxial",
    min_samples=5,
    description="Angular power spectrum coefficient C_2: energy of the direction samples' l=2 spherical-harmonic (quadrupole) mode -- magnitude-invariant, purely directional anisotropy, unlike PCA on the raw vectors.",
)
def spherical_quadrupole_power(arr, sample_rate=None):
    return _angular_power(_unit_directions(arr), _real_sph_harmonics_l2, 5)


@register_feature(
    "spherical_spectral_ratio",
    family="spherical_wave",
    scope="triaxial",
    min_samples=5,
    description="Ratio of quadrupole to dipole angular power (C_2 / C_1): how much directional structure is higher-order/anisotropic vs. simple mean-direction bias.",
)
def spherical_spectral_ratio(arr, sample_rate=None):
    u = _unit_directions(arr)
    c1 = _angular_power(u, _real_sph_harmonics_l1, 3)
    c2 = _angular_power(u, _real_sph_harmonics_l2, 5)
    return float(c2 / c1) if c1 > 1e-12 else 0.0
