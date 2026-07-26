import numpy as np

from imu_features import FeatureEngine, IMUWindow
from imu_features.families import spherical_wave as sw


def test_dispersion_zero_for_constant_direction():
    arr = np.tile([1.0, 2.0, 3.0], (50, 1))  # constant direction, varying magnitude
    assert sw.spherical_dispersion(arr, None) < 1e-9


def test_dispersion_high_for_isotropic_random_directions():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3000, 3))  # isotropic -> near-uniform on the sphere
    assert sw.spherical_dispersion(arr, None) > 0.8


def test_concentration_higher_for_tight_cluster_than_isotropic():
    rng = np.random.default_rng(0)
    tight = np.tile([0.0, 0.0, 1.0], (500, 1)) + 0.02 * rng.standard_normal((500, 3))
    isotropic = rng.standard_normal((500, 3))
    assert sw.spherical_concentration(tight, None) > sw.spherical_concentration(isotropic, None)


def test_geodesic_path_length_zero_for_still_direction():
    arr = np.tile([0.0, 0.0, 1.0], (50, 1))
    assert sw.geodesic_path_length(arr, None) < 1e-9


def test_geodesic_path_length_positive_when_direction_moves():
    n = 50
    t = np.linspace(0, np.pi, n)
    arr = np.stack([np.sin(t), np.zeros(n), np.cos(t)], axis=1)
    assert sw.geodesic_path_length(arr, None) > 1.0


def _oscillating_direction(n, sr, amp_deg, freq):
    t = np.arange(n) / sr
    theta = np.radians(amp_deg) * np.sin(2 * np.pi * freq * t)
    return np.stack([np.sin(theta), np.zeros(n), np.cos(theta)], axis=1)


def test_geodesic_dominant_frequency_tracks_oscillation_rate():
    sr = 100.0
    slow = _oscillating_direction(500, sr, amp_deg=20, freq=1.0)
    fast = _oscillating_direction(500, sr, amp_deg=20, freq=4.0)
    f_slow = sw.geodesic_dominant_frequency(slow, sr)
    f_fast = sw.geodesic_dominant_frequency(fast, sr)
    assert f_fast > f_slow


def test_geodesic_spectral_energy_nonnegative():
    arr = _oscillating_direction(300, 100.0, amp_deg=15, freq=2.0)
    assert sw.geodesic_spectral_energy(arr, 100.0) >= 0.0


def test_dipole_power_higher_for_clustered_than_isotropic():
    rng = np.random.default_rng(0)
    clustered = np.tile([1.0, 0.0, 0.0], (1000, 1)) + 0.05 * rng.standard_normal((1000, 3))
    isotropic = rng.standard_normal((1000, 3))
    assert sw.spherical_dipole_power(clustered, None) > sw.spherical_dipole_power(isotropic, None)


def test_quadrupole_power_higher_for_equatorial_band_than_isotropic():
    rng = np.random.default_rng(0)
    n = 4000
    theta = rng.uniform(0, 2 * np.pi, n)
    # directions confined near the equator (z ~ 0): strong l=2 anisotropy
    band = np.stack([np.cos(theta), np.sin(theta), 0.05 * rng.standard_normal(n)], axis=1)
    isotropic = rng.standard_normal((n, 3))
    assert sw.spherical_quadrupole_power(band, None) > sw.spherical_quadrupole_power(isotropic, None)


def test_spectral_ratio_finite_and_nonnegative():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((200, 3))
    val = sw.spherical_spectral_ratio(arr, None)
    assert np.isfinite(val)
    assert val >= 0.0


def test_zero_vector_rows_do_not_produce_nan():
    arr = np.zeros((50, 3))
    assert np.isfinite(sw.spherical_dispersion(arr, None))
    assert np.isfinite(sw.spherical_concentration(arr, None))
    assert np.isfinite(sw.spherical_dipole_power(arr, None))


def test_engine_selects_spherical_wave_family():
    rng = np.random.default_rng(0)
    accel = rng.standard_normal((200, 3)) + [0, 0, 9.81]
    window = IMUWindow(sample_rate=100.0, accel=accel)
    engine = FeatureEngine(families=["spherical_wave"])
    feats = engine.extract(window)
    assert "accel_triaxial_spherical_wave_spherical_dispersion" in feats
    assert all(np.isfinite(v) for v in feats.values())
