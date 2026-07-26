import numpy as np

from imu_features import FeatureEngine, IMUWindow
from imu_features.families import random_matrix as rm


def test_independent_noise_has_no_significant_modes():
    rng = np.random.default_rng(0)
    n = 2000
    accel = rng.standard_normal((n, 3))
    gyro = rng.standard_normal((n, 3))
    window = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro)

    spectrum = rm.rmt_spectrum(window)
    assert spectrum["significant_mode_count"] <= 1  # allow rare sampling fluctuation
    assert spectrum["effective_rank"] > 4.5  # close to the full p=6 for pure noise


def test_common_driving_signal_produces_one_dominant_broadly_shared_mode():
    rng = np.random.default_rng(0)
    n = 2000
    t = np.arange(n) / 100.0
    s = np.sin(2 * np.pi * 1.3 * t)
    accel = np.outer(s, [1.0, 2.0, -1.5]) + 0.05 * rng.standard_normal((n, 3))
    gyro = np.outer(s, [0.5, -2.0, 1.0]) + 0.05 * rng.standard_normal((n, 3))
    window = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro)

    spectrum = rm.rmt_spectrum(window)
    leading = rm.rmt_leading_mode(window)

    assert spectrum["significant_mode_count"] == 1
    assert spectrum["largest_eigenvalue_ratio"] > 0.9
    assert spectrum["effective_rank"] < 1.5
    assert leading["effective_channel_count"] > 5.0  # spread across ~all 6 channels


def test_coupling_confined_to_two_channels_gives_low_effective_channel_count():
    rng = np.random.default_rng(0)
    n = 2000
    accel = rng.standard_normal((n, 3))
    gyro = rng.standard_normal((n, 3))
    gyro[:, 0] = accel[:, 0] + 0.05 * rng.standard_normal(n)  # only these two channels coupled
    window = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro)

    leading = rm.rmt_leading_mode(window)
    assert 1.5 < leading["effective_channel_count"] < 3.0


def test_effective_channel_count_bounded_by_channel_dimension():
    rng = np.random.default_rng(0)
    n = 500
    accel = rng.standard_normal((n, 3))
    gyro = rng.standard_normal((n, 3))
    mag = rng.standard_normal((n, 3))
    window = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro, mag=mag)

    leading = rm.rmt_leading_mode(window)
    assert 1.0 <= leading["effective_channel_count"] <= 9.0 + 1e-6


def test_mag_included_when_present_increases_channel_dimension():
    rng = np.random.default_rng(0)
    n = 500
    accel = rng.standard_normal((n, 3))
    gyro = rng.standard_normal((n, 3))
    mag = rng.standard_normal((n, 3))
    window_no_mag = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro)
    window_with_mag = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro, mag=mag)

    upper_no_mag = rm.rmt_spectrum(window_no_mag)["mp_upper_bound"]
    upper_with_mag = rm.rmt_spectrum(window_with_mag)["mp_upper_bound"]
    # more channels (p) at fixed n means a wider noise bulk (larger q = p/n)
    assert upper_with_mag > upper_no_mag


def test_engine_produces_rmt_keys_and_skips_without_gyro():
    rng = np.random.default_rng(0)
    n = 100
    accel = rng.standard_normal((n, 3))
    gyro = rng.standard_normal((n, 3))
    window = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro)
    engine = FeatureEngine(families=["random_matrix"])
    feats = engine.extract(window)
    assert "fusion_random_matrix_rmt_spectrum_significant_mode_count" in feats
    assert "fusion_random_matrix_rmt_leading_mode_participation_ratio" in feats
    assert all(np.isfinite(v) for v in feats.values())

    accel_only_window = IMUWindow(sample_rate=100.0, accel=accel)
    assert engine.extract(accel_only_window) == {}
