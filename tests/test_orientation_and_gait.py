import numpy as np

from imu_features import FeatureEngine, IMUWindow
from imu_features.families import gait, orientation


def _tilted_static_window(n=200, sr=100.0, roll_deg=30.0):
    roll = np.radians(roll_deg)
    g = 9.80665
    accel_vec = np.array([0.0, g * np.sin(roll), g * np.cos(roll)])
    accel = np.tile(accel_vec, (n, 1)) + 0.01 * np.random.default_rng(0).standard_normal((n, 3))
    gyro = np.zeros((n, 3))
    return IMUWindow(sample_rate=sr, accel=accel, gyro=gyro)


def test_complementary_tilt_recovers_static_roll():
    window = _tilted_static_window(roll_deg=30.0)
    result = orientation.complementary_tilt(window)
    assert abs(result["roll_mean"] - 30.0) < 2.0


def test_engine_skips_fusion_feature_when_sensor_missing():
    accel = np.tile([0.0, 0.0, 9.80665], (100, 1))
    window = IMUWindow(sample_rate=100.0, accel=accel)  # no gyro
    engine = FeatureEngine(families=["orientation"])
    feats = engine.extract(window)
    assert feats == {}


def test_engine_includes_fusion_keys_when_sensors_present():
    window = _tilted_static_window()
    engine = FeatureEngine(features=["complementary_tilt"])
    feats = engine.extract(window)
    assert "fusion_orientation_complementary_tilt_roll_mean" in feats


def test_heading_requires_mag_and_accel():
    n = 100
    rng = np.random.default_rng(0)
    accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
    mag = np.tile([20.0, 0.0, -40.0], (n, 1)) + 0.1 * rng.standard_normal((n, 3))
    window = IMUWindow(sample_rate=100.0, accel=accel, mag=mag)
    result = orientation.tilt_compensated_heading(window)
    assert 0.0 <= result["heading_mean"] < 360.0
    assert result["heading_circular_std"] >= 0.0


def _walking_signal(n=1000, sr=100.0, step_freq=1.8):
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    return 9.81 + 1.2 * np.abs(np.sin(2 * np.pi * step_freq * t)) + 0.05 * rng.standard_normal(n)


def test_gait_step_count_matches_expected_order_of_magnitude():
    sr = 100.0
    duration_s = 10.0
    step_freq = 1.8
    x = _walking_signal(n=int(duration_s * sr), sr=sr, step_freq=step_freq)
    n_steps = gait.step_count(x, sr)
    # |sin(2*pi*f*t)| has a peak every half-period, i.e. at 2*f
    expected = duration_s * step_freq * 2
    assert abs(n_steps - expected) <= 3


def test_gait_cadence_positive_for_walking_signal():
    sr = 100.0
    x = _walking_signal(n=1000, sr=sr)
    assert gait.cadence(x, sr) > 0


def test_gait_step_interval_cv_nonnegative():
    sr = 100.0
    x = _walking_signal(n=1000, sr=sr)
    assert gait.step_interval_cv(x, sr) >= 0.0


def test_gait_family_registered_and_selectable():
    engine = FeatureEngine(families=["gait"])
    accel = np.tile(
        [0.0, 0.0, 9.81], (500, 1)
    ) + np.stack([np.zeros(500), np.zeros(500), 1.0 * np.sin(2 * np.pi * 1.8 * np.arange(500) / 100.0)], axis=1)
    window = IMUWindow(sample_rate=100.0, accel=accel)
    feats = engine.extract(window)
    assert any("_gait_step_count" in k for k in feats)
