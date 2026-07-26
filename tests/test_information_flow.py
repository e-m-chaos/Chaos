import numpy as np

from imu_features import FeatureEngine, IMUWindow
from imu_features.families import information_flow as info


def _window(accel_mag, gyro_mag, sr=100.0):
    n = len(accel_mag)
    accel = np.stack([accel_mag, np.zeros(n), np.zeros(n)], axis=1)
    gyro = np.stack([gyro_mag, np.zeros(n), np.zeros(n)], axis=1)
    return IMUWindow(sample_rate=sr, accel=accel, gyro=gyro)


def test_accel_driving_gyro_shows_dominant_forward_flow():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.standard_normal(n)
    g = np.zeros(n)
    for t in range(1, n):
        g[t] = 0.8 * a[t - 1] + 0.2 * rng.standard_normal()
    window = _window(a, g)

    result = info.transfer_entropy(window)
    assert result["accel_to_gyro"] > result["gyro_to_accel"]
    assert result["net"] > 0


def test_gyro_driving_accel_shows_dominant_reverse_flow():
    rng = np.random.default_rng(0)
    n = 500
    g = rng.standard_normal(n)
    a = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.8 * g[t - 1] + 0.2 * rng.standard_normal()
    window = _window(a, g)

    result = info.transfer_entropy(window)
    assert result["gyro_to_accel"] > result["accel_to_gyro"]
    assert result["net"] < 0


def test_independent_signals_show_much_weaker_flow_than_true_coupling():
    rng = np.random.default_rng(0)
    n = 500
    a_indep = rng.standard_normal(n)
    g_indep = rng.standard_normal(n)
    independent_result = info.transfer_entropy(_window(a_indep, g_indep))

    a_coupled = rng.standard_normal(n)
    g_coupled = np.zeros(n)
    for t in range(1, n):
        g_coupled[t] = 0.8 * a_coupled[t - 1] + 0.2 * rng.standard_normal()
    coupled_result = info.transfer_entropy(_window(a_coupled, g_coupled))

    assert coupled_result["accel_to_gyro"] > independent_result["accel_to_gyro"]


def test_engine_produces_information_flow_keys_and_skips_without_gyro():
    rng = np.random.default_rng(0)
    n = 100
    window = _window(rng.standard_normal(n), rng.standard_normal(n))
    engine = FeatureEngine(families=["information_flow"])
    feats = engine.extract(window)
    assert "fusion_information_flow_transfer_entropy_accel_to_gyro" in feats
    assert "fusion_information_flow_transfer_entropy_net" in feats
    assert all(np.isfinite(v) for v in feats.values())

    accel_only = IMUWindow(sample_rate=100.0, accel=rng.standard_normal((n, 3)))
    assert engine.extract(accel_only) == {}
