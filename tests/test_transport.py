import numpy as np

from imu_features import FeatureEngine, IMUWindow
from imu_features.families import transport


def test_gaussian_distance_low_for_gaussian_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    assert transport.wasserstein_gaussian_distance(x, None) < 0.1


def test_gaussian_distance_high_for_bimodal_data():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-5, 0.5, 1000), rng.normal(5, 0.5, 1000)])
    assert transport.wasserstein_gaussian_distance(x, None) > 1.0


def test_gaussian_distance_zero_for_constant_signal():
    x = np.ones(50)
    assert transport.wasserstein_gaussian_distance(x, None) == 0.0


def test_uniform_distance_low_for_uniform_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 2000)
    assert transport.wasserstein_uniform_distance(x, None) < 0.1


def test_uniform_distance_high_for_gaussian_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    # a Gaussian is far more concentrated than a uniform spread over its own range
    assert transport.wasserstein_uniform_distance(x, None) > transport.wasserstein_gaussian_distance(x, None)


def test_uniform_distance_zero_for_constant_signal():
    x = np.ones(50)
    assert transport.wasserstein_uniform_distance(x, None) == 0.0


def test_split_half_distance_low_for_stationary_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert transport.wasserstein_split_half_distance(x, None) < 0.3


def test_split_half_distance_high_for_regime_shift():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(500), 10.0 + rng.standard_normal(500)])
    assert transport.wasserstein_split_half_distance(x, None) > 5.0


def test_engine_selects_transport_family():
    rng = np.random.default_rng(0)
    accel = rng.standard_normal((200, 3)) + [0, 0, 9.81]
    window = IMUWindow(sample_rate=100.0, accel=accel)
    engine = FeatureEngine(families=["transport"])
    feats = engine.extract(window)
    assert "accel_x_transport_wasserstein_gaussian_distance" in feats
    assert all(np.isfinite(v) for v in feats.values())
