import numpy as np
import pytest

from imu_features import REGISTRY, FeatureEngine, IMUWindow


def _make_window(n=256, sr=50.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    freq = 2.0
    accel = np.stack(
        [
            np.sin(2 * np.pi * freq * t),
            np.cos(2 * np.pi * freq * t),
            9.81 + 0.05 * np.sin(2 * np.pi * freq * t),
        ],
        axis=1,
    )
    gyro = 0.1 * rng.standard_normal((n, 3))
    return IMUWindow(sample_rate=sr, accel=accel, gyro=gyro)


def test_extract_returns_expected_keys():
    window = _make_window()
    engine = FeatureEngine(families=["statistical"])
    feats = engine.extract(window)
    assert "accel_x_statistical_mean" in feats
    assert "gyro_mag_statistical_std" in feats
    assert all(np.isfinite(v) for v in feats.values())


def test_family_selection_is_disjoint():
    window = _make_window()
    stat_engine = FeatureEngine(families=["statistical"])
    freq_engine = FeatureEngine(families=["frequency"])
    stat_feats = stat_engine.extract(window)
    freq_feats = freq_engine.extract(window)
    assert all("_statistical_" in k for k in stat_feats)
    assert all("_frequency_" in k for k in freq_feats)


def test_dominant_frequency_detection():
    window = _make_window(n=512, sr=100.0)
    engine = FeatureEngine(features=["dominant_frequency"])
    feats = engine.extract(window)
    assert abs(feats["accel_x_frequency_dominant_frequency"] - 2.0) < 0.5


def test_triaxial_features_have_triaxial_key():
    window = _make_window()
    engine = FeatureEngine(families=["geometrical"])
    feats = engine.extract(window)
    assert "accel_triaxial_geometrical_pca_linearity" in feats
    assert not any(k.endswith("_x_geometrical_pca_linearity") for k in feats)


def test_full_engine_all_families_finite_on_clean_signal():
    window = _make_window(n=300, sr=100.0)
    engine = FeatureEngine(families=REGISTRY.families())
    feats = engine.extract(window)
    assert len(feats) > 100
    non_finite = {k: v for k, v in feats.items() if not np.isfinite(v)}
    # sample_entropy can legitimately be undefined (nan) for some signals;
    # everything else should be finite.
    assert all("sample_entropy" in k for k in non_finite)


def test_extract_many_returns_dataframe_like():
    windows = [_make_window(seed=i) for i in range(3)]
    engine = FeatureEngine(families=["statistical", "magnitude"])
    result = engine.extract_many(windows)
    assert len(result) == 3


def test_unknown_selection_raises():
    with pytest.raises(ValueError):
        FeatureEngine(families=["does_not_exist"])
