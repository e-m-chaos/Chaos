import numpy as np

from imu_features import FeatureEngine, IMUWindow
from imu_features.families import coupling


def _window(accel, gyro, sr=100.0):
    return IMUWindow(sample_rate=sr, accel=np.asarray(accel, dtype=float), gyro=np.asarray(gyro, dtype=float))


def test_helicity_positive_and_alignment_one_when_parallel():
    n = 50
    base = np.tile([1.0, 2.0, 3.0], (n, 1))
    window = _window(accel=base, gyro=2.0 * base)  # gyro is a positive scalar multiple of accel
    helicity = coupling.kinematic_helicity(window)
    alignment = coupling.kinematic_alignment(window)

    assert helicity["mean"] > 0
    assert helicity["chirality_index"] == 1.0
    assert abs(alignment["alignment_index_mean"] - 1.0) < 1e-9
    assert alignment["cross_magnitude_mean"] < 1e-9  # parallel vectors: zero cross product


def test_helicity_negative_and_alignment_minus_one_when_antiparallel():
    n = 50
    base = np.tile([1.0, 0.0, 0.0], (n, 1))
    window = _window(accel=base, gyro=-3.0 * base)
    helicity = coupling.kinematic_helicity(window)
    alignment = coupling.kinematic_alignment(window)

    assert helicity["mean"] < 0
    assert helicity["chirality_index"] == -1.0
    assert abs(alignment["alignment_index_mean"] - (-1.0)) < 1e-9


def test_orthogonal_vectors_have_zero_helicity_and_alignment():
    n = 50
    accel = np.tile([1.0, 0.0, 0.0], (n, 1))
    gyro = np.tile([0.0, 5.0, 0.0], (n, 1))
    window = _window(accel=accel, gyro=gyro)
    helicity = coupling.kinematic_helicity(window)
    alignment = coupling.kinematic_alignment(window)

    assert abs(helicity["mean"]) < 1e-9
    assert abs(alignment["alignment_index_mean"]) < 1e-9
    # cross product of orthogonal unit-scaled vectors has magnitude |a||g|
    assert abs(alignment["cross_magnitude_mean"] - 5.0) < 1e-9


def test_zero_gyro_gives_defined_zero_alignment_not_nan():
    n = 30
    accel = np.tile([1.0, 2.0, 3.0], (n, 1))
    gyro = np.zeros((n, 3))
    window = _window(accel=accel, gyro=gyro)
    alignment = coupling.kinematic_alignment(window)
    helicity = coupling.kinematic_helicity(window)

    assert np.isfinite(alignment["alignment_index_mean"])
    assert alignment["alignment_index_mean"] == 0.0
    assert helicity["mean"] == 0.0
    assert helicity["chirality_index"] == 0.0


def test_engine_produces_coupling_keys_when_accel_and_gyro_present():
    n = 100
    rng = np.random.default_rng(0)
    accel = rng.standard_normal((n, 3))
    gyro = rng.standard_normal((n, 3))
    window = _window(accel=accel, gyro=gyro)

    engine = FeatureEngine(families=["coupling"])
    feats = engine.extract(window)

    assert "fusion_coupling_kinematic_helicity_mean" in feats
    assert "fusion_coupling_kinematic_alignment_alignment_index_mean" in feats
    assert all(np.isfinite(v) for v in feats.values())


def test_engine_skips_coupling_when_gyro_missing():
    n = 50
    accel = np.random.default_rng(0).standard_normal((n, 3))
    window = IMUWindow(sample_rate=100.0, accel=accel)  # no gyro
    engine = FeatureEngine(families=["coupling"])
    assert engine.extract(window) == {}
