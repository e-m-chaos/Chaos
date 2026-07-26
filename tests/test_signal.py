import numpy as np
import pytest

from imu_features import IMUWindow


def test_valid_window():
    accel = np.random.randn(100, 3)
    gyro = np.random.randn(100, 3)
    w = IMUWindow(sample_rate=100.0, accel=accel, gyro=gyro)
    assert w.n_samples == 100
    assert set(w.sensors()) == {"accel", "gyro"}
    assert w.magnitude("accel").shape == (100,)
    assert w.duration == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        IMUWindow(sample_rate=100.0, accel=np.zeros((100, 3)), gyro=np.zeros((50, 3)))


def test_missing_sensor_raises():
    with pytest.raises(ValueError):
        IMUWindow(sample_rate=100.0)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        IMUWindow(sample_rate=100.0, accel=np.zeros((100, 2)))


def test_non_positive_sample_rate_raises():
    with pytest.raises(ValueError):
        IMUWindow(sample_rate=0.0, accel=np.zeros((10, 3)))


def test_magnitude_missing_sensor_raises():
    w = IMUWindow(sample_rate=100.0, accel=np.zeros((10, 3)))
    with pytest.raises(KeyError):
        w.magnitude("gyro")
