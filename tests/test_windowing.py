import numpy as np
import pytest

from imu_features import segment_signal, sliding_indices


def test_sliding_indices_no_overlap():
    idx = list(sliding_indices(100, window_size=20, step=20))
    assert idx == [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def test_sliding_indices_with_overlap():
    idx = list(sliding_indices(50, window_size=20, step=10))
    assert idx == [(0, 20), (10, 30), (20, 40), (30, 50)]


def test_segment_signal_produces_correct_window_count():
    accel = np.random.randn(500, 3)
    windows = list(segment_signal({"accel": accel}, sample_rate=100.0, window_seconds=1.0, overlap=0.5))
    assert len(windows) > 0
    for w in windows:
        assert w.n_samples == 100
        assert w.sample_rate == 100.0


def test_segment_signal_rejects_bad_overlap():
    with pytest.raises(ValueError):
        list(segment_signal({"accel": np.zeros((10, 3))}, sample_rate=10.0, window_seconds=1.0, overlap=1.0))


def test_segment_signal_requires_sensor():
    with pytest.raises(ValueError):
        list(segment_signal({}, sample_rate=10.0, window_seconds=1.0))
