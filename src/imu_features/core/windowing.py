"""Sliding-window segmentation of continuous IMU streams into IMUWindows."""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .signal import IMUWindow


def sliding_indices(n_samples: int, window_size: int, step: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) index pairs for fixed-size, fixed-step windows."""
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    start = 0
    while start + window_size <= n_samples:
        yield start, start + window_size
        start += step


def segment_signal(
    sensors: Dict[str, np.ndarray],
    sample_rate: float,
    window_seconds: float,
    overlap: float = 0.5,
    timestamps: Optional[np.ndarray] = None,
) -> Iterator[IMUWindow]:
    """Slice continuous sensor arrays (each shape (N, 3)) into IMUWindows.

    `overlap` is the fraction of each window shared with the next one
    (0 = no overlap, 0.5 = 50% overlap).
    """
    if not sensors:
        raise ValueError("at least one sensor array must be provided")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    n_samples = next(iter(sensors.values())).shape[0]
    window_size = int(round(window_seconds * sample_rate))
    step = max(1, int(round(window_size * (1 - overlap))))

    for start, end in sliding_indices(n_samples, window_size, step):
        kwargs = {name: arr[start:end] for name, arr in sensors.items()}
        ts = timestamps[start:end] if timestamps is not None else None
        yield IMUWindow(sample_rate=sample_rate, timestamps=ts, **kwargs)
