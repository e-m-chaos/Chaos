"""Core data container for a single IMU window (accel/gyro/mag samples)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

_SENSOR_NAMES = ("accel", "gyro", "mag")


@dataclass
class IMUWindow:
    """A fixed-length slice of synchronized IMU samples.

    Each provided sensor array must have shape (n_samples, 3) — one row per
    sample, columns are the x/y/z axes. At least one sensor must be given,
    and all provided sensors must share the same number of samples.
    """

    sample_rate: float
    accel: Optional[np.ndarray] = None
    gyro: Optional[np.ndarray] = None
    mag: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        lengths = set()
        for name in _SENSOR_NAMES:
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"'{name}' must have shape (n_samples, 3), got {arr.shape}")
            setattr(self, name, arr)
            lengths.add(arr.shape[0])

        if not lengths:
            raise ValueError("at least one of accel/gyro/mag must be provided")
        if len(lengths) > 1:
            raise ValueError(f"sensor arrays must share the same length, got lengths {lengths}")

    @property
    def n_samples(self) -> int:
        for name in _SENSOR_NAMES:
            arr = getattr(self, name)
            if arr is not None:
                return arr.shape[0]
        raise RuntimeError("unreachable: __post_init__ guarantees at least one sensor")

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    def sensors(self) -> Dict[str, np.ndarray]:
        """Return the present sensor arrays keyed by name (accel/gyro/mag)."""
        return {name: getattr(self, name) for name in _SENSOR_NAMES if getattr(self, name) is not None}

    def magnitude(self, sensor: str) -> np.ndarray:
        """Euclidean norm of the given sensor's triaxial samples, per sample."""
        arr = getattr(self, sensor)
        if arr is None:
            raise KeyError(f"sensor '{sensor}' not present in this window")
        return np.linalg.norm(arr, axis=1)
