"""imu_features: an extensible feature-engineering engine for IMU signals
(accelerometer, gyroscope, magnetometer), spanning statistical, magnitude,
frequency, geometrical, mechanical, cross-axis, nonlinear/entropy,
topological, and wavelet feature families.
"""

from . import families  # noqa: F401  (side effect: registers built-in features)
from .core import (
    REGISTRY,
    FeatureEngine,
    FeatureRegistry,
    FeatureSpec,
    IMUWindow,
    register_feature,
    segment_signal,
    sliding_indices,
)

__version__ = "0.1.0"

__all__ = [
    "IMUWindow",
    "segment_signal",
    "sliding_indices",
    "REGISTRY",
    "FeatureRegistry",
    "FeatureSpec",
    "register_feature",
    "FeatureEngine",
    "families",
]
