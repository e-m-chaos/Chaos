"""Orientation (sensor-fusion) feature family.

Unlike every other family, orientation estimation genuinely needs more than
one sensor at once: attitude comes from fusing gyroscope integration (fast,
accurate short-term, drifts long-term) with accelerometer tilt (drift-free,
noisy short-term) — the classic complementary-filter idea — and heading
comes from tilt-compensating a magnetometer reading with that same attitude.
These are registered with ``scope="fusion"``: each takes the whole
``IMUWindow`` and returns several named sub-values, and each declares which
sensors it needs via ``requires`` so the engine skips it when they're absent.

Angular rate (gyro) is assumed to be in degrees/second, matching most
wearable/consumer IMUs; adapt the ``dt`` scaling in `_complementary_tilt` if
your gyro reports radians/second.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import register_feature


def _complementary_tilt(accel, gyro, sample_rate, alpha=0.98):
    """Fuse accel-derived tilt with integrated gyro rate into roll/pitch (degrees)."""
    n = len(accel)
    dt = 1.0 / sample_rate
    roll = np.empty(n)
    pitch = np.empty(n)

    roll[0] = np.degrees(np.arctan2(accel[0, 1], accel[0, 2]))
    pitch[0] = np.degrees(np.arctan2(-accel[0, 0], np.sqrt(accel[0, 1] ** 2 + accel[0, 2] ** 2)))

    for i in range(1, n):
        roll_acc = np.degrees(np.arctan2(accel[i, 1], accel[i, 2]))
        pitch_acc = np.degrees(
            np.arctan2(-accel[i, 0], np.sqrt(accel[i, 1] ** 2 + accel[i, 2] ** 2))
        )
        roll_gyro = roll[i - 1] + gyro[i, 0] * dt
        pitch_gyro = pitch[i - 1] + gyro[i, 1] * dt
        roll[i] = alpha * roll_gyro + (1 - alpha) * roll_acc
        pitch[i] = alpha * pitch_gyro + (1 - alpha) * pitch_acc

    return roll, pitch


@register_feature(
    "complementary_tilt",
    family="orientation",
    scope="fusion",
    requires=("accel", "gyro"),
    min_samples=4,
    description="Complementary-filter roll/pitch fusing accelerometer tilt with integrated gyro rate.",
)
def complementary_tilt(window):
    roll, pitch = _complementary_tilt(window.accel, window.gyro, window.sample_rate)
    return {
        "roll_mean": float(np.mean(roll)),
        "roll_std": float(np.std(roll)),
        "roll_range": float(np.ptp(roll)),
        "pitch_mean": float(np.mean(pitch)),
        "pitch_std": float(np.std(pitch)),
        "pitch_range": float(np.ptp(pitch)),
    }


def _circular_stats_degrees(angles_deg):
    rad = np.radians(angles_deg)
    mean_sin, mean_cos = np.mean(np.sin(rad)), np.mean(np.cos(rad))
    mean_angle = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0
    resultant_length = np.hypot(mean_cos, mean_sin)
    circular_std = np.degrees(np.sqrt(-2.0 * np.log(max(resultant_length, 1e-12))))
    return mean_angle, circular_std


@register_feature(
    "tilt_compensated_heading",
    family="orientation",
    scope="fusion",
    requires=("accel", "mag"),
    min_samples=4,
    description="Tilt-compensated compass heading from accelerometer + magnetometer (circular mean/spread, degrees).",
)
def tilt_compensated_heading(window):
    accel, mag = window.accel, window.mag
    roll = np.arctan2(accel[:, 1], accel[:, 2])
    pitch = np.arctan2(-accel[:, 0], np.sqrt(accel[:, 1] ** 2 + accel[:, 2] ** 2))

    mx, my, mz = mag[:, 0], mag[:, 1], mag[:, 2]
    mx2 = mx * np.cos(pitch) + mz * np.sin(pitch)
    my2 = mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll) - mz * np.sin(roll) * np.cos(pitch)
    heading = np.degrees(np.arctan2(-my2, mx2)) % 360.0

    mean_heading, circular_std = _circular_stats_degrees(heading)
    return {"heading_mean": float(mean_heading), "heading_circular_std": float(circular_std)}
