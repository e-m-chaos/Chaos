"""Minimal end-to-end example: synthesize a short IMU recording, segment it
into windows, and extract a full feature table with the giant feature engine.

Run with: python examples/quickstart.py
"""

import numpy as np

from imu_features import REGISTRY, FeatureEngine, segment_signal


def synthesize_walking(duration_s=10.0, sample_rate=100.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    step_freq = 1.8  # Hz, roughly a walking cadence

    accel = np.stack(
        [
            0.6 * np.sin(2 * np.pi * step_freq * t) + 0.05 * rng.standard_normal(n),
            0.3 * np.sin(2 * np.pi * step_freq * t + 0.5) + 0.05 * rng.standard_normal(n),
            9.81 + 1.2 * np.abs(np.sin(2 * np.pi * step_freq * t)) + 0.1 * rng.standard_normal(n),
        ],
        axis=1,
    )
    gyro = np.stack(
        [
            30 * np.sin(2 * np.pi * step_freq * t) + rng.standard_normal(n),
            10 * np.cos(2 * np.pi * step_freq * t) + rng.standard_normal(n),
            5 * rng.standard_normal(n),
        ],
        axis=1,
    )
    return {"accel": accel, "gyro": gyro}


def main():
    sample_rate = 100.0
    sensors = synthesize_walking(duration_s=10.0, sample_rate=sample_rate)

    windows = list(
        segment_signal(sensors, sample_rate=sample_rate, window_seconds=2.0, overlap=0.5)
    )
    print(f"Segmented into {len(windows)} windows")

    print("Registered feature families:", REGISTRY.families())
    print("Total registered feature definitions:", len(REGISTRY))

    # Use every family: a per-sensor, per-channel/triaxial sweep across
    # statistical, magnitude, frequency, geometrical, mechanical, cross-axis,
    # nonlinear, topological, and wavelet feature families, plus the
    # multi-sensor fusion families (orientation, since accel+gyro are both
    # present) and gait (step detection on the acceleration channels).
    engine = FeatureEngine(families=REGISTRY.families())
    table = engine.extract_many(windows)

    if hasattr(table, "shape"):
        print(f"Feature table shape: {table.shape}")
    else:
        print(f"Feature table shape: ({len(table)}, {len(table[0])})")
    first_window_features = engine.extract(windows[0])
    print("Example feature values from window 0:")
    for key in sorted(first_window_features)[:10]:
        print(f"  {key}: {first_window_features[key]:.4f}")

    print("Orientation fusion features (accel+gyro):")
    for key, value in sorted(first_window_features.items()):
        if key.startswith("fusion_"):
            print(f"  {key}: {value:.4f}")

    print("Gait features (accel z-axis):")
    for key, value in sorted(first_window_features.items()):
        if "_gait_" in key and "_z_" in key:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
