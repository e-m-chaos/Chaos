# imu-features

An extensible feature-engineering engine for IMU signals (accelerometer,
gyroscope, magnetometer). Give it a window of raw sensor samples and it
extracts a large, well-organized set of engineered features spanning eleven
feature families — statistical, magnitude, frequency, geometrical,
mechanical, cross-axis, nonlinear/entropy, topological, wavelet,
orientation (sensor fusion), and gait — with a plug-in registry so new
families or individual features are a few lines of code away, no engine
changes required.

See [`docs/FEATURE_TAXONOMY.md`](docs/FEATURE_TAXONOMY.md) for the research
behind each family and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
how the engine and plug-in registry work.

## Install

```bash
pip install -e ".[pandas]"
```

(`pandas` is optional — only needed if you want `extract_many` to return a
DataFrame instead of a list of dicts.)

## Quickstart

```python
import numpy as np
from imu_features import IMUWindow, FeatureEngine, segment_signal, REGISTRY

sample_rate = 100.0
n = 1000
accel = np.random.randn(n, 3) + [0, 0, 9.81]
gyro = np.random.randn(n, 3)

# Segment a continuous recording into 2s / 50%-overlap windows
windows = list(segment_signal({"accel": accel, "gyro": gyro},
                               sample_rate=sample_rate,
                               window_seconds=2.0, overlap=0.5))

# Pick whole families...
engine = FeatureEngine(families=["statistical", "frequency", "geometrical"])
# ...or the entire engine
full_engine = FeatureEngine(families=REGISTRY.families())

features = engine.extract(windows[0])          # dict[str, float]
table = full_engine.extract_many(windows)       # pandas.DataFrame (or list[dict])
```

Feature keys are flat and self-describing:
`accel_x_statistical_mean`, `gyro_mag_frequency_dominant_frequency`,
`accel_triaxial_geometrical_pca_linearity`, ...

Run `python examples/quickstart.py` for a full synthetic-walking example.

## Feature families

| Family | Examples |
|---|---|
| `statistical` | mean, std, RMS, skewness, kurtosis, zero-crossing rate, waveform length |
| `magnitude` | signal magnitude area (SMA), movement intensity, resultant energy |
| `frequency` | dominant frequency, spectral entropy, spectral centroid, band-power ratios |
| `geometrical` | PCA linearity/planarity/sphericity, roll/pitch/azimuth, bounding-box volume |
| `mechanical` | jerk/snap RMS, kinetic-energy proxy, gravity deviation |
| `crossaxis` | Pearson correlation, covariance, and max cross-correlation between axis pairs |
| `nonlinear` | Shannon/sample/approximate/permutation entropy, Hurst exponent, DFA, fractal dimension |
| `topological` | 0-dim persistent homology (total/max/entropy), recurrence rate, determinism, laminarity |
| `wavelet` | Haar wavelet per-level energy ratios and energy entropy |
| `orientation` | complementary-filter roll/pitch (accel+gyro fusion), tilt-compensated heading (accel+mag fusion) |
| `gait` | step count, cadence, step-interval mean/CV (regularity) |

## Development

```bash
pip install -e ".[dev,pandas]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
