# IMU Feature Taxonomy — Research Notes

This document is the research backing `imu_features`: a survey of the
feature families used across IMU-driven fields (human activity recognition,
gesture/gait analysis, structural health monitoring, robotics state
estimation, tremor/biomechanics analysis) and how each family is realized in
the library.

An IMU (Inertial Measurement Unit) reports triaxial accelerometer data
(and often gyroscope and/or magnetometer data) as a time series
`x(t), y(t), z(t)` per sensor. Almost every downstream task — classifying an
activity, detecting a gesture, estimating gait quality, flagging a
mechanical fault — starts by converting a *window* of raw samples into a
fixed-length vector of engineered features. The families below are the
established ways of doing that conversion, organized by what kind of
structure in the signal each one captures.

## 1. Statistical (time-domain) features

The baseline of virtually every published HAR feature set (see Bulling,
Blanke & Schiele's 2014 survey on activity recognition from accelerometer
data for the canonical list). These describe the *distribution* and simple
*morphology* of a channel's amplitude over a window, independent of when in
the window things happened:

- Central tendency: mean, median
- Dispersion: standard deviation, variance, range, IQR, mean absolute
  deviation, coefficient of variation
- Shape: skewness, kurtosis
- Energy: RMS, signal energy (sum of squares)
- Morphology/counting: zero-crossing rate, mean-crossing rate, waveform
  length, slope sign changes, peak count

Why they matter: cheap to compute, robust, and surprisingly discriminative
for distinguishing motion intensity and regularity (e.g. sitting vs.
running has wildly different variance and RMS).

## 2. Magnitude features

The vector magnitude `|a| = sqrt(x^2 + y^2 + z^2)` is orientation-invariant
— unlike x/y/z individually, it doesn't change when the device is worn at a
different angle. Statistics of the magnitude signal (mean, std, ...) are
obtained "for free" in this library because the engine treats the norm as a
fourth pseudo-channel and reuses every channel-scope family on it.

The **magnitude family module** holds descriptors that need the three raw
axes jointly rather than the norm alone:

- **Signal Magnitude Area (SMA)**: mean of `|x| + |y| + |z|` — a classic
  energy-expenditure proxy from accelerometry-based physical activity
  research (Bouten et al.).
- Resultant energy, movement intensity, magnitude variability, peak
  resultant.

## 3. Frequency-domain features

Computed from the (mean-removed) FFT of a channel. Human and mechanical
motion has characteristic frequency content — walking cadence, tremor rate,
rotor imbalance — that time-domain statistics can't see directly:

- Dominant frequency and its amplitude
- Spectral energy, spectral entropy (how "tonal" vs. "broadband" the
  spectrum is), spectral flatness (Wiener entropy)
- Spectral centroid, spread, skewness, kurtosis — the spectrum's own
  "distribution shape," analogous to family #1 but in frequency space
- Spectral roll-off (85% energy cutoff frequency)
- Coarse band-power ratios tuned to human-motion bandwidths (0–3 Hz
  postural, 3–6 Hz locomotion/cadence, 6 Hz+ fast/vibratory)

## 4. Geometrical features

Descriptors of the *shape* the triaxial trajectory traces in 3-D space over
the window, and quasi-static orientation:

- **PCA shape descriptors** (linearity, planarity, sphericity, dominant
  variance ratio): eigen-decompose the 3×3 covariance matrix of the
  windowed samples. This is the same "structure tensor" idea used in point
  cloud/LiDAR shape classification, borrowed here to classify whether a
  motion is essentially 1-D (line), 2-D (planar swing), or fills 3-D space.
- **Orientation angles** (roll, pitch, azimuth, inclination): derived from
  the mean vector, valid under the quasi-static assumption that gravity (for
  accelerometer) or a stable field (for magnetometer) dominates the mean.
- Bounding-box volume of the trajectory.

## 5. Mechanical / physical features

Quantities with a direct physical reading when applied to acceleration
channels:

- **Jerk** (`d(accel)/dt`) and **snap** (`d²(accel)/dt²`): rate of change of
  acceleration/jerk, RMS and peak. Widely used in gait smoothness research —
  smoother, more efficient movement has lower jerk.
- Kinetic-energy-shaped proxy (`0.5 * mean(x^2)`), dynamic/static power
  ratio (how much of the signal's mean-square value is DC/gravity vs.
  AC/motion), a velocity-change proxy via numerical integration.
- **Gravity deviation**: how far the mean resultant magnitude sits from
  standard gravity (9.80665 m/s²) — near zero at rest, large under dynamic
  acceleration or free-fall.

## 6. Cross-axis features

Pairwise relationships *between* the x/y/z channels of one sensor:
Pearson correlation, covariance, and best-lag cross-correlation. These
surface coordinated, multi-axis motion (e.g. a rotational or swinging
motion couples two axes with a phase offset) that per-axis statistics miss
entirely.

## 7. Nonlinear-dynamics / information-theoretic features

Motion signals are not simple periodic or i.i.d. random processes — they
sit somewhere in between, and this family, borrowed from dynamical-systems
and biomedical signal analysis (heart-rate variability, gait, tremor),
quantifies *where*:

- **Entropy measures**: Shannon entropy (amplitude histogram), sample
  entropy and approximate entropy (regularity/self-similarity of the
  signal's own sub-patterns, from Richman & Moorman / Pincus), permutation
  entropy (complexity of the ordinal patterns of consecutive samples,
  Bandt & Pompe).
- **Long-range correlation**: Hurst exponent via rescaled-range (R/S)
  analysis, and the detrended fluctuation analysis (DFA) scaling exponent —
  both distinguish random-walk-like signals from persistent or
  anti-persistent ones.
- **Fractal dimension**: Higuchi's and Petrosian's estimators, which
  quantify how "space-filling"/complex a waveform is independent of its
  amplitude.

## 8. Topological features

Descriptors of the *shape* of the signal's reconstructed phase-space
trajectory, grounded in computational topology. A 1-D channel is lifted
into a point cloud via **time-delay (Takens) embedding**
(`[x(t), x(t+tau), x(t+2*tau), ...]`), then:

- **0-dimensional persistent homology**, computed exactly and efficiently
  as the minimum spanning tree of the embedded point cloud — MST edge
  weights are precisely the birth/death "lifetimes" in the H0 persistence
  diagram (a standard equivalence; see Carlsson's "Topology and data" for
  the underlying theory). We extract total persistence, max persistence,
  and persistence entropy from that diagram — genuine topological
  invariants, computed with nothing more exotic than `scipy.sparse.csgraph`.
- **Recurrence Quantification Analysis (RQA)**: build a recurrence matrix
  from the embedded trajectory (Eckmann, Kamphorst & Ruelle) and derive
  recurrence rate, determinism (fraction of recurrences forming diagonal
  lines → periodic/deterministic structure), and laminarity (fraction
  forming vertical lines → laminar/stationary states).

Both techniques are implemented with only numpy/scipy — no dependency on
heavier TDA libraries (`gudhi`, `ripser`) is required for this feature set,
though those remain natural extensions for higher-dimensional persistence
(H1 loops, etc.) if a project needs them.

## 9. Wavelet features

Multi-resolution energy distribution via a Haar discrete wavelet transform,
implemented directly on numpy (no `PyWavelets` dependency). Each
decomposition level splits the signal into a coarser approximation and a
detail (high-frequency residual); the relative energy across levels
(`wavelet_energy_ratio_d1..d4`, `..._approx`) and the entropy of that
distribution (`wavelet_energy_entropy`) summarize *where in scale* a
signal's power sits — complementary to family #3's *where in frequency*
view, and often more robust to non-stationary bursts (e.g. a single sharp
impact) because wavelets are localized in time as well as scale.

## Extending the taxonomy

The registry (`imu_features.core.registry`) is a plain decorator-based
plug-in point — `docs/ARCHITECTURE.md` explains how to add a new family or
feature without touching the engine. Natural next candidates, not yet
implemented:

- **Orientation/attitude family**: full quaternion or Euler-angle
  estimation via complementary or Kalman/Madgwick filtering across a
  gyroscope + accelerometer (+ magnetometer) fusion, rather than the
  quasi-static angle estimate in family #4.
- **Step/gait family**: peak-based step detection, stride time, cadence,
  gait symmetry/regularity indices — specializations of families #1, #3,
  and #7 tuned to locomotion.
- **Higher-dimensional persistent homology** (H1/H2) via `gudhi`/`ripser`
  as an optional extra, for projects that want loop/void structure beyond
  the H0 features already included.
- **Cross-sensor features**: correlation/coherence between accelerometer
  and gyroscope channels (as opposed to family #6's within-sensor,
  cross-axis view).
