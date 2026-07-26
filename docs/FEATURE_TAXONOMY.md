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

## 10. Orientation (sensor-fusion) features

Every family above operates on one sensor at a time. Orientation is
different — real attitude estimation requires *fusing* sensors: gyroscope
integration is accurate over short timescales but drifts, while
accelerometer-derived tilt is drift-free but noisy and only valid when
gravity dominates the reading. The classic remedy is a **complementary
filter**: blend integrated gyro rate with accelerometer tilt, weighted so
each contributes where it's reliable (Mahony/Madgwick filters formalize the
same idea; the complementary filter here is the simplest member of that
family). We compute roll/pitch this way and summarize each with mean, std,
and range over the window.

Heading is a second fusion problem: a raw magnetometer reading only gives a
usable compass bearing once **tilt-compensated** using the accelerometer's
roll/pitch — otherwise tilting the device rotates the apparent heading.
`tilt_compensated_heading` implements the standard tilt-compensation
formula and reports the heading's circular mean and circular standard
deviation (heading is an angle, so ordinary mean/std would be wrong at the
0°/360° wrap).

Because these need more than one sensor, they're registered with
`scope="fusion"` (see `docs/ARCHITECTURE.md`): each declares which sensors
it `requires`, and the engine silently omits it for a window missing one.

## 11. Gait / step-detection features

A specialization of families #1 and #3 tuned specifically to locomotion:
peak-based step detection on an acceleration channel (`scipy.signal.find_peaks`
with an amplitude threshold and a minimum inter-step distance) yields step
count and cadence (steps/minute) directly, and the coefficient of variation
of the resulting step intervals is a standard gait-regularity index used in
fall-risk and rehabilitation research — a more regular gait has a lower CV.

## 12. Coupling features (exterior algebra between sensors)

Every family above (including orientation) either operates on one sensor,
or fuses sensors specifically to *estimate attitude*. The coupling family
instead asks a narrower algebraic question: at each instant, how related
are the accelerometer vector and the gyroscope vector, independent of what
either means physically on its own?

Treating `accel(t)` and `gyro(t)` as two vectors in R^3 at each sample, two
classical vector-algebra operations answer that:

- **Dot product** `accel(t) . gyro(t)`: a pseudoscalar directly analogous
  to *kinetic helicity* in fluid mechanics (`v . omega`, a measure of how
  "corkscrew-like" a flow is) and to the linear/angular momentum coupling
  term in rigid-body dynamics. It is large and positive when translation
  and rotation are aligned (e.g. a hand drilling or a spinning coin/top
  moving along its own axis), large and negative for the mirror-image
  motion, and near zero when the two are orthogonal or either is small.
  Averaged over the window it gives `kinematic_helicity_mean`; its sign
  pattern gives a **chirality index** — the net fraction of samples with
  positive vs. negative helicity, i.e. whether the window's motion has a
  consistent handedness.
- **Cross product** `accel(t) x gyro(t)`: the corresponding bivector (in 3D
  the Clifford/geometric-algebra wedge product of two vectors collapses to
  the ordinary cross product). Its magnitude is the coupling strength with
  the sign convention removed — large whenever accel and gyro are far from
  parallel/anti-parallel, regardless of which.
- **Normalized alignment index**: dividing the dot product by both vector
  norms gives `cos(theta)` between the two vectors, in `[-1, 1]` — unitless
  and therefore comparable across sensors, subjects, or recording sessions
  regardless of amplitude, unlike the raw (unit-mixing) helicity value.

Because it inherently needs both sensors at once, `coupling` is registered
with `scope="fusion"` (the same mechanism introduced for `orientation`) and
requires `("accel", "gyro")`.

## 13. Spherical-wave features (non-Euclidean geometry)

Every family so far treats a channel as living in flat, Euclidean space —
even the topological family's phase-space embedding is a point cloud in
flat R^dim. This family instead puts the *direction* of a triaxial vector,
`u(t) = v(t) / |v(t)|`, on the unit 2-sphere S^2: a curved Riemannian
manifold of constant positive curvature. Two pieces of mathematics that are
native to that curved setting, not borrowed from flat-space analysis:

- **Directional (Fisher) statistics.** On a line, dispersion is variance.
  On a sphere, the curvature-correct analog is `1 - R`, where
  `R = |mean(u_i)|` is the *mean resultant length* — Fisher's spherical
  variance (Fisher, 1953). Its companion is the concentration parameter
  `kappa` of the fitted von Mises-Fisher distribution (the sphere's analog
  of a Gaussian), estimated via the standard `p=3` approximation
  `kappa ~= R(3-R^2)/(1-R^2)` (Mardia & Jupp, *Directional Statistics*):
  large `kappa` means the direction samples cluster tightly around one
  point on the sphere; small `kappa` means they're widely spread.
- **Spherical harmonics as the sphere's "waves."** On a line, waves are
  sines and cosines — eigenfunctions of the flat Laplacian — and a Fourier
  transform expands a signal in that basis. On a sphere, the eigenfunctions
  of the Laplace-Beltrami operator are the *spherical harmonics* `Y_lm`,
  and expanding the direction samples in that basis gives an *angular power
  spectrum* `C_l`: the sphere's exact analog of a Fourier power spectrum,
  but for waves that live on curved space. `spherical_dipole_power` (`C_1`)
  and `spherical_quadrupole_power` (`C_2`) are the sphere's two lowest
  nontrivial wave modes, estimated from the empirical harmonic moments of
  the direction samples — the same estimator used to build angular power
  spectra from scattered directional data in other fields (e.g. CMB
  temperature maps, crystallographic pole figures). Quadrupole power is
  magnitude-invariant and purely about directional anisotropy, unlike the
  `geometrical` family's PCA descriptors, which operate on the raw (not
  normalized) vectors and mix magnitude with direction.
- **Geodesic (great-circle) distance as the family's literal wave.** The
  distance `arccos(u(t) . u(t+1))` between consecutive direction samples is
  the sphere's intrinsic angular-speed signal — the non-Euclidean analog of
  a first difference, since it's measured along the curved surface rather
  than as a straight Euclidean chord. Treating that speed sequence as a
  time series and taking its FFT (`geodesic_dominant_frequency`,
  `geodesic_spectral_energy`) measures oscillation *of the manifold
  trajectory itself* — a genuinely different "frequency" than anything in
  the `frequency` family, which never leaves flat amplitude space.
  `geodesic_path_length` is the total distance traveled along the sphere,
  the curved-space analog of `waveform_length`.

## Extending the taxonomy

The registry (`imu_features.core.registry`) is a plain decorator-based
plug-in point — `docs/ARCHITECTURE.md` explains how to add a new family or
feature without touching the engine. Natural next candidates, not yet
implemented:

- **Higher-dimensional persistent homology** (H1/H2) via `gudhi`/`ripser`
  as an optional extra, for projects that want loop/void structure beyond
  the H0 features already included in family #8.
- **Madgwick/Mahony quaternion fusion**: the complementary filter in family
  #10 is deliberately the simplest fusion scheme; a full quaternion-based
  Madgwick filter would give a more accurate, gimbal-lock-free attitude at
  the cost of more state to carry across the window.
- **Time-varying coupling**: family #12 summarizes helicity/alignment
  scalars over the whole window; a natural extension is their own
  time-domain or frequency-domain features (e.g. does the alignment index
  itself oscillate at the step frequency?), rather than only mean/std.
- **Higher-degree spherical harmonics** (`l=3+`) for family #13, or a full
  angular bispectrum, for projects that need finer-grained directional
  shape than the dipole/quadrupole power already captures.
