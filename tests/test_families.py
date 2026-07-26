import numpy as np

from imu_features.families import (
    crossaxis,
    frequency,
    geometrical,
    magnitude,
    mechanical,
    nonlinear,
    statistical,
    topological,
    wavelet,
)


def test_statistical_mean():
    x = np.array([1.0, 2.0, 3.0])
    assert statistical.mean(x, 100.0) == 2.0


def test_statistical_zero_crossing_rate():
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    assert statistical.zero_crossing_rate(x, None) == 1.0


def test_frequency_dominant_frequency_sine():
    sr = 100.0
    t = np.arange(500) / sr
    x = np.sin(2 * np.pi * 5 * t)
    f = frequency.dominant_frequency(x, sr)
    assert abs(f - 5.0) < 0.3


def test_frequency_spectral_entropy_bounded():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    val = frequency.spectral_entropy(x, 100.0)
    assert 0.0 <= val <= 1.0 + 1e-9


def test_geometrical_pca_linearity_for_line():
    arr = np.zeros((100, 3))
    arr[:, 0] = np.linspace(-1, 1, 100)
    val = geometrical.pca_linearity(arr, None)
    assert val > 0.95


def test_geometrical_pca_sphericity_for_isotropic_noise():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((2000, 3))
    val = geometrical.pca_sphericity(arr, None)
    assert val > 0.5


def test_mechanical_jerk_zero_for_constant_signal():
    x = np.ones(50)
    assert mechanical.jerk_rms(x, 100.0) == 0.0


def test_mechanical_gravity_deviation_near_zero_for_gravity_vector():
    arr = np.tile([0.0, 0.0, 9.80665], (50, 1))
    val = mechanical.gravity_deviation(arr, None)
    assert abs(val) < 1e-6


def test_nonlinear_shannon_entropy_higher_for_noise_than_constant():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(500)
    constant = np.ones(500)
    assert nonlinear.shannon_entropy(noise, None) > nonlinear.shannon_entropy(constant, None)


def test_nonlinear_permutation_entropy_bounded():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    val = nonlinear.permutation_entropy(x, None)
    assert 0.0 <= val <= 1.0 + 1e-9


def test_topological_h0_persistence_nonnegative():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    val = topological.h0_total_persistence(x, None)
    assert val >= 0


def test_topological_recurrence_rate_bounded():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    val = topological.recurrence_rate(x, None)
    assert 0.0 <= val <= 1.0


def test_topological_determinism_high_for_periodic_signal():
    t = np.arange(300) / 50.0
    x = np.sin(2 * np.pi * 2.0 * t)
    val = topological.determinism(x, None)
    assert val > 0.5


def test_wavelet_energy_entropy_nonnegative():
    x = np.sin(np.linspace(0, 20, 256))
    val = wavelet.wavelet_energy_entropy(x, None)
    assert val >= 0


def test_wavelet_energy_ratios_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    energies = wavelet._level_energies(x, wavelet._DEFAULT_LEVELS)
    total = sum(energies)
    assert total > 0
    ratios = [e / total for e in energies]
    assert abs(sum(ratios) - 1.0) < 1e-9


def test_crossaxis_corr_perfect_for_identical_axes():
    a = np.linspace(0, 1, 100)
    arr = np.stack([a, a, np.zeros(100)], axis=1)
    val = crossaxis.corr_xy(arr, None)
    assert abs(val - 1.0) < 1e-6


def test_crossaxis_corr_zero_for_orthogonal_signals():
    arr = np.zeros((100, 3))
    arr[:, 0] = np.ones(100)  # constant -> zero std -> defined as 0
    arr[:, 1] = np.linspace(-1, 1, 100)
    val = crossaxis.corr_xy(arr, None)
    assert val == 0.0


def test_magnitude_sma_nonnegative():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 3))
    val = magnitude.signal_magnitude_area(arr, None)
    assert val >= 0


def test_magnitude_peak_resultant_at_least_movement_intensity():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 3))
    assert magnitude.peak_resultant(arr, None) >= magnitude.movement_intensity(arr, None)
