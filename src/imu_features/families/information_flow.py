"""Information-flow feature family: directed (causal) coupling between
sensors, as opposed to the merely correlational relationships everywhere
else in this engine.

Every cross-sensor feature elsewhere is symmetric: Pearson correlation,
the coupling family's dot/cross product, and the random-matrix family's
eigenvalue spectrum all treat "accel relates to gyro" the same as "gyro
relates to accel." **Transfer entropy** (Schreiber, 2000) breaks that
symmetry. It asks: does knowing the source signal's recent past reduce
uncertainty about the target signal's next value, *beyond* what the
target's own past already tells you? Formally, for source X and target Y:

    TE(X -> Y) = sum p(y_t+1, y_t, x_t) * log[ p(y_t+1 | y_t, x_t) / p(y_t+1 | y_t) ]

TE(X -> Y) != TE(Y -> X) in general — that asymmetry is the entire point.
A high TE(accel -> gyro) alongside a low TE(gyro -> accel) means linear
motion is *driving* angular motion (beyond simple synchrony), such as a
limb's translational swing inducing its own rotation; the reverse pattern
would suggest rotation-first motion (e.g. a twist inducing translation).

This implementation is the standard histogram/plug-in estimator: each
signal is discretized into `n_bins` quantile-balanced symbols, and the
probabilities above are estimated from empirical joint counts. This
estimator is known to carry a small positive bias for finite samples
(confirmed here too: independent noise at typical window lengths yields a
small nonzero TE in both directions rather than exactly zero) — treat
absolute TE values cautiously and prefer the directional *comparison*
(`net`) or comparisons across windows/subjects, where the bias is roughly
constant and cancels out.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..core.registry import register_feature


def _discretize(x, n_bins=4):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)).copy()
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)


def _transfer_entropy(source, target, n_bins=4):
    source_sym = _discretize(source, n_bins)
    target_sym = _discretize(target, n_bins)
    y_next, y_now, x_now = target_sym[1:], target_sym[:-1], source_sym[:-1]
    n = len(y_next)

    joint3 = Counter(zip(y_next, y_now, x_now))
    joint_yx = Counter(zip(y_now, x_now))
    joint_yy = Counter(zip(y_next, y_now))
    marg_y = Counter(y_now)

    te = 0.0
    for (yn, yc, xc), count in joint3.items():
        p_joint = count / n
        p_next_given_yx = count / joint_yx[(yc, xc)]
        p_next_given_y = joint_yy[(yn, yc)] / marg_y[yc]
        if p_next_given_yx > 0 and p_next_given_y > 0:
            te += p_joint * np.log(p_next_given_yx / p_next_given_y)
    return float(te)


@register_feature(
    "transfer_entropy",
    family="information_flow",
    scope="fusion",
    requires=("accel", "gyro"),
    min_samples=32,
    description=(
        "Directed information flow (transfer entropy) between the accelerometer and "
        "gyroscope resultant-magnitude signals, in both directions, plus their net "
        "(dominant-direction) difference."
    ),
)
def transfer_entropy(window):
    a_mag = window.magnitude("accel")
    g_mag = window.magnitude("gyro")
    te_a_to_g = _transfer_entropy(a_mag, g_mag)
    te_g_to_a = _transfer_entropy(g_mag, a_mag)
    return {
        "accel_to_gyro": te_a_to_g,
        "gyro_to_accel": te_g_to_a,
        "net": te_a_to_g - te_g_to_a,
    }
