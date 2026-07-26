"""Time-delay (Takens) embedding, shared by the nonlinear and topological
feature families for phase-space reconstruction of a scalar signal."""

from __future__ import annotations

import numpy as np


def time_delay_embedding(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """Reconstruct a pseudo phase-space trajectory from a 1-D signal.

    Returns an array of shape (n_samples - (dim - 1) * tau, dim), where row i
    is [x[i], x[i + tau], ..., x[i + (dim - 1) * tau]].
    """
    x = np.asarray(x, dtype=float)
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("signal too short for the requested embedding dimension/delay")
    return np.array([x[i : i + n] for i in range(0, dim * tau, tau)]).T
