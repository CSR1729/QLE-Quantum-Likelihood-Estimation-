"""Probe-state preparation. See qle/README.md."""
import numpy as np


def prepare_state(alpha, beta):
    """|psi0> = cos(alpha)|0> + e^{i beta} sin(alpha)|1>, as a 2x1 column vector."""
    return np.array([[np.cos(alpha)], [np.exp(1j * beta) * np.sin(alpha)]])
