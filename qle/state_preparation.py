"""Single-qubit pure-state preparation, shared by all three scripts.

Line references: Q = QLE.m, G = Optimized_QLE_Grid.m, S = Optimized_QLE_SA.m.
"""
import numpy as np


def prepare_state(alpha, beta):
    """|psi0> = cos(alpha)|0> + e^{i beta} sin(alpha)|1>, as a 2x1 column.

    The optimized scripts follow some calls with psi0 / norm(psi0); that
    normalisation stays at those call sites, exactly where the original had it.
    """
    # MATLAB Q:46 G:73,191 S:62,181: psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];
    return np.array([[np.cos(alpha)], [np.exp(1j * beta) * np.sin(alpha)]])
