"""Time evolution under a Hamiltonian, shared by all three scripts.

Line references: Q = QLE.m, G = Optimized_QLE_Grid.m, S = Optimized_QLE_SA.m.
"""
from scipy.linalg import expm


def evolve(H, t, psi):
    """expm(-1i*H*t) * psi.

    MATLAB Q:86-87,95-96  G:85-86,101-102,197  S:65-66,83-84,185
    """
    return expm(-1j * H * t) @ psi
