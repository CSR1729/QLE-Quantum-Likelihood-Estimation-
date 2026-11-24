"""Pauli matrices and candidate Hamiltonians, moved verbatim from the Setup
sections of QLE.m, Optimized_QLE_Grid.m and Optimized_QLE_SA.m.

Line references: Q = QLE.m, G = Optimized_QLE_Grid.m, S = Optimized_QLE_SA.m.
None of the algorithms modifies these arrays in place.
"""
import numpy as np

# Pauli matrices
# MATLAB Q:9 G:12 S:12: sigX = [0, 1; 1, 0];
sigX = np.array([[0, 1], [1, 0]], dtype=float)
# MATLAB Q:10 G:13 S:13: sigY = [0, -1i; 1i, 0];
sigY = np.array([[0, -1j], [1j, 0]], dtype=complex)
# MATLAB Q:11 G:14 S:14: sigZ = [1, 0; 0, -1];
sigZ = np.array([[1, 0], [0, -1]], dtype=float)
# MATLAB Q:12 G:15 S:15: sigX2 = 2 * sigX;
sigX2 = 2 * sigX
# MATLAB Q:13 G:16 S:16: sigZ2 = 2 * sigZ;
sigZ2 = 2 * sigZ

# Candidate Hamiltonians
# MATLAB Q:16 G:19 S:19: H1 = sigX+sigZ;
H1 = sigX + sigZ
# MATLAB Q:17 G:20 S:20: H2 = 1.2 * [1, 0; 0, 0] - 0.8 * [0, 0; 0, 1];
H2 = 1.2 * np.array([[1, 0], [0, 0]], dtype=float) - 0.8 * np.array([[0, 0], [0, 1]], dtype=float)
# MATLAB Q:18 G:21 S:21: H3 = 2*sigY+[1, 0; 0, 2];
H3 = 2 * sigY + np.array([[1, 0], [0, 2]], dtype=float)
# MATLAB Q:19 G:22 S:22: H4 = 4*(1/sqrt(2)) * [1, 1; 1, -1];
H4 = 4 * (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=float)
# MATLAB Q:20 G:23 S:23: H5 = sigX+H4;
H5 = sigX + H4
# MATLAB Q:21 G:24 S:24: H6= [1, 2; 2, -1];
H6 = np.array([[1, 2], [2, -1]], dtype=float)

# Defined only in Optimized_QLE_Grid.m (never used in any candidate list)
# MATLAB G:25: H7 = 0.5 * sigX + 0.8 * sigY;
H7 = 0.5 * sigX + 0.8 * sigY
# MATLAB G:26: H8 = [2, 0; 0, -1];
H8 = np.array([[2, 0], [0, -1]], dtype=float)
# MATLAB G:27: H9 = [1, 1i; -1i, 1];
H9 = np.array([[1, 1j], [-1j, 1]], dtype=complex)
# MATLAB G:28: H10 = [3, 2; 2, 0];
H10 = np.array([[3, 2], [2, 0]], dtype=float)
# MATLAB G:29: H11 = [0, 2 + 1i; 2 - 1i, 1];
H11 = np.array([[0, 2 + 1j], [2 - 1j, 1]], dtype=complex)
# MATLAB G:30: H12 = 0.6 * sigZ + 0.4 * sigY;
H12 = 0.6 * sigZ + 0.4 * sigY


def random_hermitian(rng):
    """H13 of Optimized_QLE_Grid.m. Unused there, but it draws 8 normal samples
    from the run's random stream, so the Grid script calls it at the original point."""
    # MATLAB G:31: A = randn(2) + 1i * randn(2);
    A = rng.randn(2) + 1j * rng.randn(2)
    # MATLAB G:32: H13 = (A + A') / 2;
    H13 = (A + A.conj().T) / 2
    return H13


def default_list():
    """MATLAB Q:27 G:38 S:30: H_list = {sigX, sigX2, sigZ, sigZ2};"""
    return [sigX, sigX2, sigZ, sigZ2]


def six_list_original():
    """Order used by QLE.m (commented alternative, Q:26): {H1, H2, H3, H4, H5, H6}."""
    return [H1, H2, H3, H4, H5, H6]


def six_list_optimized():
    """Order used by both optimized scripts (commented alternative, G:37 S:29):
    {H1, H4, H3, H2, H5, H6}. Index 2 here is H4, not H2."""
    return [H1, H4, H3, H2, H5, H6]
