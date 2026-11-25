"""Pauli matrices and the candidate Hamiltonians. See qle/README.md."""
import numpy as np

# Pauli matrices
sigX = np.array([[0, 1], [1, 0]], dtype=float)
sigY = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigZ = np.array([[1, 0], [0, -1]], dtype=float)
sigX2 = 2 * sigX
sigZ2 = 2 * sigZ

# Six-candidate set from the paper
H1 = sigX + sigZ
H2 = 1.2 * np.array([[1, 0], [0, 0]], dtype=float) - 0.8 * np.array([[0, 0], [0, 1]], dtype=float)
H3 = 2 * sigY + np.array([[1, 0], [0, 2]], dtype=float)
H4 = 4 * (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=float)
H5 = sigX + H4
H6 = np.array([[1, 2], [2, -1]], dtype=float)

# Extra examples defined in the MATLAB Grid script; not used in any list
H7 = 0.5 * sigX + 0.8 * sigY
H8 = np.array([[2, 0], [0, -1]], dtype=float)
H9 = np.array([[1, 1j], [-1j, 1]], dtype=complex)
H10 = np.array([[3, 2], [2, 0]], dtype=float)
H11 = np.array([[0, 2 + 1j], [2 - 1j, 1]], dtype=complex)
H12 = 0.6 * sigZ + 0.4 * sigY


def random_hermitian(rng):
    """A random 2x2 Hermitian matrix (H13 in the MATLAB Grid script, unused there).
    It consumes 8 normal draws from rng, so the Grid script still calls it."""
    A = rng.randn(2) + 1j * rng.randn(2)
    H13 = (A + A.conj().T) / 2
    return H13


def default_list():
    """{sigX, 2 sigX, sigZ, 2 sigZ}: the default candidate set in all three scripts."""
    return [sigX, sigX2, sigZ, sigZ2]


def six_list_original():
    """The six-candidate set in the order used by the original QLE: H1..H6."""
    return [H1, H2, H3, H4, H5, H6]


def six_list_optimized():
    """The six-candidate set in the order used by both optimised scripts: H1, H4, H3, H2, H5, H6."""
    return [H1, H4, H3, H2, H5, H6]
