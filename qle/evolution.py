"""Time evolution under a Hamiltonian. See qle/README.md."""
from scipy.linalg import expm


def evolve(H, t, psi):
    """Schrodinger evolution (hbar = 1): returns exp(-i H t) |psi>."""
    return expm(-1j * H * t) @ psi
