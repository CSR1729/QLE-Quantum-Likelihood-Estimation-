"""Helpers. See qle/README.md."""
import numpy as np

from .matlab_compat import matlab_eig, matlab_min


def Min_Energy_Gap(list):
    """Smallest non-zero energy gap |E2 - E1| among a list of 2x2 Hamiltonians.
    Defined in all three MATLAB scripts but never called."""
    N = len(list)
    gap = np.zeros((N, 1))
    for i in range(1, N + 1):
        D = matlab_eig(list[i - 1])
        gap[i - 1, 0] = np.abs(D[1, 0] - D[0, 0])
    delta = matlab_min(gap[gap > 0].reshape(-1, 1))
    return delta
