"""Helpers defined identically in all three scripts (and unused by them).

Line references: Q = QLE.m, G = Optimized_QLE_Grid.m, S = Optimized_QLE_SA.m.
"""
import numpy as np

from .matlab_compat import matlab_eig, matlab_min


# MATLAB Q:133-135 G:245-247 S:231-233:
# % Min_Energy_Gap - Find the minimal energy gap in a given list of Hamiltonians
# %   INPUT:  list - a list of 2*2 Hermitian matrices
# %   OUTPUT: delta - the minimal energy gap
# MATLAB Q:136 G:248 S:234: function delta = Min_Energy_Gap(list)
def Min_Energy_Gap(list):
    # MATLAB: N = length(list);
    N = len(list)
    # MATLAB: gap = zeros(N,1);
    gap = np.zeros((N, 1))
    # MATLAB: for i = 1:N
    for i in range(1, N + 1):
        # MATLAB: D = eig(list{i});
        D = matlab_eig(list[i - 1])
        # MATLAB: gap(i) = abs(D(2)-D(1));
        gap[i - 1, 0] = np.abs(D[1, 0] - D[0, 0])
    # MATLAB: end
    # MATLAB: delta = min(gap(gap>0));
    delta = matlab_min(gap[gap > 0].reshape(-1, 1))
    return delta
# MATLAB: end
