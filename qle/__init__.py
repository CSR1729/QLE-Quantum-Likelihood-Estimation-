"""Quantum Likelihood Estimation (QLE): shared building blocks and three algorithms.

Shared pieces (moved verbatim from the original scripts):
    hamiltonians       Pauli matrices and candidate Hamiltonian lists
    state_preparation  |psi0> = cos(alpha)|0> + e^{i beta} sin(alpha)|1>
    evolution          expm(-iHt) |psi>
    measurement        basis {u1, u2}, projectors, outcome probabilities, sampling, likelihood
    utils              Min_Energy_Gap (defined but unused in all three originals)

Algorithms (each keeps its own loop, time/parameter choice and entropy code):
    algorithms.qle_original  fixed basis/state, PGH time            (QLE.m)
    algorithms.qle_grid      grid search over (t, theta, phi, alpha, beta)  (Optimized_QLE_Grid.m)
    algorithms.qle_sa        simulated annealing over the same five settings (Optimized_QLE_SA.m)
"""
