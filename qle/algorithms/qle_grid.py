"""Optimized QLE, grid search: each round picks (t, theta, phi, alpha, beta) by
trying every point on a fixed grid and keeping the one that minimises H(F|Y).

The science is explained in qle/algorithms/README.md.
Run: python -m qle.algorithms.qle_grid [--seed N] [--threshold X] [--candidate-set default|six]
"""
import warnings

import numpy as np

from ..evolution import evolve
from ..hamiltonians import default_list, random_hermitian, six_list_optimized
from ..matlab_compat import RandomStream, clc, cli, descending_indices, disp, first_min, matlab_eig
from ..measurement import (clip_probability, likelihood, measurement_basis, projectors,
                           rotated_probability, sample_from_distribution)
from ..state_preparation import prepare_state
from ..utils import Min_Energy_Gap  # noqa: F401  (unused helper from the MATLAB original)


def run(*, seed=None, threshold=None, candidate_set="default"):
    """Try to identify each candidate in turn; return convergence rounds and success flags."""
    if candidate_set not in ("default", "six"):
        raise ValueError("candidate_set must be 'default' or 'six'")
    rng = RandomStream(seed)
    _threshold = threshold
    clc()

    # Unused random Hamiltonian from the MATLAB original. It still draws random
    # numbers, so it stays here to keep seeded runs identical.
    H13 = random_hermitian(rng)

    # Candidate Hamiltonians
    H_list = default_list()
    if candidate_set == "six":
        H_list = six_list_optimized()
    N = len(H_list)

    convergence_iters = np.full((1, N), np.nan)
    success_flags = np.zeros((1, N), dtype=bool)
    threshold = 0.99
    if _threshold is not None:
        threshold = _threshold

    # Pretend each candidate in turn is the true Hamiltonian
    for idx in range(1, N + 1):
        disp('**********************')
        disp('the right H is:'); disp(idx)
        H_true = H_list[idx - 1]
        weights = np.ones((1, N)) / N          # uniform prior
        num_iterations = 10

        iter = 1
        while iter <= num_iterations + 1:
            if np.fmax.reduce(weights.ravel(order="F")) >= threshold:
                convergence_iters[0, idx - 1] = iter
                break

            # Choose this round's experiment: the settings that minimise H(F|Y)
            best_params = optimize_entropy(weights, H_list)
            t = best_params[0, 0]
            theta = best_params[0, 1]
            phi = best_params[0, 2]
            alpha = best_params[0, 3]
            beta = best_params[0, 4]
            disp("ideal t")
            disp(t)

            psi0 = prepare_state(alpha, beta)
            psi0 = psi0 / np.linalg.norm(psi0)

            # Joint probe/register state (computed but not used, as in the MATLAB original)
            psi_all = np.tile(psi0, (1, N))
            rho_tilda_F = np.diag(weights.ravel(order="F"))
            rho_total = np.zeros((2 * N, 2 * N))
            for j in range(1, N + 1):
                proj = psi_all[:, j - 1:j] @ psi_all[:, j - 1:j].conj().T
                block = np.kron(proj, rho_tilda_F)
                rho_total = rho_total + block

            # Experiment: evolve under the true H, rotate into the chosen basis, measure once
            psi_f = evolve(H_true, t, psi0)
            psi_f = psi_f / np.linalg.norm(psi_f)
            u1, u2, W = measurement_basis(theta, phi)
            psi_final, p_u1 = rotated_probability(W, psi_f)
            outcome = sample_from_distribution(np.array([[p_u1, 1 - p_u1]]), rng)

            # Model: likelihood of that outcome under each candidate, plus
            # the information quantities H(Y) and H(Y|F) for reporting
            likelihoods = np.zeros((1, N))
            psi_all = np.zeros((2, N), dtype=complex)
            rho_Y = np.zeros((2, 2))
            H_YF = 0
            for j in range(1, N + 1):
                psi_j = evolve(H_list[j - 1], t, psi0)
                psi_j, pj = rotated_probability(W, psi_j)
                psi_all[:, j - 1:j] = psi_j
                pj = clip_probability(pj)
                likelihoods[0, j - 1] = likelihood(outcome, pj)
                rho_Y = rho_Y + (weights[0, j - 1] * psi_j) @ psi_j.conj().T
                H_YF = H_YF + weights[0, j - 1] * (-pj * np.log2(pj) - (1 - pj) * np.log2(1 - pj))

            ent_y = Entropy(np.diag(np.diag(rho_Y)))
            disp(f'Iteration {iter}:')
            disp('H(Y)='); disp(ent_y)
            disp('H(Y|F)='); disp(H_YF)
            disp('I(FY)='); disp(ent_y - H_YF)

            # Bayes' rule
            weights = weights * likelihoods
            weights[np.real(weights) < 0] = 0
            weights = weights / np.sum(weights)

            if np.fmax.reduce(weights.ravel(order="F")) >= 0.95:
                _, guess_H = first_min(-weights)
                success_flags[0, idx - 1] = guess_H == idx

            disp('weights ='); disp(weights)
            iter = iter + 1

    disp('Convergence iterations:')
    disp(convergence_iters - 1)
    disp('Did each H converge to correct hypothesis?')
    disp(success_flags)
    disp('All correct: ' + ('true' if np.all(success_flags) else 'false'))
    return {"convergence_iters": convergence_iters, "success_flags": success_flags}


def PGH(H_list, weights):
    """Alternative time heuristic, t = pi / (2 ||H1 - H2||) for the two most likely
    candidates. Defined in the MATLAB original but never called."""
    I = descending_indices(weights)
    i1 = I[0, 0]; i2 = I[0, 1]
    deltaH = H_list[i1 - 1] - H_list[i2 - 1]
    normH = np.linalg.norm(deltaH, "fro")
    t = np.pi / (2 * normH)
    return t


def Entropy(rho):
    """Von Neumann entropy S = -sum(lambda * log2 lambda) over the eigenvalues of rho."""
    rho = (rho + rho.conj().T) / 2
    if np.linalg.norm(rho - rho.conj().T, "fro") > 1e-10:
        warnings.warn('Entropy: input not Hermitian even after symmetrization.', RuntimeWarning)
    _, D = matlab_eig(rho, vectors=True)
    eigvals = np.real(np.diag(D)).reshape(-1, 1)
    eigvals = eigvals[eigvals > 1e-12].reshape(-1, 1)
    S = -np.sum(eigvals * np.log2(eigvals))
    return S


def optimize_entropy(weights, H_list):
    """Return [t, theta, phi, alpha, beta] minimising H(F|Y) over a fixed grid."""
    N = len(H_list)

    # The grid: 10 x 6 x 24 x 12 x 24 = 414,720 settings
    t_vals = np.linspace(0.1, 2 * np.pi, 10)
    sample = 24
    theta_vals = np.linspace(0, np.pi / 2, int(sample / 4))
    phi_vals = np.linspace(0, 2 * np.pi, sample)
    alpha_vals = np.linspace(0, np.pi, int(sample / 2))
    beta_vals = np.linspace(0, 2 * np.pi, sample)

    min_entropy = np.inf
    best_params = np.empty((0, 0))
    IN = np.eye(N)

    for t in t_vals:
        for alpha in alpha_vals:
            for beta in beta_vals:
                # Joint state rho_FY = sum_j w_j |j><j| (x) |psi_j><psi_j|, block-diagonal
                psi0 = prepare_state(alpha, beta)
                rho_FY = np.zeros((2 * N, 2 * N), dtype=complex)
                for j in range(1, N + 1):
                    H = H_list[j - 1]
                    psi_j = evolve(H, t, psi0)
                    rho_j = psi_j @ psi_j.conj().T
                    rho_FY[2 * j - 2:2 * j, 2 * j - 2:2 * j] = weights[0, j - 1] * rho_j

                for theta in theta_vals:
                    for phi in phi_vals:
                        # Project onto each measurement outcome
                        u1, u2, W = measurement_basis(theta, phi)
                        A_plus, A_minus = projectors(W)
                        rho_plus = np.kron(IN, A_plus) @ rho_FY @ np.kron(IN, A_plus)
                        rho_minus = np.kron(IN, A_minus) @ rho_FY @ np.kron(IN, A_minus)
                        prob_plus = np.trace(rho_plus)
                        prob_minus = np.trace(rho_minus)

                        # Skip settings where either outcome is impossible
                        if np.real(prob_plus) > 0:
                            rho_plus = rho_plus / prob_plus
                        else:
                            continue
                        if np.real(prob_minus) > 0:
                            rho_minus = rho_minus / prob_minus
                        else:
                            continue

                        # Expected entropy left after the measurement, H(F|Y)
                        ent = prob_plus * Entropy(rho_plus) + prob_minus * Entropy(rho_minus)
                        if np.real(ent) < np.real(min_entropy):
                            min_entropy = ent
                            best_params = np.array([[t, theta, phi, alpha, beta]])
    return best_params


if __name__ == "__main__":
    cli(run)
