"""Optimized QLE, simulated annealing: each round picks (t, theta, phi, alpha, beta)
by simulated annealing to minimise H(F|Y).

The science is explained in qle/algorithms/README.md.
Run: python -m qle.algorithms.qle_sa [--seed N] [--threshold X] [--candidate-set default|six]
"""
import numpy as np

from ..evolution import evolve
from ..hamiltonians import default_list, six_list_optimized
from ..matlab_compat import RandomStream, clc, cli, disp, first_min, matlab_eig
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
        disp('******************************************************************')
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
            best_params = optimize_entropy(weights, H_list, rng)
            t = best_params[0, 0]
            theta = best_params[0, 1]
            phi = best_params[0, 2]
            alpha = best_params[0, 3]
            beta = best_params[0, 4]
            disp("ideal t")
            disp(t)

            psi0 = prepare_state(alpha, beta)
            psi0 = psi0 / np.linalg.norm(psi0)

            # Experiment: evolve under the true H, rotate into the chosen basis, measure once
            psi_f = evolve(H_true, t, psi0)
            psi_f = psi_f / np.linalg.norm(psi_f)
            u1, u2, W = measurement_basis(theta, phi)
            psi_final, p_u1 = rotated_probability(W, psi_f)
            outcome = sample_from_distribution(np.array([[p_u1, 1 - p_u1]]), rng)

            # Model: likelihood of that outcome under each candidate, plus
            # the information quantities H(Y) and H(Y|F) for reporting
            likelihoods = np.zeros((1, N))
            rho_Y = np.zeros((2, 2))
            H_YF = 0
            for j in range(1, N + 1):
                psi_j, pj = rotated_probability(W, evolve(H_list[j - 1], t, psi0))
                pj = clip_probability(pj)
                likelihoods[0, j - 1] = likelihood(outcome, pj)
                rho_Y = rho_Y + weights[0, j - 1] * (psi_j @ psi_j.conj().T)
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


def optimize_entropy(weights, H_list, rng):
    """Simulated annealing over x = [t, theta, phi, alpha, beta] to minimise entropy_cost."""
    max_iters = 200
    T_init = 1.0
    alpha = 0.9                                # cooling factor (not the state angle)

    # Random starting point
    x = np.array([[rng.rand() * 2 * np.pi, rng.rand() * np.pi, rng.rand() * 2 * np.pi, rng.rand() * np.pi, rng.rand() * 2 * np.pi]])
    cost = entropy_cost(x, weights, H_list)
    T = T_init
    best_x = x.copy()
    best_cost = cost
    num_neighbors = 20

    for iter in range(1, max_iters + 1):
        # Try 20 random neighbours; step size shrinks with temperature T
        candidates = np.zeros((num_neighbors, 5))
        candidate_costs = np.zeros((num_neighbors, 1))
        for k in range(1, num_neighbors + 1):
            step = np.array([[np.pi, np.pi / 2, np.pi, np.pi / 2, np.pi]])
            x_new = x + (rng.rand(1, 5) * 2 - 1) * step * T
            # Wrap each setting back into its range
            x_new[0, 0] = np.mod(x_new[0, 0], 2 * np.pi)
            x_new[0, 1] = np.mod(x_new[0, 1], np.pi)
            x_new[0, 2] = np.mod(x_new[0, 2], 2 * np.pi)
            x_new[0, 3] = np.mod(x_new[0, 3], np.pi)
            x_new[0, 4] = np.mod(x_new[0, 4], 2 * np.pi)
            cost_new = entropy_cost(x_new, weights, H_list)
            candidates[k - 1:k, :] = x_new
            candidate_costs[k - 1, 0] = cost_new

        # Move to the best neighbour if it is better, or with
        # Metropolis probability exp(-(increase)/T) if it is worse
        min_cost, best_k = first_min(candidate_costs)
        x_best = candidates[best_k - 1:best_k, :].copy()
        if np.real(min_cost) < np.real(cost) or rng.rand() < np.exp((cost - min_cost) / T):
            x = x_best.copy()
            cost = min_cost
            if np.real(cost) < np.real(best_cost):
                best_x = x.copy()
                best_cost = cost

        T = T * alpha                          # cool down

    best_params = best_x.copy()
    print("Best entropy after annealing: %.5f" % best_cost)
    return best_params


def entropy_cost(x, weights, H_list):
    """H(F|Y): expected entropy of the candidate register after measuring with settings x."""
    t = x[0, 0]; theta = x[0, 1]; phi = x[0, 2]; alpha = x[0, 3]; beta = x[0, 4]
    N = len(H_list)

    # Joint state rho_FY = sum_j w_j |j><j| (x) |psi_j><psi_j|, block-diagonal
    psi0 = prepare_state(alpha, beta)
    rho_FY = np.zeros((2 * N, 2 * N), dtype=complex)
    for j in range(1, N + 1):
        psi_j = evolve(H_list[j - 1], t, psi0)
        rho_j = psi_j @ psi_j.conj().T
        rho_FY[2 * j - 2:2 * j, 2 * j - 2:2 * j] = weights[0, j - 1] * rho_j

    # Project onto each measurement outcome
    u1, u2, W = measurement_basis(theta, phi)
    A_plus, A_minus = projectors(W)
    rho_plus = np.kron(np.eye(N), A_plus) @ rho_FY @ np.kron(np.eye(N), A_plus)
    rho_minus = np.kron(np.eye(N), A_minus) @ rho_FY @ np.kron(np.eye(N), A_minus)
    prob_plus = np.real(np.trace(rho_plus))
    prob_minus = np.real(np.trace(rho_minus))

    # Entropy left after each outcome, weighted by its probability
    S_plus = 0; S_minus = 0
    if np.real(prob_plus) > 0:
        rho_plus = rho_plus / prob_plus
        S_plus = Entropy(rho_plus)
    if np.real(prob_minus) > 0:
        rho_minus = rho_minus / prob_minus
        S_minus = Entropy(rho_minus)
    cost = prob_plus * S_plus + prob_minus * S_minus
    return cost


def Entropy(rho):
    """Von Neumann entropy S = -sum(lambda * log2 lambda) over the eigenvalues of rho."""
    rho = (rho + rho.conj().T) / 2
    eigvals = np.real(matlab_eig(rho))
    eigvals = eigvals[eigvals > 1e-12].reshape(-1, 1)
    S = -np.sum(eigvals * np.log2(eigvals))
    return S


if __name__ == "__main__":
    cli(run)
