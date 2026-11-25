"""Original QLE: fixed probe state and measurement basis, evolution time from PGH.

The science is explained in qle/algorithms/README.md.
Run: python -m qle.algorithms.qle_original [--seed N] [--threshold X] [--candidate-set default|six]
"""
import numpy as np

from ..evolution import evolve
from ..hamiltonians import default_list, six_list_original
from ..matlab_compat import RandomStream, clc, cli, disp, first_find
from ..measurement import likelihood, measurement_basis, prob_overlap, sample_from_distribution
from ..state_preparation import prepare_state
from ..utils import Min_Energy_Gap  # noqa: F401  (unused helper from the MATLAB original)


def run(*, seed=None, threshold=None, candidate_set="default"):
    """Try to identify each candidate in turn; return rounds used per candidate (+1)."""
    if candidate_set not in ("default", "six"):
        raise ValueError("candidate_set must be 'default' or 'six'")
    rng = RandomStream(seed)
    _threshold = threshold
    clc()

    # Candidate Hamiltonians
    H_list = default_list()
    if candidate_set == "six":
        H_list = six_list_original()
    N = len(H_list)

    # Fixed measurement basis and probe state, used in every round
    theta = 0.7854
    phi = 6.2832
    u1, u2, W = measurement_basis(theta, phi)
    alpha = 1.2566
    beta = 0.5712
    psi0 = prepare_state(alpha, beta)

    threshold = 0.99
    if _threshold is not None:
        threshold = _threshold
    iter_arr = np.zeros((N, 1))

    # Pretend each candidate in turn is the true Hamiltonian
    for idx in range(1, N + 1):
        H_true = H_list[idx - 1]
        weights = np.ones((1, N)) / N          # uniform prior
        num_iterations = 10000

        iter = 1
        while iter <= num_iterations + 1 and np.fmax.reduce(weights.ravel(order="F")) < threshold:
            # Evolution time from the particle guess heuristic
            t = PGH(H_list, weights, rng)

            # Joint state of probe and classical register (printed only, not used below)
            psi_all = np.tile(psi0, (1, N))
            rho_tilda_F = np.diag(weights.ravel(order="F"))
            rho_total = np.zeros((2 * N, 2 * N))
            for j in range(1, N + 1):
                proj = psi_all[:, j - 1:j] @ psi_all[:, j - 1:j].conj().T
                block = np.kron(proj, rho_tilda_F)
                rho_total = rho_total + block

            disp(f'Iteration {iter - 1}:')
            disp('rho_tilda_F ='); disp(rho_tilda_F)
            disp('rho_total ='); disp(rho_total)

            # Experiment: evolve under the true H and measure once
            psi_f = evolve(H_true, t, psi0)
            p_u1 = prob_overlap(u1, psi_f)
            outcome = sample_from_distribution(np.array([[p_u1, 1 - p_u1]]), rng)

            # Model: how likely was that outcome under each candidate?
            likelihoods = np.zeros((1, N))
            psi_all = np.zeros((2, N), dtype=complex)
            for j in range(1, N + 1):
                psi_j = evolve(H_list[j - 1], t, psi0)
                psi_all[:, j - 1:j] = psi_j
                pj = prob_overlap(u1, psi_j)
                likelihoods[0, j - 1] = likelihood(outcome, pj)

            # Bayes' rule
            weights = weights * likelihoods
            weights = weights / np.sum(weights)

            iter = iter + 1
        iter_arr[idx - 1, 0] = iter

    disp('results:')
    disp(iter_arr)
    return {"iter_arr": iter_arr}


def PGH(H_list, weights, rng):
    """Particle guess heuristic: t = 1/||Hi - Hj|| for two distinct candidates drawn by weight."""
    i1 = discrete_sample(weights, rng)
    i2 = discrete_sample(weights, rng)
    while i2 == i1:
        i2 = discrete_sample(weights, rng)
    deltaH = H_list[i1 - 1] - H_list[i2 - 1]
    nH = np.linalg.norm(deltaH, "fro")
    t = (nH < 1e-6) * 0.1 + (nH >= 1e-6) * (1 / nH)
    return t


def discrete_sample(w, rng):
    """Draw a candidate index (1-based) with probability proportional to its weight."""
    r = rng.rand(); cw = np.cumsum(w, axis=1)
    idx = first_find(r <= cw)
    return idx


if __name__ == "__main__":
    cli(run)
