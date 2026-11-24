"""Original QLE (QLE.m): fixed state and measurement basis, PGH evolution time.

Refactored from reference/QLE.py. State preparation, evolution, measurement
and the Hamiltonians now come from shared modules; every MATLAB source line is
still listed below, at the call that implements it. The loop, PGH and
discrete_sample are unchanged. tests/test_equivalence.py checks that output is
identical to the reference for the same seed.

Run: python -m qle.algorithms.qle_original [--seed N] [--threshold X] [--candidate-set default|six]
"""
import numpy as np

from ..evolution import evolve
from ..hamiltonians import default_list, six_list_original
from ..matlab_compat import RandomStream, clc, cli, disp, first_find
from ..measurement import likelihood, measurement_basis, prob_overlap, sample_from_distribution
from ..state_preparation import prepare_state
from ..utils import Min_Energy_Gap  # noqa: F401  (defined in QLE.m, unused there)


def run(*, seed=None, threshold=None, candidate_set="default"):
    """Execute the original script and return its original result arrays."""
    if candidate_set not in ("default", "six"):
        raise ValueError("candidate_set must be 'default' or 'six'")
    rng = RandomStream(seed)
    _threshold = threshold
    # MATLAB QLE.m:1: %% Background
    # MATLAB QLE.m:2: % This script implements the original version of QLE.
    # MATLAB QLE.m:3: % The configurable pieces are denoted by comments starting with CONFIGURE.
    # MATLAB QLE.m:4: clc;
    clc()
    # MATLAB QLE.m:5: clearvars;
    pass  # Fresh run() locals implement clearvars; the RNG is not reset.
    # MATLAB QLE.m:6: 
    # MATLAB QLE.m:7: %% Setup
    # MATLAB QLE.m:8-21: Pauli matrices and candidate Hamiltonians -> qle/hamiltonians.py
    # MATLAB QLE.m:22:
    # MATLAB QLE.m:23: %CONFIGURE: The user may compose any list of candidate Hamiltonians out of
    # MATLAB QLE.m:24: %the 2*2 matrices above, or use their own matrices. The paper studies the
    # MATLAB QLE.m:25: %following two lists.
    # MATLAB QLE.m:26: %H_list = {H1, H2, H3, H4, H5, H6};
    # MATLAB QLE.m:27: H_list = {sigX, sigX2, sigZ, sigZ2};
    H_list = default_list()
    # Explicit CLI configuration of the source's commented alternative.
    if candidate_set == "six":
        H_list = six_list_original()
    # MATLAB QLE.m:28: 
    # MATLAB QLE.m:29: N = length(H_list);
    N = len(H_list)
    # MATLAB QLE.m:30: 
    # MATLAB QLE.m:31: %% Parameters
    # MATLAB QLE.m:32: % Here we use the optimal fixed values of the parameters
    # MATLAB QLE.m:33: % theta, phi, alpha, beta. These were found by going
    # MATLAB QLE.m:34: % over a grid with resolution pi/12.
    # MATLAB QLE.m:35: 
    # MATLAB QLE.m:36: % Measurement basis parameters
    # MATLAB QLE.m:37: theta = 0.7854;
    theta = 0.7854
    # MATLAB QLE.m:38: phi=6.2832;
    phi = 6.2832
    # MATLAB QLE.m:39: u1=[cos(theta/2); exp(1i*phi)*sin(theta/2)];%W gate
    # MATLAB QLE.m:40: u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
    # MATLAB QLE.m:41: W=[u1,u2];
    u1, u2, W = measurement_basis(theta, phi)
    # MATLAB QLE.m:42: 
    # MATLAB QLE.m:43: % Initial state parameters
    # MATLAB QLE.m:44: alpha = 1.2566;
    alpha = 1.2566
    # MATLAB QLE.m:45: beta   = 0.5712;
    beta = 0.5712
    # MATLAB QLE.m:46: psi0   = [cos(alpha); exp(1i*beta)*sin(alpha)];
    psi0 = prepare_state(alpha, beta)
    # MATLAB QLE.m:47: 
    # MATLAB QLE.m:48: %% Algorithm
    # MATLAB QLE.m:49: % CONFIGURE: The user may choose the convergence threshold
    # MATLAB QLE.m:50: threshold = 0.99;
    threshold = 0.99
    # An omitted override leaves the original literal unchanged.
    if _threshold is not None:
        threshold = _threshold
    # MATLAB QLE.m:51: iter_arr=zeros(N,1);
    iter_arr = np.zeros((N, 1))
    # MATLAB QLE.m:52: 
    # MATLAB QLE.m:53: % Loop through Hamiltonians
    # MATLAB QLE.m:54: for idx = 1:N
    for idx in range(1, N + 1):
        # MATLAB QLE.m:55: H_true = H_list{idx};
        H_true = H_list[idx - 1]
        # MATLAB QLE.m:56: weights = ones(1, N) / N;
        weights = np.ones((1, N)) / N
        # MATLAB QLE.m:57: num_iterations = 10000;
        num_iterations = 10000
        # MATLAB QLE.m:58: 
        # MATLAB QLE.m:59: iter = 1;
        iter = 1
        # MATLAB QLE.m:60: 
        # MATLAB QLE.m:61: while iter <= num_iterations+1 && max(weights) < threshold
        while iter <= num_iterations + 1 and np.fmax.reduce(weights.ravel(order="F")) < threshold:
            # MATLAB QLE.m:62: % Determine evolution time
            # MATLAB QLE.m:63: t = PGH(H_list, weights);
            t = PGH(H_list, weights, rng)
            # MATLAB QLE.m:64: 
            # MATLAB QLE.m:65: % Step 1: Prepare ensemble states
            # MATLAB QLE.m:66: psi_all = repmat(psi0, 1, N);
            psi_all = np.tile(psi0, (1, N))
            # MATLAB QLE.m:67: 
            # MATLAB QLE.m:68: % Build classical register density rho_tilda_F
            # MATLAB QLE.m:69: rho_tilda_F = diag(weights);
            rho_tilda_F = np.diag(weights.ravel(order="F"))
            # MATLAB QLE.m:70: 
            # MATLAB QLE.m:71: % Build joint density rho_total = \sum_j |psi_j><psi_j| \otimes rho_tilda_F
            # MATLAB QLE.m:72: % Here psi_all(:,j) is psi_j
            # MATLAB QLE.m:73: rho_total = zeros(2*N);
            rho_total = np.zeros((2 * N, 2 * N))
            # MATLAB QLE.m:74: for j = 1:N
            for j in range(1, N + 1):
                # MATLAB QLE.m:75: proj = psi_all(:,j) * psi_all(:,j)';   % 2x2
                proj = psi_all[:, j - 1:j] @ psi_all[:, j - 1:j].conj().T
                # MATLAB QLE.m:76: % embed into block-diagonal positions
                # MATLAB QLE.m:77: block = kron(proj, rho_tilda_F);
                block = np.kron(proj, rho_tilda_F)
                # MATLAB QLE.m:78: rho_total = rho_total + block;
                rho_total = rho_total + block
            # MATLAB QLE.m:79: end
            # MATLAB QLE.m:80: 
            # MATLAB QLE.m:81: disp(['Iteration ', num2str(iter-1), ':']);
            disp(f'Iteration {iter - 1}:')
            # MATLAB QLE.m:82: disp('rho_tilda_F ='); disp(rho_tilda_F);
            disp('rho_tilda_F ='); disp(rho_tilda_F)
            # MATLAB QLE.m:83: disp('rho_total =');     disp(rho_total);
            disp('rho_total ='); disp(rho_total)
            # MATLAB QLE.m:84: 
            # MATLAB QLE.m:85: % Step 2: Oracle evolution and measurement
            # MATLAB QLE.m:86: U_true = expm(-1i * H_true * t);
            # MATLAB QLE.m:87: psi_f  = U_true * psi0;
            psi_f = evolve(H_true, t, psi0)
            # MATLAB QLE.m:88: p_u1   = abs(u1' * psi_f)^2;
            p_u1 = prob_overlap(u1, psi_f)
            # MATLAB QLE.m:89: outcome = sample_from_distribution([p_u1, 1 - p_u1]);
            outcome = sample_from_distribution(np.array([[p_u1, 1 - p_u1]]), rng)
            # MATLAB QLE.m:90: 
            # MATLAB QLE.m:91: % Compute likelihoods and update weights
            # MATLAB QLE.m:92: likelihoods = zeros(1, N);
            likelihoods = np.zeros((1, N))
            # MATLAB QLE.m:93: psi_all = zeros(2, N);
            psi_all = np.zeros((2, N), dtype=complex)  # MATLAB promotes on complex assignment.
            # MATLAB QLE.m:94: for j = 1:N
            for j in range(1, N + 1):
                # MATLAB QLE.m:95: U_j = expm(-1i * H_list{j} * t);
                # MATLAB QLE.m:96: psi_j = U_j * psi0;
                psi_j = evolve(H_list[j - 1], t, psi0)
                # MATLAB QLE.m:97: psi_all(:,j) = psi_j;
                psi_all[:, j - 1:j] = psi_j
                # MATLAB QLE.m:98: pj = abs(u1' * psi_j)^2;
                pj = prob_overlap(u1, psi_j)
                # MATLAB QLE.m:99: likelihoods(j) = (outcome==0)*pj + (outcome==1)*(1-pj);
                likelihoods[0, j - 1] = likelihood(outcome, pj)
            # MATLAB QLE.m:100: end
            # MATLAB QLE.m:101: 
            # MATLAB QLE.m:102: weights = weights .* likelihoods;
            weights = weights * likelihoods
            # MATLAB QLE.m:103: weights = weights / sum(weights);
            weights = weights / np.sum(weights)
            # MATLAB QLE.m:104: 
            # MATLAB QLE.m:105: iter = iter + 1;
            iter = iter + 1
        # MATLAB QLE.m:106: end
        # MATLAB QLE.m:107: iter_arr(idx)=iter;
        iter_arr[idx - 1, 0] = iter
    # MATLAB QLE.m:108: end
    # MATLAB QLE.m:109: disp('results:')
    disp('results:')
    # MATLAB QLE.m:110: disp(iter_arr);
    disp(iter_arr)
    # MATLAB QLE.m:111: 
    # MATLAB QLE.m:112: %% Helper functions
    return {"iter_arr": iter_arr}


# MATLAB QLE.m:113: function t = PGH(H_list, weights)
def PGH(H_list, weights, rng):
    # MATLAB QLE.m:114: i1 = discrete_sample(weights);
    i1 = discrete_sample(weights, rng)
    # MATLAB QLE.m:115: i2 = discrete_sample(weights);
    i2 = discrete_sample(weights, rng)
    # MATLAB QLE.m:116: while i2 == i1
    while i2 == i1:
        # MATLAB QLE.m:117: i2 = discrete_sample(weights);
        i2 = discrete_sample(weights, rng)
    # MATLAB QLE.m:118: end
    # MATLAB QLE.m:119: deltaH = H_list{i1} - H_list{i2};
    deltaH = H_list[i1 - 1] - H_list[i2 - 1]
    # MATLAB QLE.m:120: nH = norm(deltaH, 'fro');
    nH = np.linalg.norm(deltaH, "fro")
    # MATLAB QLE.m:121: t = (nH < 1e-6) * 0.1 + (nH >= 1e-6) * (1 /  nH);
    t = (nH < 1e-6) * 0.1 + (nH >= 1e-6) * (1 / nH)
    return t
# MATLAB QLE.m:122: end
# MATLAB QLE.m:123: 
# MATLAB QLE.m:124: function idx = discrete_sample(w)
def discrete_sample(w, rng):
    # MATLAB QLE.m:125: r = rand(); cw = cumsum(w);
    r = rng.rand(); cw = np.cumsum(w, axis=1)
    # MATLAB QLE.m:126: idx = find(r <= cw, 1, 'first');
    idx = first_find(r <= cw)
    return idx
# MATLAB QLE.m:127: end
# MATLAB QLE.m:128:
# MATLAB QLE.m:129-131: function outcome = sample_from_distribution(p) -> qle/measurement.py
# MATLAB QLE.m:132:
# MATLAB QLE.m:133-144: function delta = Min_Energy_Gap(list) -> qle/utils.py


if __name__ == "__main__":
    cli(run)
