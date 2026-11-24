"""Optimized QLE, simulated annealing (Optimized_QLE_SA.m).

Each round, (t, theta, phi, alpha, beta) is chosen by simulated annealing to
minimise H(F|Y). Refactored from reference/Optimized_QLE_SA.py: state
preparation, evolution, measurement and the Hamiltonians now come from shared
modules; every MATLAB source line is still listed below, at the call that
implements it. The loop, optimize_entropy, entropy_cost and Entropy are
unchanged. tests/test_equivalence.py checks that output is identical to the
reference for the same seed.

Run: python -m qle.algorithms.qle_sa [--seed N] [--threshold X] [--candidate-set default|six]
"""
import numpy as np

from ..evolution import evolve
from ..hamiltonians import default_list, six_list_optimized
from ..matlab_compat import RandomStream, clc, cli, disp, first_min, matlab_eig
from ..measurement import (clip_probability, likelihood, measurement_basis, projectors,
                           rotated_probability, sample_from_distribution)
from ..state_preparation import prepare_state
from ..utils import Min_Energy_Gap  # noqa: F401  (defined in Optimized_QLE_SA.m, unused there)


def run(*, seed=None, threshold=None, candidate_set="default"):
    """Execute the original script and return its original result arrays."""
    if candidate_set not in ("default", "six"):
        raise ValueError("candidate_set must be 'default' or 'six'")
    rng = RandomStream(seed)
    _threshold = threshold
    # MATLAB Optimized_QLE_SA.m:1: %% Background
    # MATLAB Optimized_QLE_SA.m:2: % This script implements the optimized version of QLE developed in the
    # MATLAB Optimized_QLE_SA.m:3: % article "Optimal Quantum Likelihood Estimation". This version finds the
    # MATLAB Optimized_QLE_SA.m:4: % optimal parameters for each iteration using simulated annealing.
    # MATLAB Optimized_QLE_SA.m:5: % The configurable pieces are denoted by comments starting with CONFIGURE.
    # MATLAB Optimized_QLE_SA.m:6: 
    # MATLAB Optimized_QLE_SA.m:7: clc;
    clc()
    # MATLAB Optimized_QLE_SA.m:8: clearvars;
    pass  # Fresh run() locals implement clearvars; the RNG is not reset.
    # MATLAB Optimized_QLE_SA.m:9: 
    # MATLAB Optimized_QLE_SA.m:10: %% Setup
    # MATLAB Optimized_QLE_SA.m:11-24: Pauli matrices and candidate Hamiltonians -> qle/hamiltonians.py
    # MATLAB Optimized_QLE_SA.m:25: 
    # MATLAB Optimized_QLE_SA.m:26: %CONFIGURE: The user may compose any list of candidate Hamiltonians out of
    # MATLAB Optimized_QLE_SA.m:27: %the 2*2 matrices above, or use their own matrices. The paper studies the
    # MATLAB Optimized_QLE_SA.m:28: %following two lists.
    # MATLAB Optimized_QLE_SA.m:29: %H_list = {H1, H4, H3, H2, H5, H6};
    # MATLAB Optimized_QLE_SA.m:30: H_list = {sigX, sigX2, sigZ, sigZ2};
    H_list = default_list()
    # Explicit CLI configuration of the source's commented alternative.
    if candidate_set == "six":
        H_list = six_list_optimized()
    # MATLAB Optimized_QLE_SA.m:31: N = length(H_list);
    N = len(H_list)
    # MATLAB Optimized_QLE_SA.m:32: 
    # MATLAB Optimized_QLE_SA.m:33: convergence_iters = NaN(1, N);
    convergence_iters = np.full((1, N), np.nan)
    # MATLAB Optimized_QLE_SA.m:34: success_flags = false(1, N);
    success_flags = np.zeros((1, N), dtype=bool)
    # MATLAB Optimized_QLE_SA.m:35: 
    # MATLAB Optimized_QLE_SA.m:36: %% Algorithm
    # MATLAB Optimized_QLE_SA.m:37: % CONFIGURE: The user may choose the convergence threshold.
    # MATLAB Optimized_QLE_SA.m:38: threshold = 0.99;
    threshold = 0.99
    # An omitted override leaves the original literal unchanged.
    if _threshold is not None:
        threshold = _threshold
    # MATLAB Optimized_QLE_SA.m:39: for idx = 1:N
    for idx in range(1, N + 1):
        # MATLAB Optimized_QLE_SA.m:40: disp('******************************************************************')
        disp('******************************************************************')
        # MATLAB Optimized_QLE_SA.m:41: disp('the right H is:'); disp(idx);
        disp('the right H is:'); disp(idx)
        # MATLAB Optimized_QLE_SA.m:42: H_true = H_list{idx};
        H_true = H_list[idx - 1]
        # MATLAB Optimized_QLE_SA.m:43: weights = ones(1, N) / N;
        weights = np.ones((1, N)) / N
        # MATLAB Optimized_QLE_SA.m:44: num_iterations = 10;
        num_iterations = 10
        # MATLAB Optimized_QLE_SA.m:45: 
        # MATLAB Optimized_QLE_SA.m:46: iter = 1;
        iter = 1
        # MATLAB Optimized_QLE_SA.m:47: while iter <= num_iterations+1
        while iter <= num_iterations + 1:
            # MATLAB Optimized_QLE_SA.m:48: if max(weights) >= threshold
            if np.fmax.reduce(weights.ravel(order="F")) >= threshold:
                # MATLAB Optimized_QLE_SA.m:49: convergence_iters(idx) = iter;
                convergence_iters[0, idx - 1] = iter
                # MATLAB Optimized_QLE_SA.m:50: break;
                break
            # MATLAB Optimized_QLE_SA.m:51: end
            # MATLAB Optimized_QLE_SA.m:52: 
            # MATLAB Optimized_QLE_SA.m:53: best_params = optimize_entropy(weights, H_list);
            best_params = optimize_entropy(weights, H_list, rng)
            # MATLAB Optimized_QLE_SA.m:54: t     = best_params(1);
            t = best_params[0, 0]
            # MATLAB Optimized_QLE_SA.m:55: theta = best_params(2);
            theta = best_params[0, 1]
            # MATLAB Optimized_QLE_SA.m:56: phi   = best_params(3);
            phi = best_params[0, 2]
            # MATLAB Optimized_QLE_SA.m:57: alpha = best_params(4);
            alpha = best_params[0, 3]
            # MATLAB Optimized_QLE_SA.m:58: beta  = best_params(5);
            beta = best_params[0, 4]
            # MATLAB Optimized_QLE_SA.m:59: disp("ideal t")
            disp("ideal t")
            # MATLAB Optimized_QLE_SA.m:60: disp(t);
            disp(t)
            # MATLAB Optimized_QLE_SA.m:61: 
            # MATLAB Optimized_QLE_SA.m:62: psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];
            psi0 = prepare_state(alpha, beta)
            # MATLAB Optimized_QLE_SA.m:63: psi0 = psi0 / norm(psi0);
            psi0 = psi0 / np.linalg.norm(psi0)
            # MATLAB Optimized_QLE_SA.m:64:
            # MATLAB Optimized_QLE_SA.m:65: U_true = expm(-1i * H_true * t);
            # MATLAB Optimized_QLE_SA.m:66: psi_f  = U_true * psi0;
            psi_f = evolve(H_true, t, psi0)
            # MATLAB Optimized_QLE_SA.m:67: psi_f = psi_f / norm(psi_f);
            psi_f = psi_f / np.linalg.norm(psi_f)
            # MATLAB Optimized_QLE_SA.m:68:
            # MATLAB Optimized_QLE_SA.m:69: u1 = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
            # MATLAB Optimized_QLE_SA.m:70: u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
            # MATLAB Optimized_QLE_SA.m:71: W = [u1, u2];
            u1, u2, W = measurement_basis(theta, phi)
            # MATLAB Optimized_QLE_SA.m:72:
            # MATLAB Optimized_QLE_SA.m:73: psi_final = W * psi_f;
            # MATLAB Optimized_QLE_SA.m:74: psi_final = psi_final / norm(psi_final);
            # MATLAB Optimized_QLE_SA.m:75: p_u1 = abs(psi_final(1))^2;
            psi_final, p_u1 = rotated_probability(W, psi_f)
            # MATLAB Optimized_QLE_SA.m:76: outcome = sample_from_distribution([p_u1, 1 - p_u1]);
            outcome = sample_from_distribution(np.array([[p_u1, 1 - p_u1]]), rng)
            # MATLAB Optimized_QLE_SA.m:77: 
            # MATLAB Optimized_QLE_SA.m:78: likelihoods = zeros(1, N);
            likelihoods = np.zeros((1, N))
            # MATLAB Optimized_QLE_SA.m:79: rho_Y = zeros(2);
            rho_Y = np.zeros((2, 2))
            # MATLAB Optimized_QLE_SA.m:80: H_YF = 0;
            H_YF = 0
            # MATLAB Optimized_QLE_SA.m:81: 
            # MATLAB Optimized_QLE_SA.m:82: for j = 1:N
            for j in range(1, N + 1):
                # MATLAB Optimized_QLE_SA.m:83: U_j = expm(-1i * H_list{j} * t);
                # MATLAB Optimized_QLE_SA.m:84: psi_j = W * (U_j * psi0);
                # MATLAB Optimized_QLE_SA.m:85: psi_j = psi_j / norm(psi_j);
                # MATLAB Optimized_QLE_SA.m:86: pj = abs(psi_j(1))^2;
                psi_j, pj = rotated_probability(W, evolve(H_list[j - 1], t, psi0))
                # MATLAB Optimized_QLE_SA.m:87: pj = min(max(real(pj), 1e-12), 1 - 1e-12);
                pj = clip_probability(pj)
                # MATLAB Optimized_QLE_SA.m:88: likelihoods(j) = (outcome == 0)*pj + (outcome == 1)*(1 - pj);
                likelihoods[0, j - 1] = likelihood(outcome, pj)
                # MATLAB Optimized_QLE_SA.m:89: rho_Y = rho_Y + weights(j) * (psi_j * psi_j');
                rho_Y = rho_Y + weights[0, j - 1] * (psi_j @ psi_j.conj().T)
                # MATLAB Optimized_QLE_SA.m:90: H_YF = H_YF + weights(j) * (-pj*log2(pj) - (1 - pj)*log2(1 - pj));
                H_YF = H_YF + weights[0, j - 1] * (-pj * np.log2(pj) - (1 - pj) * np.log2(1 - pj))
            # MATLAB Optimized_QLE_SA.m:91: end
            # MATLAB Optimized_QLE_SA.m:92: 
            # MATLAB Optimized_QLE_SA.m:93: ent_y = Entropy(diag(diag(rho_Y)));
            ent_y = Entropy(np.diag(np.diag(rho_Y)))
            # MATLAB Optimized_QLE_SA.m:94: disp(['Iteration ', num2str(iter), ':']);
            disp(f'Iteration {iter}:')
            # MATLAB Optimized_QLE_SA.m:95: disp('H(Y)='); disp(ent_y);
            disp('H(Y)='); disp(ent_y)
            # MATLAB Optimized_QLE_SA.m:96: disp('H(Y|F)='); disp(H_YF);
            disp('H(Y|F)='); disp(H_YF)
            # MATLAB Optimized_QLE_SA.m:97: disp('I(FY)='); disp(ent_y - H_YF);
            disp('I(FY)='); disp(ent_y - H_YF)
            # MATLAB Optimized_QLE_SA.m:98: 
            # MATLAB Optimized_QLE_SA.m:99: weights = weights .* likelihoods;
            weights = weights * likelihoods
            # MATLAB Optimized_QLE_SA.m:100: weights(weights < 0) = 0;
            weights[np.real(weights) < 0] = 0
            # MATLAB Optimized_QLE_SA.m:101: weights = weights / sum(weights);
            weights = weights / np.sum(weights)
            # MATLAB Optimized_QLE_SA.m:102: 
            # MATLAB Optimized_QLE_SA.m:103: if max(weights) >= 0.95
            if np.fmax.reduce(weights.ravel(order="F")) >= 0.95:
                # MATLAB Optimized_QLE_SA.m:104: [~, guess_H] = max(weights);
                _, guess_H = first_min(-weights)
                # MATLAB Optimized_QLE_SA.m:105: success_flags(idx) = (guess_H == idx);
                success_flags[0, idx - 1] = guess_H == idx
            # MATLAB Optimized_QLE_SA.m:106: end
            # MATLAB Optimized_QLE_SA.m:107: 
            # MATLAB Optimized_QLE_SA.m:108: disp('weights ='); disp(weights);
            disp('weights ='); disp(weights)
            # MATLAB Optimized_QLE_SA.m:109: iter = iter + 1;
            iter = iter + 1
        # MATLAB Optimized_QLE_SA.m:110: end
    # MATLAB Optimized_QLE_SA.m:111: end
    # MATLAB Optimized_QLE_SA.m:112: 
    # MATLAB Optimized_QLE_SA.m:113: disp('Convergence iterations:');
    disp('Convergence iterations:')
    # MATLAB Optimized_QLE_SA.m:114: disp(convergence_iters - 1);
    disp(convergence_iters - 1)
    # MATLAB Optimized_QLE_SA.m:115: disp('Did each H converge to correct hypothesis?');
    disp('Did each H converge to correct hypothesis?')
    # MATLAB Optimized_QLE_SA.m:116: disp(success_flags);
    disp(success_flags)
    # MATLAB Optimized_QLE_SA.m:117: disp(['All correct: ', mat2str(all(success_flags))]);
    disp('All correct: ' + ('true' if np.all(success_flags) else 'false'))
    # MATLAB Optimized_QLE_SA.m:118: 
    # MATLAB Optimized_QLE_SA.m:119: 
    # MATLAB Optimized_QLE_SA.m:120: %% FUNCTIONS
    # MATLAB Optimized_QLE_SA.m:121: 
    # MATLAB Optimized_QLE_SA.m:122: % Simulated Annealing optimization
    return {"convergence_iters": convergence_iters, "success_flags": success_flags}


# MATLAB Optimized_QLE_SA.m:123: function best_params = optimize_entropy(weights, H_list)
def optimize_entropy(weights, H_list, rng):
    # MATLAB Optimized_QLE_SA.m:124: max_iters = 200;
    max_iters = 200
    # MATLAB Optimized_QLE_SA.m:125: T_init = 1.0;
    T_init = 1.0
    # MATLAB Optimized_QLE_SA.m:126: alpha = 0.9;
    alpha = 0.9
    # MATLAB Optimized_QLE_SA.m:127: 
    # MATLAB Optimized_QLE_SA.m:128: x = [rand*2*pi, rand*pi, rand*2*pi, rand*pi, rand*2*pi];
    x = np.array([[rng.rand() * 2 * np.pi, rng.rand() * np.pi, rng.rand() * 2 * np.pi, rng.rand() * np.pi, rng.rand() * 2 * np.pi]])
    # MATLAB Optimized_QLE_SA.m:129: cost = entropy_cost(x, weights, H_list);
    cost = entropy_cost(x, weights, H_list)
    # MATLAB Optimized_QLE_SA.m:130: T = T_init;
    T = T_init
    # MATLAB Optimized_QLE_SA.m:131: 
    # MATLAB Optimized_QLE_SA.m:132: best_x = x;
    best_x = x.copy()
    # MATLAB Optimized_QLE_SA.m:133: best_cost = cost;
    best_cost = cost
    # MATLAB Optimized_QLE_SA.m:134: 
    # MATLAB Optimized_QLE_SA.m:135: num_neighbors = 20;
    num_neighbors = 20
    # MATLAB Optimized_QLE_SA.m:136: 
    # MATLAB Optimized_QLE_SA.m:137: for iter = 1:max_iters
    for iter in range(1, max_iters + 1):
        # MATLAB Optimized_QLE_SA.m:138: candidates = zeros(num_neighbors, 5);
        candidates = np.zeros((num_neighbors, 5))
        # MATLAB Optimized_QLE_SA.m:139: candidate_costs = zeros(num_neighbors, 1);
        candidate_costs = np.zeros((num_neighbors, 1))
        # MATLAB Optimized_QLE_SA.m:140: 
        # MATLAB Optimized_QLE_SA.m:141: for k = 1:num_neighbors
        for k in range(1, num_neighbors + 1):
            # MATLAB Optimized_QLE_SA.m:142: step = [pi, pi/2, pi, pi/2, pi];
            step = np.array([[np.pi, np.pi / 2, np.pi, np.pi / 2, np.pi]])
            # MATLAB Optimized_QLE_SA.m:143: x_new = x + (rand(1,5)*2 - 1) .* step * T;
            x_new = x + (rng.rand(1, 5) * 2 - 1) * step * T
            # MATLAB Optimized_QLE_SA.m:144: 
            # MATLAB Optimized_QLE_SA.m:145: % wrap
            # MATLAB Optimized_QLE_SA.m:146: x_new(1) = mod(x_new(1), 2*pi);
            x_new[0, 0] = np.mod(x_new[0, 0], 2 * np.pi)
            # MATLAB Optimized_QLE_SA.m:147: x_new(2) = mod(x_new(2), pi);
            x_new[0, 1] = np.mod(x_new[0, 1], np.pi)
            # MATLAB Optimized_QLE_SA.m:148: x_new(3) = mod(x_new(3), 2*pi);
            x_new[0, 2] = np.mod(x_new[0, 2], 2 * np.pi)
            # MATLAB Optimized_QLE_SA.m:149: x_new(4) = mod(x_new(4), pi);
            x_new[0, 3] = np.mod(x_new[0, 3], np.pi)
            # MATLAB Optimized_QLE_SA.m:150: x_new(5) = mod(x_new(5), 2*pi);
            x_new[0, 4] = np.mod(x_new[0, 4], 2 * np.pi)
            # MATLAB Optimized_QLE_SA.m:151: 
            # MATLAB Optimized_QLE_SA.m:152: cost_new = entropy_cost(x_new, weights, H_list);
            cost_new = entropy_cost(x_new, weights, H_list)
            # MATLAB Optimized_QLE_SA.m:153: 
            # MATLAB Optimized_QLE_SA.m:154: candidates(k,:) = x_new;
            candidates[k - 1:k, :] = x_new
            # MATLAB Optimized_QLE_SA.m:155: candidate_costs(k) = cost_new;
            candidate_costs[k - 1, 0] = cost_new
        # MATLAB Optimized_QLE_SA.m:156: end
        # MATLAB Optimized_QLE_SA.m:157: 
        # MATLAB Optimized_QLE_SA.m:158: [min_cost, best_k] = min(candidate_costs);
        min_cost, best_k = first_min(candidate_costs)
        # MATLAB Optimized_QLE_SA.m:159: x_best = candidates(best_k, :);
        x_best = candidates[best_k - 1:best_k, :].copy()
        # MATLAB Optimized_QLE_SA.m:160: 
        # MATLAB Optimized_QLE_SA.m:161: if min_cost < cost || rand < exp((cost - min_cost)/T)
        if np.real(min_cost) < np.real(cost) or rng.rand() < np.exp((cost - min_cost) / T):
            # MATLAB Optimized_QLE_SA.m:162: x = x_best;
            x = x_best.copy()
            # MATLAB Optimized_QLE_SA.m:163: cost = min_cost;
            cost = min_cost
            # MATLAB Optimized_QLE_SA.m:164: 
            # MATLAB Optimized_QLE_SA.m:165: if cost < best_cost
            if np.real(cost) < np.real(best_cost):
                # MATLAB Optimized_QLE_SA.m:166: best_x = x;
                best_x = x.copy()
                # MATLAB Optimized_QLE_SA.m:167: best_cost = cost;
                best_cost = cost
            # MATLAB Optimized_QLE_SA.m:168: end
        # MATLAB Optimized_QLE_SA.m:169: end
        # MATLAB Optimized_QLE_SA.m:170: 
        # MATLAB Optimized_QLE_SA.m:171: T = T * alpha;
        T = T * alpha
    # MATLAB Optimized_QLE_SA.m:172: end
    # MATLAB Optimized_QLE_SA.m:173: 
    # MATLAB Optimized_QLE_SA.m:174: best_params = best_x;
    best_params = best_x.copy()
    # MATLAB Optimized_QLE_SA.m:175: fprintf("Best entropy after annealing: %.5f\n", best_cost);
    print("Best entropy after annealing: %.5f" % best_cost)
    return best_params
# MATLAB Optimized_QLE_SA.m:176: end
# MATLAB Optimized_QLE_SA.m:177: 
# MATLAB Optimized_QLE_SA.m:178: function cost = entropy_cost(x, weights, H_list)
def entropy_cost(x, weights, H_list):
    # MATLAB Optimized_QLE_SA.m:179: t = x(1); theta = x(2); phi = x(3); alpha = x(4); beta = x(5);
    t = x[0, 0]; theta = x[0, 1]; phi = x[0, 2]; alpha = x[0, 3]; beta = x[0, 4]
    # MATLAB Optimized_QLE_SA.m:180: N = length(H_list);
    N = len(H_list)
    # MATLAB Optimized_QLE_SA.m:181: psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];
    psi0 = prepare_state(alpha, beta)
    # MATLAB Optimized_QLE_SA.m:182: rho_FY = zeros(2*N);
    rho_FY = np.zeros((2 * N, 2 * N), dtype=complex)  # MATLAB promotes on complex assignment.
    # MATLAB Optimized_QLE_SA.m:183: 
    # MATLAB Optimized_QLE_SA.m:184: for j = 1:N
    for j in range(1, N + 1):
        # MATLAB Optimized_QLE_SA.m:185: psi_j = expm(-1i * H_list{j} * t) * psi0;
        psi_j = evolve(H_list[j - 1], t, psi0)
        # MATLAB Optimized_QLE_SA.m:186: rho_j = psi_j * psi_j';
        rho_j = psi_j @ psi_j.conj().T
        # MATLAB Optimized_QLE_SA.m:187: rho_FY(2*j-1:2*j, 2*j-1:2*j) = weights(j) * rho_j;
        rho_FY[2 * j - 2:2 * j, 2 * j - 2:2 * j] = weights[0, j - 1] * rho_j
    # MATLAB Optimized_QLE_SA.m:188: end
    # MATLAB Optimized_QLE_SA.m:189: 
    # MATLAB Optimized_QLE_SA.m:190: u1 = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
    # MATLAB Optimized_QLE_SA.m:191: u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
    # MATLAB Optimized_QLE_SA.m:192: W = [u1, u2];
    u1, u2, W = measurement_basis(theta, phi)
    # MATLAB Optimized_QLE_SA.m:193: 
    # MATLAB Optimized_QLE_SA.m:194: A_plus  = W' * [1, 0; 0, 0] * W;
    # MATLAB Optimized_QLE_SA.m:195: A_minus = eye(2) - A_plus;
    A_plus, A_minus = projectors(W)
    # MATLAB Optimized_QLE_SA.m:196: 
    # MATLAB Optimized_QLE_SA.m:197: rho_plus  = kron(eye(N), A_plus) * rho_FY * kron(eye(N), A_plus);
    rho_plus = np.kron(np.eye(N), A_plus) @ rho_FY @ np.kron(np.eye(N), A_plus)
    # MATLAB Optimized_QLE_SA.m:198: rho_minus = kron(eye(N), A_minus) * rho_FY * kron(eye(N), A_minus);
    rho_minus = np.kron(np.eye(N), A_minus) @ rho_FY @ np.kron(np.eye(N), A_minus)
    # MATLAB Optimized_QLE_SA.m:199: 
    # MATLAB Optimized_QLE_SA.m:200: prob_plus  = real(trace(rho_plus));
    prob_plus = np.real(np.trace(rho_plus))
    # MATLAB Optimized_QLE_SA.m:201: prob_minus = real(trace(rho_minus));
    prob_minus = np.real(np.trace(rho_minus))
    # MATLAB Optimized_QLE_SA.m:202: 
    # MATLAB Optimized_QLE_SA.m:203: S_plus = 0; S_minus = 0;
    S_plus = 0; S_minus = 0
    # MATLAB Optimized_QLE_SA.m:204: if prob_plus > 0
    if np.real(prob_plus) > 0:
        # MATLAB Optimized_QLE_SA.m:205: rho_plus = rho_plus / prob_plus;
        rho_plus = rho_plus / prob_plus
        # MATLAB Optimized_QLE_SA.m:206: S_plus = Entropy(rho_plus);
        S_plus = Entropy(rho_plus)
    # MATLAB Optimized_QLE_SA.m:207: end
    # MATLAB Optimized_QLE_SA.m:208: if prob_minus > 0
    if np.real(prob_minus) > 0:
        # MATLAB Optimized_QLE_SA.m:209: rho_minus = rho_minus / prob_minus;
        rho_minus = rho_minus / prob_minus
        # MATLAB Optimized_QLE_SA.m:210: S_minus = Entropy(rho_minus);
        S_minus = Entropy(rho_minus)
    # MATLAB Optimized_QLE_SA.m:211: end
    # MATLAB Optimized_QLE_SA.m:212: 
    # MATLAB Optimized_QLE_SA.m:213: cost = prob_plus * S_plus + prob_minus * S_minus;
    cost = prob_plus * S_plus + prob_minus * S_minus
    return cost
# MATLAB Optimized_QLE_SA.m:214: end
# MATLAB Optimized_QLE_SA.m:215: 
# MATLAB Optimized_QLE_SA.m:216: function S = Entropy(rho)
def Entropy(rho):
    # MATLAB Optimized_QLE_SA.m:217: rho = (rho + rho') / 2;
    rho = (rho + rho.conj().T) / 2
    # MATLAB Optimized_QLE_SA.m:218: eigvals = real(eig(rho));
    eigvals = np.real(matlab_eig(rho))
    # MATLAB Optimized_QLE_SA.m:219: eigvals = eigvals(eigvals > 1e-12);
    eigvals = eigvals[eigvals > 1e-12].reshape(-1, 1)
    # MATLAB Optimized_QLE_SA.m:220: S = -sum(eigvals .* log2(eigvals));
    S = -np.sum(eigvals * np.log2(eigvals))
    return S
# MATLAB Optimized_QLE_SA.m:221: end
# MATLAB Optimized_QLE_SA.m:222: 
# MATLAB Optimized_QLE_SA.m:223-229: function outcome = sample_from_distribution(p) -> qle/measurement.py
# MATLAB Optimized_QLE_SA.m:230: 
# MATLAB Optimized_QLE_SA.m:231-242: function delta = Min_Energy_Gap(list) -> qle/utils.py


if __name__ == "__main__":
    cli(run)
