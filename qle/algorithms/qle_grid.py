"""Optimized QLE, grid search (Optimized_QLE_Grid.m).

Each round, (t, theta, phi, alpha, beta) is chosen by exhaustive grid search to
minimise H(F|Y). Refactored from reference/Optimized_QLE_Grid.py: state
preparation, evolution, measurement and the Hamiltonians now come from shared
modules; every MATLAB source line is still listed below, at the call that
implements it. The loop, optimize_entropy, Entropy and PGH are unchanged.
tests/test_equivalence.py checks that output is identical to the reference
for the same seed.

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
from ..utils import Min_Energy_Gap  # noqa: F401  (defined in Optimized_QLE_Grid.m, unused there)


def run(*, seed=None, threshold=None, candidate_set="default"):
    """Execute the original script and return its original result arrays."""
    if candidate_set not in ("default", "six"):
        raise ValueError("candidate_set must be 'default' or 'six'")
    rng = RandomStream(seed)
    _threshold = threshold
    # MATLAB Optimized_QLE_Grid.m:1: %% Background
    # MATLAB Optimized_QLE_Grid.m:2: % This script implements the optimized version of QLE developed in the
    # MATLAB Optimized_QLE_Grid.m:3: % article "Optimal Quantum Likelihood Estimation". This version finds the
    # MATLAB Optimized_QLE_Grid.m:4: % optimal parameters for each iteration by searching over a grid.
    # MATLAB Optimized_QLE_Grid.m:5: % The configurable pieces are denoted by comments starting with CONFIGURE.
    # MATLAB Optimized_QLE_Grid.m:6: 
    # MATLAB Optimized_QLE_Grid.m:7: clc;
    clc()
    # MATLAB Optimized_QLE_Grid.m:8: clearvars;
    pass  # Fresh run() locals implement clearvars; the RNG is not reset.
    # MATLAB Optimized_QLE_Grid.m:9: 
    # MATLAB Optimized_QLE_Grid.m:10: %% Setup
    # MATLAB Optimized_QLE_Grid.m:11-30: Pauli matrices and H1-H12 -> qle/hamiltonians.py
    # MATLAB Optimized_QLE_Grid.m:31: A = randn(2) + 1i * randn(2);
    # MATLAB Optimized_QLE_Grid.m:32: H13 = (A + A') / 2;
    H13 = random_hermitian(rng)  # unused, but consumes the same random draws at the same point
    # MATLAB Optimized_QLE_Grid.m:33:
    # MATLAB Optimized_QLE_Grid.m:34: %CONFIGURE: The user may compose any list of candidate Hamiltonians out of
    # MATLAB Optimized_QLE_Grid.m:35: %the 2*2 matrices above, or use their own matrices. The paper studies the
    # MATLAB Optimized_QLE_Grid.m:36: %following two lists.
    # MATLAB Optimized_QLE_Grid.m:37: %H_list = {H1, H4, H3, H2, H5, H6};
    # MATLAB Optimized_QLE_Grid.m:38: H_list = {sigX, sigX2, sigZ, sigZ2};
    H_list = default_list()
    # Explicit CLI configuration of the source's commented alternative.
    if candidate_set == "six":
        H_list = six_list_optimized()
    # MATLAB Optimized_QLE_Grid.m:39: 
    # MATLAB Optimized_QLE_Grid.m:40: N = length(H_list);
    N = len(H_list)
    # MATLAB Optimized_QLE_Grid.m:41: 
    # MATLAB Optimized_QLE_Grid.m:42: convergence_iters = NaN(1, N);
    convergence_iters = np.full((1, N), np.nan)
    # MATLAB Optimized_QLE_Grid.m:43: success_flags = false(1, N);
    success_flags = np.zeros((1, N), dtype=bool)
    # MATLAB Optimized_QLE_Grid.m:44: 
    # MATLAB Optimized_QLE_Grid.m:45: %% Algorithm
    # MATLAB Optimized_QLE_Grid.m:46: % CONFIGURE: The user may choose the convergence threshold
    # MATLAB Optimized_QLE_Grid.m:47: threshold = 0.99;
    threshold = 0.99
    # An omitted override leaves the original literal unchanged.
    if _threshold is not None:
        threshold = _threshold
    # MATLAB Optimized_QLE_Grid.m:48: 
    # MATLAB Optimized_QLE_Grid.m:49: for idx = 1:N
    for idx in range(1, N + 1):
        # MATLAB Optimized_QLE_Grid.m:50: disp('**********************')
        disp('**********************')
        # MATLAB Optimized_QLE_Grid.m:51: disp('the right H is:'); disp(idx);
        disp('the right H is:'); disp(idx)
        # MATLAB Optimized_QLE_Grid.m:52: H_true = H_list{idx};
        H_true = H_list[idx - 1]
        # MATLAB Optimized_QLE_Grid.m:53: weights = ones(1, N) / N;
        weights = np.ones((1, N)) / N
        # MATLAB Optimized_QLE_Grid.m:54: num_iterations = 10;
        num_iterations = 10
        # MATLAB Optimized_QLE_Grid.m:55: 
        # MATLAB Optimized_QLE_Grid.m:56: iter = 1;
        iter = 1
        # MATLAB Optimized_QLE_Grid.m:57: while iter <= num_iterations+1
        while iter <= num_iterations + 1:
            # MATLAB Optimized_QLE_Grid.m:58: if max(weights) >= threshold
            if np.fmax.reduce(weights.ravel(order="F")) >= threshold:
                # MATLAB Optimized_QLE_Grid.m:59: convergence_iters(idx) = iter;
                convergence_iters[0, idx - 1] = iter
                # MATLAB Optimized_QLE_Grid.m:60: break;
                break
            # MATLAB Optimized_QLE_Grid.m:61: end
            # MATLAB Optimized_QLE_Grid.m:62: 
            # MATLAB Optimized_QLE_Grid.m:63: %t = PGH(H_list, weights);
            # MATLAB Optimized_QLE_Grid.m:64: best_params = optimize_entropy(weights,H_list);
            best_params = optimize_entropy(weights, H_list)
            # MATLAB Optimized_QLE_Grid.m:65: t     = best_params(1);
            t = best_params[0, 0]
            # MATLAB Optimized_QLE_Grid.m:66: theta = best_params(2);
            theta = best_params[0, 1]
            # MATLAB Optimized_QLE_Grid.m:67: phi   = best_params(3);
            phi = best_params[0, 2]
            # MATLAB Optimized_QLE_Grid.m:68: alpha = best_params(4);
            alpha = best_params[0, 3]
            # MATLAB Optimized_QLE_Grid.m:69: beta  = best_params(5);
            beta = best_params[0, 4]
            # MATLAB Optimized_QLE_Grid.m:70: disp("ideal t")
            disp("ideal t")
            # MATLAB Optimized_QLE_Grid.m:71: disp(t);
            disp(t)
            # MATLAB Optimized_QLE_Grid.m:72: 
            # MATLAB Optimized_QLE_Grid.m:73: psi0   = [cos(alpha); exp(1i*beta)*sin(alpha)];
            psi0 = prepare_state(alpha, beta)
            # MATLAB Optimized_QLE_Grid.m:74: psi0 = psi0 / norm(psi0);
            psi0 = psi0 / np.linalg.norm(psi0)
            # MATLAB Optimized_QLE_Grid.m:75: 
            # MATLAB Optimized_QLE_Grid.m:76: psi_all = repmat(psi0, 1, N);
            psi_all = np.tile(psi0, (1, N))
            # MATLAB Optimized_QLE_Grid.m:77: rho_tilda_F = diag(weights);
            rho_tilda_F = np.diag(weights.ravel(order="F"))
            # MATLAB Optimized_QLE_Grid.m:78: rho_total = zeros(2*N);
            rho_total = np.zeros((2 * N, 2 * N))
            # MATLAB Optimized_QLE_Grid.m:79: for j = 1:N
            for j in range(1, N + 1):
                # MATLAB Optimized_QLE_Grid.m:80: proj = psi_all(:,j) * psi_all(:,j)';
                proj = psi_all[:, j - 1:j] @ psi_all[:, j - 1:j].conj().T
                # MATLAB Optimized_QLE_Grid.m:81: block = kron(proj, rho_tilda_F);
                block = np.kron(proj, rho_tilda_F)
                # MATLAB Optimized_QLE_Grid.m:82: rho_total = rho_total + block;
                rho_total = rho_total + block
            # MATLAB Optimized_QLE_Grid.m:83: end
            # MATLAB Optimized_QLE_Grid.m:84: 
            # MATLAB Optimized_QLE_Grid.m:85: U_true = expm(-1i * H_true * t);
            # MATLAB Optimized_QLE_Grid.m:86: psi_f  = U_true * psi0;
            psi_f = evolve(H_true, t, psi0)
            # MATLAB Optimized_QLE_Grid.m:87: psi_f = psi_f / norm(psi_f);
            psi_f = psi_f / np.linalg.norm(psi_f)
            # MATLAB Optimized_QLE_Grid.m:88: u1=[cos(theta/2); exp(1i*phi)*sin(theta/2)];
            # MATLAB Optimized_QLE_Grid.m:89: u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
            # MATLAB Optimized_QLE_Grid.m:90: W=[u1,u2];
            u1, u2, W = measurement_basis(theta, phi)
            # MATLAB Optimized_QLE_Grid.m:91: psi_final=W*psi_f;
            # MATLAB Optimized_QLE_Grid.m:92: psi_final = psi_final / norm(psi_final);
            # MATLAB Optimized_QLE_Grid.m:93: p_u1   = abs(psi_final(1))^2;
            psi_final, p_u1 = rotated_probability(W, psi_f)
            # MATLAB Optimized_QLE_Grid.m:94: outcome = sample_from_distribution([p_u1, 1 - p_u1]);
            outcome = sample_from_distribution(np.array([[p_u1, 1 - p_u1]]), rng)
            # MATLAB Optimized_QLE_Grid.m:95: 
            # MATLAB Optimized_QLE_Grid.m:96: likelihoods = zeros(1, N);
            likelihoods = np.zeros((1, N))
            # MATLAB Optimized_QLE_Grid.m:97: psi_all = zeros(2, N);
            psi_all = np.zeros((2, N), dtype=complex)  # MATLAB promotes on complex assignment.
            # MATLAB Optimized_QLE_Grid.m:98: rho_Y=zeros(2,2);
            rho_Y = np.zeros((2, 2))
            # MATLAB Optimized_QLE_Grid.m:99: H_YF=0;
            H_YF = 0
            # MATLAB Optimized_QLE_Grid.m:100: for j = 1:N
            for j in range(1, N + 1):
                # MATLAB Optimized_QLE_Grid.m:101: U_j = expm(-1i * H_list{j} * t);
                # MATLAB Optimized_QLE_Grid.m:102: psi_j = U_j * psi0;
                psi_j = evolve(H_list[j - 1], t, psi0)
                # MATLAB Optimized_QLE_Grid.m:103: psi_j=W*psi_j;
                # MATLAB Optimized_QLE_Grid.m:104: psi_j = psi_j / norm(psi_j);
                # MATLAB Optimized_QLE_Grid.m:106: pj = abs(psi_j(1))^2;
                psi_j, pj = rotated_probability(W, psi_j)
                # MATLAB Optimized_QLE_Grid.m:105: psi_all(:,j) = psi_j;
                psi_all[:, j - 1:j] = psi_j
                # MATLAB Optimized_QLE_Grid.m:107: pj = min(max(real(pj), 1e-12), 1 - 1e-12);
                pj = clip_probability(pj)
                # MATLAB Optimized_QLE_Grid.m:108: likelihoods(j) = (outcome==0)*pj + (outcome==1)*(1-pj);
                likelihoods[0, j - 1] = likelihood(outcome, pj)
                # MATLAB Optimized_QLE_Grid.m:109: rho_Y=rho_Y+weights(j)*psi_j*psi_j';
                rho_Y = rho_Y + (weights[0, j - 1] * psi_j) @ psi_j.conj().T
                # MATLAB Optimized_QLE_Grid.m:110: H_YF=H_YF+weights(j)*(-pj*log2(pj)-(1-pj)*log2(1-pj));
                H_YF = H_YF + weights[0, j - 1] * (-pj * np.log2(pj) - (1 - pj) * np.log2(1 - pj))
            # MATLAB Optimized_QLE_Grid.m:111: end
            # MATLAB Optimized_QLE_Grid.m:112: 
            # MATLAB Optimized_QLE_Grid.m:113: ent_y=Entropy(diag(diag(rho_Y)));
            ent_y = Entropy(np.diag(np.diag(rho_Y)))
            # MATLAB Optimized_QLE_Grid.m:114: disp(['Iteration ', num2str(iter), ':']);
            disp(f'Iteration {iter}:')
            # MATLAB Optimized_QLE_Grid.m:115: disp('H(Y)='); disp(ent_y);
            disp('H(Y)='); disp(ent_y)
            # MATLAB Optimized_QLE_Grid.m:116: disp('H(Y|F)='); disp(H_YF);
            disp('H(Y|F)='); disp(H_YF)
            # MATLAB Optimized_QLE_Grid.m:117: disp('I(FY)='); disp(ent_y-H_YF);
            disp('I(FY)='); disp(ent_y - H_YF)
            # MATLAB Optimized_QLE_Grid.m:118: 
            # MATLAB Optimized_QLE_Grid.m:119: weights = weights .* likelihoods;
            weights = weights * likelihoods
            # MATLAB Optimized_QLE_Grid.m:120: weights(weights < 0) = 0;
            weights[np.real(weights) < 0] = 0
            # MATLAB Optimized_QLE_Grid.m:121: weights = weights / sum(weights);
            weights = weights / np.sum(weights)
            # MATLAB Optimized_QLE_Grid.m:122: 
            # MATLAB Optimized_QLE_Grid.m:123: if max(weights) >= 0.95
            if np.fmax.reduce(weights.ravel(order="F")) >= 0.95:
                # MATLAB Optimized_QLE_Grid.m:124: [~, guess_H] = max(weights);
                _, guess_H = first_min(-weights)
                # MATLAB Optimized_QLE_Grid.m:125: success_flags(idx) = (guess_H == idx);
                success_flags[0, idx - 1] = guess_H == idx
            # MATLAB Optimized_QLE_Grid.m:126: end
            # MATLAB Optimized_QLE_Grid.m:127: 
            # MATLAB Optimized_QLE_Grid.m:128: disp('weights ='); disp(weights);
            disp('weights ='); disp(weights)
            # MATLAB Optimized_QLE_Grid.m:129: iter = iter + 1;
            iter = iter + 1
        # MATLAB Optimized_QLE_Grid.m:130: end
    # MATLAB Optimized_QLE_Grid.m:131: end
    # MATLAB Optimized_QLE_Grid.m:132: 
    # MATLAB Optimized_QLE_Grid.m:133: disp('Convergence iterations:');
    disp('Convergence iterations:')
    # MATLAB Optimized_QLE_Grid.m:134: disp(convergence_iters-1);
    disp(convergence_iters - 1)
    # MATLAB Optimized_QLE_Grid.m:135: disp('Did each H converge to correct hypothesis?');
    disp('Did each H converge to correct hypothesis?')
    # MATLAB Optimized_QLE_Grid.m:136: disp(success_flags);
    disp(success_flags)
    # MATLAB Optimized_QLE_Grid.m:137: disp(['All correct: ', mat2str(all(success_flags))]);
    disp('All correct: ' + ('true' if np.all(success_flags) else 'false'))
    # MATLAB Optimized_QLE_Grid.m:138: 
    # MATLAB Optimized_QLE_Grid.m:139: %% Helper functions
    return {"convergence_iters": convergence_iters, "success_flags": success_flags}


# MATLAB Optimized_QLE_Grid.m:140: function t = PGH(H_list, weights)
def PGH(H_list, weights):
    # MATLAB Optimized_QLE_Grid.m:141: [~, I] = sort(weights, 'descend');
    I = descending_indices(weights)
    # MATLAB Optimized_QLE_Grid.m:142: i1 = I(1); i2 = I(2);
    i1 = I[0, 0]; i2 = I[0, 1]
    # MATLAB Optimized_QLE_Grid.m:143: deltaH = H_list{i1} - H_list{i2};
    deltaH = H_list[i1 - 1] - H_list[i2 - 1]
    # MATLAB Optimized_QLE_Grid.m:144: normH = norm(deltaH, 'fro');
    normH = np.linalg.norm(deltaH, "fro")
    # MATLAB Optimized_QLE_Grid.m:145: t = pi / (2 * normH);
    t = np.pi / (2 * normH)
    return t
# MATLAB Optimized_QLE_Grid.m:146: end
# MATLAB Optimized_QLE_Grid.m:147: 
# MATLAB Optimized_QLE_Grid.m:148-150: function outcome = sample_from_distribution(p) -> qle/measurement.py
# MATLAB Optimized_QLE_Grid.m:151: 
# MATLAB Optimized_QLE_Grid.m:152: %Entropy - Von Neumann entropy of a quantum state
# MATLAB Optimized_QLE_Grid.m:153: % S = Entropy(rho) returns the von Neumann entropy of the density matrix
# MATLAB Optimized_QLE_Grid.m:154: % rho.
# MATLAB Optimized_QLE_Grid.m:155: function S = Entropy(rho)
def Entropy(rho):
    # MATLAB Optimized_QLE_Grid.m:156: % Force Hermitian: symmetric average
    # MATLAB Optimized_QLE_Grid.m:157: rho = (rho + rho') / 2;
    rho = (rho + rho.conj().T) / 2
    # MATLAB Optimized_QLE_Grid.m:158: 
    # MATLAB Optimized_QLE_Grid.m:159: % Check if it's still significantly non-Hermitian
    # MATLAB Optimized_QLE_Grid.m:160: if norm(rho - rho', 'fro') > 1e-10
    if np.linalg.norm(rho - rho.conj().T, "fro") > 1e-10:
        # MATLAB Optimized_QLE_Grid.m:161: warning('Entropy: input not Hermitian even after symmetrization.');
        warnings.warn('Entropy: input not Hermitian even after symmetrization.', RuntimeWarning)
    # MATLAB Optimized_QLE_Grid.m:162: end
    # MATLAB Optimized_QLE_Grid.m:163: 
    # MATLAB Optimized_QLE_Grid.m:164: [~, D] = eig(rho);
    _, D = matlab_eig(rho, vectors=True)
    # MATLAB Optimized_QLE_Grid.m:165: eigvals = real(diag(D));
    eigvals = np.real(np.diag(D)).reshape(-1, 1)
    # MATLAB Optimized_QLE_Grid.m:166: eigvals = eigvals(eigvals > 1e-12); % filter near-zero
    eigvals = eigvals[eigvals > 1e-12].reshape(-1, 1)
    # MATLAB Optimized_QLE_Grid.m:167: 
    # MATLAB Optimized_QLE_Grid.m:168: S = -sum(eigvals .* log2(eigvals));
    S = -np.sum(eigvals * np.log2(eigvals))
    return S
# MATLAB Optimized_QLE_Grid.m:169: end
# MATLAB Optimized_QLE_Grid.m:170: 
# MATLAB Optimized_QLE_Grid.m:171: %optimize_entropy - Find the best t and parameters minimizing H(Y|F)
# MATLAB Optimized_QLE_Grid.m:172: %   INPUT:  weights - 1x4 array of weights for the four hypotheses
# MATLAB Optimized_QLE_Grid.m:173: %   OUTPUT: best_params - [t, theta, phi, alpha, beta]
# MATLAB Optimized_QLE_Grid.m:174: function best_params = optimize_entropy(weights, H_list)
def optimize_entropy(weights, H_list):
    # MATLAB Optimized_QLE_Grid.m:175: N=length(H_list);
    N = len(H_list)
    # MATLAB Optimized_QLE_Grid.m:176: 
    # MATLAB Optimized_QLE_Grid.m:177: t_vals = linspace(0.1, 2*pi, 10);
    t_vals = np.linspace(0.1, 2 * np.pi, 10)
    # MATLAB Optimized_QLE_Grid.m:178: sample = 24;
    sample = 24
    # MATLAB Optimized_QLE_Grid.m:179: theta_vals = linspace(0, pi/2, sample/4);
    theta_vals = np.linspace(0, np.pi / 2, int(sample / 4))
    # MATLAB Optimized_QLE_Grid.m:180: phi_vals   = linspace(0, 2*pi, sample);
    phi_vals = np.linspace(0, 2 * np.pi, sample)
    # MATLAB Optimized_QLE_Grid.m:181: alpha_vals = linspace(0, pi, sample/2);
    alpha_vals = np.linspace(0, np.pi, int(sample / 2))
    # MATLAB Optimized_QLE_Grid.m:182: beta_vals  = linspace(0, 2*pi, sample);
    beta_vals = np.linspace(0, 2 * np.pi, sample)
    # MATLAB Optimized_QLE_Grid.m:183: 
    # MATLAB Optimized_QLE_Grid.m:184: min_entropy = Inf;
    min_entropy = np.inf
    # MATLAB Optimized_QLE_Grid.m:185: best_params = [];
    best_params = np.empty((0, 0))
    # MATLAB Optimized_QLE_Grid.m:186: IN = eye(N);
    IN = np.eye(N)
    # MATLAB Optimized_QLE_Grid.m:187: 
    # MATLAB Optimized_QLE_Grid.m:188: for t = t_vals
    for t in t_vals:
        # MATLAB Optimized_QLE_Grid.m:189: for alpha = alpha_vals
        for alpha in alpha_vals:
            # MATLAB Optimized_QLE_Grid.m:190: for beta = beta_vals
            for beta in beta_vals:
                # MATLAB Optimized_QLE_Grid.m:191: psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];
                psi0 = prepare_state(alpha, beta)
                # MATLAB Optimized_QLE_Grid.m:192: 
                # MATLAB Optimized_QLE_Grid.m:193: % Evolve under all Hamiltonians
                # MATLAB Optimized_QLE_Grid.m:194: rho_FY = zeros(2*N, 2*N);
                rho_FY = np.zeros((2 * N, 2 * N), dtype=complex)  # MATLAB promotes on complex assignment.
                # MATLAB Optimized_QLE_Grid.m:195: for j = 1:N
                for j in range(1, N + 1):
                    # MATLAB Optimized_QLE_Grid.m:196: H = H_list{j};
                    H = H_list[j - 1]
                    # MATLAB Optimized_QLE_Grid.m:197: psi_j = expm(-1i * H * t) * psi0;
                    psi_j = evolve(H, t, psi0)
                    # MATLAB Optimized_QLE_Grid.m:198: rho_j = psi_j * psi_j';
                    rho_j = psi_j @ psi_j.conj().T
                    # MATLAB Optimized_QLE_Grid.m:199: rho_FY(2*j-1:2*j, 2*j-1:2*j) = weights(j) * rho_j;
                    rho_FY[2 * j - 2:2 * j, 2 * j - 2:2 * j] = weights[0, j - 1] * rho_j
                # MATLAB Optimized_QLE_Grid.m:200: end
                # MATLAB Optimized_QLE_Grid.m:201: 
                # MATLAB Optimized_QLE_Grid.m:202: for theta = theta_vals
                for theta in theta_vals:
                    # MATLAB Optimized_QLE_Grid.m:203: for phi = phi_vals
                    for phi in phi_vals:
                        # MATLAB Optimized_QLE_Grid.m:204: u1 = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
                        # MATLAB Optimized_QLE_Grid.m:205: u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
                        # MATLAB Optimized_QLE_Grid.m:206: W = [u1, u2];
                        u1, u2, W = measurement_basis(theta, phi)
                        # MATLAB Optimized_QLE_Grid.m:207:
                        # MATLAB Optimized_QLE_Grid.m:208: A_plus  = W' * [1,0;0,0] * W;
                        # MATLAB Optimized_QLE_Grid.m:209: A_minus = eye(2) - A_plus;
                        A_plus, A_minus = projectors(W)
                        # MATLAB Optimized_QLE_Grid.m:210: 
                        # MATLAB Optimized_QLE_Grid.m:211: rho_plus = kron(IN, A_plus) * rho_FY * ...
                        # MATLAB Optimized_QLE_Grid.m:212: kron(IN, A_plus);
                        rho_plus = np.kron(IN, A_plus) @ rho_FY @ np.kron(IN, A_plus)
                        # MATLAB Optimized_QLE_Grid.m:213: rho_minus = kron(IN, A_minus) * rho_FY * ...
                        # MATLAB Optimized_QLE_Grid.m:214: kron(IN, A_minus);
                        rho_minus = np.kron(IN, A_minus) @ rho_FY @ np.kron(IN, A_minus)
                        # MATLAB Optimized_QLE_Grid.m:215: 
                        # MATLAB Optimized_QLE_Grid.m:216: prob_plus = trace(rho_plus);
                        prob_plus = np.trace(rho_plus)
                        # MATLAB Optimized_QLE_Grid.m:217: prob_minus = trace(rho_minus);
                        prob_minus = np.trace(rho_minus)
                        # MATLAB Optimized_QLE_Grid.m:218: 
                        # MATLAB Optimized_QLE_Grid.m:219: if prob_plus > 0
                        if np.real(prob_plus) > 0:
                            # MATLAB Optimized_QLE_Grid.m:220: rho_plus = rho_plus / prob_plus;
                            rho_plus = rho_plus / prob_plus
                        # MATLAB Optimized_QLE_Grid.m:221: else
                        else:
                            # MATLAB Optimized_QLE_Grid.m:222: continue;
                            continue
                        # MATLAB Optimized_QLE_Grid.m:223: end
                        # MATLAB Optimized_QLE_Grid.m:224: 
                        # MATLAB Optimized_QLE_Grid.m:225: if prob_minus > 0
                        if np.real(prob_minus) > 0:
                            # MATLAB Optimized_QLE_Grid.m:226: rho_minus = rho_minus / prob_minus;
                            rho_minus = rho_minus / prob_minus
                        # MATLAB Optimized_QLE_Grid.m:227: else
                        else:
                            # MATLAB Optimized_QLE_Grid.m:228: continue;
                            continue
                        # MATLAB Optimized_QLE_Grid.m:229: end
                        # MATLAB Optimized_QLE_Grid.m:230: 
                        # MATLAB Optimized_QLE_Grid.m:231: ent = prob_plus * Entropy(rho_plus) + ...
                        # MATLAB Optimized_QLE_Grid.m:232: prob_minus * Entropy(rho_minus);
                        ent = prob_plus * Entropy(rho_plus) + prob_minus * Entropy(rho_minus)
                        # MATLAB Optimized_QLE_Grid.m:233: 
                        # MATLAB Optimized_QLE_Grid.m:234: if ent < min_entropy
                        if np.real(ent) < np.real(min_entropy):
                            # MATLAB Optimized_QLE_Grid.m:235: min_entropy = ent;
                            min_entropy = ent
                            # MATLAB Optimized_QLE_Grid.m:236: best_params = [t, theta, phi, alpha, beta];
                            best_params = np.array([[t, theta, phi, alpha, beta]])
                        # MATLAB Optimized_QLE_Grid.m:237: end
                    # MATLAB Optimized_QLE_Grid.m:238: end
                # MATLAB Optimized_QLE_Grid.m:239: end
            # MATLAB Optimized_QLE_Grid.m:240: end
        # MATLAB Optimized_QLE_Grid.m:241: end
    # MATLAB Optimized_QLE_Grid.m:242: end
    return best_params
# MATLAB Optimized_QLE_Grid.m:243: end
# MATLAB Optimized_QLE_Grid.m:244: 
# MATLAB Optimized_QLE_Grid.m:245-256: function delta = Min_Energy_Gap(list) -> qle/utils.py


if __name__ == "__main__":
    cli(run)
