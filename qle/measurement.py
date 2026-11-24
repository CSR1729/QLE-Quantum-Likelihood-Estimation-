"""Measurement: basis, projectors, outcome probabilities, sampling and likelihood.

Line references: Q = QLE.m, G = Optimized_QLE_Grid.m, S = Optimized_QLE_SA.m.

Two probability functions are kept on purpose. They are mathematically equal
(the first row of W is u1'), but they are different floating-point paths, and
each script keeps the one it originally used:
    prob_overlap         |u1' * psi|^2                       (QLE.m)
    rotated_probability  |(W*psi / norm(W*psi))(1)|^2        (both optimized scripts)
"""
import numpy as np


def measurement_basis(theta, phi):
    """Measurement vectors u1, u2 and W = [u1, u2] for angles (theta, phi)."""
    # MATLAB Q:39 G:88,204 S:69,190: u1=[cos(theta/2); exp(1i*phi)*sin(theta/2)];
    u1 = np.array([[np.cos(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2)]])
    # MATLAB Q:40 G:89,205 S:70,191: u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
    u2 = np.array([[np.exp(-1j * phi) * np.sin(theta / 2)], [-np.cos(theta / 2)]])
    # MATLAB Q:41 G:90,206 S:71,192: W=[u1,u2];
    W = np.column_stack((u1, u2))
    return u1, u2, W


def projectors(W):
    """A_plus = W' * |0><0| * W and A_minus = I - A_plus."""
    # MATLAB G:208 S:194: A_plus  = W' * [1,0;0,0] * W;
    A_plus = W.conj().T @ np.array([[1, 0], [0, 0]], dtype=float) @ W
    # MATLAB G:209 S:195: A_minus = eye(2) - A_plus;
    A_minus = np.eye(2) - A_plus
    return A_plus, A_minus


def prob_overlap(u1, psi):
    """QLE.m form: p = abs(u1' * psi)^2.   MATLAB Q:88,98"""
    return np.abs((u1.conj().T @ psi)[0, 0]) ** 2


def rotated_probability(W, psi):
    """Optimized-script form: rotate by W, renormalise, take |first amplitude|^2.

    Returns (rotated state, probability); Optimized_QLE_Grid.m reuses the
    rotated state for psi_all and rho_Y, Optimized_QLE_SA.m for rho_Y.
    MATLAB G:91-93,103-104,106  S:73-75,84-86
    """
    psi_rot = W @ psi
    psi_rot = psi_rot / np.linalg.norm(psi_rot)
    p = np.abs(psi_rot[0, 0]) ** 2
    return psi_rot, p


def clip_probability(p):
    """MATLAB G:107 S:87: pj = min(max(real(pj), 1e-12), 1 - 1e-12);"""
    return np.fmin(np.fmax(np.real(p), 1e-12), 1 - 1e-12)


def sample_from_distribution(p, rng):
    """Outcome 0 with probability p(1), else 1.

    MATLAB Q:129-131 G:148-150 S:223-229:
    if rand() < p(1), outcome = 0; else, outcome = 1; end
    """
    outcome = 0 if rng.rand() < np.asarray(p).ravel(order="F")[0] else 1
    return outcome


def likelihood(outcome, pj):
    """MATLAB Q:99 G:108 S:88: likelihoods(j) = (outcome==0)*pj + (outcome==1)*(1-pj);"""
    return (outcome == 0) * pj + (outcome == 1) * (1 - pj)
