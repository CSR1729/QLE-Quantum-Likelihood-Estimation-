"""Single-qubit projective measurement. See qle/README.md."""
import numpy as np


def measurement_basis(theta, phi):
    """Basis vectors u1, u2 (Bloch angles theta, phi) and the gate W = [u1, u2]."""
    u1 = np.array([[np.cos(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2)]])
    u2 = np.array([[np.exp(-1j * phi) * np.sin(theta / 2)], [-np.cos(theta / 2)]])
    W = np.column_stack((u1, u2))
    return u1, u2, W


def projectors(W):
    """Projectors onto the two outcomes: A_plus = W' |0><0| W and A_minus = I - A_plus."""
    A_plus = W.conj().T @ np.array([[1, 0], [0, 0]], dtype=float) @ W
    A_minus = np.eye(2) - A_plus
    return A_plus, A_minus


def prob_overlap(u1, psi):
    """P(outcome 0) = |<u1|psi>|^2. Used by the original QLE."""
    return np.abs((u1.conj().T @ psi)[0, 0]) ** 2


def rotated_probability(W, psi):
    """P(outcome 0) computed as |first amplitude of W|psi>|^2, after renormalising.
    Same value as prob_overlap; used by the optimised versions. Also returns W|psi>."""
    psi_rot = W @ psi
    psi_rot = psi_rot / np.linalg.norm(psi_rot)
    p = np.abs(psi_rot[0, 0]) ** 2
    return psi_rot, p


def clip_probability(p):
    """Keep p inside [1e-12, 1 - 1e-12] so log2(p) and log2(1 - p) stay finite."""
    return np.fmin(np.fmax(np.real(p), 1e-12), 1 - 1e-12)


def sample_from_distribution(p, rng):
    """Simulate one shot: outcome 0 with probability p(1), otherwise 1."""
    outcome = 0 if rng.rand() < np.asarray(p).ravel(order="F")[0] else 1
    return outcome


def likelihood(outcome, pj):
    """P(observed outcome | candidate j), where pj is that candidate's P(outcome 0)."""
    return (outcome == 0) * pj + (outcome == 1) * (1 - pj)
