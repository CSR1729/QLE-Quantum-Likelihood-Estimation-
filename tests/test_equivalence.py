"""The refactored package must behave exactly like the original scripts.

For the same seed, each refactored algorithm must return identical arrays,
print byte-identical output and raise the same warnings as its file in
reference/. Shared helpers are also compared one-to-one with the reference
helpers on random inputs.

Run:            python -m pytest -q
Include Grid:   QLE_SLOW=1 python -m pytest -q   (full Grid runs take ~15 min each)
"""
import contextlib
import importlib.util
import io
import os
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference")
sys.path.insert(0, ROOT)


def _load_reference(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(REF, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The reference scripts import a top-level "matlab_compat"; load reference/'s copy under that name.
_load_reference("matlab_compat", "matlab_compat.py")
ref_qle = _load_reference("ref_QLE", "QLE.py")
ref_grid = _load_reference("ref_Optimized_QLE_Grid", "Optimized_QLE_Grid.py")
ref_sa = _load_reference("ref_Optimized_QLE_SA", "Optimized_QLE_SA.py")

from qle import hamiltonians, measurement, state_preparation, utils  # noqa: E402
from qle.algorithms import qle_grid, qle_original, qle_sa  # noqa: E402
from qle.matlab_compat import RandomStream  # noqa: E402


def _capture(run, **kwargs):
    out = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
        warnings.simplefilter("always")
        result = run(**kwargs)
    return result, out.getvalue(), [str(w.message) for w in caught]


class RunEquivalence(unittest.TestCase):
    def assert_same_run(self, ref_run, new_run, **kwargs):
        ref_result, ref_out, ref_warn = _capture(ref_run, **kwargs)
        new_result, new_out, new_warn = _capture(new_run, **kwargs)
        self.assertEqual(ref_result.keys(), new_result.keys())
        for key in ref_result:
            np.testing.assert_array_equal(ref_result[key], new_result[key], err_msg=f"{key} {kwargs}")
        self.assertEqual(ref_out, new_out, f"printed output differs for {kwargs}")
        self.assertEqual(ref_warn, new_warn)

    def test_qle_original(self):
        for seed in (0, 1, 2):
            with self.subTest(candidate_set="default", seed=seed):
                self.assert_same_run(ref_qle.run, qle_original.run, seed=seed)

    def test_qle_original_six(self):
        # Several candidates hit the 10,000-round cap here, so one seed takes a few minutes.
        self.assert_same_run(ref_qle.run, qle_original.run, seed=0, candidate_set="six")

    def test_qle_original_threshold(self):
        self.assert_same_run(ref_qle.run, qle_original.run, seed=3, threshold=0.9)

    def test_qle_sa(self):
        for seed in (0, 1, 2):
            with self.subTest(candidate_set="default", seed=seed):
                self.assert_same_run(ref_sa.run, qle_sa.run, seed=seed)
        self.assert_same_run(ref_sa.run, qle_sa.run, seed=0, candidate_set="six")
        self.assert_same_run(ref_sa.run, qle_sa.run, seed=4, threshold=0.95)

    @unittest.skipUnless(os.environ.get("QLE_SLOW"), "set QLE_SLOW=1 to run the full Grid comparison")
    def test_qle_grid(self):
        self.assert_same_run(ref_grid.run, qle_grid.run, seed=1)


class HelperEquivalence(unittest.TestCase):
    """Fast checks of every moved helper against its reference definition."""

    def setUp(self):
        self.gen = np.random.default_rng(12345)

    def test_hamiltonian_lists(self):
        for name, mine in (("default", hamiltonians.default_list()),
                           ("six_original", hamiltonians.six_list_original()),
                           ("six_optimized", hamiltonians.six_list_optimized())):
            with self.subTest(name):
                self.assertTrue(all(m.dtype in (float, complex) for m in mine))
        sX, sY, sZ = hamiltonians.sigX, hamiltonians.sigY, hamiltonians.sigZ
        self.assertTrue(np.array_equal(hamiltonians.H4, 4 * (1 / np.sqrt(2)) * np.array([[1., 1], [1, -1]])))
        self.assertTrue(np.array_equal(hamiltonians.six_list_original()[1], hamiltonians.H2))
        self.assertTrue(np.array_equal(hamiltonians.six_list_optimized()[1], hamiltonians.H4))
        self.assertTrue(np.array_equal(hamiltonians.H12, 0.6 * sZ + 0.4 * sY))
        self.assertTrue(np.array_equal(hamiltonians.default_list()[1], 2 * sX))

    def test_random_hermitian_draws(self):
        a, b = RandomStream(7), RandomStream(7)
        A = a.randn(2) + 1j * a.randn(2)
        np.testing.assert_array_equal(hamiltonians.random_hermitian(b), (A + A.conj().T) / 2)
        self.assertEqual(a.rand(), b.rand())  # stream left at the same position

    def test_state_and_measurement(self):
        for _ in range(200):
            alpha, beta, theta, phi = self.gen.uniform(0, 2 * np.pi, 4)
            psi = np.array([[np.cos(alpha)], [np.exp(1j * beta) * np.sin(alpha)]])
            np.testing.assert_array_equal(state_preparation.prepare_state(alpha, beta), psi)
            u1 = np.array([[np.cos(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2)]])
            u2 = np.array([[np.exp(-1j * phi) * np.sin(theta / 2)], [-np.cos(theta / 2)]])
            W = np.column_stack((u1, u2))
            m1, m2, mW = measurement.measurement_basis(theta, phi)
            for x, y in ((m1, u1), (m2, u2), (mW, W)):
                np.testing.assert_array_equal(x, y)
            A_plus = W.conj().T @ np.array([[1, 0], [0, 0]], dtype=float) @ W
            mp, mm = measurement.projectors(W)
            np.testing.assert_array_equal(mp, A_plus)
            np.testing.assert_array_equal(mm, np.eye(2) - A_plus)
            self.assertEqual(measurement.prob_overlap(u1, psi), np.abs((u1.conj().T @ psi)[0, 0]) ** 2)
            r = W @ psi
            r = r / np.linalg.norm(r)
            mr, mpj = measurement.rotated_probability(W, psi)
            np.testing.assert_array_equal(mr, r)
            self.assertEqual(mpj, np.abs(r[0, 0]) ** 2)

    def test_sampling_likelihood_gap(self):
        for p in (0.0, 0.3, 0.5, 0.999, 1.0):
            a, b = RandomStream(11), RandomStream(11)
            for _ in range(50):
                self.assertEqual(measurement.sample_from_distribution(np.array([[p, 1 - p]]), a),
                                 ref_sa.sample_from_distribution(np.array([[p, 1 - p]]), b))
        for outcome in (0, 1):
            for pj in (1e-12, 0.25, 1 - 1e-12):
                self.assertEqual(measurement.likelihood(outcome, pj),
                                 (outcome == 0) * pj + (outcome == 1) * (1 - pj))
        for p in (-1.0, 0.0, 1e-13, 0.4, 1.0, 2.0):
            self.assertEqual(measurement.clip_probability(p), np.fmin(np.fmax(np.real(p), 1e-12), 1 - 1e-12))
        lists = [hamiltonians.default_list(), hamiltonians.six_list_original()]
        for lst in lists:
            np.testing.assert_array_equal(utils.Min_Energy_Gap(lst), ref_qle.Min_Energy_Gap(lst))

    def test_grid_helpers_unchanged(self):
        for _ in range(50):
            A = self.gen.normal(size=(8, 8)) + 1j * self.gen.normal(size=(8, 8))
            rho = A @ A.conj().T
            rho = rho / np.trace(rho)
            self.assertEqual(qle_grid.Entropy(rho), ref_grid.Entropy(rho))
            self.assertEqual(qle_sa.Entropy(rho), ref_sa.Entropy(rho))
            w = self.gen.dirichlet(np.ones(4)).reshape(1, 4)
            self.assertEqual(qle_grid.PGH(hamiltonians.default_list(), w), ref_grid.PGH(ref_lists_default(), w))
            x = self.gen.uniform(0, np.pi, (1, 5))
            self.assertEqual(qle_sa.entropy_cost(x, w, hamiltonians.default_list()),
                             ref_sa.entropy_cost(x, w, ref_lists_default()))


def ref_lists_default():
    sX = np.array([[0, 1], [1, 0]], dtype=float)
    sZ = np.array([[1, 0], [0, -1]], dtype=float)
    return [sX, 2 * sX, sZ, 2 * sZ]


if __name__ == "__main__":
    unittest.main()
