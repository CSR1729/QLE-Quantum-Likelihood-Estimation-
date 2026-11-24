"""Small, explicit MATLAB language adapters; no QLE algorithm lives here."""
import argparse
import sys

import numpy as np
from scipy import linalg


class RandomStream:
    """One seedable PCG64 stream, with MATLAB array shapes and fill order.

    This does not reproduce MATLAB's random number sequence for a given seed.
    """

    def __init__(self, seed=None):
        self.generator = np.random.Generator(np.random.PCG64(seed))

    @staticmethod
    def _shape(shape):
        return (shape[0], shape[0]) if len(shape) == 1 else shape

    def rand(self, *shape):
        if not shape:
            return self.generator.random()
        shape = self._shape(shape)
        return self.generator.random(int(np.prod(shape))).reshape(shape, order="F")

    def randn(self, *shape):
        if not shape:
            return self.generator.standard_normal()
        shape = self._shape(shape)
        return self.generator.standard_normal(int(np.prod(shape))).reshape(shape, order="F")


def matlab_eig(a, vectors=False):
    """MATLAB eig dispatches Hermitian inputs to a Hermitian eigensolver.

    Keep eigenvectors even when discarded at the MATLAB call site. General
    (potentially non-Hermitian) user matrices still use the general solver.
    """
    if np.array_equal(a, a.conj().T):
        if vectors:
            values, v = linalg.eigh(a, driver="ev", check_finite=True)
            return v, np.diag(values)
        return linalg.eigvalsh(a, driver="ev", check_finite=True).reshape(-1, 1)
    if vectors:
        values, v = linalg.eig(a)
        return v, np.diag(values)
    return linalg.eigvals(a).reshape(-1, 1)


def matlab_min(a):
    """min([]) is empty, not an invented fallback energy gap."""
    return np.fmin.reduce(a.ravel(order="F")) if a.size else a.copy()


def first_min(a):
    """Real-valued min: omit NaNs and choose the first equal minimum."""
    flat = a.ravel(order="F")
    value = np.fmin.reduce(flat)
    index = 1 if np.isnan(value) else int(np.flatnonzero(flat == value)[0]) + 1
    return value, index


def descending_indices(a):
    """Stable descending sort of real weights, with missing values first."""
    flat = a.ravel(order="F")
    indices = np.lexsort((np.arange(flat.size), -flat, ~np.isnan(flat)))
    return (indices + 1).reshape(a.shape, order="F")


def first_find(condition):
    """One-based find(...,1,'first'); preserve the empty result."""
    found = np.flatnonzero(np.asarray(condition).ravel(order="F"))
    return int(found[0]) + 1 if found.size else np.empty((0, 0))


def clc():
    """Clear an interactive terminal; redirected output is an intact transcript."""
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")


def _real_text(value, integer, scale):
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    if value == 0:
        return "0"
    return f"{value:.0f}" if integer else f"{value / scale:.4f}"


def format_disp(value):
    """Short, loose MATLAB-style output, at a fixed 80-column display width.

    MATLAB Command Window wrapping is environment-dependent; this fixes it
    explicitly for reproducible transcripts. No array elements are truncated.
    """
    if isinstance(value, str):
        return value + "\n"
    a = np.asarray(value)
    if not a.size:
        return ""
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError("QLE display supports scalars and matrices only")
    if a.dtype.kind == "b":
        return "".join("   " + "   ".join(str(int(v)) for v in row) + "\n" for row in a) + "\n"
    complex_output = np.iscomplexobj(a) and np.any(a.imag != 0)
    components = np.concatenate((a.real.ravel(), a.imag.ravel())) if complex_output else a.real.ravel()
    finite = components[np.isfinite(components)]
    integer = not complex_output and bool(np.all(finite == np.trunc(finite)))
    nonzero = np.abs(finite[finite != 0])
    scale = 1.0
    if not integer and nonzero.size:
        largest = float(np.max(nonzero))
        if largest >= 1e3 or largest < 1e-3:
            scale = 10.0 ** np.floor(np.log10(largest))
    rows = []
    for row in a:
        formatted = []
        for v in row:
            re = _real_text(v.real, integer, scale)
            if complex_output:
                im = _real_text(abs(v.imag), False, scale)
                sign = "-" if np.signbit(v.imag) else "+"
                re = f"{re} {sign} {im}i"
            formatted.append(re)
        rows.append(formatted)
    width = max(len(v) for row in rows for v in row)
    width = max(width, 7 if not integer else 1)
    per_block = max(1, 80 // (width + 3))
    result = ""
    if scale != 1:
        result += f"   {scale:.1e} *\n\n"
    for start in range(0, a.shape[1], per_block):
        stop = min(start + per_block, a.shape[1])
        if a.shape[1] > per_block:
            result += (f"  Column {start + 1}\n\n" if stop == start + 1
                       else f"  Columns {start + 1} through {stop}\n\n")
        result += "".join("   " + "   ".join(v.rjust(width) for v in row[start:stop]) + "\n" for row in rows)
        result += "\n"
    return result


def disp(value):
    sys.stdout.write(format_disp(value))


def cli(run):
    parser = argparse.ArgumentParser(description=run.__module__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--candidate-set", choices=("default", "six"), default="default")
    args = parser.parse_args()
    run(seed=args.seed, threshold=args.threshold, candidate_set=args.candidate_set)
