# Quantum Likelihood Estimation (QLE)

Python implementation of the original and optimised QLE algorithms (MATLAB
originals: `QLE.m`, `Optimized_QLE_Grid.m`, `Optimized_QLE_SA.m`; see the paper
"Optimal Quantum Likelihood Estimation"). The code is split into shared physics
building blocks and three algorithm files. No algorithmic logic was added or
changed: `tests/test_equivalence.py` checks that every algorithm gives the same
results and printed output as the original line-by-line port in `reference/`.

## Problem statement

A qubit evolves under an unknown Hamiltonian H. You know H is one of N
candidates (by default {σX, 2σX, σZ, 2σZ}). The goal is to identify which one,
with posterior confidence of at least 0.99, using as few single-shot
measurements as possible.

Each measurement round:

1. **Prepare** |ψ₀⟩ = cos α |0⟩ + e^{iβ} sin α |1⟩ (`state_preparation.py`)
2. **Evolve** for time t: e^{−iHt} |ψ₀⟩ (`evolution.py`)
3. **Measure** in the basis {u₁, u₂} set by (θ, φ), giving one bit (`measurement.py`)
4. **Update** the weights by Bayes' rule: wⱼ ← wⱼ · P(outcome | Hⱼ), then renormalise
5. **Repeat** until max(w) ≥ threshold

The three algorithms differ only in how they choose (t, θ, φ, α, β) each round:

| Module | Choice of settings | Round cap |
|---|---|---|
| `qle.algorithms.qle_original` | θ, φ, α, β fixed; t = 1/‖Hᵢ−Hⱼ‖_F for two candidates sampled by weight (PGH) | 10,000 |
| `qle.algorithms.qle_sa` | simulated annealing over all five, minimising H(F\|Y) | 10 |
| `qle.algorithms.qle_grid` | exhaustive grid (≈415k points) over all five, minimising H(F\|Y) | 10 |

H(F|Y) is the expected remaining uncertainty about the hypothesis after the
measurement, so minimising it maximises the information I(F;Y) = H(F) − H(F|Y)
gained per shot.

## Layout

```
qle/
  matlab_compat.py      MATLAB adapters (RNG stream, disp, eig, find, ...), unchanged
  hamiltonians.py       Pauli matrices, H1–H13, candidate lists
  state_preparation.py  prepare_state(alpha, beta)
  evolution.py          evolve(H, t, psi) = expm(-iHt) psi
  measurement.py        measurement_basis, projectors, prob_overlap, rotated_probability,
                        clip_probability, sample_from_distribution, likelihood
  utils.py              Min_Energy_Gap (defined in all three originals, never called)
  algorithms/
    qle_original.py     Bayesian loop + PGH + discrete_sample
    qle_grid.py         loop + grid optimize_entropy + Entropy (+ unused sorted PGH)
    qle_sa.py           loop + annealing optimize_entropy + entropy_cost + Entropy
reference/              the original single-file ports, unmodified
tests/test_equivalence.py
```

Every moved line keeps its `# MATLAB <file>.m:<line>` comment, so each line of
Python can still be traced back to the MATLAB source.

### What is shared and what is not, and why

- **Shared:** state preparation, evolution, measurement basis, projectors,
  outcome sampling, likelihood, candidate Hamiltonians, `Min_Energy_Gap`. These
  were the same expression in every file.
- **Two probability functions, kept separate on purpose.** `prob_overlap`
  computes |u₁†ψ|² (original QLE). `rotated_probability` computes
  |(Wψ/‖Wψ‖)₁|² (both optimised versions). They are mathematically equal but
  are different floating-point paths, and merging them could change sampled
  outcomes for a given seed.
- **Normalisation stays at the call sites** (`psi0 / norm(psi0)` and
  `psi_f / norm(psi_f)` in the optimised scripts), exactly where the originals
  had it.
- **Not shared:** each algorithm's `Entropy`. The Grid version warns on
  non-Hermitian input and computes eigenvalues with eigenvectors; the SA version
  computes eigenvalues only. Also not shared: the two cost functions (they treat
  zero-probability outcomes differently) and the two `PGH` functions (different
  rules; Grid's is never called).
- **Candidate order:** the six-Hamiltonian list is ordered (H1, H2, H3, H4, H5,
  H6) in the original QLE but (H1, H4, H3, H2, H5, H6) in both optimised
  versions. Both orders are kept (`six_list_original`, `six_list_optimized`), so
  column k does not refer to the same matrix across algorithms.
- **H13 in Grid** is unused but draws 8 random numbers before anything else;
  `random_hermitian(rng)` is called at the same point so seeded runs are
  unchanged.

## Usage

```bash
pip install -r requirements.txt

python -m qle.algorithms.qle_original --seed 1
python -m qle.algorithms.qle_sa       --seed 1 --candidate-set six
python -m qle.algorithms.qle_grid     --seed 1 --threshold 0.95
```

All three take the same options:

- `--seed N`: seeds a NumPy PCG64 stream. This is not MATLAB's generator, so
  numbers won't match a MATLAB run with the same seed.
- `--threshold X`: confidence needed to stop (default 0.99). Keep it below 1.
- `--candidate-set default|six`: the default list, or the paper's second list
  of six.

From Python:

```python
from qle.algorithms import qle_original, qle_sa
qle_original.run(seed=1)["iter_arr"]
qle_sa.run(seed=1)["convergence_iters"], qle_sa.run(seed=1)["success_flags"]
```

`qle_original` prints two 8×8 matrices on every round; pipe through `| tail -8`
if you only want the final results.

## Reading the results

- **`qle_original`:** `iter_arr` holds the number of measurements **+ 1** for
  each true H.
- **Optimised versions:** the printed "Convergence iterations" already has 1
  subtracted; the returned `convergence_iters` does not. `NaN` means the
  threshold was not reached within 11 rounds.
- **`success_flags`:** set only once max(w) ≥ 0.95, and true if the top-weighted
  candidate is the true one.
- **Printed entropies (optimised versions):** `H(Y)`, `H(Y|F)` and `I(FY)` are
  the outcome entropy, the average outcome entropy given each hypothesis, and
  their difference. "Best entropy after annealing" (SA) is H(F|Y) for the chosen
  settings.

### Known behaviour of the original QLE on the six-set

H4 = 2√2·H1, and the fixed measurement angles (θ = π/4, φ ≈ 2π) lie along the
rotation axis of both. The outcome probability is then 0.389 for every t, so H1
and H4 can never be told apart and those runs stop at the 10,000-round cap.
The optimised versions choose the basis every round and do not have this
problem.

## Tests

```bash
python -m pytest -q                  # QLE + SA full runs, all shared helpers (~10 min)
QLE_SLOW=1 python -m pytest -q       # also compares a full Grid run (~30 min more)
```

The Grid search takes about 85 s per measurement round. Moving code into
functions adds a few percent to that.
