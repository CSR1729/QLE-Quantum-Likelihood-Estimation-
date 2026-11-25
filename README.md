# Quantum Likelihood Estimation (QLE) in Python

A Python version of the MATLAB code accompanying

> A. Levi, Z. Ossi, E. Cohen and A. Te'eni, *Optimal Quantum Likelihood Estimation*,
> Quantum Sci. Technol. **11**, 015029 (2026). [doi:10.1088/2058-9565/ae2b31](https://www.doi.org/10.1088/2058-9565/ae2b31)
> · MATLAB code: [smitke6/Optimal-Quantum-Likelihood-Estimation](https://github.com/smitke6/Optimal-Quantum-Likelihood-Estimation)

It computes the same thing as the MATLAB scripts, statement for statement. The
only difference is the organisation: parts that were copied into all three
MATLAB files (state preparation, evolution, measurement, Hamiltonians) now live
in one shared file each.

| Where | What it explains |
|---|---|
| this file | the problem, one measurement round, how to run, how to read the output |
| [`qle/README.md`](qle/README.md) | the physics in each shared file |
| [`qle/algorithms/README.md`](qle/algorithms/README.md) | how each of the three algorithms chooses its experiments |

---

## The problem

A qubit sits in a black box and evolves under a Hamiltonian **H** that you cannot
see. You are told H is one of N known candidates, for example

    { σx, 2σx, σz, 2σz }

and you must find out which one. The only thing you can do is run experiments.
Each experiment gives you a single bit, so the question is: **which experiments
should you run to identify H with as few shots as possible?**

This is *Hamiltonian learning* reduced to a finite set of hypotheses.

## One measurement round

Every algorithm here repeats the same five steps:

```
 1. choose settings      t, θ, φ, α, β            ← the only step the algorithms do differently
 2. prepare the probe    |ψ₀⟩ = cos α |0⟩ + e^{iβ} sin α |1⟩
 3. evolve               |ψ(t)⟩ = e^{−iHt} |ψ₀⟩
 4. measure              in the basis {|u₁⟩, |u₂⟩} set by (θ, φ)  →  one bit y ∈ {0, 1}
 5. update beliefs       wⱼ ← wⱼ · P(y | Hⱼ) / Σₖ wₖ P(y | Hₖ)      (Bayes' rule)
```

- **The weights.** The vector **w** = (w₁ … w_N) is your current belief: wⱼ is the
  probability that Hⱼ is the true Hamiltonian. It starts uniform, wⱼ = 1/N.
- **The likelihood.** P(y | Hⱼ) is the probability that candidate j would have
  produced the bit you saw. You can compute it exactly, because you know Hⱼ:
  pⱼ = |⟨u₁| e^{−iHⱼt} |ψ₀⟩|² is the chance of outcome 0, and 1 − pⱼ the chance
  of outcome 1.
- **Stopping.** Rounds repeat until one weight passes a threshold (0.99 by default).

In this simulation, step 4 is also simulated. The code evolves the state under a
chosen "true" candidate and flips a biased coin with probability p_true. On real
hardware, that coin flip would be an actual measurement.

## Why the choice of settings matters

A bit only helps if the candidates *disagree* about what it will be. If every
candidate predicts outcome 0 with the same probability, the shot tells you
nothing. The quantity that measures how much a shot helps is the
**mutual information**

    I(F;Y) = H(F) − H(F|Y)

- **F** is the unknown candidate.
- **Y** is the measurement outcome.
- **H(F)** is your current uncertainty: the Shannon entropy of the weights.
- **H(F|Y)** is the uncertainty you expect to have left after seeing the bit.

The three algorithms differ only in how they pick (t, θ, φ, α, β):

| Algorithm | How it picks settings | Typical shots (default set, 0.99) |
|---|---|---|
| `qle_original` | θ, φ, α, β fixed once; t from a heuristic (PGH) | ≈ 18–50 |
| `qle_sa` | simulated annealing, minimising H(F\|Y) every round | ≈ 1–3 |
| `qle_grid` | exhaustive grid search, minimising H(F\|Y) every round | ≈ 2–3 |

Picking the most informative experiment each round is the main idea of the
paper. Here it cuts the number of shots by roughly a factor of 10.

---

## Install and run

```bash
pip install -r requirements.txt          # numpy, scipy

python -m qle.algorithms.qle_original --seed 1
python -m qle.algorithms.qle_sa       --seed 1
python -m qle.algorithms.qle_grid     --seed 1     # slow: ~85 s per round
```

Run these from the folder that contains `qle/`. All three take the same options:

| Option | Meaning | Default |
|---|---|---|
| `--seed N` | makes the run repeatable | random |
| `--threshold X` | stop when a weight reaches X (keep it below 1) | 0.99 |
| `--candidate-set six` | use the paper's second set {H1 … H6} instead of {σx, 2σx, σz, 2σz} | default |

`--seed` seeds NumPy's random generator, not MATLAB's, so a single run won't
match a MATLAB run number for number. Averages over many runs do match the
paper's published results.

From Python:

```python
from qle.algorithms import qle_sa
result = qle_sa.run(seed=1)
result["convergence_iters"], result["success_flags"]
```

## Reading the output

Each script tries every candidate in turn as the "true" one and reports how
many shots it needed to identify it.

- **`qle_original`** prints `results:` followed by one number per candidate.
  Each number is **shots + 1**, so `13` means 12 measurements. It also prints
  the weights and two matrices every round. To see only the final answer:
  `python -m qle.algorithms.qle_original --seed 1 | tail -6`.
- **`qle_sa` and `qle_grid`** print, for each round:
  - `ideal t`: the evolution time chosen.
  - `H(Y)`, `H(Y|F)`, `I(FY)`: the information this round's measurement was
    expected to give (explained in [`qle/algorithms/README.md`](qle/algorithms/README.md)).
  - `weights`: the beliefs after the update.

  At the end they print:
  - `Convergence iterations`: the number of shots per candidate. `NaN` means
    it didn't reach the threshold within 10 shots.
  - `Did each H converge to correct hypothesis?`: 1 = yes, 0 = no.
  - `All correct: true/false`.

## Folder layout

```
qle/
  hamiltonians.py         the candidate Hamiltonians
  state_preparation.py    step 2: the probe state
  evolution.py            step 3: e^{-iHt}
  measurement.py          step 4: basis, outcome probabilities, coin flip, likelihood
  utils.py                an unused helper kept from the MATLAB code
  matlab_compat.py        makes Python behave like MATLAB where it matters
  algorithms/
    qle_original.py       original QLE
    qle_sa.py             optimised QLE, simulated annealing
    qle_grid.py           optimised QLE, grid search
requirements.txt
```
