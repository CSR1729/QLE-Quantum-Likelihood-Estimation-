# The shared files: the physics of one round

These files hold the parts that every algorithm uses. They follow the order of a
measurement round: candidates → probe state → evolution → measurement.

All states are single-qubit pure states, written as 2×1 column vectors in the
basis {|0⟩, |1⟩}, with ħ = 1.

---

## `hamiltonians.py`: the candidates

**What it is:** the list of Hamiltonians the true one is drawn from.

Any 2×2 Hermitian matrix can be written as

    H = a₀ I + a·σ = a₀ I + aₓ σx + a_y σy + a_z σz

where σx, σy, σz are the Pauli matrices. The identity part a₀ only adds a global
phase, which no measurement can see. The vector **a** is what matters:
- its **direction** is the axis the qubit's Bloch vector rotates around;
- its **length** |a| sets the rotation speed (angle 2|a|t after time t).

**The default set** {σx, 2σx, σz, 2σz} gives two rotation axes (x and z), each at
two speeds. To identify the true H, you must tell the axes apart *and* the
speeds apart.

**The six-candidate set** H1…H6 is the harder example studied in the paper:

| | Matrix | Rotation axis and speed |
|---|---|---|
| H1 | σx + σz | axis (1, 0, 1)/√2, \|a\| = √2 |
| H2 | diag(1.2, −0.8) | z axis, \|a\| = 1 (plus 0.2·I) |
| H3 | 2σy + diag(1, 2) | mostly y, tilted toward −z |
| H4 | 2√2 (σx + σz) | same axis as H1, 2√2 ≈ 2.8× faster (\|a\| = 4) |
| H5 | σx + H4 | axis tilted a little from H4's toward x |
| H6 | 2σx + σz | between x and z, closer to x |

H1 and H4 share an axis and differ only in speed, which makes this set hard.

**Orders:** `six_list_original()` gives H1…H6, the order used by the original
QLE. `six_list_optimized()` gives H1, H4, H3, H2, H5, H6, the order used by both
optimised scripts. The order only changes which column of the output belongs to
which matrix.

**Unused extras:** H7–H12 and `random_hermitian` (H13) are examples from the
MATLAB grid script and aren't in any list. The grid script still calls
`random_hermitian` because it draws random numbers, and removing it would shift
every later random draw.

---

## `state_preparation.py`: the probe state

```
|ψ₀⟩ = cos α |0⟩ + e^{iβ} sin α |1⟩
```

This covers every pure qubit state. On the Bloch sphere it is the point with
polar angle 2α and azimuth β:
- α = 0 is |0⟩ (north pole);
- α = π/2 is |1⟩ (south pole);
- β sets the longitude.

**Why the probe matters:** evolution rotates the Bloch vector around each
candidate's axis. A probe lying *on* a candidate's axis doesn't move at all, so
it carries no information about that candidate's speed. A good probe sits where
the candidates' rotations pull it in different directions.

---

## `evolution.py`: the black box

```
|ψ(t)⟩ = e^{−iHt} |ψ₀⟩
```

This is the Schrödinger equation solved for a time-independent H. `scipy.linalg.expm`
computes the matrix exponential exactly; for a qubit, it is a rotation of the
Bloch vector by angle 2|a|t around the axis of **a**.

**What the evolution time t controls:** how far apart two candidates' states end
up.
- If t is too short, nothing has moved yet and every candidate looks the same.
- If t is too long, phases wrap around (rotations are periodic), and two
  different candidates can line up again.

The best t depends on how different the candidates you still believe in are.
That is why the original QLE picks t from the difference between two candidates
(see the algorithms README).

---

## `measurement.py`: reading out one bit

A projective measurement asks: "is the qubit in state |u₁⟩, or in the
orthogonal state |u₂⟩?"

```
|u₁⟩ = cos(θ/2) |0⟩ + e^{iφ} sin(θ/2) |1⟩
|u₂⟩ = e^{−iφ} sin(θ/2) |0⟩ − cos(θ/2) |1⟩
```

- (θ, φ) is the measurement axis on the Bloch sphere.
- Outcome 0 means "found in |u₁⟩", outcome 1 means "found in |u₂⟩".
- **Born rule:** P(0) = |⟨u₁|ψ⟩|², and P(1) = 1 − P(0).
- On the Bloch sphere, P(0) = (1 + **r**·**n**)/2, where **r** is the state's
  Bloch vector and **n** the measurement axis.

The functions in this file:

| Function | What it does |
|---|---|
| `measurement_basis(θ, φ)` | Builds u₁, u₂ and the gate W = [u₁ u₂]. W rotates the measurement axis onto z, so measuring W\|ψ⟩ in {\|0⟩, \|1⟩} is the same as measuring \|ψ⟩ in {u₁, u₂}. |
| `projectors(W)` | A₊ = \|u₁⟩⟨u₁\| and A₋ = I − A₊, the two outcome projectors. The optimisers use them to work out the state after each possible outcome. |
| `prob_overlap(u₁, ψ)` | P(0) = \|⟨u₁\|ψ⟩\|². Used by the original QLE. |
| `rotated_probability(W, ψ)` | The same P(0), computed as \|(W\|ψ⟩)₁\|². Used by the optimised versions. The two are mathematically equal; each script keeps the form its MATLAB original used. |
| `clip_probability(p)` | Keeps p inside [10⁻¹², 1 − 10⁻¹²], so log₂ p stays finite and no weight can be set to exactly zero by one unlucky shot. |
| `sample_from_distribution(p, rng)` | The simulated experiment: a biased coin with P(0) = p. |
| `likelihood(y, pⱼ)` | P(y \| Hⱼ): pⱼ if y = 0, 1 − pⱼ if y = 1. This is the factor Bayes' rule multiplies each weight by. |

**Why the basis matters:** a measurement only helps if candidates give different
P(0). If the measurement axis lines up with a candidate's rotation axis, that
candidate's P(0) stays the same for every t, and so does that of any other
candidate sharing the axis. The original QLE's fixed basis has exactly this
problem with H1 and H4: P(0) = 0.389 for both, at every t, so they can never be
told apart.

---

## `utils.py`

`Min_Energy_Gap(list)` returns the smallest non-zero gap |E₂ − E₁| between the
two energy levels among a list of Hamiltonians. The gap sets how fast a
Hamiltonian evolves (period 2π/gap). It is defined in all three MATLAB scripts
but never called; it's kept here for completeness.

---

## `matlab_compat.py`: not physics

This file makes Python behave like MATLAB in the few places where they differ
and the difference would change results or output:

| Piece | Why it's needed |
|---|---|
| `RandomStream` | One seedable random generator for the whole run. It fills arrays in MATLAB's (column) order. It is not MATLAB's generator, so seeded numbers differ from MATLAB. |
| `matlab_eig` | Returns eigenvalues the way MATLAB's `eig` does (ascending, using the Hermitian solver for Hermitian input). |
| `matlab_min`, `first_min`, `descending_indices`, `first_find` | MATLAB's `min`, `max`, `sort` and `find`: ignore NaN, choose the first index on ties, count from 1. |
| `disp`, `clc` | MATLAB-style printing and screen clearing. |
| `cli` | The `--seed`, `--threshold` and `--candidate-set` command-line options. |

This is also why loops in the algorithms run `for j in range(1, N + 1)` and
index with `[j - 1]`: the code counts from 1 like MATLAB, so it can be checked
line by line against the original.
