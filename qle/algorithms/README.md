# The three algorithms

All three run the same loop (see the top-level README):

```
for each candidate H_true in the list:            # pretend it's the hidden one
    w = uniform
    repeat until max(w) ≥ threshold (or the round cap is hit):
        choose settings  t, θ, φ, α, β             ← this is where they differ
        prepare |ψ₀⟩, evolve under H_true, measure → bit y
        for each candidate j: pⱼ = P(0 | Hⱼ), wⱼ ← wⱼ · P(y | Hⱼ)
        normalise w
    record how many rounds it took
```

So the only question each algorithm answers is: **what experiment do I run next?**

---

## `qle_original.py`: original QLE

**Strategy:** keep the probe and the measurement basis fixed forever, and adapt
only the evolution time t.

### Fixed settings

```
θ = 0.7854 (π/4),  φ = 6.2832 (2π)    →  measurement axis (1, 0, 1)/√2 on the Bloch sphere
α = 1.2566 (2π/5), β = 0.5712         →  fixed probe state
```

The MATLAB comment says these values were found beforehand by a grid search with
resolution π/12. For the default set {σx, 2σx, σz, 2σz}, the measurement axis
sits halfway between x and z, so it is sensitive to rotations around both axes.

### Evolution time: the particle guess heuristic (PGH)

The PGH comes from adaptive Hamiltonian learning (Wiebe et al., 2014). Each round:

1. Draw two different candidates i and j at random, each with probability equal
   to its current weight.
2. Set **t = 1 / ‖Hᵢ − Hⱼ‖_F**, the inverse Frobenius norm of their difference.
   If the two are numerically identical, it uses t = 0.1.

The idea: two Hamiltonians whose difference has size Δ have dephased by about
one radian after a time 1/Δ. That is roughly when their measurement statistics
first differ clearly, and before they wrap around and look alike again. As the
weights concentrate on the last few rivals, the pairs drawn are mostly those
rivals, so t adapts to separate exactly them.

### Output

- **Every round** it prints `rho_tilda_F` = diag(w), the classical register
  holding the current beliefs, and `rho_total` = Σⱼ |ψ₀⟩⟨ψ₀| ⊗ diag(w), the
  joint probe-and-register state before evolution. Both are for display only;
  nothing uses them.
- **At the end** it prints `results:`, the number of rounds per candidate **+ 1**.
- **Round cap:** 10,000.

### Limitation

The fixed basis is tuned for one candidate set. On the six-candidate set, the
measurement axis (1, 0, 1)/√2 is exactly the rotation axis of H1 and H4. A
rotation never changes the component of the Bloch vector along its own axis, so
both give P(0) = 0.389 at every t. No number of shots can tell them apart, and
those runs stop at the 10,000-round cap. This is the motivation for choosing the
settings anew every round.

---

## The optimised versions: choosing the most informative experiment

Both `qle_sa.py` and `qle_grid.py` choose **all five** settings (t, θ, φ, α, β)
anew every round. They pick the settings that minimise

```
H(F|Y)  =  Σ_y  P(y) · H(F | Y = y)
```

the uncertainty about the candidate F that you **expect** to have left after
seeing the bit Y. Your current uncertainty H(F) doesn't depend on the settings,
so this is the same as maximising the information the shot gives:
I(F;Y) = H(F) − H(F|Y).

### How H(F|Y) is computed (`entropy_cost` in SA, inside `optimize_entropy` in Grid)

1. **Build the joint state** of a classical "which candidate" register F and the
   evolved probe:

   ```
   ρ_FY = Σⱼ wⱼ |j⟩⟨j| ⊗ |ψⱼ⟩⟨ψⱼ|,      |ψⱼ⟩ = e^{−iHⱼt} |ψ₀⟩
   ```

   It is block-diagonal: block j is wⱼ |ψⱼ⟩⟨ψⱼ|, a 2×2 block.

2. **Project onto each outcome** with A₊ = |u₁⟩⟨u₁| and A₋ = I − A₊ (applied
   to the qubit, block by block):

   ```
   ρ₊ = (I ⊗ A₊) ρ_FY (I ⊗ A₊),     P(+) = Tr ρ₊
   ```

3. **Normalise and take the von Neumann entropy.** After normalising, ρ₊'s
   eigenvalues are exactly wⱼ P(+|Hⱼ) / P(+), which is the Bayesian posterior
   after seeing "+". So S(ρ₊ / P(+)) is the Shannon entropy of your beliefs if
   the outcome is +.

4. **Average over outcomes:** cost = P(+) S(ρ₊) + P(−) S(ρ₋) = H(F|Y).

In other words, the cost function is "how uncertain will I be after this shot,
on average", written in the language of quantum states.

### What gets printed each round

| Line | Meaning |
|---|---|
| `ideal t` | the chosen evolution time |
| `H(Y)=` | Shannon entropy of the predicted outcome, averaged over candidates: how unpredictable the bit is |
| `H(Y\|F)=` | Σⱼ wⱼ h(pⱼ), the average randomness of the bit *given* the candidate (h is the binary entropy) |
| `I(FY)=` | H(Y) − H(Y\|F), the information this shot carries about F |
| `weights =` | the beliefs after the Bayes update |

Example (SA, default set, true H = σx, first round):

```
Best entropy after annealing: 1.18872     ← H(F|Y) for the chosen settings
H(Y|F) ≈ 0                                 ← every candidate predicts the bit almost surely
I(FY)  = 0.8113                            ← = H(F) − H(F|Y) = 2 − 1.1887 bits
weights = 1 0 0 0                           ← solved in one shot
```

The best experiments make each candidate predict the outcome with near
certainty (H(Y|F) ≈ 0) while the candidates *disagree* with each other
(H(Y) large).

**Round cap:** 10 rounds. If the threshold isn't reached, the result is `NaN`.
`success_flags` records whether the highest-weight candidate is the true one,
once some weight reaches 0.95.

---

## `qle_sa.py`: simulated annealing

**Strategy:** search the five-dimensional settings space x = (t, θ, φ, α, β)
with simulated annealing.

| Step | What happens |
|---|---|
| Start | a random point: t, φ, β ∈ [0, 2π); θ, α ∈ [0, π) |
| Each of 200 iterations | Try 20 random neighbours x + (uniform in [−1, 1]) · (π, π/2, π, π/2, π) · T. Wrap each coordinate back into its range. |
| Move | Go to the best neighbour if it lowers the cost. If it raises the cost by Δ, go there anyway with probability e^{−Δ/T} (the Metropolis rule, which lets the search escape local minima). |
| Cool | T ← 0.9 T, starting at T = 1. After about 50 iterations T < 0.01, so the search becomes a fine local refinement. |
| Result | the best point seen overall |

Cost: 1 + 200 × 20 = 4,001 evaluations of H(F|Y) per round. About 0.1 s per
round; a whole run on the default set takes about 5 s.

The search is random, so two runs can choose slightly different settings. In
practice the result is consistently close to optimal.

---

## `qle_grid.py`: exhaustive grid search

**Strategy:** evaluate H(F|Y) on every point of a fixed grid and take the best.

| Setting | Grid | Points |
|---|---|---|
| t | 0.1 … 2π | 10 |
| α | 0 … π | 12 |
| β | 0 … 2π | 24 |
| θ | 0 … π/2 | 6 |
| φ | 0 … 2π | 24 |

That's 10 × 12 × 24 × 6 × 24 = **414,720 settings per round**.

- **Why θ only goes to π/2:** measuring along axis **n** or −**n** is the same
  measurement with the outcomes swapped, so the upper hemisphere already covers
  every measurement.
- **Efficiency:** the evolution e^{−iHⱼt}|ψ₀⟩ is computed once per (t, α, β) and
  reused for all 144 (θ, φ) pairs.
- **Skipped settings:** if either outcome has zero probability, the setting is
  skipped.

Cost: about 85 s per round in Python, so roughly 15–25 minutes for a full run on
the default set. The result is deterministic for a given weight vector; only
the simulated measurement outcomes are random.

`PGH` in this file is a different heuristic, t = π / (2‖H₁ − H₂‖) for the two
highest-weight candidates. The MATLAB original defines it but never calls it,
so it's unused here too.

---

## Results

Average number of shots to reach 0.99, from Python runs of this code:

| Candidate set | Original QLE | SA | Grid |
|---|---|---|---|
| {σx, 2σx, σz, 2σz} | 18, 21, 31, 50 (200 runs) | 1.05, 2.15, 2.85, 3.0 (20 runs) | 2, 2, 3, 2 (1 run) |
| six-set (optimised order H1, H4, H3, H2, H5, H6) | H1 and H4 never converge | 3.5–4.3 per candidate (20 runs) | not run (very slow) |

The default-set numbers agree with the results the authors published in their
MATLAB repository. On the six-set, the SA averages are somewhat lower than
theirs on two candidates. The repository doesn't say which settings produced
those numbers.
