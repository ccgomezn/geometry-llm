## Map boost dose check (gemma3-12b-8bit) — preregistered

Model **Gemma 3 12B** (chat format), same located subspaces and controls as the map boost run. Positive control: at the boost layer, final position, add D(α)·u, u ∝ (final-norm scale) ⊙ (W_U[true answer] − W_U[other name]) (unit length), D(α) = (α − 1)·s, s = mean |P(h − μ)| of the map arm. Arms and controls at α ∈ {16, 32, 64} added; α 2–8 answers reused from the boost run. Paired scene-bootstrap 99% intervals (2000 resamples).

- Layer 28: s = 183, mean residual norm 6.43e+04; D(8) = 1.28e+03, D(64) = 1.16e+04.
- Layer 32: s = 259, mean residual norm 5.32e+04; D(8) = 1.82e+03, D(64) = 1.63e+04.
- Clean answers rescored here agree with the boost run's on 1.000 of questions.

**1. Positive control (answer direction).** Criterion at D(8): decisive gain ≥ +0.20 with its 99% interval above +0.10.

| layer | α | dose D(α) | decisive | gain [99% CI] | first step only gain | easy gains (own / agree / C larger) |
|---|---|---|---|---|---|---|
| 28 | 2 | 183 | 0.32 | +0.00 [+0.00, +0.00] | +0.00 | +0.00 / +0.00 / +0.00 |
| 28 | 4 | 550 | 0.33 | +0.01 [+0.00, +0.02] | +0.00 | +0.00 / +0.00 / +0.00 |
| 28 | 8 | 1.28e+03 | 0.35 | +0.03 [+0.01, +0.06] | +0.00 | +0.00 / +0.00 / +0.00 |
| 28 | 16 | 2.75e+03 | 0.36 | +0.04 [+0.02, +0.08] | +0.01 | +0.00 / +0.00 / +0.01 |
| 28 | 32 | 5.68e+03 | 0.51 | +0.19 [+0.12, +0.26] | +0.10 | +0.02 / +0.00 / +0.09 |
| 28 | 64 | 1.16e+04 | 1.00 | +0.68 [+0.62, +0.75] | +0.48 | +0.03 / +0.01 / +0.16 |
| 32 | 2 | 259 | 0.33 | +0.01 [+0.00, +0.02] | +0.00 | +0.00 / +0.00 / +0.00 |
| 32 | 4 | 778 | 0.35 | +0.03 [+0.01, +0.05] | +0.01 | +0.00 / +0.00 / +0.00 |
| 32 | 8 | 1.82e+03 | 0.36 | +0.04 [+0.01, +0.07] | +0.01 | +0.00 / +0.00 / +0.01 |
| 32 | 16 | 3.89e+03 | 0.45 | +0.13 [+0.08, +0.18] | +0.06 | +0.01 / +0.00 / +0.06 |
| 32 | 32 | 8.04e+03 | 0.98 | +0.66 [+0.59, +0.73] | +0.47 | +0.03 / +0.01 / +0.15 |
| 32 | 64 | 1.63e+04 | 1.00 | +0.68 [+0.62, +0.75] | +0.48 | +0.03 / +0.01 / +0.16 |

Moves the answer at D(8): layer 28 **no**; layer 32 **no**.

**2. Larger α (arms against their variance-matched controls).** Verdicts over the six new conditions (α 16–64); α 2–8 rows are the boost run's, shown for reference.

Arm **map**, at higher dose: **partial (the decisive contrast is above 0 in some condition, but no condition meets every criterion)**

| layer | α | decisive | gain | control gain | contrast [99% CI] | first step only gain [99% lower] | easy gains (own / agree / C larger) | works |
|---|---|---|---|---|---|---|---|---|
| 28 | 2 | 0.32 | -0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 4 | 0.32 | -0.00 | -0.00 | +0.00 [-0.01, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 8 | 0.32 | +0.00 | +0.00 | +0.00 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 16 | 0.33 | +0.01 | -0.00 | +0.01 [-0.00, +0.03] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 32 | 0.34 | +0.02 | -0.01 | +0.03 [+0.01, +0.05] | +0.00 [+0.00] | -0.01 / +0.00 / +0.00 | no |
| 28 | 64 | 0.35 | +0.03 | +0.01 | +0.02 [-0.01, +0.05] | +0.01 [+0.00] | -0.02 / +0.00 / +0.00 | no |
| 32 | 2 | 0.32 | -0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 4 | 0.32 | -0.00 | -0.01 | +0.00 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 8 | 0.32 | -0.00 | -0.00 | -0.00 [-0.02, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 16 | 0.32 | +0.00 | +0.01 | -0.01 [-0.03, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 32 | 0.33 | +0.01 | +0.01 | -0.00 [-0.02, +0.02] | +0.00 [-0.01] | +0.00 / +0.00 / +0.01 | no |
| 32 | 64 | 0.34 | +0.02 | +0.02 | -0.00 [-0.03, +0.03] | +0.00 [-0.01] | +0.00 / -0.01 / +0.01 | no |

Arm **M's step**, at higher dose: **partial (the decisive contrast is above 0 in some condition, but no condition meets every criterion)**

| layer | α | decisive | gain | control gain | contrast [99% CI] | first step only gain [99% lower] | easy gains (own / agree / C larger) | works |
|---|---|---|---|---|---|---|---|---|
| 28 | 2 | 0.32 | +0.00 | -0.01 | +0.01 [-0.00, +0.02] | -0.00 [-0.01] | +0.00 / +0.00 / +0.00 | no |
| 28 | 4 | 0.32 | -0.00 | -0.00 | +0.00 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 8 | 0.32 | +0.00 | -0.00 | +0.01 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 16 | 0.33 | +0.01 | -0.01 | +0.01 [-0.00, +0.03] | +0.00 [+0.00] | +0.00 / +0.00 / -0.00 | no |
| 28 | 32 | 0.33 | +0.01 | -0.01 | +0.02 [-0.00, +0.04] | +0.01 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 64 | 0.35 | +0.03 | -0.00 | +0.04 [+0.01, +0.07] | +0.00 [-0.01] | -0.01 / +0.00 / +0.00 | no |
| 32 | 2 | 0.32 | +0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 4 | 0.32 | -0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 8 | 0.32 | -0.00 | -0.00 | -0.00 [-0.02, +0.01] | +0.00 [-0.01] | +0.00 / +0.00 / +0.00 | no |
| 32 | 16 | 0.33 | +0.01 | -0.00 | +0.01 [-0.01, +0.03] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 32 | 0.33 | +0.01 | +0.00 | +0.01 [-0.01, +0.04] | +0.00 [-0.01] | +0.00 / +0.00 / +0.00 | no |
| 32 | 64 | 0.37 | +0.06 | +0.01 | +0.05 [+0.02, +0.08] | -0.01 [-0.02] | -0.02 / -0.00 / -0.02 | no |

**Overall (preregistered): still too small to tell.**
- Layer 28: the positive control first reaches the threshold at D(64) = 1.16e+04. The map arm was tested at that dose (α 64) and above, and no condition works: **not read at a dose that moves the answer**.
- Layer 32: the positive control first reaches the threshold at D(32) = 8.04e+03. The map arm was tested at that dose (α 32) and above, and no condition works: **not read at a dose that moves the answer**.

Exploratory: mean shift of the decisive margin toward the true answer (logits; clean margins are large), with the size of each edit.

| layer | α | push (D) | map real | map control | M's step real | M's step control |
|---|---|---|---|---|---|---|
| 28 | 2 | +0.14 (183) | -0.02 | -0.03 | -0.03 (116) | -0.04 |
| 28 | 4 | +0.40 (550) | -0.01 | -0.05 | -0.02 (349) | -0.03 |
| 28 | 8 | +0.95 (1.28e+03) | +0.01 | -0.07 | -0.01 (815) | -0.06 |
| 28 | 16 | +2.50 (2.75e+03) | +0.16 | -0.13 | +0.01 (1.75e+03) | -0.10 |
| 28 | 32 | +10.13 (5.68e+03) | +0.28 | -0.24 | +0.11 (3.61e+03) | -0.13 |
| 28 | 64 | +47.23 (1.16e+04) | -0.38 | -0.11 | -0.18 (7.34e+03) | -0.27 |
| 32 | 2 | +0.38 (259) | -0.05 | -0.02 | -0.05 (161) | -0.04 |
| 32 | 4 | +1.18 (778) | -0.09 | -0.03 | -0.07 (483) | -0.03 |
| 32 | 8 | +2.66 (1.82e+03) | -0.21 | -0.01 | -0.21 (1.13e+03) | -0.08 |
| 32 | 16 | +7.69 (3.89e+03) | -0.42 | -0.04 | -0.41 (2.42e+03) | -0.11 |
| 32 | 32 | +32.36 (8.04e+03) | -0.80 | -0.23 | -0.77 (4.99e+03) | -0.25 |
| 32 | 64 | +75.23 (1.63e+04) | -2.16 | -1.22 | -1.34 (1.01e+04) | -0.83 |

Exploratory: first α at which the controls cost an easy kind more than 0.05: map layer 28: none up to α 64; map layer 32: none up to α 64; M's step layer 28: none up to α 64; M's step layer 32: none up to α 64.

Exploratory: on the decisive questions, share of answers following C's sentence / M's sentence at the new α (clean 0.69 / 0.31): map L28 α16 0.68 / 0.32; map L28 α32 0.67 / 0.33; map L28 α64 0.66 / 0.34; map L32 α16 0.69 / 0.31; map L32 α32 0.68 / 0.32; map L32 α64 0.68 / 0.32; M's step L28 α16 0.69 / 0.31; M's step L28 α32 0.68 / 0.32; M's step L28 α64 0.66 / 0.34; M's step L32 α16 0.68 / 0.32; M's step L32 α32 0.68 / 0.32; M's step L32 α64 0.64 / 0.36.

Preregistered verdict: **still too small to tell**. Rules: docs/preregistration_map_boost_dose.md.
