## Map boost at the answer position (gemma3-12b-8bit) — preregistered

Model **Gemma 3 12B** (chat format). Locate: ridge probes for M's step and C's own step (per axis) at the final position of the flow set (seed 1601, 2400 questions); amplify on new scenes (seed 1701): h ← h + (α − 1)·P(h − μ) at one layer, final position only. Control: mean of 3 random subspaces of the same rank and projected variance (±10%) in the top principal components. Balanced accuracy; gains are boosted − clean; contrast = gain − control gain, paired scene-bootstrap 99% CI (2000 resamples).

- Layer 28: held-out R² of M's step 0.81, of C's own step 0.83; mean residual norm 6.43e+04; mean size of P(h − μ), i.e. the change per unit of α − 1: map 183 (control 176, top 64 components); M's step 116 (control 113, top 64 components).
- Layer 32: held-out R² of M's step 0.79, of C's own step 0.82; mean residual norm 5.32e+04; mean size of P(h − μ), i.e. the change per unit of α − 1: map 259 (control 253, top 128 components); M's step 161 (control 156, top 128 components).

Clean balanced accuracy on the decisive questions: 0.32; first step only 0.52; own step only 0.97, agree 0.99, conflict, C larger 0.84.

**Arm: map** — preregistered verdict: **no benefit**

| layer | α | decisive | gain | control gain | contrast [99% CI] | first step only gain [99% lower] | easy gains (own / agree / C larger) | works |
|---|---|---|---|---|---|---|---|---|
| 28 | 2 | 0.32 | -0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 4 | 0.32 | -0.00 | -0.00 | +0.00 [-0.01, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 8 | 0.32 | +0.00 | +0.00 | +0.00 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 2 | 0.32 | -0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 4 | 0.32 | -0.00 | -0.01 | +0.00 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 8 | 0.32 | -0.00 | -0.00 | -0.00 [-0.02, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |

**Arm: M's step** — preregistered verdict: **no benefit**

| layer | α | decisive | gain | control gain | contrast [99% CI] | first step only gain [99% lower] | easy gains (own / agree / C larger) | works |
|---|---|---|---|---|---|---|---|---|
| 28 | 2 | 0.32 | +0.00 | -0.01 | +0.01 [-0.00, +0.02] | -0.00 [-0.01] | +0.00 / +0.00 / +0.00 | no |
| 28 | 4 | 0.32 | -0.00 | -0.00 | +0.00 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 28 | 8 | 0.32 | +0.00 | -0.00 | +0.01 [-0.01, +0.02] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 2 | 0.32 | +0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 4 | 0.32 | -0.00 | -0.00 | +0.00 [-0.00, +0.01] | +0.00 [+0.00] | +0.00 / +0.00 / +0.00 | no |
| 32 | 8 | 0.32 | -0.00 | -0.00 | -0.00 [-0.02, +0.01] | +0.00 [-0.01] | +0.00 / +0.00 / +0.00 | no |

Transfer (exploratory: no condition works), for the condition with the largest decisive contrast: M's step, layer 28, α 8.

| question | n | clean accuracy | gain | contrast with control [95% CI] |
|---|---|---|---|---|
| C vs M | 150 | 0.91 | +0.00 | +0.00 [+0.00, +0.01] |
| M vs R | 150 | 0.88 | +0.01 | +0.01 [+0.00, +0.02] |
| farthest | 150 | 0.66 | -0.01 | +0.01 [-0.00, +0.02] |
| closest | 150 | 0.73 | +0.00 | +0.00 [-0.01, +0.01] |

Overall transfer accuracy change +0.00. (Would read: specific.)

Exploratory: weight of M's step relative to C's own step in the margin, clean 0.23, M's step|L28|a8 0.23. On the decisive questions the answer follows C's sentence in 0.69 (clean) and 0.69 (boosted), M's sentence in 0.31 and 0.31. Logit lens: not computed.

Preregistered verdicts: map: **no benefit**; M's step: **no benefit**. Rules: docs/preregistration_map_boost.md.
