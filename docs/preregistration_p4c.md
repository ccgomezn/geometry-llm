# Preregistration: P4c, confirming the sentence-site 2-D code, and a replication on Gemma 3 27B

*Committed 2026-11-10, before any P4c data exists.*

## Why

On Gemma 3 12B, a 2-D subspace learned by DAS over the moved object's sentence (layer 14) moved 0.81 [0.78, 0.83] of the affected answers on held-out pairs, against 0.00 for a random 2-D subspace. Setting it halfway gave in-between answers on 6% of the questions that called for one (chance ≈ 14%). Those results are **exploratory**: the site and layer were chosen after seeing P3. This run confirms them on fresh data with the rule fixed in advance, and replicates them on a larger model.

## Runs

1. **Gemma 3 12B (confirmation).**
   - Site: the moved object's sentence. Layer 14. k = 2.
   - 300 training pairs (seed 105) and 300 test pairs (seed 106): new scenes, never used before.
   - 2 epochs, lr 1e-3, the same procedure as the exploratory run. Random-subspace control as before.
   - Then the halfway test (P4i) on 150 of the test pairs, α ∈ {0, ¼, ½, ¾, 1, 1½}.
2. **Gemma 3 27B (replication).**
   - The same procedure at the matched relative depth: layer 18 of 62 (14/48 × 62 ≈ 18).
   - 3-object scenes, `closer_than` excluded. Seeds 105/106.
   - The halfway test as above.

## Decision rules

- **2-D code confirmed** (per model): overall interchange accuracy (ALL) ≥ 0.70, and the random-subspace control ≤ 0.05.
- **Halfway test** (per model, α = ½, questions whose answer at the in-between position differs from both stated layouts):
  - **switch** if the upper end of the 95% CI of the in-between-answer rate is below chance;
  - **coordinate** if the rate is ≥ 0.5;
  - **unclear** otherwise.

The per-question-type ≥ 0.7 criterion of the original preregistration is reported but is not the decision rule here; the exploratory run already showed closest/farthest are rare and noisy.
