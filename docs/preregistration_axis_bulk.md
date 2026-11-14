# Preregistration: which C–R axis questions use the erased map directions? (bulk test)

*Committed 2026-11-14, before any data of this test exists. Gemma 3 12B instruct, bfloat16 (Colab A100).*

## Why

The fair-control amendment (`docs/preregistration_erase_fair_control.md`) confirmed that erasing the map hurts the C–R
axis questions ("which is farther east / north, C or R?") beyond a same-size random erasure: +0.055 [+0.040, +0.071].
An exploratory split of those questions (same data, after the result) by how the two steps combine on the asked axis
(d1 = R→M step, d2 = M→C step) suggested two things:

1. **Behaviour:** the model answers mostly by the sign of C's own step d2. Accuracy is 0.97–0.98 when that sign gives
   the answer, 0.51 when d2 = 0, and 0.32 when the steps conflict and |d1| > |d2| (the sign of d2 is then wrong).
2. **Erasure:** in that last cell, where the composed position and C's own step disagree, erasing the map *raised*
   accuracy (0.32 → 0.45; fair control 0.38; excess −0.073 [−0.128, −0.021]). If the answers read C's composed
   position, erasing it should hurt there. The erased directions may instead carry C's own step, which the shortcut
   uses.

This test checks both on new data, with many questions in each cell.

## Design

- **Questions:** only C–R axis questions (*more east*, *more north*, either order) from new 3-object chains R → M → C
  (seed 1301; scenes and margins as in all chain tests: |d1 + d2| ≥ 1). Each question is put in one cell by the
  steps on its axis:

  | cell | rule | answer given by |
  |---|---|---|
  | own step only | d1 = 0 | sign of d2 |
  | agree | d1·d2 > 0 | both |
  | first step only | d2 = 0 | sign of d1 |
  | conflict, C larger | d1·d2 < 0, \|d2\| > \|d1\| | sign of d2 |
  | conflict, M larger | d1·d2 < 0, \|d1\| > \|d2\| | sign of d1 (d2's sign is wrong) |

- **Sizes:** 1,500 questions in each conflict cell, 750 in each other cell (6,000 in all). Scenes are drawn in order
  and a scene's C–R axis questions are kept while their cell is not yet full.
- **Conditions:** clean; map erased (the confirmatory test's bases, layers 12–48, every token except the first); fair
  random control (the amendment's variance-matched bases). Nothing is refitted.
- **Excess drop** = fair-control accuracy − map-erased accuracy, 95% scene-bootstrap CI (1,000 resamples).

## Decision rules

1. **Behaviour (shortcut):** clean accuracy in *conflict, M larger*, with a scene-bootstrap CI. Upper bound < 0.50:
   **the model follows C's own step** (it answers below chance where that step misleads). Lower bound > 0.50: **the
   model composes the steps**. Otherwise **undecided**.
2. **Erasure (primary):** excess drop in *conflict, M larger*:
   - lower bound > 0 and excess ≥ 0.03: **the answers read C's composed position from the map**;
   - upper bound < 0: **the erased directions support the own-step shortcut, not the composed position**;
   - otherwise: **undecided**.
3. **Replication:** excess drop pooled over all 6,000 questions, lower bound > 0: **the C–R axis effect replicates**
   (the pool is balanced by cell, not the natural mix; reported as such).

## Reported, not part of the rules

Clean, map-erased and fair accuracy and excess drop in every cell; the same split by question type (east / north)
and by which object is named first.
