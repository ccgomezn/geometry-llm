# Preregistration: do composed "more east / more north" answers use the map? (confirmatory erasure)

*Committed 2026-11-14, before any data of this test exists. Gemma 3 12B instruct, bfloat16 (Colab A100).*

## Why

The closed-loop erasure (`docs/preregistration_erase_closed_loop.md`, RESULTS.md 2026-11-14) ended with the verdict
"the map could not be erased": on fresh scenes the linear probe still read positions with R² 0.13 at layers 20 and
24 (the rule needed < 0.10 everywhere). An exploratory breakdown of its answers by question type found that erasing
the map hurts the single-axis comparisons (*more east*, *more north*) beyond a random erasure in every class,
including the composed C–R questions (+0.06 [+0.02, +0.10] and +0.04 [+0.01, +0.08]), and the same with P5c's bases
once the first token is spared, while distance and direction questions were not hurt beyond random. Two weaknesses
remain: the erasure did not pass its own check, and the random control (isotropic directions) was not neutral (on
direction questions it hurt more than the map erasure). This test fixes both and tests the axis finding on new data.

## Design

### The erasure

- **Layers and positions:** every layer 12–48, every token except the first, h ← h − (h·B − c)Bᵀ with c = μ·B, as in
  the closed-loop test.
- **Fit:** 4,000 new 3-object scenes (seed 1101), starting from the closed-loop test's final bases. Each round: run
  the model under the current erasure, capture the object-list activations after the erasure at every erased layer,
  fit the Study 1 ridge probe (scene-grouped 3-fold CV); where its R² ≥ **0.02** (was 0.05), run INLP up to 5 steps
  (stop at R² < 0.02) and add the directions; c recomputed from the erased run's mean.
- **Generalisation check during the fit (dev set):** when every layer is below 0.02 on the fit scenes, capture 1,000
  further scenes (seed 1103) under the erasure and fit the linear probe per layer. If any layer has R² ≥ 0.07, those
  scenes join the fit set and the rounds continue; the next dev set uses seed 1104. At most 2 dev additions, at most
  15 rounds in all. The validation scenes below are never used in fitting.

### Random controls (same layers, positions and rank per layer)

- **Matched (primary):** at each layer, random directions drawn from a family that runs from isotropic (α = 0) to
  the activations' top principal directions (α large): w ~ (Σ + floor)^(α/2) z, orthonormalised, with Σ the
  covariance of clean object-list activations (first 1,000 fit scenes, scene-centred) and the null space given the
  smallest positive eigenvalue. α (0–8) is set by bisection so that the variance the random basis removes, tr(RᵀΣR),
  equals the probe basis's tr(BᵀΣB) within 2%. c from the clean mean. Seeded by layer.
- **Isotropic (secondary):** random orthonormal directions as in the closed-loop test.

### Is the erasure effective? (fresh scenes)

1,000 new scenes (seed 1102): linear probe at every erased layer and MLP probe at 14, 21, 28, 36, 44 under the map
erasure; the same five layers under each random erasure. **Effective** if the linear R² < 0.10 at every erased layer.

### Test

- **New** test set: 600 chain scenes R → M → C (seed 1203), all questions passing the 1-unit / 10° margins except
  *closer than*. The exploratory finding came from P5c's set (seed 202), which is not reused.
- Conditions: clean, map erased, matched random, isotropic random.
- **Excess drop** = (clean − map-erased accuracy) − (clean − random accuracy) = random accuracy − map-erased accuracy,
  per question, 95% scene-bootstrap CI (1,000 resamples).
- **Primary subset: CR axis** = questions comparing C and R (no M) of type *more east* or *more north*. C's position
  relative to R is stated nowhere; it must be composed from the two sentences.

## Decision rule

1. Linear R² ≥ 0.10 at any erased layer on the fresh scenes: **the map could not be erased**; no verdict on use.
2. Otherwise, if the matched random erasure leaves composed accuracy below chance + 0.15: **uninformative**.
3. Otherwise, CR-axis excess drop against the **matched** random control:
   - CI lower bound > 0 and excess ≥ 0.03: **confirmed: composed axis answers use the map**;
   - CI lower bound > 0 and excess < 0.03: **small effect**;
   - otherwise: **not confirmed**.

## Reported, not part of the rule

Fit rounds, dev checks and ranks; every probe on the fresh scenes; α and removed variance per layer; excess drops
against both controls for every class × type family (axis, distance, direction) and for composed vs stated-only
overall; accuracy per class and type.

## What each outcome leads to

- **Confirmed:** the claim "Gemma reads composed coordinates from its map to answer axis comparisons" goes into the
  write-up. Next: 27B and 4–5-object scenes; why distance/direction do not show it.
- **Small effect / not confirmed:** the exploratory finding was weaker than it looked or depended on the control;
  report it as such and decide with the author.
- **Could not be erased / uninformative:** report; decide with the author.
