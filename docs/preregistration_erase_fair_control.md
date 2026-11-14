# Preregistration (amendment): a fair random control for the verified erasure

*Committed 2026-11-14, before any data of this control exists. Gemma 3 12B instruct, bfloat16 (Colab A100).
Amends `docs/preregistration_erase_axis_confirm.md`.*

## Why

The confirmatory erasure removed the map completely (fresh-scene linear R² ≤ 0.009 at every layer 12–48, MLP ≤ −0.02),
but its verdict was "uninformative": its "matched" random control removed far more variance than the map erasure. The
map lives in low-variance directions (equal-rank isotropic directions remove a median 10× more), and the matching only
searched towards higher variance. This amendment adds the control that test needed. Nothing else changes: same bases,
same test set (600 chains, seed 1203), same clean and map-erased answers (already scored; not rescored).

## The control

At each layer 12–48, a random orthonormal basis of the map basis's rank, drawn as w ~ (Σ + floor)^(α/2) z with α now
searched over **[−8, 8]** (negative α leans towards low-variance directions), set by bisection so that the variance it
removes, tr(RᵀΣR), equals the map basis's within 2%. Σ: covariance of clean, scene-centred object-list activations
on the 4,000 fit scenes (seed 1101; full rank). Seeded by layer; c from the clean mean; applied at every token except
the first. The achieved ratio is also reported on the 1,000 validation scenes (seed 1102), which play no part in it.

## Checks (reported)

Linear and MLP probes at layers 14, 21, 28, 36, 44 on the validation scenes under this control (it should leave the
map readable); the match ratio per layer on fit and validation scenes.

## Decision rules

Excess drop = control accuracy − map-erased accuracy, per question, 95% scene-bootstrap CI (1,000 resamples).

0. If the control leaves composed accuracy below chance + 0.15: **uninformative** (both readouts).
1. **Primary A (as preregistered for the confirmatory test), C–R axis questions:** CI lower bound > 0 and excess ≥ 0.03:
   **confirmed: composed axis answers use the map**; CI lower bound > 0 and excess < 0.03: **small effect**;
   otherwise **not confirmed**.
2. **Primary B (for the paper), all composed questions:** excess ≥ 0.10 and CI lower bound > 0: **map used**;
   CI lower bound > 0: **partial use**; excess ≤ 0.03: **map not used beyond generic damage**; otherwise **unclear**.

Both are reported; they answer different questions and no multiplicity correction is applied.

## Reported, not part of the rules

Excess drops for every class × type family; stated-only overall; accuracy per condition.
