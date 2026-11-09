# E1 preregistration

*Committed before any P2 (probe) result exists. Copied verbatim from `docs/specs/2026-11-09-e1-spatial-map-design.md` §4. Analyses not listed here are exploratory and are labelled as such in RESULTS.md.*

**Thresholds, set in advance (written to `docs/preregistration.md` and committed before P2 runs; Claude's call).**

| Outcome | Requires |
|---|---|
| Support for H-map | **All** of these: (1) Probe R² ≥ 0.6 at the best layer, beating every control by ≥ 0.3. (2) Multi-hop R² within 0.1 of 1-hop. (3) Joint consistency ≥ 0.6 at the readout position or the end of the list. (4) 2-D DAS interchange accuracy ≥ 0.7 on every question type. |
| Support for H-rel | **All** of these: (1) R² < 0.3, or a clear drop with hop count. (2) Patching effects that are concentrated on the relation-sentence tokens. (3) Joint consistency < 0.2 at the readout positions. |
| Mixed | Anything in between. The result is reported as such, with the per-layer and per-question-type breakdown. |

Every metric is reported with 95% bootstrap confidence intervals. The sample sizes must be large enough for intervals of ±0.05 on the key metrics.

**Branches if a phase comes back weak (Claude's call).**
- **Weak P2 probes.**
  - Try a small nonlinear (MLP) probe and the other readout positions before concluding.
  - Run P3 anyway: patching does not assume the code is linear.
- **No usable direction from P2 or DAS.** Skip the slider.
- **The minimum complete negative result** is P1 + P2 + P3, with all controls. That gets written up.

**Priority if time or budget is tight** (no fixed deadline; the goal is to publish as soon as the result is solid).
1. The consistency result in P3 comes first.
2. Things to cut first, in this order: the scale check, comparing DAS at larger k, the random-init control, the near-tie set.


## Fixed analysis choices

- Probe readout: per-object position in the object list; per-scene-centred ridge regression; scene-level 80/20 split (seed 0).
- Best layer = the layer with the highest held-out R² for the real probe; every reported R² is held-out.
- Multi-hop comparison: R² on objects with hop 1 vs hop ≥ 2 (centred coordinates), at the best layer.
- Joint consistency (P3) uses only question types with ≥ 1 diagnostic question per pair; pairs with no diagnostic question are dropped and counted.
- CIs: 95% percentile bootstrap, 1000 resamples, resampling scenes (P2) or pairs (P3, P4).

## Clarification (2026-11-09, before any model data)

- DAS criterion (4): "interchange accuracy" is computed on **diagnostic questions only** (those whose answer differs between the base and source maps). Counting control questions would let a model that ignores the intervention score highly. All-question accuracy and control stability are reported alongside. This makes the criterion stricter.

## Clarification (2026-11-09, after P1, before any P2–P4 data)

- **P1 outcome.** No model reached the gate on 4-object scenes. Best: Gemma 3 27B, 68% [67, 69]. The preregistered fallback (3-object scenes) was applied. On 3-object scenes, Gemma 3 27B passes: 76% [75, 77], with accuracy above chance on 2-hop questions of +0.27 [0.26, 0.29]. Gemma 3 12B narrowly fails (74.5%). **Chosen model: Gemma 3 27B; all P2–P4 scenes have 3 objects** (hop depth ≤ 2, so the multi-hop comparison is hop 1 vs hop 2).
- **Question types at chance.** Question types where the chosen model's P1 accuracy CI on 3-object scenes includes chance are excluded from the P3 joint-consistency and P4 DAS/slider headline metrics, and reported separately. For Gemma 3 27B this is `closer_than` (0.53 [0.50, 0.56]).
- Analyses on 4–5-object scenes with this model, if any, are exploratory.
