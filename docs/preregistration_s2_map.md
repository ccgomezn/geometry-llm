# Preregistration: Study 2, could the model's own map tell whether a point is inside the triangle?

*Committed 2026-11-13, before any data of this test exists. Gemma 3 12B (Colab, one A100). Follows the Study 2 pilot
(`docs/preregistration_s2_pilot.md`, verdict **fail**: balanced accuracy 0.51 on the hard "inside" questions, answers
"no" 93% of the time) and the exploratory score check (the yes − no score adds nothing beyond the size shortcut).*

## Why

The pilot shows the model cannot judge "inside the triangle" from text, not even in its yes/no scores. Study 1 found
that the model builds an accurate map of such scenes but its answers do not read it. This test asks whether that map,
decoded from the model's activations, carries enough of the shape to answer the question the model fails. If it
does, the model holds what it needs and still does not use it, even when the map is the only route. That is a
stronger form of Study 1's finding. If it does not, the map is too coarse for shapes, which explains the failure
another way.

## Design

- **Map read-out.** Residual stream at the object-list positions, at layers 14, 18, 21, 24 and 28. Ridge probes
  (GroupRidge, scene-grouped CV, as in P2/P9) for scene-centred (x, y) of all four objects, fitted on 2,000 new
  triangle scenes (seed 711, the same generator and prompt format as the pilot).
- **Primary layer:** the layer with the highest training-split CV R². It is fixed when the probes are fitted, before
  any test scene is decoded. The other layers are reported.
- **Test scenes:** the pilot's 400 scenes (seed 702) and all their ladder questions (4,017).
- **Map answers:** decode every object's position from its list activation in the bare description (no question).
  Then answer each question geometrically from the decoded positions, using the same rule that defines the correct
  answers. For rung 4, "inside" is the strict interior of the decoded triangle.
- **Model answers:** the pilot's answers (the candidate with the highest log-probability). The script checks that the
  question texts match exactly.
- **Size rule:** the pilot's magnitude rule (map-free size cues from the stated sentences), fitted on the pilot's 2,000
  training scenes (seed 701).
- **Metric:** balanced accuracy with a scene-bootstrap 95% CI (1,000 resamples). Map − model and map − size rule are
  paired differences over the same resampled scenes.

## Decision rule (primary layer, rung 4, hard subset: inside vs box_hard)

1. **No evidence the map carries the shape:** the CI lower bound of (map − size rule) is ≤ 0.
2. Otherwise, **the map can tell inside** if all of these hold:
   - map balanced accuracy ≥ 0.65;
   - its CI lower bound > 0.5;
   - the CI lower bound of (map − model) > 0.
3. Otherwise: **partial**.

## Reported, not part of the rule

- Every layer.
- Rungs 1–3: map vs model, and agreement. As in Study 1's P9, the map should match or beat the model on chain
  questions.
- Probe CV R² on the training scenes and R² on the test scenes.
