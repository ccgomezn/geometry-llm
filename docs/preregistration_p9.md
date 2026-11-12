# Preregistration: P9, would the model's own map have answered better?

*Committed 2026-11-13, before any P9 data exists. Gemma 3 12B (Colab, one A100). 3-object chains, `closer_than`
excluded.*

## Why

The object-list probe reads two-step positions as accurately as one-step ones (P2: 0.74 vs 0.76), yet the answers use
the parent's offset only partially (P8b, exploratory: on questions about C and R alone, the model updates correctly in
0.39 of cases when C's own sentence moves C, but in 0.21 when only its parent's sentence moves C to the same place).
If the decodable map composes the chain while the answers do not, then answering from the map should beat the model
on exactly the questions that need composition.

## Design

- **Map read-out.** Residual stream at the object-list positions, layer 21 (the P2 peak; primary). Ridge probe for
  scene-centred (x, y), fitted as in P2 (scene-grouped CV for the penalty) on 2,000 new 3-object scenes (seed 601).
- **Test.** 300 new chains R → M → C (seed 602), every question passing the 1-unit / 10° margins. Classes as in P7:
  **CR** (involves C and R), **CM** (only C and M), **MR** (no C).
- **Map answers.** Decode every object's position from its list activation in the bare description (no question), then
  answer each question geometrically from the decoded positions (the same rule that defines correct answers).
- **Model answers.** The model's own answer (highest candidate log-probability), as everywhere else.
- **Metric.** Accuracy against the true layout, per class; difference = map − model, with a scene-bootstrap 95% CI.
- **Reported, not part of the rule:** the same at layers 18, 24 and 28; agreement between map and model answers.

## Decision rule (layer 21, CR questions)

- **The map would have answered better:** map − model ≥ 0.05 and the CI excludes 0.
- **The model answers better:** model − map ≥ 0.05 and the CI excludes 0.
- **Comparable:** otherwise.

## What each outcome means

- *Map better*: the model holds a more accurate answer than the one it gives, in a representation its answer pathway
  does not read (P3, P5c, P6, P7). That would be a concrete case of a decodable world state that is more faithful than
  the output.
- *Model better* or *comparable*: the decodable map is not a better source of answers; the unused map is not a missed
  opportunity, at least through a linear read-out.

## Caveat stated in advance

The read-out is a linear probe fitted by us. A better map answer shows that the information is present and usable by a
linear decoder, not that the model could have used it without further computation.
