# Preregistration: Study 2 pilot, can the model tell whether a point is inside a triangle?

*Committed 2026-11-13, before any Study 2 data exists; revised the same day after code review, still before any data
(see "Revision" at the end). Gemma 3 12B instruct (Colab, one A100). Answers only: no
probes, no interventions. Design: `docs/specs/2026-11-13-study2-triangle-dial-design.md`.*

## Why

Study 1 found that Gemma 3 builds an accurate map of a scene described in text but answers without it, using a
shortcut over each stated sentence. Every Study 1 question could be answered that way. Study 2 asks whether the model
uses its map when the shortcut stops working. Its main tests (erasing the map, moving a corner inside it) only mean
something if the model can answer the "inside the triangle" questions in the first place, well above any map-free
shortcut. This pilot checks that before any main-study work.

## Scenes

- Four objects: corners A, B, C and point P, with single-token names (people pool for even-numbered scenes, objects pool
  for odd ones, as in Study 1).
- Integer grid, north = +y, east = +x. A at the origin; B and C drawn uniformly from [-8, 8]²; triangle area ≥ 4.
  A triangle is kept only if it admits P positions in **all three** strata below, so its shape carries no information
  about which stratum P is drawn from.
- P's position is drawn uniformly from the eligible grid points of a stratum chosen first: **inside** the triangle (probability 0.5), **box_hard**, which is
  outside the triangle but inside its bounding box (0.35), or **outside_box** (0.15). P is never on a corner and is at
  least 0.5 units from every edge.
- Described as relations only: "B is … of A. C is … of B. P is … of X", where X is a random corner. This is followed
  by the object list and the question, with Study 1's prompt format and templates.

## The ladder (same scenes)

| Rung | Question | Kept when |
|---|---|---|
| 1 direct | "Which is farther north/east, P or X?", where X is P's parent | that axis differs by ≥ 1 |
| 2 chain | the same, with each corner that is not P's parent | that axis differs by ≥ 1 |
| 3 extremes | "Is P farther north/east/south/west than all of A, B and C? Answer yes or no." | P differs by ≥ 1 from the most extreme corner on that axis |
| 4 inside | "Is P inside the triangle formed by A, B and C? Answer yes or no." | always, one per scene |

## Data

- **Test:** 400 new scenes (seed 702) with every ladder question.
- **Baseline training:** 2,000 other scenes (seed 701). No model is involved.
- **Answers:** the candidate with the highest log-probability, scored as in Study 1.

## Metric

- **Balanced accuracy on every rung (chance 0.5).** The positive class is "yes" on rungs 3–4 and "the first-named
  object is the answer" on rungs 1–2, where argument order is random.
- **CI:** a 95% scene-bootstrap with 1,000 resamples; resamples missing a class are dropped.
- **Rung 4 is reported three ways:** on all scenes; on the **hard** subset (inside vs box_hard); and per stratum.

## Baselines (rung 4, hard subset)

- **Box rule** (map-free): "inside" iff P is inside the bounding box. It is at chance on the hard subset by
  construction.
- **Feature rule** (map-free): logistic regression on the stated offsets only (B − A, C − B, P − its parent, and which
  corner is P's parent), fitted on the training scenes.
- **Magnitude rule** (map-free): standardised logistic regression on size cues: |dx| and |dy| of each stated sentence,
  the length of P's offset, the three side lengths, the area, and which corner is P's parent. It knows how big things
  are but not where they point. (On the test scenes it scores about 0.69 without any model.)
- **Centroid rule** (reference only, never part of the decision): "inside" iff P's distance to the centroid, divided by
  the mean centroid-to-corner distance, is below a threshold fitted on the training scenes. It needs true positions,
  so it is itself a rough map method.

## Decision rule (rung 4, hard subset)

Let *best map-free* be whichever of the box, feature and magnitude rules has the highest hard-subset balanced accuracy,
and *gap* = model − best map-free, with a **paired** scene-bootstrap 95% CI (same resampled scenes for both).

- **Go:** balanced accuracy ≥ 0.65, its CI lower bound > 0.5, and the gap's CI lower bound > 0.
- **Weak pass:** both CI conditions hold but balanced accuracy < 0.65.
- **Fail:** anything else.

The 0.65 floor guards headroom. Below it, erasing the map has too little room to show an effect in the main study.

## What each outcome leads to

- **Go:** write the main-study plan (probes, erasure by rung, map vs model, moving a corner), each test with its own
  preregistration.
- **Weak pass or fail:** stop and decide with the author. Any change (bigger triangles, other wording) gets a new
  preregistration. Nothing is tuned silently.

## Reported, not part of the rule

- Rungs 1–3. Rungs 1–2 should be roughly at Study 1's levels (a sanity check).
- Rung 4 per stratum.
- The centroid rule.
- The probability the model puts on the allowed answers.

## Revision (2026-11-13, before any data)

A code review found that size-only shortcuts from the stated sentences (how long P's offset is, the triangle's area)
reach about 0.6–0.7 balanced accuracy on the hard subset, while the first version's map-free baselines sat near 0.5,
and that the generator made triangle size depend on P's stratum. Changes: triangles must admit all three strata; the
magnitude rule joins the map-free baselines; "above the baselines" became a paired CI on the gap instead of a point
comparison. No Study 2 data existed when this was changed (git history: the pilot notebook had not been run).
