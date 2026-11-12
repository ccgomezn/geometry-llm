# Preregistration: P6, is the map used when chaining gets hard?

*Committed 2026-11-12, before any P6 data exists. Gemma 3 12B (Colab, one A100).*

## Why

P5c erased the linearly decodable coordinates at every token (layers 12–30) and found no cost beyond a random erasure
of the same rank, on 3-object scenes. One explanation is that 3-object scenes are too easy to need a map: every answer
can be reached by chaining at most two stated relations. If the map is a resource the model draws on when chaining is
long, erasing it should start to hurt on questions about objects many steps from the root.

The behaviour screen gives headroom: on 4–5-object scenes, Gemma 3 12B answers questions whose deepest object is 1, 2, 3
and 4 steps from the root with accuracy 0.77, 0.63, 0.59 and 0.56.

## Design

Identical to P5c (`docs/preregistration_p5.md`) except for the scenes and the question grouping.

- **Scenes.** 4 and 5 objects, alternating, from the same generator and settings as the behaviour screen
  (`generate_scene` defaults: compound offsets 0.3, shuffled sentence order 0.2). `closer_than` excluded.
- **Fit.** Object-list activations of 2,000 new scenes (seed 301), every layer 12–30, scene-centred; INLP on the
  (x, y) ridge probe until held-out R² < 0.1 or 20 rounds (rank ≤ 40).
- **Erasure.** Mean-preserving removal of the basis at every token position; configurations: layer 14, 21 or 28 alone,
  and every layer 12–30 with its own basis. Control: a random orthonormal basis of the same rank at the same layers.
- **Test.** 300 new scenes (seed 302); every question passing the 1-unit / 10° margins, at most 24 per scene (a random
  subset, fixed by the seed).
- **Question depth.** The largest hop (steps from the root) among the objects the question names or offers as
  candidates. Groups: *shallow* (depth ≤ 1), *middle* (depth 2), *deep* (depth ≥ 3).
- **Metric.** Accuracy against the layout. Drop = clean − erased; *excess drop* = drop with the probe basis − drop
  with the random basis, same configuration; scene-bootstrap 95% CIs.
- **Headroom check.** Chance for a question is 1 / (number of candidates). Clean accuracy on deep questions must exceed
  their mean chance by ≥ 0.15, or the test is uninformative.

## Decision rule

- **Map used when chaining is hard:** at some configuration, the deep excess drop ≥ 0.10 **and** exceeds the shallow
  excess drop by ≥ 0.05.
- **Still not used:** the deep excess drop ≤ 0.03 at every configuration, the probe's held-out R² after erasure is
  ≤ 0.1 at every erased layer, and the headroom check passes.
- **Uninformative:** the headroom check fails.
- **Unclear:** otherwise.

Reported, not part of the rule: excess drop for the middle group; the probe's R² before erasure by layer; clean
accuracy by depth.

## Caveat stated in advance

As in P5c, only the linear subspace that the list-position probe reads is removed. "Not used" means not used in that
subspace.
