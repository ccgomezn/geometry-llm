# Preregistration: sum-of-parts test, does the model combine the two sentences, or only store them side by side?

*Committed 2026-11-15, before any data of this test exists. Gemma 3 12B instruct, bfloat16 (Colab, one GPU). Probes
only: no interventions, no questions asked.*

## Why

Study 1 decodes every object's position from the hidden states (P2: R² 0.74 at the object's list mention, layer 21),
including two-step objects whose position needs two sentences ("M is 3 east of R", "C is 2 west and 2 north of M").
But a position is a *sum* of stated offsets. If the model kept each sentence's offset side by side and never combined
them, a linear probe could still read the position: the probe would be doing the adding. That reading would explain
why erasing or moving the "map" barely changes the answers (P4 slider, P5, P10, the closed-loop erasure): there
would be no finished map to use, only its parts.

The exploratory P2d (2026-11-10, same model, the P2 scenes) argued against it: distance to the root, which is not a sum
of the offsets, was readable for two-step objects (R² 0.62) where an additive baseline got −0.05. That baseline was
weak, though: it was additive also *within* a sentence (dx and dy coded separately), and the run was not
preregistered. This test repeats it on new scenes with the strongest side-by-side baseline, at the two places that
matter, with fixed layers and a decision rule.

## Scenes

3,000 new scenes (seed 1501), generated as Study 1's main set (3–6 objects, alternating people and object names,
single-token names for the tokenizer, offsets −5…5, compound offsets with p = 0.3). No question is appended: the
prompt is the scene description and the object list, as in P2. Split 80/20 by scene (`scene_split`, seed 0).
**Rows:** two-step objects only (parent is a one-step object), about 2,850 train and 700 test rows.

## Targets (not sums of the offsets)

For each two-step object, relative to the root: **distance** (primary), **|x|** and **|y|** (secondary).

## The parts baseline (the best a side-by-side store can do)

One one-hot code per sentence over that sentence's joint (dx, dy) (121 values each): the object's own sentence and its
parent's sentence, 242 features. A linear readout of them can express any function of the own sentence plus any
function of the parent's sentence, so it is the best case for "each sentence kept separately". It is fitted exactly
like the probe (ridge, penalty by scene-grouped CV, train rows only). Expected from the scenes alone (computed before
any model data): distance 0.15, |x| 0.21, |y| 0.21. Reference from the true position (x, y, x², y², xy): 0.95, 0.91, 0.92.

## Sites and layers

- **List mention** (the object's name in the object list, P2's site), **layer 21** (best two-step distance layer in P2d).
- **Sentence end** (the last token of the object's own sentence, P2b's site; patching it moves the answers, P3),
  **layer 24** (best layer in P2b).

The primary layers are fixed here from the earlier exploratory runs, so no layer is chosen on the test data. A sweep
over layers 0, 6, 12, 14, 16, 18, 21, 24, 27, 30, 36, 42, 48 is reported, not part of the rule.

## Metric

Per site, target and layer: held-out R² of a ridge probe on the activations, R² of the parts baseline, and the **gap**
= probe − baseline, with a paired 95% scene-bootstrap CI (1,000 resamples; the same test scenes for both).

## Decision rule (distance, the primary layer of each site)

- **Composed by the model** at a site: gap ≥ 0.15 and CI lower bound > 0. The verdict lists the sites that pass.
- **Parts only:** at both sites the gap's CI upper bound < 0.05.
- **Undecided:** anything else.

## What each outcome means

- **Composed:** the model itself combines the two sentences into something non-additive (a position, a distance, or a
  joint code of both steps), so the map is not an artefact of the probe. Composed at the list mention but not at the
  sentence end would mean the combined position lives where the answers do not read (P3, P7), fitting the
  "built but not read" account. Composed at the sentence end too would mean it is present where the answers read,
  and the shortcut ignores it there.
- **Parts only:** the decodable map of two-step objects is assembled by the probe; the model never built it. The paper
  would say so, and the "unused map" becomes "no composed map".

## Limits (stated now)

- Beating the baseline shows an *interaction* of the two sentences. It does not by itself show a metric map; a joint
  code of the two offsets would also beat it. The reported generalisation check below speaks to this.
- One model (12B). Probes, so this is about what is present, not what is used.

## Reported, not part of the rule

- |x| and |y| at the primary layers; the layer sweep at both sites.
- Shuffled-target control at the primary layers (train targets permuted; expected R² ≤ 0).
- **Unseen combinations:** R² of probe and baseline on test rows whose (own offset, parent offset) pair never occurs in
  the training rows. A probe that reads a smooth position should hold up there; one that reads a lookup code of seen
  pairs should not.
