# Preregistration: was the map boost too small? (map boost, dose check)

*Committed 2026-11-22, after the map boost results (`docs/preregistration_map_boost.md`, results 30321f6) and before
any data or code of this check. Gemma 3 12B it, chat format, same setup as the map boost run.*

## Why

Amplifying the map's directions at the answer position did nothing. In both arms (map, M's step), at layers 28 and 32
and α ∈ {2, 4, 8}:

- the decisive questions stayed at 0.32;
- the answer margins moved by about 0.3 logits (median decisive margin about 12), no more than for random subspaces
  of the same variance;
- at most 0.6% of answers flipped.

Two readings remain open:

- **Not read.** The pathway that writes the answer does not read these directions at the answer position.
- **Too small.** At α = 8 the edit adds only about 2–3% of the mean residual norm. An edit that small might not move
  the answer along any direction, so the null says nothing about map use.

This check separates the two. It adds a positive control: an edit of the same size along a direction that should
move the answer. It also raises α.

## Design

**Model, sets and directions.** Everything is as in the map boost run:

- the same model and loading rule (bfloat16 on a GPU with at least 35 GB, 8-bit weights otherwise);
- the same fitting set, test set (seed 1701, including the transfer questions) and layers (28, 32);
- the same located subspaces S (both arms) and the same three variance-matched control subspaces per arm and layer,
  reused from the boost run's cache. They do not depend on α.

The clean answers are scored again in this run, and all gains are measured against them.

**Dose unit.** For each layer, s is the mean size of P(h − μ) for the map arm over the fitting set, as reported by the
boost run (183 at layer 28, 259 at layer 32). The boost at α adds a vector of mean size (α − 1)·s. The dose
**D(α) = (α − 1)·s** is used to match the positive control to the boost.

**Positive control: answer direction.** At the same layer and position as the boost, add a fixed-size vector along
the direction in the residual stream that most directly raises the true answer's log-probability over the other name:

- u ∝ (1 + g) ⊙ (W_U[t_true] − W_U[t_other]), unit length;
- g is the final RMSNorm weight (Gemma's norm scales by 1 + g);
- W_U is the unembedding;
- t_true and t_other are the first tokens of the two candidate names, exactly as the scorer reads them.

The added vector is D(α)·u, for α ∈ {2, 4, 8, 16, 32, 64}. It applies to all C–R axis questions of the test set.

This control uses the answer label, so it is not a method. It only asks whether an edit of this size, at this layer
and position, can move the answer at all.

**Larger α.** The two arms and their controls are run as in the boost test, with α ∈ {16, 32, 64}, at both layers, on
all test questions (C–R axis and transfer).

**Measures.** As in the boost test:

- balanced accuracy per question kind, boosted minus clean;
- paired scene-bootstrap intervals (2,000 resamples);
- for the arms, the contrast with the mean of the three controls.

## Criteria

**1. Can an edit of the boost's size move the answer?** At each layer, the positive control at dose D(8) (the largest
boost already run):

- **moves the answer** if the decisive gain is at least +0.20 and its 99% interval is above +0.10;
- **does not move the answer** otherwise.

**2. Does a stronger boost help?** For each arm, the six new conditions (2 layers × α ∈ {16, 32, 64}) are judged with
the boost test's rule for whether a condition works, unchanged:

- decisive gain ≥ +0.10;
- contrast ≥ +0.05 with its 99% interval above 0;
- first-step-only gain with its 99% interval above 0;
- no easy kind down by more than 0.05.

The arm's verdict at higher dose uses the boost test's three labels (wins, no benefit, partial), over these six
conditions only.

**Overall verdict.**

- **The map is not read (dose-checked):** the positive control moves the answer at D(8) in at least one layer, and in
  that layer no condition of either arm works at any α (old or new).
- **The map wins at a higher dose:** some new condition works in either arm. In addition, the preregistered transfer
  test of the boost prereg is run for the working condition with the largest decisive gain.
- **Still too small to tell:** the positive control does not move the answer at D(8) in either layer, and no new
  condition works.
  - The dose at which the positive control first meets the criterion-1 threshold is then reported.
  - Whether the arms were tested at or above that dose is reported too.
  - If they were and no condition works, the conclusion is "not read at a dose that moves the answer". If they were
    not, the question stays open.

## What each outcome means

- **Not read (dose-checked):** an edit of this size can move the answer, but not when it lies along the map's
  directions. The answer pathway does not consult the map at the answer position. Connecting the map to the answer
  needs something the model does not already have, which is the framework's learned-connector stage.
- **Wins at higher dose:** the map can drive the answer if it is made much louder. The next step is to check that this
  is map use and not damage (prove step, and the transfer test).
- **Still too small:** the boost test was underpowered by design. Gemma's residual stream at these layers is very
  large, so the boost needs a dose scale that first moves the answer through some known direction.

## Reported, not part of the criteria (exploratory)

- The positive control's decisive gain at every dose and both layers, as a dose–response curve.
- The arms' and controls' margin shift per unit of dose, compared with the positive control's.
- At which α the controls first cost the easy kinds more than 0.05 (where generic damage begins).
- The share of decisive answers following C's sentence and M's sentence at each new condition.

## Limits

- One model, the same scenes and directions as the boost test. The test set is reused, so this is a check on that
  test, not a fresh sample.
- The positive control reads the answer label. It measures what an edit of a given size can do, not a way to improve
  the model.
- The answer direction ignores the effect of the final norm's division by the residual's size. Gemma's final residual is
  large, so the same size of edit may matter less there than at the logit.
