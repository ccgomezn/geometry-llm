# Preregistration: does the map win if it is made louder at the answer position? (map boost)

*Committed 2026-11-22, before any data of this test exists and before its code is written. Gemma 3 12B it, chat
format.*

## Why

This is the first test of a find–amplify–prove framework: locate where the internal map lives, make it count in the
answer, then show that the answer really comes from it.

What we know about Gemma 3 12B on the "which is farther east / north, C or R?" questions (chain R → M → C):

- At the final (answer) position, M's step and C's own step are both linearly readable by layer 28 (R² 0.84 and 0.89).
  A truth probe there gets 0.86 of the decisive questions; the model gets 0.32
  (`docs/preregistration_flow.md`).
- The answer is written in layers 32–36 by about 20 late components. They vote on C's own sentence; "brakes" damp
  them (`docs/preregistration_components.md`, `docs/preregistration_components2.md`).
- Removing the shortcut writers lifts the decisive questions only from 0.32 to 0.46.
- The same pattern holds in base models and with worked examples (`docs/preregistration_base_plain.md`,
  `docs/preregistration_base_fewshot.md`).

Earlier pushes toward the map acted elsewhere and failed:

- **P4:** moving the decoded position at layer 21 of the object list changed almost no answers.
- **P10:** boosting attention to the object list gave no benefit.

Neither acted at the answer position, where the flow test later found the composed information arrives. This test
asks: if the map's directions are amplified at the answer position, just before the answer is written, does the map
win the vote?

## Design

**Model and format.** Gemma 3 12B it, chat format, the one-word answer scored as in the flow test (sign of
log p(C) − log p(R)). The model is loaded in bfloat16 on a GPU with at least 35 GB, and in 8-bit weights otherwise.

**Fitting set (locate).** The flow set: seed 1601, the same 2,400 questions.

- Residual stream at the final position, at the two boost layers.
- Ridge probes (same regularization as the flow test) for M's step d1 and C's own step d2, fitted separately on east
  and north questions. This gives four directions per layer: d1-east, d2-east, d1-north, d2-north.
- No answer labels are used, only the two steps.
- The mean final-position activation μ (over the fitting set) is stored per layer.

**Test set.** New chains from the flow set's generator with seed 1701, and nothing from the fitting set:

- **C–R axis questions:** own step only 200, agree 200, first step only 300, conflict C larger 200, conflict M larger
  (decisive) 300.
- **Transfer questions, from the same scenes, never used for fitting:** up to 600 other questions with one-word name
  answers:
  - farther east / north between C and M, and between M and R;
  - farthest / closest object.

**Boost layers.** Two layers, from the flow test's layer grid: L_a, the first layer of the final window (28 in
Gemma 3 12B), and L_b = L_a + 4 (32). The boost is applied at one layer at a time, only at the final position.

**Boosts.** With P the orthogonal projection onto a subspace S (orthonormalized directions), the activation h at the
final position becomes h + (α − 1) · P(h − μ), for α ∈ {2, 4, 8}. Two arms:

- **map:** S = span of all four directions (d1 and d2, east and north). This amplifies the map as a whole, without
  favouring either step.
- **M's step:** S = span of the two d1 directions. This amplifies only the part the answer under-weights.

**Control.** For each arm, layer and α, three random subspaces of the same rank are drawn from the span of the top 64
principal components of the fitting-set activations at that layer. A draw is kept only if its total projected variance
is within ±10% of that of S, so the same α adds about the same amount of change. The control effect is the mean over
the three draws.

**Measures.**

- Balanced accuracy per question kind, boosted minus clean, with 95% and 99% scene-bootstrap intervals (2000
  resamples).
- The contrast: the real boost's gain minus the control's gain, paired by question.

## Criteria

Question kinds: decisive = conflict M larger; easy = own step only, agree and conflict C larger. Each arm has six
conditions (2 layers × 3 values of α), so the intervals used for the decision are 99%.

A condition **works** if all of these hold:

- the decisive gain is at least +0.10;
- the contrast on the decisive questions is at least +0.05, with its 99% interval above 0;
- the gain on first step only is above 0 (its 99% interval above 0);
- no easy kind loses more than 0.05 balanced accuracy (point estimate).

The second and third conditions together rule out a "follow M's sentence" flip; the fourth rules out simply reversing
the rule.

**Verdict per arm.**

- **the map wins when amplified:** at least one condition works.
- **no benefit:** in every condition, the decisive contrast's 99% interval includes 0 or lies below it.
- **partial:** otherwise, reported condition by condition, including help that comes with damage on the easy kinds.

**Generality (secondary, preregistered).** For the arm and condition that work with the largest decisive gain:

- the change on the transfer questions, against the same control;
- the boost **transfers** if transfer accuracy does not drop by more than 0.03 and at least one transfer type gains
  ≥ 0.03 with its 95% contrast interval above 0;
- it is **specific** if no transfer type gains;
- it is **harmful** if transfer accuracy drops by more than 0.03.

## What each outcome means

- **The map wins (map arm):** the information is in a format the answer pathway can use, and the failure is one of
  volume. Making the map louder lets it outvote the rule, without fitting any answer labels. This would be the
  framework's first positive step.
- **Only the M's-step arm works:** the map is usable, but the fix has to target what the answer neglects; the general
  "amplify the map" version is not enough.
- **No benefit:** loudness is not the problem. The pathway that writes the answer does not read these directions, so
  a learned readout is needed (the next stage of the framework).
- **Transfer:** the boost helps questions it was not fitted for, which is the core of a general framework. **Specific
  or harmful:** the locate step generalizes, but amplification has to be tuned per question type.

## Reported, not part of the criteria (exploratory)

- All kinds at every condition, for both arms and the control.
- The weight of M's step relative to C's own step in the margin, clean and boosted.
- The share of answers following C's sentence and M's sentence.
- Logit lens at the later layers under the best condition.

## Limits

- One model (Gemma 3 12B, chat format), three-object chains, one-word answers.
- The probe directions come from the C–R axis questions, so the transfer test is demanding by design.
- A positive result shows the amplified directions can drive the answer. It does not yet show that the model then
  "uses its map" in the read-back / blind sense; that is the framework's prove step, to be preregistered if this
  works.
- Gemma in 8-bit (if run on a small GPU) differs from the bfloat16 flow run in precision.

## Clarifications (added with the code, before any data)

- **Control pool.** Variance-matched random subspaces need the top 64 components to be able to reach S's variance.
  If S's projected variance is below that of the 64th–61st components, the pool doubles (128, 256, …) until a match
  is possible. The report states the number of components used.
- **Draws.** Random draws tilt the Gaussian rows by a random power of the component variances, so that both low- and
  high-variance targets can be matched; a draw is kept only if within ±10%.
- **Not computed.** The exploratory logit lens is not computed in this run.
