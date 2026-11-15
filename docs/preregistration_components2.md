# Preregistration: Stage 2b, removing the shortcut while keeping the names; who rewrites it?

*Committed 2026-11-15, before any data of this test exists. Gemma 3 12B instruct, bfloat16, eager attention (Colab,
one GPU).*

## Why

Stage 2 (`docs/preregistration_components.md`, verdict *ineffective*) found by exact attribution that the shortcut
("answer C iff C's own sentence points the asked way") is written at the final position by a few late attention heads
(layers 41–47; the top 10 write +36 logits of it, the model's net is +22). Replacing them with their mean output did
not remove it, for two reasons seen after the fact: these heads copy names, so their mean output adds a bias that
depends on which names C and R are (81–86% of the change it caused), and the shortcut's coefficient came back at full
size, so something downstream rewrites it (self-repair).

This stage removes the shortcut from those heads with an intervention that keeps the names, asks again whether the
rest of the model then answers from both steps, and measures which components rewrite the shortcut.

## Questions

New C–R axis questions built as in Stage 2 (`axis_bulk_items`, seed 1901): *own step only* 300, *agree* 300,
*conflict, C larger* 300, *conflict, M larger* 600. *First step only* is left out: C has no step on the asked axis
there, so it has no twin. Each question gets a **twin**: the same scene and question with C's own step on the asked
axis negated (M's sentence, the names, the other axis and the wording otherwise unchanged; `flow.flip_step`). In the
twin, "C's own sentence points the asked way" is reversed. On *conflict, M larger* the twin has the same correct
answer (M's step is larger), so the twin differs only in what the shortcut says. Questions whose twin puts two
objects on the same point are dropped (count reported).

## Components (fixed in advance, from Stage 2's discovery set)

Own-direction writers, top 20 by own_yes coefficient: L44H10, L47H2, L45H1, L41H0, L41H11, L42H0, L41H5, L41H3, L35H8,
L47H10 (top 10), L47H8, L35H3, L41H4, MLP40, L35H12, L46H4, L47H7, MLP43, L41H1, L32H7. Matched controls: three draws
each of 20 and of 10 random components of the same kind and layer (MLPs within 4 layers), excluding the top 20
own-direction writers and the top 10 first-step writers of Stage 2.

## Interventions (at the final position only)

Heads: their slice of the attention output before the output projection; MLPs: their addition to the residual stream.

- **Twin mean** (the ablation): each selected component's value is replaced by the average of its values in the clean
  runs of the question and of its twin. What flips with C's own direction is removed; names, M's step and everything
  else the two runs share are kept.
- **Swap** (exploratory): each selected component's value is replaced by its value in the twin.

Conditions: clean; twin mean of the top 20 (primary) and of the top 10; twin mean of each control draw (20 and 10);
swap of the top 20 and of the top 10.

## Measures

Each question's logit difference logit(C) − logit(R) at the final position, regressed on own_yes (C's own step
points the asked way), truth (±1), d1 (M's step), d2 (C's step) and the question order. **Shortcut** = the own_yes
coefficient. Truth is in the regression so that a read-out that turns towards the correct answer is not counted as
shortcut (the two are separated by *conflict, M larger*, where they disagree). Scene-bootstrap 95% CIs.

## Decision rule (top 20, twin mean, against the mean of its three control draws)

- **Ineffective:** the shortcut with the top 20 removed is more than half the control's (the model rewrites it, or
  it is written elsewhere).
- **A correct path underneath the shortcut:** otherwise, if accuracy on *conflict, M larger* is ≥ 0.65 with the top
  20 removed, at least 0.15 above the control, CI lower bound of that gain > 0.
- **No correct path underneath:** otherwise, if that accuracy's CI upper bound is < 0.60.
- **Undecided:** otherwise.

## Reported, not part of the rule

- Per-cell accuracy and the regression for every condition; the same rule applied to the top 10.
- **Linear prediction:** the clean logit difference minus the clean own_yes coefficient × own_yes, i.e. what the
  answers would be if the shortcut were removed and nothing else changed (Stage 2: the graded part weighs M's step
  1.20 per unit and C's 0.90). A correct path under this rule may just be that graded part; the prediction says how
  much of it to expect.
- **Swap:** on *own step only* (M's step is 0), how often the answer follows the twin's shortcut, i.e. whether these
  components are sufficient to set the answer.
- **Who rewrites the shortcut:** exact attribution (as in Stage 2) on 600 of the test questions (150 per cell), clean
  and with the top 20 and top 10 removed by twin mean. For every component, the change in its own_yes coefficient,
  summed by group: the removed components; other heads in the same layers (Gemma's post-attention norm rescales a layer's
  heads together); later components; earlier components (must be 0: nothing before the removal changes). The final
  norm's scale, removed vs clean. The top 10 components by increase in own_yes.

## What each outcome means

- **Ineffective:** the shortcut is not stored in these components alone; the attribution says which components take
  over. If they are later heads of the same kind, the model has backup writers (as in other circuits); if the change
  is spread over everything, it is a norm effect.
- **A correct path underneath:** with the shortcut gone from the final read-out, the model's remaining read-out weighs
  both steps and gives the right answer, so the composed position is effectively read once the one-sentence vote is
  silenced. Compare with the linear prediction: if accuracy is near it, the correct path is the graded part already
  seen in Stage 2.
- **No correct path underneath:** removing the shortcut leaves no read-out that turns the two steps into the right
  answer.

## Limits

- Direct effects and self-repair at the final position only; the hop of C's direction onto the question tokens
  (layers 20–24) is not touched.
- The twin also changes C's position (and so its distance to everything); the twin mean removes all of what the
  selected components carry about C's own step, not only the direction word.
- Components selected by a linear regression in Stage 2; one model.
