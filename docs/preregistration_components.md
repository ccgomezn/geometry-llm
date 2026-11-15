# Preregistration: Stage 2, which components write the answer, and is there a correct path underneath the shortcut?

*Committed 2026-11-15, before any data of this test exists. Gemma 3 12B instruct, bfloat16, eager attention (Colab,
one GPU).*

## Why

The flow map (`docs/preregistration_flow.md`, verdict: *reaches the answer position, the read-out ignores it*) found
that on "which is farther east / north, C or R?" both steps, and the right answer, are linearly present at the final
position by layer 28 (truth probe 0.86 on *conflict, M larger*, where the model gets 0.32), and that the answer is
written in layers ~32–36. The read-out follows one rule: answer C if C's own sentence points the asked way, otherwise
R; M's step only scales the confidence (≈ 1.9 logits per unit against ≈ 30 for C's direction).

This stage asks which components write that read-out, how each weighs the two steps, and whether, with the
own-direction writers removed, the rest of the model answers from the information it holds (a correct path underneath
the shortcut) or not.

## Questions

Two new sets of C–R axis questions built as in the flow map (`axis_bulk_items`, cells by the two steps on the asked
axis): **discovery** (seed 1701), 300 per cell (1,500); **test** (seed 1801), 300 per cell and 500 in *first step only*
and *conflict, M larger* (1,900). Components are selected on the discovery set only.

## Discovery: direct attribution

At the final position, the residual stream is the embedding plus each layer's attention and MLP addition. With the
final norm's scale fixed per question, logit(C) − logit(R) is the sum of their direct contributions; each attention
addition splits exactly into its 16 heads (Gemma's post-attention norm is a per-question scale), and each head's part
splits exactly into the source tokens it attends to (grouped: chat template, M's sentence, C's sentence, object list,
C's name in the question, R's name in the question, other question tokens, final position). Checks: the parts sum to
the model's logit difference (r > 0.99 expected), and each head's source parts sum to the head.

Each component's contribution (and the model's total) is regressed on **own_yes** (1 if C's own step points the asked
way, else 0), **d1** (M's step, signed, per unit), **d2** (C's step, per unit) and the question order (C named first or
second). A component that reads C's composed position would weigh d1 and d2 equally; the shortcut is own_yes.

**Selection** (discovery only): the **own-direction writers** are the components (heads and MLPs, all layers) with the
largest own_yes coefficient: top 5, 10 (primary) and 20. The **first-step writers** are the top 10 by the d1
coefficient.

## Test: mean ablation at the final position

On the test set, each selected component's output at the final position is replaced by its mean over the discovery
set (heads: their slice of the attention output before the output projection; MLPs: their addition to the residual
stream). Conditions: clean; own-direction writers ablated (5, 10, 20); a **matched control**, three draws of 10
random components of the same kind and layer (MLPs within 4 layers), none of them selected; first-step writers
ablated; first-step writers amplified ×3 around their mean (exploratory).

*Shortcut-right* questions: *own step only*, *agree*, *conflict, C larger* (the shortcut gives the right answer).
*Decisive* questions: *conflict, M larger*, *first step only* (only M's step can give it).

## Decision rule (top 10 own-direction writers vs the matched control)

- **Ineffective:** shortcut-right accuracy falls by less than 0.20 beyond the control (the shortcut is spread over
  more components than this).
- **A correct path underneath the shortcut:** decisive accuracy ≥ 0.65 with the writers ablated, at least 0.15 above
  the control, CI lower bound > 0 (scene bootstrap).
- **No correct path underneath:** decisive accuracy's CI upper bound < 0.60 (removing the shortcut leaves the model
  unable to answer from what it holds).
- **Undecided:** otherwise.

## Reported, not part of the rule

The top components with their coefficients and where their signal comes from (source groups); the model's own
coefficients; the sum over all components except the top 10; accuracy per cell and the regression of the logit
difference for every condition (5, 20, first-step writers ablated and amplified).

## What each outcome means

- **A correct path underneath:** the model contains a read-out that uses both steps, masked by a few own-direction
  writers. The shortcut is a small set of components that win, not the only route.
- **No correct path underneath:** nothing at the final position turns the information it holds into the right answer;
  the model's only read-out of "farther east" is "does C's sentence say east". The composed map is then present but
  never read for this question.
- **Ineffective:** the shortcut is distributed; the coefficients still show who writes it, but no causal claim.

## Limits

- Direct effects at the final position only; components that act through later layers, and the hop into the question
  tokens (layers 20–24), are not ablated here.
- Mean ablation replaces a component with its average, not with "nothing"; removing a large MLP can do generic
  damage, which the matched control is there to measure.
- Selection by a linear regression on one question type; one model.
