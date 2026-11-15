# Preregistration: flow map, where is the first step lost on the way to the answer?

*Committed 2026-11-15, before any data of this test exists. Gemma 3 12B instruct, bfloat16 (Colab, one GPU).*

## Why

On "Which is farther east, C or R?" about a chain R → M → C ("M is 3 units east of R. C is 2 units west of M."), the
model answers with the direction of C's own sentence (axis bulk test): 0.97–0.98 when that direction gives the answer,
**0.53 when C's sentence says nothing about the asked axis and only M's step decides**, 0.34 when the steps conflict
and M's is larger. Yet the sum-of-parts test showed that the end of C's own sentence holds C's position combined from
both sentences (distance R² 0.61, beyond any side-by-side store), and that site is the one whose copy moves the answers
(P3). E3 and P8 showed that copying C's sentence from a run where only M moved carries little of M's step into the
answers.

So the first step (R → M) is in the model but drops out somewhere between the description and the answer. The
hand-off results (P3q) give the route the answer takes: the sentences feed the question's tokens around layers 16–31,
and those feed the final position around layers 32–46. This test locates where along that route the first step is
lost, by measuring at each hop and layer both whether it is *present* (probes) and whether it *moves the answer*
(patching), next to the same measurements for C's own step, which does get through.

## Questions

C–R axis questions only ("which is farther east / north, C or R?", either order), new 3-object chains (seed 1601),
built as the axis bulk test (`axis_bulk_items`), split into its five cells by the two steps on the asked axis
(d1 = R → M, d2 = M → C): own step only 400, agree 400, first step only 600, conflict C larger 400, conflict M larger
600 (2,400). *Truth* = sign(d1 + d2), +1 when C is farther in the asked direction. Everything below is in this C frame,
independent of which object the question names first. Scenes split 80/20 for the probes (`scene_split`, seed 0).

## Measurements

**1. Presence.** Residual stream at every second layer (0, 2, …, 48) at six token positions: the last token of M's
sentence, the last token of C's sentence, C's name in the question, R's name in the question, the token after the
second-named object, and the final position (where the answer is read out). At each position and layer, ridge probes
(scene-grouped CV penalty) for d1, d2 and truth, fitted **separately for east and for north questions** (the asked
axis decides which step value is relevant) and averaged; held-out R² for d1 and d2, held-out accuracy for truth.

**2. Use (flow).** For the first 100 questions of each cell, two counterfactual prompts with the same words except one
direction word: C's own step flipped on the asked axis ("own"), or M's step flipped on the asked axis ("first"; C moves
with M). Questions where that step is 0 on the axis, or where the flip puts two objects on one point, or where the
prompts do not align token by token, are skipped. The residual stream at one site (M's sentence, C's sentence, the
question tokens before the final position, or the final position) at one layer (0, 4, …, 44, 46) is copied from the
counterfactual run into the clean run. Effect = change in logit(C) − logit(R), signed so that + is the way the flipped
step points, reported as a share of the full effect of the counterfactual prompt. Built-in checks: copying the flipped
sentence at layer 0 reproduces the full effect exactly.

**3. Logit lens.** Final norm and unembedding applied to the final position's residual at each layer: the share of
questions per cell where logit(C) − logit(R) points to the right answer. Check: the last layer matches the model's
answer (correlation ≈ 1).

## Decision rule (presence of the first step at the answer position)

P1 = best held-out R² of d1 at the final position over layers 28–48; P2 = the same for d2.

- **Uninformative:** P2 < 0.5 (C's own step, which does drive the answer, is not linearly readable there either).
- **Lost before the answer position:** P1 < 0.25 (with P2 ≥ 0.5).
- **Reaches the answer position (the read-out ignores it):** P1 ≥ 0.5.
- **Partial:** otherwise.

**Where it is lost** (reported with the verdict): the first hop, in the order C's sentence end (best d1 R² over layers
12–32), question tokens (best over C's name, R's name and the token after them, layers 16–40), final position (P1),
where the best d1 R² is below 0.25.

## Reported, not part of the rule

- Truth-probe accuracy at the final position on held-out *conflict, M larger* and *first step only* questions, best
  layer in 28–48, next to the model's accuracy on the same questions. (The best layer is chosen on the test rows, so
  this is slightly optimistic; a large gap is needed to mean anything.)
- The full presence table, the logit lens per cell and layer, the full effects per cell with scene-bootstrap CIs, the
  shares per site and layer.

## What each outcome means

- **Lost at C's sentence end:** the asked-axis part of M's step is not kept where C is described (the compose test's
  combined position would then be mostly the other axis or distance only).
- **Lost at the question tokens:** C's sentence holds the first step, but what the question tokens take from it is
  only C's own step. The failure is in the transfer (attention heads moving C's relation into the question), which is
  where Stage 2 (heads) would look.
- **Reaches the answer position but the read-out ignores it:** the information arrives where the answer is made and
  the final read-out weights it little. If the truth probe there also beats the model clearly, the model "knows" the
  right answer in its last position and says the shortcut anyway.

## Limits

- Probes show presence, patching shows effect; a linear read of d1 does not show the model computes with it. A truth
  probe can succeed by adding linearly readable d1 and d2, which is still the information the answer needs.
- One model (12B), one question type, 3-object chains.

## Next (not binding)

Stage 2 depends on where the step is lost: attention heads at that hop (which heads move C's information, what they
read, whether any carries M's step), with a held-out causal check.
