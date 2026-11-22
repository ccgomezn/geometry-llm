# Design: connect the map to the answers (development phase)

*2026-11-23. A working design, not a preregistration. The confirmation phase gets its own preregistration once the
recipe is frozen.*

## Goal

We want a general claim of this form: LLMs already build an internal map of a described layout, and a light
fine-tune that connects that map to the answers improves their geometry, including on tasks the fine-tune never saw.

What we know so far (Gemma 3 12B):

- The map is computed and carried to the answer position. Held-out R² at layer 28 is 0.81 for M's step and 0.83 for
  C's own step; a truth probe there gets 0.86 of the decisive questions, against the model's 0.32.
- The answer comes from a rule instead: it follows C's own sentence.
- Amplifying the map at the answer position does nothing, even at doses where a push of the same size along the
  answer direction flips nearly every answer (map boost and dose check).

So the map is present but not connected. The question is whether a small amount of training connects it, and whether
the connection generalizes.

## What counts as success

All four of these, in the confirmation phase:

1. **Transfer.** Gains on question types, scene sizes and a benchmark the training never touched.
2. **Through the map.** The mechanism checks below show that the fine-tuned model answers from the map, not by
   re-reading the sentences better.
3. **No damage.** No real loss on easy spatial questions or on general, non-spatial questions.
4. **Second model.** The same frozen recipe works on Qwen3-8B.

A fine-tune that improves answers but fails check 2 is reported as "improves, but not through the map". That is
useful, but it is not the headline.

## Two phases

**Development (Gemma 3 12B only).** We iterate freely: training mix, LoRA size and layers, learning rate, data size.
Every round is logged in `results/connect_dev/`, failures included. Only the development sets below may be looked at.

**Confirmation.** We freeze the recipe, preregister it, then run it once on the confirmation-only sets and on Qwen3-8B.
Nothing in those sets is evaluated during development. That separation is what makes heavy iteration legitimate.

## Data

All data comes from our scene generator, in chat format with one-word answers (where the map is outvoted). Seeds are
disjoint from every earlier study and between the splits.

| split | contents | used in |
|---|---|---|
| train | 3-object chains (R → M → C) plus 2- and 3-object scenes with direct relations; "which is farther east / north" for every pair (C–R, C–M, M–R), all five C–R kinds (own step only, agree, first step only, conflict C larger, conflict M larger); about 10–20k questions | development |
| dev test | new scenes of the same kind, plus new object names | development |
| dev transfer | farthest / closest (two-dimensional distance, never trained) | development |
| confirmation only | compass direction questions; "is X closer to Y than Z?"; 4- and 5-object chains (deeper composition); StepGame (Shi et al., AAAI 2022; dataset and citation to be checked before use); Qwen3-8B | confirmation |

The training mix contains one-step questions and both conflict kinds. A model that learns "follow M's sentence" or
"follow C's sentence" therefore cannot fit it; it has to combine the steps.

## The fine-tune

- **LoRA on Gemma 3 12B it**, loss on the answer token only.
- **Starting point: LoRA only on layers 28 and above,** after the map has reached the answer position. The fine-tune
  then cannot build a new map in earlier layers, and must work with the one that is already there.
- **Development knobs:** rank (8–32), layers (≥ 28 or all), attention and/or MLP, data size, learning rate.
- **Compute:** A100 in bfloat16, about 1–2 hours per round. 4-bit QLoRA on an L4 only if no A100 is available, noted
  per round.

## Mechanism checks (every development round)

- **M1, the map is still there.** Held-out R² of M's step and C's own step at the answer position, layer 28, before
  and after the fine-tune.
- **M2, the map now drives the answer.** The dose check rerun on the fine-tuned model. Before, amplifying the map did
  nothing. Connected, it should move the decisive answers well beyond the variance-matched random controls.
- **M3, no re-reading.** From layer 28 on, block the answer position (and the question tokens) from attending to M's
  sentence. The map has arrived by then, so a model that reads it should stay right on the decisive questions. A
  model that re-reads M's sentence should fall back to the rule. Blocking attention to an unrelated sentence is the
  control.
- **M4, erasing the map hurts.** Project the map's directions out at layer 28. The fine-tuned model's decisive accuracy
  should fall toward the rule, more than with a variance-matched random erasure.
- **Comparison fine-tune (when a recipe looks promising).** Train the same LoRA while the map directions are projected
  out during training. If it learns just as well, it found another route, and the map is not the mechanism.

## Development bar (to move to confirmation)

- In-distribution decisive accuracy at least 0.80 (base 0.32).
- A dev-transfer gain on farthest / closest over the base model.
- M2, M3 and M4 point to map use.
- No easy kind down by more than 0.05.

If no recipe reaches this bar after about ten rounds, we write up what the rounds showed. For example: "the map can be
used but only question by question", or "fine-tuning finds a re-reading route instead".

## Risks

- **A better re-reading.** The fine-tune may learn to combine the sentences directly, as step-by-step reasoning does.
  M3, M4 and the comparison fine-tune are there to catch this.
- **A comparison-only rule.** The fine-tune may learn a rule for farther-east/north questions only. The transfer sets
  catch this.
- **The map is noisy (R² about 0.8).** A map-reading model may stay below what careful step-by-step reasoning gets
  (0.81–0.99). We compare against the one-word baseline, the setting where the map is outvoted today.
- **General damage.** A small set of non-spatial questions is scored before and after.

## First development round

1. Data generator for the splits above, with tests (seeds disjoint, labels from geometry, both conflict kinds present).
2. LoRA training script (Colab A100), saving the adapter to Drive.
3. Evaluation: dev test, dev transfer, the easy kinds, M1–M4 and the general check, in one notebook run.
