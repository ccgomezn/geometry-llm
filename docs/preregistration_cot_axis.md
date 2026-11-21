# Preregistration: does step-by-step reasoning fix the one-sentence rule?

*Committed 2026-11-21, before any data of this test exists and before its code is written.*

## Why

Asked "which is farther east / north, C or R?" about a chain R → M → C with a one-word answer, Gemma 3 12B and 27B,
Qwen3-8B and Llama 3.1 8B all follow C's own sentence with a default towards R. On the decisive questions (M's step
larger and opposite) they are right on 14–36%, although a linear probe at the answer position is right on 86–98%
(flow set, `docs/preregistration_flow.md`, `docs/preregistration_cross_family.md`). In Study 2 (triangles), the same
models failed with one word but succeeded when allowed to reason step by step, and every reply wrote explicit (x, y)
coordinates (`docs/preregistration_s2_reasoning.md`).

This test asks whether written reasoning also overcomes the one-sentence rule on the decisive questions. If it does,
the one-word failure is a failure of the fast read-out, not of composing two steps, and the next test can ask whether
the hidden map shapes the coordinates the model writes. If it does not, the models fail to compose two relative steps
even in text.

## Models

- **Gemma 3 12B** instruct (the paper's main model): bfloat16 on a GPU with at least 35 GB; otherwise 8-bit weights
  (bitsandbytes LLM.int8, activations in bfloat16). Which one was used is reported.
- **Qwen3-8B** (non-thinking mode, as in all earlier Qwen runs; the reasoning is asked for in the reply) and **Llama
  3.1 8B Instruct**, bfloat16.

## Questions

The flow set's generator (`axis_bulk_items`, seed 1601, the flow test's question counts) for each model's tokenizer.
From it, the first questions of each kind, in generation order: **300 decisive** (*conflict, M larger*) and **75 each**
of *own step only*, *agree*, *first step only* and *conflict, C larger* (600 per model).

## Conditions (same loaded weights, same questions)

- **One word.** The flow prompt as before ("… Answer with one word."); the answer is the sign of logit(C) − logit(R)
  at the final position.
- **Step by step.** The same prompt with "Answer with one word." replaced by: "Think it through step by step, then end
  your reply with a final line of the form 'Answer: X', where X is one of the two names." Greedy decoding, up to 1024
  new tokens. The answer is the name in the last "Answer: …" line; a reply without one, or naming neither object,
  counts as wrong.

## Measures and criteria (per model)

Balanced accuracy (mean of the accuracies on questions whose answer is C and on those whose answer is R), with 95%
scene-bootstrap intervals; differences are paired over the same questions.

- **reasoning fixes the rule**: on the decisive questions, step-by-step balanced accuracy at least 0.80 with its lower
  bound above 0.5, the paired difference (step by step − one word) with its lower bound above 0, and step-by-step
  balanced accuracy on the other four kinds pooled at least 0.80 (so that the gain is not bought by breaking the easy
  questions);
- **reasoning helps**: otherwise, if the paired difference on the decisive questions has its lower bound above 0;
- **no help**: otherwise.

The headline is Gemma 3 12B; the other two models are reported with their own verdicts.

## Reported, not part of the criteria (exploratory)

- Accuracy by kind of question in both conditions; replies without a final answer; replies that hit the token limit.
- **Written coordinates.** Share of replies that write at least one (x, y) pair, and accuracy with and without them.
- **Written map.** For replies that give coordinates to both C and R, whether the written positions put C on the
  correct side of R on the asked axis, and whether the final answer agrees with the written positions. This separates
  errors in building the written map from errors in reading it.
- The one-word condition's accuracy compared with the bfloat16 flow run of the same model (a check on the 8-bit
  weights, when used).

## Limits

Greedy decoding with one instruction wording; three models of 8–12B. A positive result shows that the models can
compose the two steps when they write their reasoning, not that the written reasoning uses the hidden map.
