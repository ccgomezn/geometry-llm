# Preregistration: is the outvoted map a product of instruction tuning? Base models in plain text

*Committed 2026-11-22, before any data of this test exists and before its code is written.*

## Why

Every result so far comes from instruct models answering in their chat format. On the decisive "which is farther
east / north, C or R?" questions (chain R → M → C, M's step larger than C's and opposite), Gemma 3 12B, Qwen3-8B and
Llama 3.1 8B Instruct follow C's own sentence and are right on fewer than half (balanced accuracy, flow set, chat
format: 0.37, 0.37, 0.19), although M's step reaches the answer position and a linear probe there is right far more
often (`docs/preregistration_flow.md`, `docs/preregistration_cross_family.md`). Written reasoning re-reads the
sentences, and cannot use the processed state when they are hidden (`docs/preregistration_readback.md`,
`docs/preregistration_blind_reply.md`).

So far we do not know whether the one-sentence rule comes from pretraining or is added by instruction tuning and its
answer format. This test asks the pretrained models, as they are, in plain text.

## Runs

Each run uses one model in one format:

| family | base model, plain text | instruct model, plain text (format control) |
|---|---|---|
| Qwen3 | `Qwen/Qwen3-8B-Base` | `Qwen/Qwen3-8B` |
| Llama 3.1 | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma 3 | `google/gemma-3-12b-pt` | `google/gemma-3-12b-it` |

Gemma 3 12B is loaded in 8-bit weights on a GPU with less than 35 GB, and in bfloat16 otherwise. The other models are
loaded in bfloat16. The instruct models in their chat format are the existing flow runs
(`results/flow/*_flow_seed1601_*`); they are not rerun.

**Plain-text prompt.** The tokenizer's begin-of-text token comes first, if it has one. Then the same scene text as the
chat prompt (the relation sentences, a blank line, "The objects are: …."), a blank line, `Question: ` followed by the
question, a newline, and `Answer:`. There is no chat template, no "Answer with one word.", and no examples. The answer
is the sign of log p(" C") − log p(" R") at the final position (each name with its leading space; every name is a
single token in these tokenizers).

**Questions.** The flow set's generator with the same seed and counts: seed 1601; own step only 400, agree 400, first
step only 600, conflict C larger 400, conflict M larger 600. A base model and its instruct model share a tokenizer, so
they get the same questions as the chat-format flow run of that family.

**Probes.** The residual stream at the final position, at every second layer. Ridge probes are fitted separately for
east and north questions, on the same 80/20 scene split (seed 0) as the flow test. Measures:

- the best held-out R² of M's step (d1) and of C's own step (d2) in the final window (7/12 of the depth to the last
  layer, `layer_grid`);
- the truth probe on the held-out decisive questions, at the best layer of the final window. This is chosen as in the
  flow test, and so is slightly optimistic.

Batch size is a memory setting only.

## Measures and criteria (per run)

**Behaviour.** Balanced accuracy per question kind (C-correct and R-correct questions weighted equally), with 95%
scene-bootstrap intervals (1000 resamples).

**Verdict.**

- **uninformative** if the lower bound on *own step only* is at most 0.60. On these questions M's step is 0, so a
  model that does not answer them from the text cannot be judged on the others.
- **reads the composed position** if the lower bound is above 0.5 both on the decisive questions and on *first step
  only*. On first step only, C's own step is 0 and the answer is M's step; the one-sentence rule is at 0.5 there.
- **the map is outvoted** if all of these hold:
  - the upper bound on the decisive questions is below 0.5;
  - the best R² at the final position is at least 0.5 for both M's step and C's own step;
  - on the held-out decisive questions, the truth probe's Wilson 95% lower bound is above the Wilson upper bound of
    the model's accuracy.
- **partial** otherwise, reported item by item.

**Reading, per family, stated in advance.** The instruct model in chat format fails in all three families.

- If the base model **reads the composed position**, the rule comes with instruction tuning or its answer format. The
  instruct model in plain text separates the two:
  - if it also reads the composed position, the format brings the rule;
  - if its map is outvoted, instruction tuning does.
- If the base model's **map is outvoted**, the failure is already in the pretrained model.

**Overall.**

- **The failure comes from pretraining** if the map is outvoted in every informative base model.
- **Instruction tuning or the format brings the failure** if every informative base model reads the composed
  position.
- **Mixed** otherwise, listed per family.

If a model cannot be loaded (gated weights without access), the verdict rests on the others and says so.

## Reported, not part of the criteria (exploratory)

- Plain accuracy per kind.
- How often the answer follows the one-sentence rule (C if C's own step points the asked way, else R) on the kinds
  where the rule and the truth disagree.
- Agreement between a base model's answers and its instruct model's chat answers, question by question.
- The share of answers naming the first-named object.
- The probe's accuracy at the first layer of the final window, and its mean over the window.

## Limits

Zero-shot, closed-set scoring between the two names: a base model may prefer other continuations, and only the
relative probability of the two names is read. The task, the layers and the probes are those of the flow test, so the
results are comparable with it but share its scope (three-object chains, one kind of question, about 8–12B
parameters). Gemma in 8-bit on a small GPU differs from the bfloat16 chat-format run in precision as well as format.
