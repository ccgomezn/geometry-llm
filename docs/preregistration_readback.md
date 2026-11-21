# Preregistration: where does written reasoning get M's step, from its sentence or from the processed state?

*Committed 2026-11-21, before any data of this test exists and before its code is written.*

## Why

On the decisive "which is farther east / north, C or R?" questions (chain R → M → C, M's step larger than C's and
opposite), the one-word answer follows C's own sentence, although M's step reaches the answer position, where a probe
reads the right answer (`docs/preregistration_flow.md`). Asked to reason step by step, Gemma 3 12B and Qwen3-8B answer
these questions correctly (0.81 and 0.99; `docs/preregistration_cot_axis.md`). This test asks where the written
reasoning takes M's step from: by re-reading M's sentence, or from the state the model computed after it (C's sentence,
the object list, the question and the answer position, where the composed information is).

## Design

For each decisive question, a **flipped** version negates M's step on the asked axis (`flip_step`; objects placed
relative to M move with it, the words are otherwise unchanged), which flips the correct answer. Questions whose flipped
prompt does not have the same tokens outside M's sentence, the same length and the same sentence spans are skipped (as
in the flow test).

The prompt is split into the **prefix** (chat template up to M's sentence; identical in both versions), **M's
sentence** (its token span) and the **downstream part** (every position after M's sentence, to the end of the prompt).
Four conditions, greedy step-by-step replies (the step-by-step prompt of `docs/preregistration_cot_axis.md`, up to 1024
new tokens):

| condition | M's sentence | downstream part |
|---|---|---|
| original | original | original |
| flipped | flipped | flipped |
| mixed A | original | flipped |
| mixed B | flipped | original |

In a mixed condition the prompt with the given M's sentence is run, and the output of every decoder layer at every
downstream position is replaced by the other version's (recorded in a separate pass over the same batch). Since the
downstream part is the end of the prompt, the reply then attends to M's sentence from one version and to the processed
state from the other at every layer, and its first token comes from the downstream version's answer position.

## Models and questions

Gemma 3 12B instruct (8-bit weights on a 24 GB GPU, bfloat16 on 35 GB or more, as in the step-by-step test) and
Qwen3-8B (non-thinking mode). Llama 3.1 8B is excluded in advance: its step-by-step accuracy (0.57) is too low for the
comparison below. Questions: the first 200 decisive questions with a valid flipped version, from the flow set's
generator (seed 1601) for each model's tokenizer.

## Measures and criteria (per model)

The answer is the name in the last "Answer: …" line (the corrected parser); a reply without one is left out of the
following shares and counted separately.

- *acc* = mean accuracy in the original and flipped conditions (each against its own correct answer).
- *f_text* = in the two mixed conditions pooled, the share of answers that equal the correct answer of the version
  that supplied M's sentence.
- **Index** I = (f_text − (1 − acc)) / (2·acc − 1): 1 if the reasoning follows M's sentence as reliably as the model
  answers at all, 0 if it follows the downstream state as reliably. 95% scene-bootstrap interval (the four conditions
  of a question resampled together).

Verdict:

- **uninformative** if acc is below 0.75 or either unmixed condition is below 0.70;
- **the reasoning re-reads the sentence** if the lower bound of I is at least 0.70;
- **the reasoning reads the processed state** if the upper bound of I is at most 0.30;
- **both** otherwise.

## Reported, not part of the criteria (exploratory)

f_text separately in mixed A and mixed B; replies without an answer per condition; for replies that write positions,
whether the written position of M (relative to R, on the asked axis) and the written position of C follow M's sentence
or the downstream state, and whether a reply mixes the two (writes M from the sentence but C from the state, or the
reverse).

## Limits

Two models, one kind of question, one wording. The downstream part holds more than a map: it also carries copies of
M's step and the one-sentence rule, so "reads the processed state" would mean the reasoning uses what the model
computed after M's sentence, not specifically a two-dimensional map.
