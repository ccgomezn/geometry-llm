# Preregistration: does the account hold outside Gemma? Qwen3-8B and Llama 3.1 8B

*Committed 2026-11-20, before any data of this test exists and before its code is written. Instruct models in
bfloat16, the whole model on one GPU (Colab L4, 24 GB, or larger).*

## Why

The paper's account ("The Outvoted Map") rests on two models of one family, Gemma 3 12B and 27B. In both:

- **Behaviour.** On "which is farther east / north, C or R?" about a chain R → M → C, the model follows C's own
  sentence, and when M's step is larger and points the other way (*conflict, M larger*, the decisive questions) it is
  right on fewer than half of them (flow set: 12B 0.40, 27B 0.23).
- **Flow.** M's step reaches the final position, where it can be read linearly (held-out R² 0.84 and 0.89), and a
  linear probe for the correct answer there is right far more often than the model on the held-out decisive questions
  (12B 0.86 [0.78, 0.91] vs 0.32 [0.24, 0.41]; 27B 0.98 [0.93, 0.99] vs 0.25 [0.18, 0.34]).

The obvious objection is that this is a property of one model family. This test runs the same measurements on two
instruct models from two other families with similar size: **Qwen3-8B** (`Qwen/Qwen3-8B`, non-thinking mode, as in
all earlier Qwen runs of this project) and **Llama 3.1 8B Instruct** (`meta-llama/Llama-3.1-8B-Instruct`).

## What is run, unchanged

`scripts/flow_test.py` as committed (`docs/preregistration_flow.md`): seed 1601, the same question counts per kind
(own step only 400, agree 400, first step only 600, conflict C larger 400, conflict M larger 600), the same 80/20 scene
split (seed 0), probes, logit lens, copying and decision rule. Layers and rule windows are scaled to the depth by
`layer_grid`, as for 27B (`docs/preregistration_replication_27b.md`): probes at every second layer, final window from
7/12 of the depth to the last layer. Batch size is a memory setting only.

The questions are drawn by the same generator with the same seed, but names must be single tokens in each model's
tokenizer, so the scenes are not identical to Gemma's; they are identical between reruns of the same model.

The tokenizer and the six probe sites were checked on 100 questions per model before this was written (tokenizers
only, no model run).

## Criteria (stated in advance, applied to each model separately)

1. **Behaviour (the failure).** Clean accuracy on all *conflict, M larger* questions is below 0.5.
2. **Flow.** The flow rule's verdict is *reaches the answer position (the read-out ignores it)*: at the final
   position, within the final window, the best held-out R² of C's own step is at least 0.5 and that of M's step is at
   least 0.5.
3. **Probe beats the model.** On the held-out decisive questions, the truth probe at the final position, at the best
   layer of the final window (chosen as in the paper, and so slightly optimistic), has a Wilson 95% lower bound above
   the Wilson 95% upper bound of the model's accuracy on the same questions.

Verdict per model:

- **replicates** if 1, 2 and 3 hold;
- **the failure does not replicate** if 1 fails (the model is right on most decisive questions); 2 and 3 are then
  reported but not judged, since there is no failure to explain;
- **partial** otherwise, reported item by item.

Overall: **the account holds across families** if both models replicate; if one replicates, the report says which;
if Llama 3.1 cannot be run (no access to the gated weights), the verdict rests on Qwen3-8B alone and says so.

## Reported, not part of the criteria

Everything `flow_test.py` reports (accuracy by kind, R² of both steps by site and layer, logit lens, the copying
shares), and, for criterion 3, the probe's accuracy at the first layer of the final window and its mean over the
window, which involve no choice of layer.

## How the outcome is used

Whatever the outcome, it is reported in the paper's next version and in the reply to reviewers. If the failure does
not replicate in a model whose map does reach the answer position, that model is a contrast case: a model that may be
using what Gemma outvotes. This test does not establish that; it would motivate the follow-up.

## Limits

Two models of about 8B parameters, one task, behaviour and probes only: no component analysis or removal, which on
Gemma needed exact attribution through its sandwich norms and would need new work for each architecture.
