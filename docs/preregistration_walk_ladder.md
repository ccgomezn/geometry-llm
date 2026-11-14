# Preregistration amendment: walk pilot on larger models (Gemma 3 27B, Qwen3-32B, Llama 3.3 70B)

*Committed 2026-11-15, after the Gemma 3 12B walk pilot and before any data from these three models. Everything not
stated here is as in `docs/preregistration_walk_pilot.md`. Answers only: no probes, no interventions.*

## Why

Gemma 3 12B failed the walk pilot: it did not track where a walk had brought it back to, not even after "one block
out, one block back" (0.10–0.53 where naming the last object seen is always right). This run asks whether the
ability appears with scale, in models that answer without writing out their reasoning. Only if some model can do
the task does the question "does it read its internal map?" make sense for it.

## Models (run in this order, one A100 80 GB, H100 or G4 GPU)

| Model | Weights | Notes |
|---|---|---|
| Gemma 3 27B instruct | bf16 | same family as Study 1 |
| Qwen3-32B | bf16 | thinking switched off (`enable_thinking=False`), so no written reasoning |
| Llama 3.3 70B Instruct | 8-bit (bitsandbytes) | 140 GB in bf16 does not fit one GPU; the activations stay bf16 |

Reasoning models that always think aloud (e.g. gpt-oss) are excluded: their position tracking would sit in the
written text, not in the hidden state this programme studies. If the Hugging Face account has no access to Llama 3.3
70B, the other two run and the report says so.

## Items

The **same 2,400 walks** as the 12B pilot (seed 1401, same landmark words), in the same two formats. In Qwen3's
tokenizer 5 of the 44 words (barrel, fence, kite, piano, shoe) take two tokens; they are scored over all their tokens
(summed log-probability), as Study 1 scores multi-token answers. The script reports which words are split for each
model.

## Amendment to the baselines (disclosed: found after seeing the 12B data)

The 12B pilot's only passing cell (prefill, N = 14, ℓ = 12) was fully explained by a map-free rule the first list
missed: **the first object introduced with "There is a … here"**. That differs from "the first object seen" when an
object stands at the start. The baselines therefore add **announced k**, the k-th object introduced with "There is",
for k = 1…4. With them, the 12B verdict is **fail** (re-scored from the saved 12B answers). The cell N = 14, ℓ = 12 can
no longer pass, because "announced 1" is always right there by construction.

## Decision rule

Per model, unchanged: a cell passes at accuracy ≥ 0.60 with the gap's CI lower bound > 0 (gap = model − best
map-free baseline, now including announced 1–4). **Go** if at least one cell with ℓ ≥ 4 passes; **weak pass** if
none passes but some ℓ ≥ 4 cell has a gap CI lower bound > 0; **fail** otherwise.

## What each outcome leads to

- **At least one model goes:** the main study (probes for the running position, erasure against the matched random
  control, the ℓ ≥ 4 vs ℓ = 2 contrast, patching) is preregistered for the **smallest** model that goes, in its chosen
  cell, because the interventions are cheapest there.
- **No model goes:** walk tracking without written reasoning is absent up to 70B, at least in this task. That result
  goes in the paper. The next step is decided with the author; the leading option is a small transformer trained from
  scratch on the task's answers only (no map supervision) with a shortcut dial.

## Reported, not part of the rule

- Accuracy against ℓ and N per model and format: the scale curve, including the 12B run.
- The ℓ = 2 control per model.
- Every baseline per cell, and the probability each model puts on the 4 allowed answers.
