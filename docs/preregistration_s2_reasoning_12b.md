# Preregistration addendum: step-by-step reasoning on Gemma 3 12B

*Committed 2026-11-14, before any 12B data of this test exists.*

## Why

On Gemma 3 27B, step-by-step reasoning tells inside from outside (hard subset 0.73, vs 0.54 with one word;
`docs/preregistration_s2_reasoning.md`), and the coordinates it writes are almost always correct. The next test asks
whether the written coordinates come from the model's internal map, by erasing that map during reasoning. The probes
and erasure tools exist for 12B, so this run checks that 12B can do the task step by step first.

## What is the same

Everything in `docs/preregistration_s2_reasoning.md`: the 400 rung-4 questions (seed 702), the prompt, greedy decoding
with up to 1,536 new tokens, the answer parsing, the comparisons (the 12B pilot's one-word answers and the size rule),
the metric and the decision rule. Script: `scripts/s2_reasoning.py --model gemma3-12b`.

## What differs

- Model: `google/gemma-3-12b-it`, bfloat16, on a Colab A100 (the whole model fits).
- Batch size 8 instead of 16 (memory for long replies). Greedy replies do not depend on it beyond floating-point noise.

## What each outcome leads to

- **Can tell inside:** design the erasure-during-reasoning test on 12B, under its own preregistration.
- **Partial or no:** decide with the author (for example, run that test on 27B instead).
