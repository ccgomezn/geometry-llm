# Preregistration (amendment to `docs/preregistration_base_plain.md`): base models with worked examples

*Committed 2026-11-22, after the zero-shot base/plain results (f071f61) and before this test's code is written.*

## Why

Zero-shot, the base models fail like the instruct models: Qwen3-8B-Base and Gemma 3 12B pt "the map is outvoted";
Llama 3.1 8B base uninformative (own step only 0.57; it names the first-named object in 79% of questions). Overall
verdict: "the failure comes from pretraining". The obvious objection is that a base model given a bare
`Question: … Answer:` does not know what is being asked. This test gives it worked examples first. It has two arms:

- **neutral examples.** These teach the format only: in every example, the one-sentence rule and the truth agree. If
  the maps are still outvoted, the zero-shot format did not hide the use of the map.
- **informative examples.** These include questions where M's step decides the answer, so no one-sentence rule fits
  all of them. This asks whether a few demonstrations make the answer depend on the composed position.

## Runs

Base models only, each in both arms (six runs): `Qwen/Qwen3-8B-Base`, `meta-llama/Llama-3.1-8B`,
`google/gemma-3-12b-pt`. Gemma is loaded in 8-bit weights on a GPU with less than 35 GB, and in bfloat16 otherwise.

**Questions, scoring, probes.** As in `docs/preregistration_base_plain.md`:

- the same 2,400 flow-set questions (seed 1601, same counts);
- the answer is the sign of log p(" C") − log p(" R") at the final position;
- ridge probes at the final position, at every second layer, on the same 80/20 scene split (seed 0).

Batch size is a memory setting only.

**Prompt.** The begin-of-text token (if any), then six worked examples, then the test item in exactly the zero-shot
plain format. Each example is a complete plain item with its correct answer: the scene text, a blank line,
`Question: …`, a newline, `Answer: <name>`. Examples are separated from each other and from the test item by a blank
line. There is no instruction text.

**Examples.**

- **Source.** The examples come from a separate bank: the flow set's generator with seed 7 (three-object chains, C–R
  axis questions, 200 per question kind). None of them are test questions.
- **Slots.** Each arm has six slots, each fixed by a question kind and its correct answer (C = the end of the chain,
  R = the root):
  - **neutral:** own step only (C), own step only (R), agree (C), agree (R), conflict C larger (C), conflict C
    larger (R). The one-sentence rule answers all six correctly.
  - **informative:** own step only (C), agree (R), first step only (C), conflict C larger (R), conflict M larger (C),
    conflict M larger (R). Following C's own sentence fails 3 of the 6; following M's sentence fails 2.
- **Axes.** Slots alternate east/north (3 each).
- **Matching.** For each test scene, each slot takes the first bank question that matches its kind, answer and axis,
  whose scene shares no name with the test scene or with the examples already chosen.
- **Order.** The six examples are shuffled with a generator seeded by the test scene's index. All questions of a test
  scene share its examples.

## Measures and criteria (per run)

Balanced accuracy per question kind, with 95% scene-bootstrap intervals (1000 resamples), as before. The verdict rules
are those of `docs/preregistration_base_plain.md`, with one change. **Reads the composed position** now also needs
the lower bound above 0.5 on *conflict C larger*. A model that learned "follow M's sentence" from the informative
examples would pass the decisive and first-step-only questions, but it fails here.

- **uninformative:** the lower bound on own step only is at most 0.60.
- **reads the composed position:** the lower bound is above 0.5 on the decisive questions, on first step only, and on
  conflict C larger.
- **the map is outvoted:** all of these hold:
  - the upper bound on the decisive questions is below 0.5;
  - the best R² at the final position is at least 0.5 for both M's step and C's own step (final window);
  - the truth probe's Wilson lower bound on the held-out decisive questions is above the model's Wilson upper bound.
- **partial** otherwise, reported item by item.

**Overall, per arm, over the informative base models.**

- **neutral arm:**
  - if the map is outvoted in every informative base model: **the zero-shot format did not hide map use**;
  - if every informative base model reads the composed position: **the zero-shot format hid map use** (the zero-shot
    verdict would then reflect the format, not pretraining);
  - **mixed** otherwise.
- **informative arm:**
  - if every informative base model reads the composed position: **examples that need M's step make the answer read
    the composed position**;
  - if the map is outvoted in every informative base model: **the map stays outvoted even with such examples**;
  - **mixed** otherwise.

If a model cannot be loaded, the verdict rests on the others and says so.

## Reported, not part of the criteria (exploratory)

- Plain accuracy per kind.
- How often the answer follows the one-sentence rule (C's own sentence), and how often it follows M's sentence.
- The share of answers naming the first-named object.
- The weight of M's step relative to C's own step in the margin (least squares on all questions; 1 for a model that
  composes the two).
- Agreement with the zero-shot plain run of the same model, question by question.

## Limits

A positive result in the informative arm would show that a few demonstrations can make the answer depend on both
steps. It would not show that the answer reads the internal composed state rather than re-reading the two sentences;
the probes cannot tell these apart. Six examples make the prompt about seven times longer. The examples differ
between test scenes, so they add variance, but not bias between question kinds. Scope as in the zero-shot test.
