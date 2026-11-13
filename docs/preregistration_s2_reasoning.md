# Preregistration: Study 2, can the model tell whether a point is inside the triangle when it thinks step by step?

*Committed 2026-11-13, before any data of this test exists. Gemma 3 27B instruct, bfloat16 (Colab G4).*

## Why

With a one-word answer, both Gemma 3 12B and 27B are near chance on "is P inside the triangle?" (hard subset 0.51 and
0.54) and below a size rule (0.63). The 12B model's internal map, decoded with linear probes, is at chance too, while
it answers one-dimensional questions better than the model does. This test asks whether the model can solve the task
when it may write out its reasoning first. If it can, the skill exists but needs explicit steps: one-step answers and
the internal map handle order along one axis, not 2-D shape. If it cannot, the task is beyond this model in text.

## Design

- **Questions:** the pilot's 400 rung-4 questions (seed 702), the same scenes and wording
  (`docs/preregistration_s2_pilot.md`).
- **Prompt:** the pilot's prompt with one change. The instruction " Answer with one word." becomes " Think it through
  step by step, then end your reply with a final line of the form 'Answer: yes' or 'Answer: no'." No hint about how to
  reason (for example, no mention of coordinates).
- **Decoding:** greedy, up to 1,536 new tokens.
- **Reading the answer:** the last "Answer: yes/no" in the reply (case and markdown bold ignored). A reply without one
  counts as wrong.
- **Comparisons:** the same model's one-word answers from its pilot; the pilot's size rule (magnitude rule, fitted on
  seed 701).
- **Metric:** balanced accuracy, positive class "yes", 95% scene-bootstrap CI (1,000 resamples). Differences are paired
  over the same resampled scenes.

## Decision rule (hard subset: inside vs box_hard)

1. **Step-by-step reasoning can tell inside:** balanced accuracy ≥ 0.65, its CI lower bound > 0.5, and the CI lower
   bound of (step by step − size rule) > 0.
2. Otherwise, **partial:** its CI lower bound > 0.5 and the CI lower bound of (step by step − one word) > 0.
3. Otherwise: **no**.

## Reported, not part of the rule

- All 400 questions; "yes" rate per stratum; replies with no final answer; replies that hit the token limit; reply
  length.
- Exploratory: how many replies write (x, y) coordinates, and accuracy with and without them.
- The full replies are saved.

## What each outcome leads to

- **Can tell inside:** the result shows the gap between one-step answers (and the internal map) and explicit
  reasoning. Possible follow-up: does reading the decoded map into the prompt help, and which reasoning steps matter.
- **Partial or no:** stop and decide with the author.
