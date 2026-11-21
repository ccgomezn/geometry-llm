# Preregistration: can written reasoning use the processed state when it cannot re-read M's sentence?

*Committed 2026-11-21, before any data of this test exists and before its code is written.*

## Why

On the decisive "which is farther east / north, C or R?" questions (chain R → M → C, M's step larger than C's and
opposite), step-by-step reasoning answers correctly (Gemma 3 12B 0.81, Qwen3-8B 0.99;
`docs/preregistration_cot_axis.md`), and when M's sentence and the state computed after it disagree, the reasoning
follows the sentence (`docs/preregistration_readback.md`: index 1.00 and 0.95). The models prefer to re-read the
words. This test asks whether they *can* use the processed state when re-reading is impossible: does the reasoning
still get M's step if, while it writes, it cannot attend to M's sentence?

## Conditions

The step-by-step prompt of the earlier tests, greedy decoding, up to 1024 new tokens, four conditions per question:

| condition | prompt positions after M's sentence | reply tokens |
|---|---|---|
| **normal** | attend to everything | attend to everything |
| **blind reply** | attend to everything (the processed state holds M's step) | cannot attend to M's sentence |
| **floor** | cannot attend to M's sentence | cannot attend to M's sentence |
| **list control** | attend to everything | cannot attend to the object list |

"Cannot attend to" a span means its keys are masked in every layer for those queries. Positions are numbered as in
the normal prompt in every condition (explicit position ids), so masking does not shift later positions. In *blind
reply* the prompt is processed exactly as in *normal*; only the reply is blind to M's sentence. In *floor*, M's step is
not available anywhere the model reads (M's step appears only in M's sentence), which gives the accuracy without it.
The object list ("The objects are: …") carries no step, and its span is of similar length; the *list control* measures
the disruption caused by masking a span as such. M's sentence span is its token span in the prompt; the object list
span runs from the first token after C's sentence to the end of the list.

## Models and questions

Gemma 3 12B instruct (8-bit weights on a 24 GB GPU, bfloat16 on 35 GB or more) and Qwen3-8B (non-thinking mode), as in
the read-back test. The first 200 decisive questions of the flow set's generator (seed 1601) for each tokenizer.

## Measures and criteria (per model)

Balanced accuracy (C-correct and R-correct questions weighted equally; a reply without an answer counts as wrong),
95% scene-bootstrap intervals with the four conditions of a question resampled together.

- **Recovery** = (blind reply − floor) / (normal − floor), with its bootstrap interval.

Verdict:

- **uninformative** if normal is below 0.75, or floor's upper bound is not below normal's lower bound (M's step would
  be reachable another way), or the list control is more than 0.15 below normal (masking a span disrupts the
  reasoning by itself);
- **the processed state is usable** if recovery's lower bound is at least 0.50 and blind reply's lower bound is above
  0.5;
- **the processed state is not usable by the reply** if recovery's upper bound is at most 0.20;
- **partial** otherwise.

## Reported, not part of the criteria (exploratory)

Accuracy by condition; replies without an answer; in blind-reply replies, how often M is given a position relative to R
and whether it is on the correct side, and how often the reply says that M's position is not given or unknown.

## Limits

Two models, one kind of question. A usable processed state shows that information computed from M's sentence can be
read by the reply when the words are hidden, not that it is a two-dimensional map; and the reply can still read C's
sentence and the question, so what it recovers is M's contribution to C's position as the model encoded it after M's
sentence.
