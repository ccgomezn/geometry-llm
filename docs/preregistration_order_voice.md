# Preregistration: what the one-sentence rule follows (sentence order and voice)

*Committed 2026-11-20, before any data of this test exists. Gemma 3 12B instruct, and Gemma 3 27B instruct if the GPU
holds it; bfloat16, behaviour only (Colab, one GPU).*

## Why

On "which is farther east / north, C or R?" about chains R → M → C, both models answer C almost only when C's own
sentence points in the asked direction (flow set, `results/flow/*_items.csv`; `paper/short`). In every prompt so far
C's sentence was also (a) the last sentence of the description and (b) the one sentence whose direction word matches
the question's word exactly when C's step points the asked way. Three rules therefore fit the data equally well:

- **own step**: answer C if C's true step on the asked axis points in the asked direction;
- **word**: answer C if the sentence that mentions C's placement contains the question's direction word;
- **last sentence**: answer C if the last sentence of the description contains the question's direction word.

This test separates them by rendering the same scenes in four ways.

## Questions

New C–R axis questions built as in the flow test (`axis_bulk_items`, seed 2001): *own step only* 300, *agree* 300,
*first step only* 300, *conflict, C larger* 300, *conflict, M larger* 600. Each question is shown in four renderings of
the same scene; positions, names, the object list and the question are identical across them:

| rendering | sentence order | C's relation |
|---|---|---|
| MC-subj (as before) | M's sentence, then C's | "C is d east of M" |
| CM-subj | C's sentence, then M's | "C is d east of M" |
| MC-obj | M's sentence, then C's | "M is d west of C" (same fact, C as object) |
| CM-obj | C's sentence, then M's | "M is d west of C" |

M's sentence is always "M is d₁ … of R". Questions whose renderings fail to tokenise identically for the names (the
answer tokens) are dropped (count reported).

## Features (per question and rendering)

- own\_yes = 1 if C's own step on the asked axis points in the asked direction ($d_2 > 0$); same in all renderings.
- word\_yes = 1 if C's relation sentence contains the question's direction word: $d_2 > 0$ in the subject renderings,
  $d_2 < 0$ in the object renderings, 0 when $d_2 = 0$.
- last\_yes = 1 if the last sentence of the description contains the question's direction word: word\_yes in the MC
  renderings; $d_1 > 0$ (M's sentence) in the CM renderings.

## Measure

The model's answer: the sign of logit(C) − logit(R) at the final position (as in all earlier tests). Pooled over the four
renderings, the logit difference is regressed on own\_yes, word\_yes, last\_yes, $d_1$, $d_2$, the name order and
indicators for order (CM) and voice (object). The three rule coefficients are $b_\text{own}$, $b_\text{word}$,
$b_\text{last}$; 95% CIs by scene bootstrap (1,000 resamples; the four renderings of a scene stay together).

## Decision rule (per model)

- **The rule follows X** (X ∈ own step, word, last sentence) if, for both other features Y, the CI of $b_X − b_Y$ lies
  entirely above 0, and $b_X > 0$ with its CI above 0.
- **Mixed** otherwise; the report then names every feature whose coefficient has a CI above 0.

## Reported, not part of the rule

- Accuracy per kind of question and rendering, and the share of answers C split by own\_yes, word\_yes and last\_yes.
- Decisive accuracy per rendering, split by the correct answer, and its AUC.
- The $d_1$ coefficient (weight of M's step) per rendering: does moving M's sentence to the end change how much the
  answer uses it?
- The same analyses on 27B if it was run.

## What each outcome means for the paper

- **own step**: the rule is about C's step, not a surface match; the paper's wording ("C's own sentence points in the
  asked direction") stands and the limitation is removed.
- **word**: the rule is a lexical match between the question and the sentence that places C; the paper must say so.
- **last sentence**: the rule is recency; the paper must say so.
- **mixed**: reported as such, with the weights.

## Limits

One task family; one-word answers from instruct models; the object voice is less frequent English for spatial
relations and may itself change how well the model parses the sentence (accuracy on *own step only* in the object
renderings is reported to check this).
