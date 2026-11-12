# Preregistration: P7, what does the question read? (attention knockout)

*Committed 2026-11-13, before any P7 data exists. Gemma 3 12B (Colab, one A100). 3-object chains, `closer_than`
excluded.*

## Why

P5c and P6 erased the decodable map without behavioural cost, and P3 located the answer pathway coarsely: the stated
sentences matter early, the question tokens carry the answer from about layer 16 to 32, and the final position only at
the end. This experiment asks what the question tokens read, which would explain *why* the map is unused.

**Hypothesis (chaining at question time).** When the model reads a question about an object whose position must be
composed, the question tokens attend directly to the sentences of the whole chain and combine them there. The map lives
at the object list, which the question does not read.

## Design

- **Scenes.** 300 new chains R → M → C (seed 402), all questions passing the 1-unit / 10° margins.
- **Question classes**, by the objects a question names or offers as candidates:
  - **CR**: involves C and R (needs C's composed position);
  - **CM**: involves only C and M (needs only C's stated offset);
  - **MR**: involves only M and R (needs only M's stated offset).
- **Knockout.** In the chosen layers, the *query* positions after the object list (the question, the answer suffix, and
  the chat tokens before the answer) may not attend to one *key* span. Every other attention edge is unchanged. Spans:
  - **list**: from the end of the last sentence to the end of the object list (where the map is decoded);
  - **own**: C's sentence;
  - **parent**: M's sentence.
- **Layers.** All 48 layers (primary), and separately layers 0–15, 16–31 and 32–47 (reported).
- **Metric.** Accuracy against the layout; drop = clean accuracy − knocked-out accuracy, per class; scene-bootstrap
  95% CIs.

## Decision rules (all-layer knockout)

- **Knockout effective** (required, otherwise *uninformative*): own-span drop on CM ≥ 0.15 and parent-span drop on
  MR ≥ 0.15.
- **Chaining at question time:** parent-span drop on CR ≥ 0.15, and at least 0.10 larger than the parent-span drop on
  CM.
- **Composed position available without M's sentence:** parent-span drop on CR ≤ 0.05.
- **Unclear:** otherwise.
- Separately, **list not read** if the list-span drop is ≤ 0.03 on every class; otherwise **list read** (value
  reported).

## What each outcome means

- *Chaining at question time* + *list not read*: the answer pathway re-reads the chain of stated relations when it
  answers, and never reads the object list where the map is decodable. The map is built off the path the answers take.
- *Composed position available*: some position computed before the question (in C's sentence, the list, or elsewhere)
  suffices, contrary to the hypothesis.
- The layer windows locate when the reading happens.
