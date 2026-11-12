# Preregistration: P8, does the dependent's sentence carry its composed position? (multi-layer transfer)

*Committed 2026-11-13, before any P8 data exists. Gemma 3 12B (Colab, one A100). 3-object chains, `closer_than`
excluded.*

## Why

P7 found that questions needing C's composed position (C two steps from the root, R → M → C) barely suffer when the
question cannot attend to M's sentence (drop 0.027), but do suffer when it cannot attend to C's own sentence (0.133),
and the object list is never read. So the composed position seems to reach the question through C's sentence. Yet E3
found that copying C's word-identical sentence from a run where M (and with it C) moved, **at a single layer**, does
not transfer C's position (≤ 0.07 at every layer). One reading reconciles the two: C's sentence keeps re-reading M's
sentence at every layer, so a copy at one layer is overwritten by the next layers, which read M's unchanged sentence.

## Design

Identical to E3 (`docs/preregistration_e3.md`) except for the intervention.

- **Pairs.** 150 new chain pairs (seed 501), built like E3: only M's sentence differs between base and source (M moves,
  C moves with it, C's sentence is word-for-word the same); target = base with C at its source position, M unmoved.
  Target questions (geometric answer differs between base and target, up to 6 per pair) and controls (up to 2).
- **Intervention.** Copy the residual stream over C's sentence from the source run into the base run at **every layer
  from L to the last** (layer outputs L, L+1, …, 48; 0 = embeddings), for L ∈ {0, 4, 8, …, 48}. At L = 0 C's
  sentence is computed entirely in the source context; at L = 48 only the final outputs are copied, which no later
  computation reads (a built-in null).
- **Metrics** (model-relative, pair-bootstrap 95% CIs), as in E3: *follow_T* (patched answer = the model's target answer,
  where target and base answers differ), *stable* (patched answer = base answer, where they agree), *follow_S* (reported).

## Decision rule

- **Carried by the dependent's sentence:** at some L, follow_T ≥ 0.5 while stable ≥ 0.9. Report the latest such L (how
  late the copy still works, which bounds when the question reads C's sentence).
- **Not carried:** follow_T ≤ 0.1 at every L.
- **Partial:** otherwise.

## What each outcome means

- *Carried*: the composed position lives at C's sentence and is maintained across layers; E3's single-layer null was an
  overwrite effect. Together with P7, the answer pathway is: C's sentence composes C's position from M's sentence, and
  the question reads C's sentence in layers 16–31.
- *Not carried*: even a complete copy of C's sentence does not carry C's composed position; the question must combine
  information from elsewhere (e.g. M's sentence through other tokens), contradicting the simple reading of P7.
