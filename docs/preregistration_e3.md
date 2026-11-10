# Preregistration: E3, is a composed position carried where no stated text differs?

*Committed 2026-11-10, before any E3 data exists. Gemma 3 12B, 3-object chains, `closer_than` excluded (as E1/E2).*

## Question

In a chain R → M → C, change only M's sentence: M moves and C moves with it, while C's own sentence stays word-for-word the same. The activations over C's sentence can then differ only through what they have read from M's sentence, i.e. through C's composed position. Copying C's sentence from that run into the original run asks: **do the answers now treat C as sitting at its moved position, with M left where it was?**

## Design

- 150 pairs (seed 13), built like E2 (chains, M's sentence first, base and source token-aligned).
- **Target scene:** the base scene with C at its source absolute position (M unmoved).
- **Questions:**
  - *target questions:* the geometric answer differs between base and target (up to 6 per pair);
  - *controls:* it does not (up to 2 per pair).
  - Margins: 1 unit and 10° in base and target.
- **Intervention:** copy C's sentence activations from the source run into the base run at one layer, for layers 0–40 in steps of 2. Then ask every question.
- **Metrics** (model-relative, from clean runs; pair-bootstrap 95% CIs):
  - *follow_T* = the patched answer equals the model's answer on the target scene, over questions where its target and base answers differ;
  - *stable* = the patched answer equals its base answer, over questions where they agree;
  - *follow_S* (reported) = the answer equals the model's source answer where source and target differ.
- **Layer 0 is a sanity check.** The copied tokens are identical, so follow_T must be ≈ 0 there. It is not counted in the verdict.

## Decision rule (layers ≥ 1)

- **Carried:** at some layer follow_T ≥ 0.5 while stable ≥ 0.9. The first such layer is reported.
- **Not carried:** follow_T ≤ 0.1 at every layer.
- **Partial:** otherwise. Report the curves.

## What each outcome means

- **Carried:** C's sentence holds a position the model computed by combining two sentences, and the answers use it. This is a used, composed spatial variable, not a switch between stated facts.
- **Not carried:** the composed position read by probes (P2, P2d) is not what the answers take from C's sentence.
