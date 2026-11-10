# Preregistration — E2: when (if ever) does the model compose a position?

*Committed 2026-11-10, before any E2 data exists. Model and settings carried over from E1: Gemma 3 12B, 3-object scenes, `closer_than` excluded.*

## Question

E1 showed the answers depend on the moved object's own sentence, but only leaves were moved. For a leaf, "moving its sentence" and "moving its position" are the same edit. So E1 locates the information without identifying its format. E2 separates the two.

## Design

- **Scenes.** 3-object chains R → M → C: M is placed relative to the root R, and C relative to M. The sentence placing M comes before the sentence placing C. Sentence order is not shuffled.
- **Pairs.**
  - The *base* scene is the original.
  - The *source* scene changes only M's sentence, so M moves and C moves with it: C's offset from M is unchanged, and so is its sentence.
  - The *hybrid* scene has M at its source position but C at its base position. It is used only to get the model's own answers for the "C stays" outcome.
  - Base and source must be token-aligned outside M's sentence.
- **Intervention.** Copy M's sentence from the source run into the base run at one layer (every 2nd layer, 0–48). Then ask every question.
- **Targets** (model-relative, from clean runs):
  - *follow* = the patched answer equals the model's answer on the source;
  - *stay* = it equals the model's answer on the hybrid.
- **Question classes:**
  - *C-questions*: the model's source and hybrid answers differ, so the hypotheses disagree;
  - *M-questions*: base and source answers differ but source equals hybrid. These are a check that the patch took effect.
- **Pairs and questions.** 150 pairs; questions must pass the E1 margins (1 unit, 10°) in base, source and hybrid.

## Hypotheses and decision rule

At layer L, let follow_M(L) be the follow rate on M-questions, and follow_C(L) and stay_C(L) the follow and stay rates on C-questions. Use only layers where the patch is effective, i.e. follow_M(L) ≥ 0.8.

- **Eager composition (a position for C is computed downstream of M's sentence and later read).** Patching M after that point moves M but not C. Supported if, at some effective layer, follow_C ≤ 0.3 and stay_C ≥ 0.5. The first such layer estimates when C's position is composed.
- **Lazy / relational (C's position is rebuilt from the sentences when the question is asked).** C follows M wherever the patch is effective. Supported if follow_C ≥ follow_M − 0.15 at every effective layer.
- **Otherwise: mixed.** Report the curves.

Rates are per question, with 95% CIs from a pair-level bootstrap (1,000 resamples). Layer 0 (equivalent to editing the text) is the positive control; follow_C there should be ≈ 1.

## Exploratory (labelled as such)

The same analysis in scenes where C's sentence comes *before* M's. There, C's sentence cannot see M's, so any composition must happen later.
