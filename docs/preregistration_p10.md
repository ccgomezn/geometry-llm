# Preregistration: P10, does the question answer better if it is made to look at the map? (attention boost)

*Committed 2026-11-13, before any P10 data exists. Gemma 3 12B (Colab, one A100). 3-object chains, `closer_than`
excluded.*

## Why

P9 found that answering geometrically from the decoded object-list map beats the model on questions that need the
chain (0.80 vs 0.73 at layer 21), while P7 found that the question never reads the object list. P10 asks whether the
map is usable *as it is*: if the question tokens are pushed to attend to it, do chain answers improve?

## Design

- **Test.** 300 new chains R → M → C (seed 702), every question passing the 1-unit / 10° margins. Classes as in P7
  (object names only): **CR** (involves C and R), **CM** (only C and M), **MR** (no C).
- **Boost.** In layers 16–31 (where P7 found the question reads the sentences), add +b to the attention logits from
  every query position after the object list to a set of key positions, b ∈ {1, 2, 4, 8}. Nothing else changes.
  - **names**: the three object-name tokens in the list, where the probes read the map;
  - **control**: the list token immediately before each name (":" or ","), the same count in the same region.
- **Metric.** Accuracy against the layout, per class. Δ = boosted − clean. The contrast is Δ(names) − Δ(control),
  with a scene-bootstrap 95% CI.

## Decision rule (CR questions)

- **The map helps when attended:** for some b, Δ(names) − Δ(control) ≥ 0.03 with the CI excluding 0, and Δ(names) > 0.
- **No benefit:** for every b, the CI of Δ(names) − Δ(control) includes 0 or lies below it.
- **Unclear:** otherwise.

Reported, not part of the rule: CM and MR; every b. Four boost sizes are tested, so a single marginal hit is weak
evidence; the full curve is reported.

## What each outcome means

- *Helps*: the map is in a format the answer pathway can use; the model simply does not look.
- *No benefit*: the map is not usable by the answer pathway as it is (format mismatch), or attention alone is not
  enough. P11 (a light fine-tune) asks whether the model can learn to use it.
