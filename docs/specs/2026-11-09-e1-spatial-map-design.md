# E1 design: does an LLM build and use an internal map from a text description?

*Cristhian Gómez Neira · 2026-11-09 · Rung 1 of the "Do LLMs imagine?" project (see `llm-geometry-research-plan.md`)*

## 1. Intent

The long-term question: when an LLM solves a geometry or math problem given only in text, does it build an internal spatial representation of the figure and reason with it, the way a mathematician pictures a shape? Or does it manipulate symbols and word relations? The deepest version ("what is *seeing*, for humans and for models?") is a future direction. It shapes the framing but is out of scope here.

E1 is the first rung of a ladder: points → shapes → transformations → topology. Points on a plane are the simplest test case for geometric imagination, because the true figure is known exactly, so we can check what the model holds inside and move it.

**Why now.** The goal is a publishable result as soon as it is solid (arXiv + LessWrong/Alignment Forum, then a workshop). A clean, well-controlled **negative** result ("no map, just chained relations") counts as a result and gets written up the same way.

**Novelty position (from the literature review).** Two papers already do weaker versions of this:
- Tehenan et al. 2025 (arXiv:2506.02996) decode left/right/above/below categories and steer one answer at a time.
- An et al. 2026 (arXiv:2603.26323) find weak, "fragmented" coordinate probes (R² 0.2–0.46) in 7–8B models.

What is still open, and what E1 claims:
1. Full metric (x, y) decoding, including positions the text only implies through multi-step chains.
2. **The consistency test.** One edit inside the model moves *every* spatial answer the way a single coherent map would.
3. Steering to positions that no sentence could state.

E1 must cite and directly engage both papers, especially An et al.'s negative result.

## 2. Hypotheses and how E1 tells them apart

- **H-map.** At some layer, for each object, a low-dimensional (ideally 2-D) subspace encodes its (x, y). The model reads that subspace to answer *all* spatial questions.
- **H-rel.** No such per-object position code exists. Answers are computed at question time from the stored relation sentences.
- **Mixed.** A map exists at some layers and positions but is not what the answer is computed from, or it is used only for some question types. Finding where the model switches between map and relations is itself a reportable result.

| Test | H-map predicts | H-rel predicts |
|---|---|---|
| Probe for coordinates, by number of steps ("hops") from the reference object | Multi-hop objects decode about as well as 1-hop objects | Decoding degrades with hops, or only tracks the stated relations |
| Patching at the object's own position in the prompt (see §4) | Moves all question types consistently | Moves few or no answers. The effect lives at the relation-sentence tokens |
| 2-D DAS (Distributed Alignment Search) subspace | High interchange accuracy across all question types | Low interchange accuracy, or high only for some types |
| Steering to off-grid positions (e.g. x = 2.5) | Answers flip at the geometrically predicted thresholds | No coherent response to intermediate values |
| Behaviour errors | Driven by geometric near-ties | Driven by number of hops |

## 3. Data: the puzzle generator

- **Scenes.** 3–6 objects on an integer grid of roughly [-6, 6]². Positions are fixed by a random spanning tree of stated relations, so each object's position is determined and its hop depth from the first-mentioned object varies.
- **Relations.** "X is k units north of Y", over the 4 cardinal directions with k = 1–5, plus compound forms ("k north and m east of Y"). Several wording templates, and some scenes with sentences shuffled so references point forward.
- **Names.** Single-token in the chosen model's tokenizer, with no spatial meaning. At least two name pools (e.g. first names, nonce objects). Letters are avoided because of alphabet-order bias.
- **Readout spot.** After the description and before the question, the prompt adds a fixed line listing the objects: "The objects are: A, B, C, D." Each name in that line is the per-object readout position. It comes after all relations, so the model can attend to all of them. (Because the model reads left to right, earlier mentions of a name cannot carry the full map.) The listing line must be checked to confirm it does not change behaviour.
- **"End of list" position.** This is the last token of the object list, which comes before the question and does not depend on it. It is the second readout position. A probe there decodes *all* objects' coordinates jointly, as one multi-output regression.
  - The question's last token is used only to score answers. It is never used as a probe or patching site for map claims, because patching there would copy the answer directly.
- **Prompting.** Zero-shot, using each model's chat template.
  - No few-shot examples: other scenes in the context would contaminate the probes.
  - Qwen3 runs with thinking disabled.
- **Questions (the battery).** Every question has a short closed answer set:
  - farthest from X and closest to X (object name)
  - direction from X to Y (8 compass words)
  - which of Y or Z is farther north or east
  - is Y closer to X than Z is? (yes/no)

  Answers are scored by comparing the model's probabilities over the allowed answers, without free text generation. Each question type has a known chance level.
- **Ties.** The main set requires a margin of at least 1 unit, so every question has a unique answer. A separate near-tie set is used for the error-signature analysis.
- Ground-truth coordinates, hop depths and the expected answers are stored with every scene.

## 4. Phases and gates

**P0: Setup and benchmark.**
- Environment, model loading on the Mac's GPU, activation caching.
- Measure real throughput on the Mac and replace the estimates.

**P1: Behaviour screen. This is the gate for picking the model.**
- **Candidates:** Qwen3-1.7B, Qwen3-4B and Qwen3-8B (thinking disabled), Gemma-3-4B-it, Llama-3.2-3B-Instruct.
- **Setup:** no chain of thought; 500+ scenes run through the full battery.
- **Gate:** pick the smallest model that reaches at least 75% overall on 4-object scenes and clearly beats chance on objects that are 2+ hops from the reference object.
- **If no model passes, fall back in this order:**
  1. Qwen3-8B locally.
  2. A 14B model on a rented GPU.
  3. 3-object scenes. This is last, because it caps hop depth at 2 and weakens claim 1.
- **Error signatures, logged in this phase:** fit accuracy against hop count and against geometric margin.
- **Chain of thought on vs. off:** measure accuracy with thinking enabled too, on a subset. This is cheap and gives the behavioural side of "computed silently vs. written out".

**P2: Probes.**
- **Method:** per-scene-centred ridge regression from activations at the readout position to (x, y). Subtract each scene's mean activation and mean coordinate, because positions only exist up to translation. Run at every layer.
- **Report:** held-out R² by layer and by hop depth.
- **Controls:**
  1. Shuffled-label control task, to measure how much the probe itself learns (selectivity).
  2. Layer-0 embeddings.
  3. A randomly initialised model with the same architecture.
  4. A "stated-relation" baseline: a linear model on the one explicit relation each object appears in, which cannot recover multi-hop coordinates.
- **Geometry check:** representational similarity between activation distances and true distances.
- **Optional:** logit lens at the peak layers, to check whether spatial words are present there.

**P3: Patching and consistency (the core result).**
- **Paired scenes:** the base and source scenes differ in exactly one relation sentence, which moves one **leaf** object T. A leaf has no other object defined relative to it, so nothing else moves.
- **Sweep:** patch the source activations into the base run across layers, at three places:
  - (a) T's position in the object list
  - (b) the tokens of T's relation sentence
  - (c) the final token
- **Question sets:** for each edit, compute the expected answers under the base map and the source map.
  - *Diagnostic questions* are those where the two maps disagree.
  - *Control questions* are those where they agree; their answers must not change.
- **Metrics:**
  - per question type: how often the answer flips to the source answer
  - **joint consistency per edit:** the fraction of edits where *all* diagnostic answers follow the source map and all control answers stay put

**P4: DAS and the slider.**
- **DAS:** learn a k-dimensional subspace (k = 2 first, then compare larger k) at the best site from P3. Train on interchange interventions over the whole battery, and report interchange accuracy per question type.
- **Slider:** steer T along the probe's or DAS's x and y directions, in fine steps that include positions no sentence states (e.g. 2.5).
  - Record how the answer probabilities change across the battery.
  - Compare the observed flip points with the geometric thresholds. The score is the correlation between predicted and observed flip points.
  - Check that the steered activations stay within the model's natural range.
- **Scale check:** one larger model (8B, or 14B on a rented GPU) through P1–P3.

**Thresholds, set in advance (written to `docs/preregistration.md` and committed before P2 runs; Claude's call).**

| Outcome | Requires |
|---|---|
| Support for H-map | **All** of these: (1) Probe R² ≥ 0.6 at the best layer, beating every control by ≥ 0.3. (2) Multi-hop R² within 0.1 of 1-hop. (3) Joint consistency ≥ 0.6 at the readout position or the end of the list. (4) 2-D DAS interchange accuracy ≥ 0.7 on every question type. |
| Support for H-rel | **All** of these: (1) R² < 0.3, or a clear drop with hop count. (2) Patching effects that are concentrated on the relation-sentence tokens. (3) Joint consistency < 0.2 at the readout positions. |
| Mixed | Anything in between. The result is reported as such, with the per-layer and per-question-type breakdown. |

Every metric is reported with 95% bootstrap confidence intervals. The sample sizes must be large enough for intervals of ±0.05 on the key metrics.

**Branches if a phase comes back weak (Claude's call).**
- **Weak P2 probes.**
  - Try a small nonlinear (MLP) probe and the other readout positions before concluding.
  - Run P3 anyway: patching does not assume the code is linear.
- **No usable direction from P2 or DAS.** Skip the slider.
- **The minimum complete negative result** is P1 + P2 + P3, with all controls. That gets written up.

**Priority if time or budget is tight** (no fixed deadline; the goal is to publish as soon as the result is solid).
1. The consistency result in P3 comes first.
2. Things to cut first, in this order: the scale check, comparing DAS at larger k, the random-init control, the near-tie set.

**P5: Write-up.**
- Four core figures:
  1. behaviour and error signatures
  2. probe R² by layer and hop depth, against the baselines
  3. consistency under patching
  4. the slider demo
- A short results log kept throughout. The arXiv and blog draft is planned separately once the results exist.

## 5. How it gets made

- **Hardware.** Development and models up to 4B run on the user's MacBook Pro (M2 Max, 32 GB, MPS, 110 GB free). Large sweeps and 8B+ models go on a rented GPU by the hour when a local run would take longer than about 2 hours. NDIF is optional for the scale check.
- **Stack.** Python 3.12 with `uv`, PyTorch on MPS, Hugging Face `transformers`, `nnsight` for activations and interventions, scikit-learn for probes, matplotlib for figures. DAS is hand-written in PyTorch rather than using pyvene, which has gone quiet.
- **Location.** Everything lives in `geometry-llm/`, initialised as a git repo. Scene datasets and activation caches stay out of git and are regenerable from a seed.
- **Who builds.** Claude builds and runs straight through every phase without stopping for sign-off.
- **Reporting (user's decision).**
  - At each checkpoint (after P1, P2 and P3), Claude appends a dated entry to `results/RESULTS.md` with:
    - what was run
    - the key numbers with their confidence intervals
    - figures from `results/figures/`
    - what it means for H-map vs. H-rel
    - what runs next
  - The user reads along and steers whenever they want.

## 6. Goals, non-goals, anti-goals

- **Goals.**
  - A decisive answer on H-map vs. H-rel for one capable model, with every control.
  - The consistency figure.
  - The slider demo.
- **Non-goals for E1.**
  - Transformations (E2), topology and holes (E3), shapes (rung 2).
  - SAEs, J-lens and Natural Language Autoencoders.
  - A multi-model scale study beyond one check.
  - Any claim about phenomenal "seeing".
- **Anti-goals** (failure even if something "works"):
  - Decoding that is really the stated relations or text statistics.
  - A probe that does the work itself.
  - A cherry-picked single demo instead of aggregate statistics.
  - Claiming novelty that Tehenan et al. or An et al. already have.

## 7. Calls made by Claude (user to check)

- nnsight over TransformerLens as the main library. Both work; nnsight gives raw Hugging Face models and a path to NDIF.
- The object-list readout spot, and per-scene-centred probes.
- Leaf-object edits for P3.
- The P1 gate thresholds: 75% overall, and 2-hop clearly above chance.
- Cloud spend of up to about $50 for E1 without asking. Anything more is asked first.
- The "end of list" readout position, and zero-shot prompting with the chat template.
- The fallback order if no model passes P1.
- The thresholds set in advance, the branches if a phase comes back weak, and the cut order.
- Persistent homology is dropped from E1. A decoded 2-D point configuration has trivial topology, so it can't show anything here; it returns in E3.

## 8. Left to the builder

Exact templates and name pools, batch sizes, layer subsampling, ridge regularisation, DAS training details, file layout, and figure styling.
