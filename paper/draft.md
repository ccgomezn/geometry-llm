# A Map the Model Doesn't Read: Decodable but Unused Spatial Coordinates in a Language Model

*Cristhian Gómez Neira — draft, November 2026*

## Abstract

When people read "A is 3 units north of B; C is 2 units east of A", they can answer questions the text never answers by consulting an inner picture. Do language models do the same? We generate unseen layouts described only in text. Among eight open models, the smallest that solves 3-object layouts without written reasoning is Gemma 3 12B, and we analyse it against a preregistered set of criteria. A linear probe decodes every object's (x, y) from its mention in a list at the end of the description (R² = 0.74 at layer 21; controls ≈ 0). This includes positions the text implies only by chaining two sentences, where a baseline using the object's own sentence reaches 0.22. By every probing criterion, the model has built a map. It does not use it there. Replacing an object's representation at that position with one from a layout where the object sits elsewhere changes the joint answer in ≈ 1% of pairs at every layer. Replacing the object's *relation sentence* moves every answer consistently up to layer 14. After that, the information leaves the description altogether by layer 32, carried into the question. [DAS SENTENCE] High-quality geometric readouts can therefore sit off the causal path. Claims that a model "imagines" a scene need interventions on the representation, not just decodability.

## 1. Introduction

Whether a language model solves a spatial problem by consulting something like a picture, or by chaining stated relations, is a question about the *format* of its internal computation. Its written reasoning cannot settle the question, because written reasoning contains only words. Prior work shows that geometry can be *read out* of language models: world coordinates of places (Gurnee & Tegmark, 2023), circular features for days and months (Engels et al., 2024), and spatial relations in described scenes (Tehenan et al., 2025). An et al. (2026) find coordinate information in 7–8B models that is "transient, fragmented across task families, and weakly integrated into final predictions".

We test the property that distinguishes a usable map from a readable one: **an edit to one object's internal position should move every spatial answer consistently, and leave the others alone.** We contribute:

1. a text-only layout task with a **behaviour gate**, so that we only probe a model that solves the task silently;
2. a preregistered test of map use by **joint consistency** under activation patching, with probes, distributed alignment search (DAS) and steering as follow-ups (`docs/preregistration.md`);
3. a clean dissociation in Gemma 3 12B: coordinates that pass every probing criterion, composed across sentences, yet are causally inert where they are decodable.

## 2. Task and methods

**Scenes.** Each scene places 3–6 objects on an integer grid in [−6, 6]². The layout is fixed by a random spanning tree of relations ("X is k units north of Y", or compound "k north and m east"). Sentences use four templates, and their order is shuffled in 20% of scenes. Names are single tokens in every tokenizer tested. After the description, the prompt lists the objects ("The objects are: A, B, C."). Because attention is causal, each name in that list can see every sentence. This makes it a natural per-object readout position: it is where a completed map *could* be stored.

**Questions and scoring.** Six question types are used: farthest from X; closest to X; compass direction (8-way); which is farther north; which is farther east; and "is Y closer to X than Z?". Answers are scored by the log-probability of each allowed answer, with no free generation. Main-set answers are separated by a margin of at least 1 unit (10° for direction).

**Behaviour gate.** Preregistered: at least 0.75 accuracy on 4-object scenes and above-chance accuracy on multi-step questions, with a fallback to 3-object scenes. The smallest passing model is chosen, and question types at chance are excluded.

**Probes.** Ridge regression maps the residual stream at each object's list position to its scene-centred (x, y). Scenes are split 80/20 into training and test sets, and the penalty is chosen by scene-grouped cross-validation. Controls are shuffled labels, layer 0, a randomly initialised copy of the model, and a *stated-sentence baseline*: a probe on the object's own relation sentence alone. A probe that beats this baseline on objects two steps away has decoded a *composed* position.

**Patching (joint consistency).** Each pair of scenes differs only in the sentence placing one leaf object. We copy the source scene's activations into the base run at one site and layer: the object's list position, the end of the list, the object's relation sentence, or the whole description (positive control). We then ask every question. *Joint agreement* is the share of pairs in which **all** answers after patching equal the model's own answers on the source scene, counting only pairs where at least one answer changes; its ceiling is 1. The preregistered map criterion is joint agreement ≥ 0.6 at a readout position. The relational criterion is < 0.2 there, with more carried by the relation sentence.

**DAS and steering.** DAS learns a 2-D subspace at the readout position whose swap alone should reproduce the patching effect (Geiger et al., 2024); the preregistered threshold is ≥ 0.7 interchange accuracy on every question type. A slider then steers along the probe or DAS directions to off-grid positions, and checks whether answers flip where geometry predicts.

## 3. Results

### 3.1 Behaviour: only larger models solve layouts silently

We screened eight open models (Qwen3 1.7B–14B, Llama 3.2 3B, Gemma 3 4B–27B) on 150–500 fresh scenes each (Fig. 1). No model reached 0.75 on 4-object scenes. On the 3-object fallback, the smallest model to pass was **Gemma 3 12B** (0.76 [0.74, 0.77]; 0.68 on 4 objects; Gemma 3 27B scores 0.77). "Is Y closer to X than Z?" was at chance for every model and was excluded.

- **Silent solving is hard; written solving is not.** With thinking enabled, Qwen3-14B answers 0.98 of a 60-question 4-object sample correctly. Without it, accuracy on the full 4-object set is 0.56.
- **Error signatures differ by family.** A logistic model of errors separates two drivers. One is chain length (how many sentences must be combined). The other is near-ties (how close the competing answers are geometrically), which is analogous to distance effects in human image scanning (Kosslyn et al., 1978). Qwen3-4B's errors follow chain length (standardised coefficients −0.23 vs 0.05). Gemma 3 4B's follow near-ties (−0.08 vs 0.22). Gemma 3 12B shows both (−0.30, 0.31).
- **The object list is not a crutch.** Removing it leaves Gemma 3 12B's accuracy unchanged (0.66 with, 0.66 without).

Qwen models put almost no probability on bare one-word answers (median < 0.02, versus ≈ 1.0 for Gemma and Llama). Their forced-choice accuracies are therefore lower bounds.

![Figure 1](../results/figures/paper_fig1_behaviour.png)
*Figure 1. Left: accuracy without written reasoning (dashed line: preregistered gate). Right: what drives each model's errors.*

### 3.2 A decodable, composed map

On 3,000 fresh 3-object scenes (Fig. 2a):

- **Nothing in layers 0–11, then a map.** R² is ≈ 0 up to layer 11, jumps at layers 12–13 and peaks at **layer 21 (0.74 [0.72, 0.75])**, then declines slowly (0.54 at layer 48). Shuffled labels, layer 0 and the randomly initialised model all give 0.00.
- **Implied positions are decoded as well as stated ones.** Objects two steps from the anchor are decoded at 0.74, compared with 0.76 for objects placed by one sentence. The stated-sentence baseline reaches 0.22 on them. The position is *composed* from two sentences, not copied from one.
- **The code is per object, not one picture.** A probe reading all objects from the single end-of-list position fails (R² ≤ 0.03). Distances between activations track true distances only weakly (Spearman 0.19).

Every preregistered probing criterion for a map is met.

### 3.3 The map is not read where it is written

Patching sweeps every second layer on 100 pairs (Fig. 2b). The positive control, copying the whole description, reaches 1.00 at layer 0.

- **The decodable positions are causally inert.** Copying the moved object's list-position activations from the source gives joint agreement ≤ 0.02 at every layer, including layer 21, where the probe peaks. The end-of-list position also gives ≤ 0.01. The patch is applied: it shifts the log-probability of the base answer by 0.05–0.10 nats on average, against 3–7 nats for the relation sentence. It is simply not on the path to the answer.
- **The relation sentence carries everything.** Copying only the moved object's sentence moves every answer consistently: joint agreement is ≥ 0.89 through layer 14 and indistinguishable from copying the whole description at every layer.
- **Then the information leaves the description.** Between layers 16 and 32, both the sentence patch and the whole-description patch lose their effect (0.67 at layer 16, 0.25 at 20, 0.01 by 32). By then the question tokens have already read what they need.

[P3 CONFIRM SENTENCE]

By the preregistered criteria, the causal test for a map fails, and the test for relational use is met: < 0.2 at the readout positions, with the relation sentence carrying more.

![Figure 2](../results/figures/paper_fig2_mechanism.png)
*Figure 2. (a) Probe R² by layer, with controls. (b) Joint agreement after patching one site. (c) Interchange accuracy when only a learned 2-D subspace is swapped.*

### 3.4 Subspace swaps and steering

[P4 BLOCK]

## 4. Discussion

**What the model does.** Gemma 3 12B does compose relations into positions: from layer 12, the mention of each object in the list carries its implied (x, y), including positions reachable only by chaining sentences. It does not answer from that code. Answers depend on information that stays at the moved object's own sentence until about layer 14 and is then carried into the question between layers 16 and 32. The coordinates are computed, then ignored. Where a person would consult the picture, this model consults the sentences.

**What this implies.**

1. *Decodability is weak evidence of use, even at its best.* Our probe beats every control, including a baseline that rules out sentence copying, and still sits off the causal path. Probing results for "world models" and "spatial maps", ours included, should be read as upper bounds on what a model *could* use, unless they are paired with an intervention at the decoded site (cf. Hewitt & Liang, 2019; Vafa et al., 2024; An et al., 2026).
2. *"Imagining" is a causal claim.* The imagery debate asked whether depictive representations do work in reasoning, not whether they exist (Kosslyn et al., 1978; Pylyshyn, 2002). The same distinction applies to models. In this model, a depictive-looking code exists and does no work at the place it is written.
3. *Writing may be the model's sketchpad.* The gap between silent and written accuracy (0.56 vs 0.98 for Qwen3-14B) fits a picture in which models lack a persistent internal workspace for layouts, and text serves as one. Whether the coordinates become useful when the model writes them out is a natural next test.

**What remains open.** We did not intervene at the question tokens, where the answer is assembled. A map built *on demand* there is not ruled out. We also moved only leaf objects. For a leaf, moving its sentence and moving its position are the same edit, so P3 locates the information but does not identify its format. Moving an object that others hang from, and checking that its dependants move with it, would separate the two.

## 5. Limitations

One model (Gemma 3 12B), 3-object scenes (no model passed on 4), and five question types. Leaf-only moves; readout sites limited to the description. Single-token names on an integer grid; positions are expressed with a small set of sentence templates. Three preregistration clarifications were added. Two came after the behavioural data existed, and all three came before any probe or patching result. The third followed an independent code review that changed the scoring and the patching metric. The rescoring also changed which model the unchanged selection rule picked, from Gemma 3 27B to the smaller Gemma 3 12B. All are dated in `docs/preregistration.md`. The thinking-mode comparison uses a different scoring method (free generation) than the silent condition.

## References

- An, J., Yang, L., Wang, M., Lu, L., An, W., & Yang, E. (2026). From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs. arXiv:2603.26323.
- Engels, J., Michaud, E. J., Liao, I., Gurnee, W., & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. *ICLR 2025*. arXiv:2405.14860.
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. *CLeaR 2024*, PMLR 236, 160–187. arXiv:2303.02536.
- Gurnee, W., & Tegmark, M. (2023). Language Models Represent Space and Time. *ICLR 2024*. arXiv:2310.02207.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP-IJCNLP*, 2733–2743.
- Kosslyn, S. M., Ball, T. M., & Reiser, B. J. (1978). Visual images preserve metric spatial information: Evidence from studies of image scanning. *JEP:HPP*, 4(1), 47–60.
- Pylyshyn, Z. W. (2002). Mental imagery: In search of a theory. *BBS*, 25(2), 157–182.
- Tehenan, M., Bolivar Moya, C., Long, T., & Lin, G. (2025). Linear Spatial World Models Emerge in Large Language Models. arXiv:2506.02996.
- Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., & Mullainathan, S. (2024). Evaluating the World Model Implicit in a Generative Model. *NeurIPS 2024*. arXiv:2406.03689.
