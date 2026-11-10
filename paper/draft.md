# [TITLE]

*Cristhian Gómez Neira — draft, November 2026*

## Abstract

When people read "A is 3 units north of B; C is 2 units east of A", they can answer questions the text never answers by consulting an inner picture. Do language models? We generate unseen layouts described only in text and study Gemma 3 12B, the smallest of eight open models that solves 3-object layouts without written reasoning. A linear probe decodes each object's (x, y) from the model's activations at the object's name in a list after the description (R² = 0.74 at layer 21; controls ≈ 0). This includes positions the text implies only by chaining two sentences, so the model encodes *composed* per-object coordinates. Yet these coordinates are not where the answers come from. We copied that representation from a layout where the object sits elsewhere. In 390 pairs, the answers never followed (joint agreement 0.00). Steering along the decoded directions changed the answer in 2% of the 268 sweeps where geometry says it must. Copying the object's *sentence* instead moves every answer consistently, until the information stops being read from the description at about layer 32. [NECESSITY SENTENCE] [DAS SENTENCE] A strong, well-controlled probe can sit off the causal path. Claims that a model "imagines" a scene need interventions, not decodability alone.

## 1. Introduction

Does a language model solve a spatial problem by consulting something like a picture, or by chaining stated relations? That is a question about the *format* of its internal computation. Written reasoning cannot settle it, because written reasoning contains only words. Geometry can be *read out* of language models: world coordinates of places (Gurnee & Tegmark, 2023), circular features for days and months (Engels et al., 2024), and spatial relations in described scenes (Tehenan et al., 2025). Some of it is used: Tehenan et al. steer single answers along probed relation directions. An et al. (2026) find spatial information in 7–8B models that "can causally influence behavior" but is "transient, fragmented across task families, and weakly integrated into final predictions".

We test the property that separates a usable map from a readable one. **Editing one object's internal position should move every spatial answer consistently.** We contribute a text-only layout task with a behaviour gate, so that we analyse a model that actually solves it silently. We also contribute a preregistered causal test of map use (`docs/preregistration.md`). In Gemma 3 12B, the result is a dissociation: coordinates are linearly decodable and composed across sentences, but overwriting them changes no answer.

## 2. Task and methods

**Scenes.** Each scene places 3–6 named objects on an integer grid in [−6, 6]². Positions are fixed by a tree of relations: "X is k units north of Y", or "k north and m east". One object is the root; an object placed relative to an object that is itself placed relative to the root is "two steps" away. A *leaf* is an object nothing else is placed relative to. After the description, the prompt lists the objects ("The objects are: A, B, C."). Each name in this list can attend to every sentence, so it is a natural *readout position*: where a finished map could be stored.

**Questions.** Six types are used: farthest from X; closest to X; compass direction of Y from X; which of two objects is farther north; which is farther east; and is Y closer to X than Z? Each allowed answer is scored by its log-probability, with no generation, and correct answers win by at least 1 unit (10° for direction).

**Behaviour gate** (preregistered). A model qualifies with ≥ 0.75 accuracy on 4-object scenes and above-chance accuracy on multi-step questions; if none qualifies, 3-object scenes are used. The smallest qualifying model is analysed, and question types at chance are excluded.

**Probes.** Ridge regression maps the hidden state (residual stream) at an object's list position to its (x, y) relative to the scene centre. Scenes are split 80/20 into training and test sets. Controls are shuffled labels, layer 0, and a randomly initialised model. A *stated-sentence baseline* regresses position on the object's own stated offset and chain depth. That is what a reader who never combines sentences would know.

**Patching.** Each pair of scenes differs in the one sentence that places a leaf object. We run the model on the base scene but copy in the source scene's hidden states at one site and one layer. The sites are the object's list position, the end of the list, the object's sentence, or the whole description (positive control). *Joint agreement* is the share of pairs in which every answer then equals the model's own answer on the source scene. It counts pairs where at least one answer differs between the two scenes, and its ceiling is 1. Preregistered criteria: a map needs ≥ 0.6 at a readout position; relational processing needs < 0.2 there, with more carried by the sentence.

**Subspace swaps and steering.** Distributed alignment search (DAS; Geiger et al., 2024) learns a 2-D subspace at the readout position whose swap alone should move the answers (preregistered: ≥ 0.7 on every question type). A *slider* moves the readout representation along the probe's (x, y) directions and checks whether answers flip where geometry predicts.

## 3. Results

### 3.1 Behaviour

We screened eight models (Qwen3 1.7B–14B, Llama 3.2 3B, Gemma 3 4B–27B) on 150–500 fresh scenes per object count (Fig. 1). None reached 0.75 on 4 objects. On 3 objects, the smallest model to qualify was **Gemma 3 12B** (0.76 [0.74, 0.77]; 0.68 on 4 objects; Gemma 3 27B: 0.77). "Is Y closer to X than Z?" was at chance for every model and was excluded. The object list does not help: removing it leaves Gemma 3 12B's accuracy unchanged (0.66 with and without). Written reasoning does help. On 60 4-object questions, Qwen3-14B scored 0.98 with thinking enabled (answers parsed from the generated text) and 0.57 without (scored with an earlier version of our log-probability scoring). What drives errors differs by family (Fig. 1, right).

![Figure 1](../results/figures/paper_fig1_behaviour.png)
*Figure 1. Left: accuracy without written reasoning (dashed: preregistered gate). Qwen models put little probability on bare one-word answers (median < 0.02), so their accuracies are lower bounds. Right: standardised logistic coefficients of errors on chain depth and on how close the competing answers are.*

### 3.2 The coordinates are decodable and composed

On 3,000 fresh 3-object scenes (Fig. 2a), R² is ≈ 0 up to layer 11. It jumps at layers 12–13 and peaks at **layer 21 (0.74 [0.72, 0.75])**, and all controls give 0.00. Objects two steps from the root are decoded as well as objects placed by one sentence (0.74 vs 0.76), while the stated-sentence baseline reaches 0.22 on them. The position is therefore composed from two sentences, not copied from one. The code is per object: no probe reads all objects from the end of the list (R² ≤ 0.03), and activation distances track true distances only weakly (Spearman 0.19). The preregistered probing criteria for a map are met.

### 3.3 The answers do not read them there

The sweep patches every second layer on 100 pairs (Fig. 2b); the confirmation run uses 390 new pairs at the two best sweep layers per site.

- **Copying the readout positions moves no answer.** At the moved object's list position, joint agreement is ≤ 0.01 at every layer of the sweep, including layer 21 where the probe peaks. In the confirmation run it is **0.00 [0.00, 0.00]**, and the same holds at the end of the list. The patch takes effect: it shifts the base answer's log-probability by at most 0.13 nats on average (per layer), against 3–7 nats for the sentence patch (layers 6–24). That is far too little to change an answer.
- **The sentence carries the information until it leaves the description.** Copying the moved object's sentence gives 0.94 [0.91, 0.96] at layer 6, close to the whole-description control (0.95). At such early layers this is little more than editing the input. The finding is what happens later. Between layers 16 and 32, both the sentence patch and the whole-description patch lose their effect (layer 20: 0.18 and 0.23; layer 32: 0.01), so the information is no longer read from the description. We did not patch the question tokens, the likely destination. From layer 16 on, the whole description carries somewhat more than the sentence alone (layer 18: 0.50 vs 0.36), so other positions contribute too.

[NECESSITY BLOCK]

Against the preregistration, the P3 criterion for a map fails and the P3 criterion for relational processing is met. Because the probing criteria favour a map, the overall preregistered verdict is **mixed**.

![Figure 2](../results/figures/paper_fig2_mechanism.png)
*Figure 2. (a) Probe R² by layer, with controls. (b) Joint agreement after copying one site from the source scene (100 pairs). (c) Interchange accuracy when only a learned 2-D subspace is swapped.*

### 3.4 Subspace swaps and steering

[DAS BLOCK]

Steering confirms the patching result. We moved the leaf's list-position representation along the probe's (x, y) directions at layer 21, over ±5 units in half-unit steps, including positions no sentence states. In 268 sweeps (50 scenes), geometry predicts that some answer must flip; the model's answer flipped in 2% of them. Predicted and observed flip points are uncorrelated (r = 0.03), and every steered state stayed within the natural range of activation norms. This contrasts with Tehenan et al. (2025), who steer answers along *relation* directions: here it is the coordinate code that does not steer.

## 4. Discussion

**What we found.** Gemma 3 12B composes relations into per-object coordinates, but its answers do not depend on them where they are decodable. The causal information sits at the moved object's sentence and stops being read from the description between layers 16 and 32.

**What this implies.**

1. *Decodability bounds use from above; it does not show it.* Our probe beats every control, including one that rules out sentence copying, and still sits off the causal path. Probing evidence for "world models" and "maps" should be paired with an intervention at the decoded site (cf. Hewitt & Liang, 2019; Vafa et al., 2024; An et al., 2026).
2. *"Imagining" is a causal claim.* The imagery debate was about whether depictive representations do work in reasoning, not whether they exist (Kosslyn et al., 1978; Pylyshyn, 2002). The same distinction applies to models. Here, a depictive-looking code exists and does no detectable work where it is written.

**What remains open.** First, a map built on demand at the question tokens is not ruled out. Second, because we moved only leaves, moving a sentence and moving a position are the same edit. Our results therefore locate the information but do not identify its format. Moving an object that others depend on, and checking that its dependants move with it, would separate the two. Third, writing might serve as the workspace that silent processing lacks (0.98 vs 0.57 for Qwen3-14B). That would need testing in the same model with the same scoring.

## 5. Limitations

One model, 3-object scenes, five question types and leaf-only moves. Sentence templates are fixed and positions lie on an integer grid. Preregistration clarifications are dated in `docs/preregistration.md`: one before any model data, and two after the behaviour screen but before any probe or patching result, plus an outcome note. Rescoring after a code review changed the model the unchanged selection rule picked, from Gemma 3 27B to Gemma 3 12B. The thinking comparison mixes two scoring methods and covers 60 questions.

## References

- An, J., Yang, L., Wang, M., Lu, L., An, W., & Yang, E. (2026). From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs. arXiv:2603.26323.
- Engels, J., Michaud, E. J., Liao, I., Gurnee, W., & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. *ICLR 2025*. arXiv:2405.14860.
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. *CLeaR 2024*, PMLR 236, 160–187. arXiv:2303.02536.
- Gurnee, W., & Tegmark, M. (2023). Language Models Represent Space and Time. *ICLR 2024*. arXiv:2310.02207.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP-IJCNLP*, 2733–2743. doi:10.18653/v1/D19-1275
- Kosslyn, S. M., Ball, T. M., & Reiser, B. J. (1978). Visual images preserve metric spatial information: Evidence from studies of image scanning. *JEP:HPP*, 4(1), 47–60. doi:10.1037/0096-1523.4.1.47
- Pylyshyn, Z. W. (2002). Mental imagery: In search of a theory. *BBS*, 25(2), 157–182. doi:10.1017/S0140525X02000043
- Tehenan, M., Bolivar Moya, C., Long, T., & Lin, G. (2025). Linear Spatial World Models Emerge in Large Language Models. arXiv:2506.02996.
- Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., & Mullainathan, S. (2024). Evaluating the World Model Implicit in a Generative Model. *NeurIPS 2024*. arXiv:2406.03689.
