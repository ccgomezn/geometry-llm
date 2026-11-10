# Do Language Models Imagine? Testing for a Causally Used Spatial Map Built from Text

*Cristhian Gómez Neira — draft, November 2026*

<!-- STATUS: sections 1–4 and the behavioural results are written; sections 5–7 are filled from results/RESULTS.md once P2–P4 finish. -->

## Abstract

{{ABSTRACT}}

## 1. Introduction

When a person reads "A is 3 units north of B, and C is 2 units east of A," they can answer questions the text never answers ("is C farther from B than A is?") by consulting an inner picture. The picture does work that the sentences alone do not: a single move of one point updates every distance and direction at once, and the answers stay mutually consistent. Whether large language models do something similar, or instead chain the stated relations symbolically, is a question about the *format* of their internal computation. Their written reasoning cannot settle it, because a model's chain of thought can only contain words.

Prior interpretability work has shown that geometric structure can be *read out* of language models: world coordinates of real places (Gurnee & Tegmark, 2023), circular features for months and days (Engels et al., 2024), and coarse 2-D layouts before drawing (Nedungadi et al., 2026). Two recent studies approach our question directly. Tehenan et al. (2025) decode categorical relations (left/right/above) from short descriptions and steer a single answer. An et al. (2026) probe coordinates in 7–8B models and describe the spatial representations they find as "transient, fragmented across task families, and weakly integrated into final predictions." Neither tests the property that distinguishes a map from a list of relations: **one internal edit should move every spatial answer consistently.**

We build that test. We generate never-before-seen layouts described only in text, find models that solve them *without* written reasoning, and ask four preregistered questions of the chosen model:

1. Can each object's (x, y) be decoded linearly, including positions the text implies only through a chain of relations?
2. If one object's internal representation is replaced by the representation from a layout where that object sits elsewhere, do *all* answers follow the new layout, while answers that should not change stay put (joint consistency)?
3. Is there a 2-dimensional subspace whose swap achieves this (Distributed Alignment Search)?
4. Does steering along that subspace to positions no sentence states (e.g. x = 2.5) flip answers exactly where geometry predicts?

{{CONTRIBUTIONS}}

## 2. Related work

**Readable geometry.** Linear probes recover latitude/longitude and dates from Llama-2 activations (Gurnee & Tegmark, 2023); the authors note that none of their evidence directly shows the model uses the probed features. Circular and helical features appear for periodic and numeric concepts (Engels et al., 2024; Kantamneni & Tegmark, 2025), and symmetry in co-occurrence statistics can explain such geometry without any spatial reasoning (Karkada et al., 2026). Anthropic's line-breaking study shows a model manipulating a curved one-dimensional manifold (Gurnee et al., 2025). These works establish that geometry can be *represented*; we ask whether a new geometry is *built on the spot and used*.

**Spatial world models from text.** Tehenan et al. (2025) find linear relation directions in Llama-3.2-3B and steer single answers (74% success). An et al. (2026) report weak coordinate probes (best-layer R² ≈ 0.16–0.46) in Qwen2.5-7B and Llama-3-8B-Instruct, with patching effects strongest in middle layers. Park et al. (2025) show in-context representations reorganising into grid geometry from random walks. Vafa et al. (2024) show that a transformer trained on navigation can perform well while holding an incoherent map. Nedungadi et al. (2026) decode the generic layout a prompt implies (not the model-specific one) and intervene only on generated text. We add the joint-consistency criterion, off-grid steering, and a behaviour gate that prevents probing a model that cannot solve the task.

**Causal abstraction.** We frame map vs. relational chaining as competing high-level causal models and test their alignment with interchange interventions and DAS (Geiger et al., 2024; 2025).

**Cognitive science.** The question restates the mental-imagery debate between depictive (Kosslyn, 1980; Kosslyn et al., 2006) and propositional (Pylyshyn, 1973; 2002) accounts. Mental-scanning times grow with imagined distance (Kosslyn et al., 1978), analogous to our "error signature" analysis. Expert mathematicians recruit a non-linguistic network of bilateral frontal, intraparietal and ventrolateral temporal regions, even for abstract mathematics (Amalric & Dehaene, 2016). We test function and format only, not experience.

## 3. Task and preregistration

**Scenes.** 3–6 objects on an integer grid in [−6, 6]². Positions are fixed by a random spanning tree of relations ("X is k units north of Y"; compound forms "k north and m east"), k ∈ 1..5, using four sentence templates. Sentence order is shuffled in 20% of scenes. Names are drawn from two pools of single-token names (people; capitalised object nouns) and checked to be single tokens in the Qwen3, Llama 3.2 and Gemma 3 tokenizers. After the description, the prompt lists the objects ("The objects are: A, B, C."). Each name in this list is a per-object readout position that can attend to every relation, because causal attention prevents earlier mentions from containing the whole map.

**Questions.** Farthest from X; closest to X; compass direction (8-way) of Y from X; which of Y, Z is farther north/east; is Y closer to X than Z (yes/no). Answers are scored as the summed log-probability of each candidate continuation in answer position (no free generation). The main set requires margins of ≥ 1 unit, or ≥ 10° for direction; a separate near-tie set has margins of 0.25–1 unit.

**Preregistration.** Thresholds for "map", "relational" and "mixed" outcomes, the analysis branches, and the order in which analyses are cut were committed before any probe result existed (`docs/preregistration.md`). Two clarifications were added before the relevant data existed: DAS accuracy is computed on diagnostic questions only; after P1, the 3-object fallback and the exclusion of at-chance question types were recorded.

## 4. Behaviour: which models solve the task without writing?

We screened eight open models (Qwen3 1.7B–14B, Llama 3.2 3B, Gemma 3 4B–27B) on 150–500 fresh scenes each, answering immediately with no written reasoning (Fig. 1). No model reached the preregistered 0.75 on 4-object scenes. On the 3-object fallback, the smallest model to pass was **Gemma 3 12B** (0.76 [0.74, 0.77]; 0.68 on 4 objects), which we analyse below. One question type, "is Y closer to X than Z?", was at chance for every model and was excluded as preregistered.

Three observations frame the mechanistic results:

- **Silent solving is hard; written solving is not.** With thinking enabled, Qwen3-14B answers 0.98 of a 60-question 4-object sample correctly. Without it, accuracy on the full 4-object set is 0.56. The models can work through layouts in words; whether they can do so *internally* is the open question.
- **Error signatures differ by family.** A logistic model of errors separates two drivers. One is chain length: how many sentences must be combined. The other is near-ties: how close the competing answers are geometrically, analogous to distance effects in human mental scanning. Qwen3-4B's errors are driven almost entirely by chain length (−0.23 vs 0.05). Gemma 3 4B's are driven by near-ties (0.22 vs −0.08). Gemma 3 12B shows both (−0.30, 0.31).
- **The readout line is not a crutch.** Removing the object list from the prompt does not change accuracy (Gemma 3 12B: 0.66 with, 0.66 without). We can therefore use it as a measurement point without changing the task.

Qwen models put almost no probability on bare one-word answers (median < 0.02 on the allowed answers, versus ≈ 1.0 for Gemma and Llama). Their forced-choice accuracies are therefore lower bounds.

## 5. Is there a decodable map? (probes)

For 3,000 fresh 3-object scenes, we fit a ridge probe at every layer. It maps the residual stream at each object's name in the object list to that object's scene-centred (x, y). Scenes are split 80/20 into training and test sets, and the penalty is chosen by scene-grouped cross-validation (Fig. 2a).

- **Nothing in layers 0–11, then a map.** R² is ≈ 0 up to layer 11, jumps at layers 12–13, and peaks at **layer 21 (R² = 0.74 [0.72, 0.75])** before declining slowly (0.54 at layer 48). Shuffled labels, layer 0 and a randomly initialised copy of the model all give 0.00.
- **Implied positions are decoded as well as stated ones.** Objects two steps from the anchor, whose position requires combining two sentences, are decoded at 0.74, compared with 0.76 for objects placed by one sentence. A baseline that reads only the object's own sentence reaches 0.22 on these objects. The model has *composed* the relations into a position; it has not merely copied a sentence.
- **The map is per object, not one picture.** A probe asked to read every object's coordinates from a single position at the end of the list fails (R² ≤ 0.03). Distances between activations correlate only weakly with true distances (Spearman 0.19). The coordinates are written onto each object's own representation as a linear code, not laid out as one image.

All preregistered P2 criteria for a map were met.

## 6. Is the map used? (patching, DAS, steering)

{{P3_P4_TEXT}}

## 7. Discussion

{{DISCUSSION}}

## 8. Limitations

{{LIMITATIONS}}

## References

- Amalric, M., & Dehaene, S. (2016). Origins of the brain networks for advanced mathematics in expert mathematicians. *PNAS*, 113(18), 4909–4917. doi:10.1073/pnas.1603205113
- An, J., Yang, L., Wang, M., Lu, L., An, W., & Yang, E. (2026). From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs. arXiv:2603.26323.
- Engels, J., Michaud, E. J., Liao, I., Gurnee, W., & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. *ICLR 2025*. arXiv:2405.14860.
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. *CLeaR 2024*, PMLR 236, 160–187. arXiv:2303.02536.
- Geiger, A., et al. (2025). Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. *JMLR*, 26(83), 1–64.
- Gurnee, W., & Tegmark, M. (2023). Language Models Represent Space and Time. *ICLR 2024*. arXiv:2310.02207.
- Gurnee, W., Ameisen, E., Kauvar, I., Tarng, J., Pearce, A., Olah, C., & Batson, J. (2025). When Models Manipulate Manifolds: The Geometry of a Counting Task. Transformer Circuits Thread. arXiv:2601.04480.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP-IJCNLP*, 2733–2743.
- Kantamneni, S., & Tegmark, M. (2025). Language Models Use Trigonometry to Do Addition. arXiv:2502.00873.
- Karkada, D., Korchinski, D. J., Nava, A., Wyart, M., & Bahri, Y. (2026). Symmetry in language statistics shapes the geometry of model representations. *ICML 2026*. arXiv:2602.15029.
- Kosslyn, S. M. (1980). *Image and Mind*. Harvard University Press.
- Kosslyn, S. M., Ball, T. M., & Reiser, B. J. (1978). Visual images preserve metric spatial information: Evidence from studies of image scanning. *JEP:HPP*, 4(1), 47–60.
- Kosslyn, S. M., Thompson, W. L., & Ganis, G. (2006). *The Case for Mental Imagery*. Oxford University Press.
- Nedungadi, A., Oehmcke, S., & Lüdtke, S. (2026). Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models. arXiv:2608.30751.
- Park, C. F., Lee, A., Lubana, E. S., Yang, Y., Okawa, M., Nishi, K., Wattenberg, M., & Tanaka, H. (2025). ICLR: In-Context Learning of Representations. *ICLR 2025*. arXiv:2501.00070.
- Pylyshyn, Z. W. (1973). What the mind's eye tells the mind's brain: A critique of mental imagery. *Psychological Bulletin*, 80(1), 1–24.
- Pylyshyn, Z. W. (2002). Mental imagery: In search of a theory. *BBS*, 25(2), 157–182.
- Shepard, R. N., & Metzler, J. (1971). Mental rotation of three-dimensional objects. *Science*, 171(3972), 701–703.
- Tehenan, M., Bolivar Moya, C., Long, T., & Lin, G. (2025). Linear Spatial World Models Emerge in Large Language Models. arXiv:2506.02996.
- Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., & Mullainathan, S. (2024). Evaluating the World Model Implicit in a Generative Model. *NeurIPS 2024*. arXiv:2406.03689.
