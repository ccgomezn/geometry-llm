# Do Language Models Imagine? Testing for a Causally Used Spatial Map Built from Text

*Cristhian Gómez Neira — draft, November 2026*

<!-- STATUS: sections 1–4 and the behavioural results are written; sections 5–7 are filled from results/RESULTS.md once P2–P4 finish. -->

## Abstract

{{ABSTRACT}}

## 1. Introduction

When a person reads "A is 3 units north of B, and C is 2 units east of A," they can answer questions the text never answers ("is C farther from B than A is?") by consulting an inner picture. The picture does work that the sentences alone do not: a single move of one point updates every distance and direction at once, and the answers stay mutually consistent. Whether large language models do something similar, or instead chain the stated relations symbolically, is a question about the *format* of their internal computation. Their written reasoning cannot settle it, because a model's chain of thought can only contain words.

Prior interpretability work has shown that geometric structure can be *read out* of language models: world coordinates of real places (Gurnee & Tegmark, 2023), circular features for months and days (Engels et al., 2024), and coarse 2-D layouts during drawing (Nedungadi et al., 2026). Two recent studies approach our question directly. Tehenan et al. (2025) decode categorical relations (left/right/above) from short descriptions and steer a single answer. An et al. (2026) probe coordinates in 7–8B models and find them "transient, fragmented, weakly integrated." Neither tests the property that distinguishes a map from a list of relations: **one internal edit should move every spatial answer consistently.**

We build that test. We generate never-before-seen layouts described only in text, find models that solve them *without* written reasoning, and ask four preregistered questions of the chosen model:

1. Can each object's (x, y) be decoded linearly, including positions the text implies only through a chain of relations?
2. If one object's internal representation is replaced by the representation from a layout where that object sits elsewhere, do *all* answers follow the new layout, while answers that should not change stay put (joint consistency)?
3. Is there a 2-dimensional subspace whose swap achieves this (Distributed Alignment Search)?
4. Does steering along that subspace to positions no sentence states (e.g. x = 2.5) flip answers exactly where geometry predicts?

{{CONTRIBUTIONS}}

## 2. Related work

**Readable geometry.** Linear probes recover latitude/longitude and dates from Llama-2 activations (Gurnee & Tegmark, 2023); the authors note that no evidence showed the model *uses* these features. Circular and helical features appear for periodic and numeric concepts (Engels et al., 2024; Kantamneni & Tegmark, 2025), and symmetry in co-occurrence statistics can explain such geometry without any spatial reasoning (Karkada et al., 2026). Anthropic's line-breaking study shows a model manipulating a curved one-dimensional manifold (Gurnee et al., 2025). These works establish that geometry can be *represented*; we ask whether a new geometry is *built on the spot and used*.

**Spatial world models from text.** Tehenan et al. (2025) find linear relation directions in Llama-3.2-3B and steer single answers (74% success). An et al. (2026) report weak coordinate probes (R² ≈ 0.2–0.46) in Qwen2.5-7B and Llama-3-8B, with activation patching localised to middle layers. Park et al. (2025) show in-context representations reorganising into grid geometry from random walks. Vafa et al. (2024) show that a transformer trained on navigation can perform well while holding an incoherent map. Nedungadi et al. (2026) decode a coarse layout before drawing but intervene only on generated text. We add the joint-consistency criterion, off-grid steering, and a behaviour gate that prevents probing a model that cannot solve the task.

**Causal abstraction.** We frame map vs. relational chaining as competing high-level causal models and test their alignment with interchange interventions and DAS (Geiger et al., 2024; 2025).

**Cognitive science.** The question restates the mental-imagery debate between depictive (Kosslyn, 1980; Kosslyn et al., 2006) and propositional (Pylyshyn, 1973; 2002) accounts. Mental-scanning times grow with imagined distance (Kosslyn et al., 1978), analogous to our "error signature" analysis. Expert mathematicians recruit a non-linguistic parietal and inferior temporal network even for abstract mathematics (Amalric & Dehaene, 2016). We test function and format only, not experience.

## 3. Task and preregistration

**Scenes.** 3–6 objects on an integer grid in [−6, 6]². Positions are fixed by a random spanning tree of relations ("X is k units north of Y"; compound forms "k north and m east"), k ∈ 1..5, using four sentence templates. Sentence order is shuffled in 20% of scenes. Names are drawn from two pools of single-token names (people; capitalised object nouns) and checked to be single tokens in the Qwen3, Llama 3.2 and Gemma 3 tokenizers. After the description, the prompt lists the objects ("The objects are: A, B, C."). Each name in this list is a per-object readout position that can attend to every relation, because causal attention prevents earlier mentions from containing the whole map.

**Questions.** Farthest from X; closest to X; compass direction (8-way) of Y from X; which of Y, Z is farther north/east; is Y closer to X than Z (yes/no). Answers are scored as the summed log-probability of each candidate continuation in answer position (no free generation). The main set requires margins of ≥ 1 unit, or ≥ 10° for direction; a separate near-tie set has margins of 0.25–1 unit.

**Preregistration.** Thresholds for "map", "relational" and "mixed" outcomes, the analysis branches, and the order in which analyses are cut were committed before any probe result existed (`docs/preregistration.md`). Two clarifications were added before the relevant data existed: DAS accuracy is computed on diagnostic questions only; after P1, the 3-object fallback and the exclusion of at-chance question types were recorded.

## 4. Behaviour: which models solve the task without writing?

{{P1_TEXT}}

## 5. Is there a decodable map? (probes)

{{P2_TEXT}}

## 6. Is the map used? (patching, DAS, steering)

{{P3_P4_TEXT}}

## 7. Discussion

{{DISCUSSION}}

## 8. Limitations

{{LIMITATIONS}}

## References

- Amalric, M., & Dehaene, S. (2016). Origins of the brain networks for advanced mathematics in expert mathematicians. *PNAS*, 113(18), 4909–4917.
- An, et al. (2026). From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs. arXiv:2603.26323.
- Engels, J., Michaud, E., Liao, I., Gurnee, W., & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. arXiv:2405.14860.
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. *CLeaR*.
- Geiger, A., et al. (2025). Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. *JMLR*, 26(83).
- Gurnee, W., & Tegmark, M. (2023). Language Models Represent Space and Time. arXiv:2310.02207.
- Gurnee, W., et al. (2025). When Models Manipulate Manifolds: The Geometry of a Counting Task. transformer-circuits.pub.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP*.
- Kantamneni, S., & Tegmark, M. (2025). Language Models Use Trigonometry to Do Addition. arXiv:2502.00873.
- Karkada, D., et al. (2026). Symmetry in language statistics shapes the geometry of model representations. arXiv:2602.15029.
- Kosslyn, S. M., Ball, T. M., & Reiser, B. J. (1978). Visual images preserve metric spatial information. *JEP:HPP*, 4(1), 47–60.
- Kosslyn, S. M. (1980). *Image and Mind*. Harvard University Press.
- Kosslyn, S. M., Thompson, W. L., & Ganis, G. (2006). *The Case for Mental Imagery*. Oxford University Press.
- Nedungadi, A., Oehmcke, S., & Lüdtke, S. (2026). Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models. arXiv:2608.30751.
- Park, C. F., et al. (2025). ICLR: In-Context Learning of Representations. *ICLR*. arXiv:2501.00070.
- Pylyshyn, Z. W. (1973). What the mind's eye tells the mind's brain. *Psychological Bulletin*, 80(1), 1–24.
- Pylyshyn, Z. W. (2002). Mental imagery: In search of a theory. *BBS*, 25(2), 157–238.
- Shepard, R. N., & Metzler, J. (1971). Mental rotation of three-dimensional objects. *Science*, 171(3972), 701–703.
- Tehenan, M., Bolivar Moya, C., Long, T., & Lin, G. (2025). Linear Spatial World Models Emerge in Large Language Models. arXiv:2506.02996.
- Vafa, K., Chen, J. Y., Rambachan, A., Kleinberg, J., & Mullainathan, S. (2024). Evaluating the World Model Implicit in a Generative Model. *NeurIPS*.
