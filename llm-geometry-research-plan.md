# Do LLMs imagine? Research plan

*Cristhian Gómez Neira, Madrid, November 2026*

## 1. The question

> **When an LLM solves a spatial or geometric problem given only in text, does it build an internal spatial model and use it to get the answer, without saying so in its reasoning?**

When a mathematician proves something geometric, they don't only chain axioms and inference rules. They *see* the space: add a hole, draw a loop, deform it, and the picture stays consistent by itself. That inner picture does real work, which is why mathematicians use boards.

For an LLM, the chain of thought is its board, but it can only write words on it, never draw. **The question: is there still a hidden picture inside, doing the work a sketch would do?**

## 2. Why it's new

| Prior work | What it showed | What it leaves open |
|---|---|---|
| Gurnee & Tegmark (2023), *Language Models Represent Space and Time* | Latitude, longitude and dates can be linearly read from activations | Those maps are **stored knowledge** (an atlas), read out but never shown to be **used** |
| Months-on-a-circle, symmetry in language statistics (2026) | Geometry of representations comes from text statistics | Geometry used to **represent concepts**, not geometry the model **thinks with** |
| Anthropic, *When Models Manipulate Manifolds* (2025) | Character counts live on curved manifolds the model manipulates | A narrow task, not imagined spaces built from a description |
| *Autoregressive Mosaics* (2026) | A coarse 2D layout is decodable in text-only models before drawing | The causal test edited the **text**, never the activations |

**One line:** they asked "does the model *know* where things are?" This project asks "does the model *imagine*?" That's the step from memory to thought: a map **built on the spot** from a description never seen before, and **used** to reason.

## 3. The brain, for intuition and comparison

- **Place cells** (hippocampus) fire at specific locations; **grid cells** (entorhinal cortex) fire on a hexagonal lattice, like coordinates. O'Keefe, May-Britt and Edvard Moser shared the 2014 Nobel Prize.
- **Persistent homology on grid-cell populations** showed their activity lives on a **torus** (Gardner et al., *Nature*, 2022). Same tool as Experiment 3.
- Grid-like codes also appear when people navigate **abstract** spaces (concepts varying along two features), so the brain may reuse its map machinery for thinking in general.
- **Expert mathematicians** use a dedicated network (parietal and inferior temporal), not the language network, even for advanced topology; **blind mathematicians** use the same one (Amalric & Dehaene, PNAS 2016; follow-up on blind mathematicians). Mathematical "seeing" is spatial and structural, not about the eyes.
- **Shepard & Metzler (1971):** time to mentally rotate a shape grows with the angle, evidence that imagery is a simulation operated on step by step, not just symbols.

## 4. Text vs. space: what the distinction means inside a model

Inside the model there is **no text**, only activations. Text exists only at the edges (input, output, chain of thought). So the question is about the **format** of the information while the model computes:

- **Symbolic or relational:** activations encode facts like "A north of B" and chain them like logic steps.
- **Spatial or metric:** activations encode each object like coordinates; distances and directions behave geometrically.

Representations are usually **directions or regions in activation space across many neurons**, not single neurons (as in the brain's population codes).

### The tests that tell them apart

1. **Decoding the unstated.** From "A is 3 north of B, C is 2 east of A," can you decode coordinates for every object, and the C–B distance that was never given, *before* the answer?
2. **Geometry of the decoded space.** Do distances between object representations match true distances (representational similarity)? Is the space plane-like? (Check with persistent homology.)
3. **Consistency under intervention (decisive).** Edit A's decoded position, then ask *many* questions (farthest point, direction C→A, alignments).
   - Map: **one edit changes all answers consistently.**
   - Symbolic: only questions using that relation change, and answers become mutually inconsistent.
4. **Error signatures.** Errors growing with the **number of reasoning steps** in the description suggest symbolic chaining; errors driven by **geometric closeness** (near-ties in distance) suggest a metric map.
5. **Verbalizability.**
   - **Logit lens / J-lens:** read intermediate activations as words. Words like "north" or "east" in the middle layers mean language-like computation; coordinates decodable with no relevant words mean non-verbal spatial information.
   - **CoT on vs. off:** if it answers correctly in a single forward pass with no written reasoning, the computation lived entirely in activations. Compare what it says with CoT on.

**Caveat:** the answer may be "both," with a map for some steps or layers and symbols for others. Finding where it switches is a result in itself.

## 5. Experiments

### E1: Static map (start here, weeks 3–4)
- **Data:** generated puzzles, 3–6 points on a plane, relations in words ("3 units north of"), ground-truth coordinates known. Questions: farthest or closest point, direction, distance comparisons.
- **Probe:** linear probes for each object's (x, y) at every layer and token position. Baselines: probes on a shuffled-label control task (selectivity, Hewitt & Liang), and probes on the raw token embeddings.
- **Geometry:** representational similarity between activation distances and true distances; persistent homology (`ripser`) on the decoded configuration.
- **Intervention:** patch or steer the decoded position (move A east), then ask 5–10 downstream questions; measure consistency with the edited geometry vs. the original.
- **Success:** coordinates decodable above controls **and** edits that move all answers consistently.

### The demo: move the map, watch the answers move
Inspired by David Bau's GAN Dissection (switching "lamp" units off turns the lights off in a generated image). Here: find where A's position lives, move it, and the model's answers about the map change.
- **Directions, not single neurons:** in LLMs, information is spread across many neurons, so move a direction or small subspace.
- **Three ways to move it, from fewest to most assumptions:**
  1. **Activation patching / interchange intervention (no linearity needed):** copy activations from a run where A sits elsewhere into the original run, and check the answers follow the new position.
  2. **Distributed Alignment Search (Geiger et al.):** learn the smallest subspace where swapping moves A's position and nothing else.
  3. **Steering along a probe direction (assumes linearity):** push activations along the x/y probe direction by any amount, like a dial. If nonlinear, optimize activations until a nonlinear probe reads the target, and check the result stays within the model's natural activation range.
- **The figure:** a slider moving A east/west while the model answers live ("farthest from B?", "direction from C to A?", "is A between B and C?"). Every answer flipping at the right moment, consistently, is the visible evidence of an inner picture.

### E2: Transformations, the dynamic version
- "A moves 3 units east," "rotate the triangle 90°," "reflect across the line…"
- Does the internal map **update** after each instruction?
- **Shepard–Metzler test for LLMs:** do bigger transformations need **more layers or more tokens** before the updated position becomes decodable?

### E3: Topology, holes in the plane
- "Take R² and remove a disc centered at (0,0) of radius 1," then ask about **many specific points and paths** (is this point in the hole? can this loop shrink to a point?).
- One sentence gives one vector, and topology can't be seen in one point, so collect activations over many queried points and run **persistent homology** on that cloud.
- **Success:** a loop (H₁) appears in the activations when, and only when, the described space has a hole; two holes give two loops.

### E4: Verbalization and faithfulness
- For E1–E3: logit lens / J-lens on middle layers, plus CoT on vs. off.
- Key outcome: cases where the map is **causally used** (E1 intervention works) but **never verbalized**: hidden imagery.

## 6. Models

| Use | Models | Why |
|---|---|---|
| First experiments, on a laptop | **Qwen2.5 / Qwen3 1.5B–4B**, **Llama 3.2 3B** | Small, capable enough at simple spatial puzzles, well supported by TransformerLens and nnsight |
| Features / SAEs | **Gemma 2 2B or 9B** (Gemma Scope SAEs) | Public sparse autoencoders let you look for spatial features without training your own; check whether SAEs exist for newer Gemma versions |
| Scale check, on a rented GPU | **7B–9B** versions of the same families (Qwen 7B, Llama 8B, Gemma 9B) | Does the map get cleaner with scale? |
| Reasoning on/off | A small reasoning-tuned variant of the same family | Compare written reasoning with internal state |

**Rule:** before probing anything, check the model actually **solves** the puzzles above chance. If it can't do the task, there's nothing to find. Pick the smallest model that succeeds.

**Tools:** TransformerLens or nnsight (activations, patching), scikit-learn (probes), `ripser` / `persim` (persistent homology), PyTorch.

## 7. Plan for the month

| Week | Dates | Learn | Build |
|---|---|---|---|
| 1 | Nov 9–15 | ARENA Ch. 1, David Bau's course, *Mathematical Framework*, Belinkov's probing survey, Othello-GPT | First linear probe on a small model |
| 2 | Nov 16–22 | ARENA patching and interventions, *Toy Models of Superposition*, Gurnee & Tegmark, *Mosaics* · **residency deadline Nov 22** | First activation edit that changes an output |
| 3 | Nov 23–29 | Park et al. on linear representations and geometry of concepts, Anthropic's manifolds paper | **E1**: puzzle generator, probes, first intervention |
| 4 | Nov 30–Dec 6 | Causal abstraction (Geiger et al.) | E1 results → start **E3** or **E2**; write up |

**Reading rule:** never read more than two papers without running something. Log each paper as: what they measured · how they know (correlation or intervention) · what they couldn't test.

## 8. Risks and controls

- **Text statistics, not imagination:** geometry can appear because words co-occur that way (as with months). Use novel names, random layouts and relations never seen in training text; vary wording.
- **Probe does the work:** use linear probes, control tasks and a random-model baseline.
- **Correlation ≠ use:** only the intervention (test 3) shows the map is used. Treat probes as a first step.
- **Superposition:** the map may not be linear. Try small nonlinear probes, but report linear results first.
- **Model can't do the task:** check behavior before interpretability.

## 9. Publishing

In machine learning, and especially interpretability, **journals are rarely the main venue**. The field moves through **arXiv + workshops + conferences**, with blog posts for reach:

1. **arXiv preprint** as soon as E1 has a clean result. It's citable, timestamped and free.
2. **Blog post or thread** with the key figure, posted on LessWrong / Alignment Forum and X. That's where the interpretability community reads.
3. **Submit to a workshop**, such as the mechanistic interpretability workshops at NeurIPS or ICML. Fast review and good feedback.
4. Later, a **conference paper** (ICLR, NeurIPS, ICML) once E1–E3 are complete.
5. **Email it directly** to the authors you build on (Gurnee, the *Mosaics* authors, relevant Anthropic researchers). In this field, that is often where the impact comes from.

arXiv + blog + workshop gives you formality *and* speed, not one or the other.

## 10. Key references

- Gurnee & Tegmark (2023). *Language Models Represent Space and Time.*
- Li et al. (2023). *Emergent World Representations* (Othello-GPT); Nanda et al. on linear world representations.
- *Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models* (2026). arXiv:2608.30751
- *Symmetry in language statistics shapes the geometry of model representations* (2026). arXiv:2602.15029
- Anthropic (2025). *When Models Manipulate Manifolds: The Geometry of a Counting Task.*
- Anthropic (2026). *Verbalizable Representations Form a Global Workspace in Language Models*; *Natural Language Autoencoders.*
- Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.*
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.*
- Park, Choe & Veitch. *The Linear Representation Hypothesis and the Geometry of LLMs*; Park et al., *Geometry of Categorical and Hierarchical Concepts.*
- Geiger et al. *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.*
- Gardner et al. (2022). *Toroidal topology of population activity in grid cells.* Nature.
- Amalric & Dehaene (2016). *Origins of the brain networks for advanced mathematics in expert mathematicians.* PNAS.
- Sablé-Meyer, Dehaene et al. (2025). *A geometric shape regularity effect in the human brain.* eLife.
- Shepard & Metzler (1971). *Mental rotation of three-dimensional objects.* Science.
