# E1 Spatial Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run E1. Generate text-only spatial puzzles, pick a model that solves them, then test whether the model holds and uses a per-object (x, y) map. The tests are probes, patching with a consistency check, DAS, and a steering slider. Results are logged to `results/RESULTS.md` as they arrive.

**Architecture:**
- A small Python package `geomap` holds the pure, tested logic: scenes, questions, prompts, probes, statistics and intervention helpers.
- Thin scripts `scripts/pN_*.py` run each phase, write raw outputs to `results/pN/`, and append a dated entry to `results/RESULTS.md`.
- The model runs through `nnsight` on Hugging Face models. Activations are cached from the shared prompt prefix only. The prefix runs up to the end of the object list, so one cache serves every question about a scene.

**Tech Stack:** Python 3.12, uv, PyTorch (MPS), transformers 4.57, nnsight, scikit-learn, scipy, pandas, matplotlib, pytest.

**Spec:** `docs/specs/2026-11-09-e1-spatial-map-design.md`

## Global Constraints

- **Coordinates.** North = +y and east = +x. A relation's (dx, dy) is the subject's offset from the object.
- **Grid.** Coordinates are integers in [-6, 6]². The scene's root (first-mentioned object) is at (0, 0). Step sizes are k ∈ 1..5 per component.
- **Scenes.** 3–6 objects, positioned by a random spanning tree. Compound relations appear with p = 0.3, and sentence order is shuffled with p = 0.2.
- **Margins in the main set.** Distance-type questions need a margin ≥ 1.0 unit. Direction questions need ≥ 10° from a sector boundary.
- **Prompt.** Zero-shot, using the model's chat template. Qwen3 runs with `enable_thinking=False`. No few-shot examples.
- **Prompt layout.** `describe(scene)` + `\n\nThe objects are: A, B, C, D.` + `\n\n` + question + ` Answer with one word.`
- **Readout positions.** Each name token in the object list is a per-object readout position. The "end of list" (EOL) position is the token of the list's final `.`.
- **Never a probe or patch site.** The last token of the question.
- **Layer indexing.** Layer 0 = the embedding output. Layer l = the output of decoder block l−1, for l ∈ [0, n_layers].
- **Seeds.** Every dataset is regenerable from a seed.
- **Git.** `data/`, `cache/` and `.venv/` are gitignored. `results/` and `docs/` are committed.
- **Reported numbers.** Every number in `RESULTS.md` carries a 95% bootstrap confidence interval (1000 resamples, resampled by scene or by pair).
- **Cloud spend.** Rent a GPU when a local run would take longer than about 2 hours. Spend up to about $50 without asking; ask before spending more.
- **Gated models.** No Hugging Face token is configured, and Llama and Gemma are gated. P1 screens the Qwen3 family first. Gemma 3 4B and Llama 3.2 3B are added only if the user logs in.

## Review Focus

1. **Candidate tokenization.** A name or compass word can be one token mid-sentence but several tokens as an answer. Scoring must use the summed log-probability of the whole candidate, tokenized in answer position (Task 4 test `test_candidate_ids_in_answer_context`).
2. **Token alignment in base/source pairs.** "1 unit" vs "2 units", or a two-digit number, shifts tokens and silently patches the wrong positions. Pairs must have identical token sequences outside T's relation span (Task 3 test `test_pair_prompts_aligned`).
3. **Prefix caching.** If the chat template tokenizes the prefix-only text differently from the full prompt, cached activations sit at the wrong positions (Task 3 test `test_prefix_matches_full_prompt`).
4. **Near-ties in the main set.** These make behaviour and consistency numbers noisy (Task 2 test `test_battery_respects_margins`).
5. **Probe leakage.** If a scene's objects land on both sides of the train/test split, R² is inflated. A probe trained on random targets must score about 0 (Task 7 tests `test_split_by_scene`, `test_random_targets_r2_near_zero`).

---

### Task 1: Project setup and scene generator

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/geomap/__init__.py`, `src/geomap/scenes.py`, `results/RESULTS.md`, `results/figures/.gitkeep`
- Test: `tests/test_scenes.py`

**Interfaces:**
- Produces:
  - `Relation(subj: str, obj: str, dx: int, dy: int, template_id: int)`, a frozen dataclass.
  - `Scene(names: tuple[str, ...], coords: dict[str, tuple[int, int]], relations: tuple[Relation, ...], parent: dict[str, str | None], hop: dict[str, int], seed: int)`, a frozen dataclass.
    - `names` is the object-list order.
    - `relations` is in text order.
  - `generate_scene(rng: np.random.Generator, names_pool: Sequence[str], n_objects: int, p_compound: float = 0.3, p_shuffle: float = 0.2, bound: int = 6) -> Scene`
  - `relation_text(rel: Relation) -> str`
  - `describe(scene: Scene) -> str`
  - `leaves(scene: Scene) -> list[str]`
  - `move_leaf(scene: Scene, leaf: str, dx: int, dy: int) -> Scene`
    - Rewrites the leaf's relation offset, keeping the template and its position in the text.
    - Raises `ValueError` if the leaf would collide with another object or leave the bounds.

- [ ] **Step 1: Set up the repo.**
  - `git init`.
  - `uv init --package --name geomap --python 3.12`.
  - `uv add torch transformers nnsight scikit-learn scipy pandas matplotlib`.
  - `uv add --dev pytest`.
  - `.gitignore`: `.venv/ data/ cache/ __pycache__/ .qodo/ .DS_Store`.
  - `results/RESULTS.md` with the header `# E1 results log`.
- [ ] **Step 2: Write the failing tests.**
  - `test_coords_consistent_with_relations`: for every relation, `coords[subj] == coords[obj] + (dx, dy)`.
  - `test_root_at_origin_and_hops`: the first-mentioned object is at (0, 0) with hop 0; every other object has hop = parent's hop + 1.
  - `test_distinct_positions_in_bounds`: over 500 seeds with n ∈ 3..6, all positions are distinct and within ±6.
  - `test_relation_text_cardinal_and_compound`:
    - `Relation("Ana","Bo",0,3,0)` renders a sentence containing "3 units north of Bo".
    - `(2,-1)` renders a sentence containing "1 unit south" and "2 units east".
  - `test_move_leaf_only_moves_leaf`: the moved leaf's coordinates change and every other object's coordinates are unchanged.
  - `test_deterministic_from_seed`.
- [ ] **Step 3: Run the tests and confirm they fail.** Run `uv run pytest tests/test_scenes.py -v`. Expected: ImportError.
- [ ] **Step 4: Implement `scenes.py`.**
  - Use at least 3 cardinal and at least 3 compound templates. Use "unit" for k = 1.
  - Build the tree by attaching each new object to a uniformly chosen existing object, rejection-sampling the offset.
  - With p_shuffle, shuffle the relation order while keeping the root as the first name mentioned.
- [ ] **Step 5: Run the tests and confirm they pass.** Run `uv run pytest tests/test_scenes.py -v`. Expected: all PASS.
- [ ] **Step 6: Commit.** `git commit -m "feat: project setup and scene generator"`

### Task 2: Question battery, answers and counterfactual pairs

**Files:**
- Create: `src/geomap/questions.py`
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: `Scene`, `move_leaf`, `leaves`.
- Produces:
  - `QType(str, Enum)` with values `FARTHEST, CLOSEST, DIRECTION, MORE_NORTH, MORE_EAST, CLOSER_THAN`.
  - `COMPASS = ("north","northeast","east","southeast","south","southwest","west","northwest")`.
  - `Question(qtype: QType, args: tuple[str, ...], text: str, candidates: tuple[str, ...])`, a frozen dataclass.
  - `answer_for(q: Question, coords: Mapping[str, tuple[float, float]]) -> str`. Coordinates may be floats; the slider uses off-grid positions.
  - `margin(q: Question, coords) -> float`. For distance-type questions this is in units; for DIRECTION it is in degrees.
  - `battery(scene: Scene, rng, per_type: int = 2, min_margin: float = 1.0, min_angle: float = 10.0) -> list[Question]`
  - `make_pair(scene: Scene, rng, max_tries: int = 50) -> tuple[Scene, Scene, str]`. Returns base, source and leaf T. The source moves T to a new valid offset from the same parent, with the same template and the same singular/plural form of "unit(s)" for each component.
  - `split_questions(qs, base_coords, src_coords) -> tuple[list[Question], list[Question]]`. Returns the diagnostic questions (answers differ) and the control questions (answers agree). Only questions that meet the margins under both maps are included.

**Question texts** (exact):
- "Which object is farthest from {X}?"
- "Which object is closest to {X}?"
- "In which compass direction is {Y} from {X}?"
- "Which is farther north, {Y} or {Z}?"
- "Which is farther east, {Y} or {Z}?"
- "Is {Y} closer to {X} than {Z} is? Answer yes or no."

**Candidates:**
- FARTHEST and CLOSEST: all names except X.
- DIRECTION: `COMPASS`.
- MORE_NORTH and MORE_EAST: (Y, Z).
- CLOSER_THAN: ("yes", "no").

- [ ] **Step 1: Write the failing tests.**
  - `test_answer_for_known_layout`, with coords {A:(0,0), B:(3,0), C:(0,-4), D:(1,1)}:
    - farthest from A = C
    - closest to A = D
    - direction from A to D = "northeast"
    - more north of (B, D) = D
    - "is B closer to A than C is" = "yes"
  - `test_battery_respects_margins`: over 200 scenes, every question meets `min_margin` / `min_angle`.
  - `test_make_pair_moves_one_leaf_same_template`.
  - `test_split_questions_partition`: diagnostic answers differ between the two maps, control answers agree.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `uv run pytest tests/test_questions.py -v`.
- [ ] **Step 3: Implement `questions.py`** with Euclidean distances. DIRECTION uses atan2 binned into 8 sectors of 45° centred on the compass directions.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `uv run pytest tests/test_questions.py -v`.
- [ ] **Step 5: Commit.** `git commit -m "feat: question battery and counterfactual pairs"`

### Task 3: Prompt builder and token positions

**Files:**
- Create: `src/geomap/prompts.py`
- Test: `tests/test_prompts.py`. Uses the real `Qwen/Qwen3-1.7B` tokenizer; only the tokenizer files are downloaded.

**Interfaces:**
- Consumes: `Scene`, `describe`, `Question`.
- Produces:
  - `NAME_POOLS: dict[str, list[str]]`, with keys `"people"` (≥ 80 common first names) and `"nonce"` (≥ 80 pronounceable non-words).
  - `single_token_names(tok, pool: list[str]) -> list[str]`. Keeps names that are exactly one token both with a leading space and at the start of a sentence.
  - `PromptSpec(input_ids: list[int], list_pos: dict[str, int], eol_pos: int, relation_spans: dict[str, tuple[int, int]], prefix_len: int)`, a dataclass.
    - `relation_spans` is keyed by subject name and holds half-open token ranges.
    - `prefix_len = eol_pos + 1`.
  - `build_prompt(scene: Scene, question: Question | None, tok, family: str) -> PromptSpec`. With `question=None`, it returns only the prefix tokens.
  - `FAMILIES = {"qwen3", "gemma3", "llama3"}`. These control the chat-template kwargs, e.g. `enable_thinking=False` for qwen3.

- [ ] **Step 1: Write the failing tests.**
  - `test_prefix_matches_full_prompt`: `build_prompt(s, None).input_ids == build_prompt(s, q).input_ids[:prefix_len]` for 50 scenes and every question type.
  - `test_list_pos_decodes_to_names`: `tok.decode(ids[list_pos[n]]).strip() == n`.
  - `test_eol_is_period`: the decoded `eol_pos` token ends with ".".
  - `test_relation_span_contains_subject_and_object`.
  - `test_pair_prompts_aligned`: for 200 `make_pair` outputs:
    - the base and source prompts are the same length, and
    - every token outside T's relation span is identical.

    If a pair fails this check, `make_pair` must be resampled. Add a helper `aligned(base_spec, src_spec, leaf) -> bool`. Use it in `make_pair_aligned(scene, rng, tok, family, max_tries: int = 50) -> tuple[Scene, Scene, str]`, which lives in `prompts.py` and returns the same tuple as `make_pair`.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `uv run pytest tests/test_prompts.py -v`.
- [ ] **Step 3: Implement `prompts.py`.**
  - Map character offsets to token indices with `return_offsets_mapping=True` on the fully rendered chat text.
  - The prefix is the rendered text cut at the EOL character, then re-tokenized and asserted equal to the full prompt's first tokens.
- [ ] **Step 4: Run the tests and confirm they pass.** Run `uv run pytest tests/test_prompts.py -v`.
- [ ] **Step 5: Commit.** `git commit -m "feat: prompt builder with readout positions"`

### Task 4: Model loading, scoring, results log, P0 benchmark

**Files:**
- Create: `src/geomap/models.py`, `src/geomap/scoring.py`, `src/geomap/report.py`, `scripts/p0_benchmark.py`
- Test: `tests/test_scoring.py`. Uses a tiny randomly initialised Qwen3 config (2 layers, d = 64) with the real Qwen3 tokenizer, so no weights are downloaded.

**Interfaces:**
- Consumes: `PromptSpec`, `Question`.
- Produces:
  - `MODELS = {"qwen3-1.7b": "Qwen/Qwen3-1.7B", "qwen3-4b": "Qwen/Qwen3-4B", "qwen3-8b": "Qwen/Qwen3-8B", "gemma3-4b": "google/gemma-3-4b-it", "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"}`
  - `LoadedModel(lm, tok, key: str, family: str, n_layers: int, d_model: int)`, a dataclass.
  - `load(key: str, device: str = "mps", dtype=torch.bfloat16, random_init: bool = False) -> LoadedModel`
  - `load_tiny_for_tests() -> LoadedModel`
  - `layer_out(m: LoadedModel, layer: int)`: an nnsight proxy for the residual stream at `layer`. It must handle decoder layers that return a tuple as well as ones that return a tensor. **This is the only place that knows the module path.**
  - `Edit = Callable[[LoadedModel, list[int]], None]`. It is called inside the trace with `row_items`, the item index of each batch row.
  - `score(m, items: list[tuple[PromptSpec, Question]], edit: Edit | None = None, batch_size: int = 16) -> list[dict[str, float]]`. Returns the summed log-probability of each candidate. Each candidate is tokenized in answer position and appended to the prompt. Batches are right-padded so that the prompt positions line up across rows.
  - `predict(scores: dict[str, float]) -> str`
  - `append_entry(title: str, body_md: str, figures: Sequence[Path] = ()) -> None`. Appends `## YYYY-MM-DD HH:MM · {title}` plus the body plus relative image links to `results/RESULTS.md`.

- [ ] **Step 1: Write the failing tests.**
  - `test_candidate_ids_in_answer_context`: the token ids scored for "northeast" equal the tokenization of the answer text right after the assistant prefix.
  - `test_score_matches_manual_logprob`: on the tiny model, `score` equals a manual teacher-forced log-probability sum, to 1e-4 in float32.
  - `test_noop_edit_is_identity`: an edit that writes `layer_out` back unchanged leaves the scores unchanged.
  - `test_append_entry_writes_header`: run against a temporary results file.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `uv run pytest tests/test_scoring.py -v`.
- [ ] **Step 3: Implement `models.py`, `scoring.py` and `report.py`.**
- [ ] **Step 4: Run the tests and confirm they pass.** Run `uv run pytest tests/test_scoring.py -v`.
- [ ] **Step 5: Implement `scripts/p0_benchmark.py --model qwen3-1.7b --n 64`.**
  - Time the cost per scored item and the cost per cached prefix.
  - Extrapolate to P1 (500 scenes × battery) and P3 (100 pairs × 3 sites × every other layer).
  - Append a "P0 benchmark" entry.
- [ ] **Step 6: Run the benchmark.** Run `uv run python scripts/p0_benchmark.py --model qwen3-1.7b --n 64`. Expected: a new entry in `RESULTS.md` with measured throughput and an estimate in hours for each later phase.
- [ ] **Step 7: Commit.** `git commit -m "feat: model loading, scoring, benchmark"`

### Task 5: P1 behaviour screen, choosing the model

**Files:**
- Create: `scripts/p1_behavior.py`, `src/geomap/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `generate_scene`, `battery`, `build_prompt`, `score`, `predict`, `append_entry`.
- Produces:
  - `analysis.bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, stat=np.mean) -> tuple[float, float, float]`
  - `analysis.chance(q: Question) -> float`
  - `analysis.error_signature(df) -> dict`. Fits a logistic regression of correctness on [hop depth of the queried objects, log margin] and returns coefficients with bootstrap CIs.
  - `results/p1/{model}.parquet`, with columns `scene_id, n_objects, qtype, max_hop, margin, correct, chance, scores`.

**Decision rule (from the spec):** pick the smallest model with ≥ 75% overall accuracy on 4-object scenes and accuracy clearly above chance on questions involving objects 2+ hops from the root (bootstrap CI lower bound > chance + 0.15). If none passes, fall back to qwen3-8b, then a 14B model on a rented GPU, then 3-object scenes.

- [ ] **Step 1: Write the failing tests.**
  - `test_bootstrap_ci_covers_mean`.
  - `test_chance_values`: FARTHEST with 4 objects = 1/3, DIRECTION = 1/8, yes/no = 1/2.
  - `test_error_signature_recovers_planted_effect`: on synthetic data where correctness depends only on margin, the margin coefficient is > 0 with its CI excluding 0, and the hop coefficient's CI includes 0.
- [ ] **Step 2: Run the tests and confirm they fail, then implement `analysis.py` and confirm they pass.** Run `uv run pytest tests/test_analysis.py -v`.
- [ ] **Step 3: Implement `scripts/p1_behavior.py`.**
  - Arguments: `--models ... --n-scenes 500 --seed 1`.
  - Data: 500 scenes per object count n ∈ {3, 4, 5}, using both name pools.
  - Also a 200-scene near-tie set where margins are in [0.25, 1.0).
  - Also a chain-of-thought subset: 100 scenes on Qwen3 with thinking enabled, generated with `max_new_tokens=1024`. The answer is the last candidate string after `</think>`.
  - Also an object-list ablation: the same 200 scenes with and without the "The objects are: …" line.
  - Writes figures `results/figures/p1_accuracy.png` and `p1_error_signature.png`, and an entry containing:
    - accuracy by model × qtype × n, with CIs
    - the hop-depth breakdown
    - the effect of chain of thought on vs. off
    - the effect of the object list
    - the chosen model, with a one-line reason
- [ ] **Step 4: Run P1.** Run `uv run python scripts/p1_behavior.py --models qwen3-1.7b qwen3-4b qwen3-8b`. Expected: a "P1 behaviour screen" entry that names a chosen model, or names the fallback being triggered. **If the object list changes accuracy by more than 5 points, note it in the entry and redesign the list line before continuing.**
- [ ] **Step 5: Commit.** Commit code and results: `git commit -m "exp: P1 behaviour screen"`

### Task 6: Activation cache and preregistration

**Files:**
- Create: `src/geomap/cache.py`, `docs/preregistration.md`, `scripts/build_cache.py`
- Test: `tests/test_cache.py` (tiny model)

**Interfaces:**
- Consumes: `LoadedModel`, `layer_out`, `build_prompt(…, None, …)`.
- Produces:
  - `cache_prefix(m, specs: list[PromptSpec], layers: list[int], batch_size: int = 16) -> dict[str, torch.Tensor]`.
    - `"list"`: float16 `[n_scenes, len(layers), 6, d]`, NaN-padded, slots in `scene.names` order.
    - `"eol"`: float16 `[n_scenes, len(layers), d]`.
  - `save_cache(path, cache, scenes)` and `load_cache(path) -> (cache, scenes)`. Path: `cache/{model}/{dataset}.pt`.
  - `docs/preregistration.md`: the threshold table, the branches if a phase is weak, and the cut order, copied verbatim from spec §4. It is committed **before** any P2 numbers exist.

- [ ] **Step 1: Write the failing test.** `test_cache_matches_full_prompt_run`: the activations at `list_pos` from the prefix-only cache equal those from a full prompt with a question, to 1e-3, on the tiny model.
- [ ] **Step 2: Implement `cache.py`, confirm the test passes, and write and commit the preregistration.** Run `uv run pytest tests/test_cache.py -v`, then `git commit -m "docs: preregistration before P2"`.
- [ ] **Step 3: Build caches on the chosen model, for all layers.**
  - Main: 3,000 scenes with n ∈ 3..6.
  - Joint EOL probe: 1,000 four-object scenes.
  - Random-init control: the same 3,000 scenes on the same architecture with `random_init=True`.
  - Run: `uv run python scripts/build_cache.py --model <chosen> --dataset main --n 3000 --seed 2` (plus `--random-init`).

  Expected: `.pt` files under `cache/`, and a run summary with sizes printed.
- [ ] **Step 4: Commit the code.** `git commit -m "feat: activation cache"`

### Task 7: P2 probes and controls

**Files:**
- Create: `src/geomap/probes.py`, `scripts/p2_probes.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Consumes: `load_cache`, `Scene`, `bootstrap_ci`.
- Produces:
  - `scene_split(n_scenes: int, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]`. Splits by scene.
  - `centered_xy(acts: np.ndarray, scenes, scene_idx) -> tuple[X, Y, meta]`.
    - Per scene, it subtracts the mean activation and the mean coordinate over that scene's objects.
    - `meta` holds the scene id, name and hop.
  - `fit_probe(X, Y) -> RidgeCV`, with `alphas=np.logspace(-2, 5, 15)`.
  - `probe_by_layer(cache, scenes, split, control: Literal[None, "shuffled", "stated_relation"] = None, nonlinear: bool = False) -> pd.DataFrame`. Columns: `layer, r2, r2_lo, r2_hi, r2_hop1, r2_hop2, r2_hop3p`.
    - `"shuffled"` permutes object-level targets across the dataset before centring.
    - `"stated_relation"` regresses on [own relation (dx, dy), hop one-hot] instead of activations.
    - `nonlinear=True` uses an `MLPRegressor(hidden_layer_sizes=(256,))`.
  - `joint_eol_probe(cache_eol, scenes4, split) -> pd.DataFrame`. Predicts all 8 centred coordinates of a 4-object scene from the EOL activation.
  - `rsa_by_layer(cache, scenes) -> pd.DataFrame`. Per scene, the Spearman correlation between Euclidean distances among the centred activations and the true distances; averaged across scenes, with a CI.
  - `probe_directions(probe) -> np.ndarray [d, 2]`, the pseudo-inverse of the probe weights, used for steering in Task 10.

- [ ] **Step 1: Write the failing tests.**
  - `test_split_by_scene`: no scene id appears in both the train and test sets.
  - `test_planted_linear_map_recovered`: with synthetic activations = A·coords + noise, R² > 0.95.
  - `test_random_targets_r2_near_zero`: |R²| < 0.05.
  - `test_centering_removes_translation`: adding a per-scene offset to the coordinates does not change R².
- [ ] **Step 2: Run the tests and confirm they fail, implement, and confirm they pass.** Run `uv run pytest tests/test_probes.py -v`.
- [ ] **Step 3: Implement `scripts/p2_probes.py --model <chosen>`.**
  - Runs at every layer:
    - the real probe
    - the shuffled control
    - the layer-0 probe (layer 0 is already included)
    - the random-init model
    - the stated-relation baseline
    - RSA
    - the joint EOL probe
  - If the best real R² is < 0.3, it also runs `nonlinear=True`.
  - It saves the best layer's probe to `results/p2/probe.pkl`.
  - Figures: `p2_r2_by_layer.png` (real vs. controls, CI bands) and `p2_r2_by_hop.png`.
  - The entry compares the numbers with the preregistered thresholds and states which way the evidence points.
- [ ] **Step 4: Run P2.** Run `uv run python scripts/p2_probes.py --model <chosen>`. Expected: a "P2 probes" entry with the best layer, R² with CI, selectivity, the R² gap between hop depths, and the RSA result.
- [ ] **Step 5: Commit.** `git commit -m "exp: P2 probes"`

### Task 8: P3 patching and consistency

**Files:**
- Create: `src/geomap/patching.py`, `scripts/p3_patching.py`
- Test: `tests/test_patching.py` (tiny model)

**Interfaces:**
- Consumes: `make_pair_aligned`, `split_questions`, `battery`, `build_prompt`, `score`, `predict`, `layer_out`.
- Produces:
  - `Site = Literal["list", "relation", "eol", "all"]`. The sites:
    - `"list"` = T's position in the object list.
    - `"relation"` = T's relation span.
    - `"eol"` = the end-of-list position.
    - `"all"` = every prefix position. This is the positive control.
  - `site_positions(spec: PromptSpec, site: Site, leaf: str) -> list[int]`
  - `patch_edit(src_acts: dict[int, torch.Tensor], layer: int, positions: list[int]) -> Edit`
  - `run_pair(m, base: Scene, src: Scene, leaf: str, qs: list[Question], site: Site, layer: int) -> pd.DataFrame`. Columns: `qtype, diagnostic, base_ans, src_ans, model_ans`.
  - `consistency(df_all: pd.DataFrame) -> pd.DataFrame`. Per (site, layer) it gives:
    - `flip_rate` by qtype on diagnostic questions
    - `control_stability`
    - `joint_consistency`: the fraction of pairs where every diagnostic answer equals `src_ans` and every control answer equals `base_ans`

    Each with a CI from resampling pairs.

- [ ] **Step 1: Write the failing tests.**
  - `test_patch_with_self_is_identity`: source = base gives scores identical to no edit.
  - `test_patch_all_at_last_layer_equals_source_run`: patching every prefix position at the final layer gives the source scores, to 1e-3.
  - `test_site_positions`: "list" returns one index that decodes to T; "relation" returns T's span.
  - `test_joint_consistency_math`: on a synthetic dataframe, the metric equals a hand-computed value.
- [ ] **Step 2: Run the tests and confirm they fail, implement, and confirm they pass.** Run `uv run pytest tests/test_patching.py -v`.
- [ ] **Step 3: Implement `scripts/p3_patching.py --model <chosen>`.**
  - **Stage A (sweep):** 100 pairs × sites {list, relation, eol, all} × every other layer.
  - **Stage B (confirm):** 400 new pairs at the best two layers for each site.
  - If the P0 estimate for this run is over 2 hours locally, print the command for running it on a rented GPU and stop.
  - Figures:
    - `p3_sweep.png`: a heatmap of joint consistency over site × layer.
    - `p3_flip_by_qtype.png`.
  - The entry compares the results with the thresholds and with the H-map / H-rel table in spec §2.
- [ ] **Step 4: Run P3.** Run `uv run python scripts/p3_patching.py --model <chosen>`. Expected: a "P3 patching" entry with the best (site, layer), joint consistency with CI, and per-question-type flip rates. The positive control "all" should have joint consistency near 1. **If it doesn't, stop and debug alignment before trusting anything else.**
- [ ] **Step 5: Commit.** `git commit -m "exp: P3 patching and consistency"`

### Task 9: P4a Distributed Alignment Search

**Files:**
- Create: `src/geomap/das.py`, `scripts/p4_das.py`
- Test: `tests/test_das.py` (tiny model)

**Interfaces:**
- Consumes: `make_pair_aligned`, `split_questions`, `site_positions`, `score`.
- Produces:
  - `SubspaceIntervention(d_model: int, k: int)`, an `nn.Module`. `Q` is a `d × k` matrix kept orthonormal with `torch.nn.utils.parametrizations.orthogonal`. `forward(h_base, h_src) = h_base + Q Qᵀ (h_src − h_base)`.
  - `train_das(m, train_pairs, layer: int, site: Site = "list", k: int = 2, epochs: int = 5, lr: float = 1e-3, batch_size: int = 8) -> SubspaceIntervention`.
    - Loss: cross-entropy of a softmax over the candidates' summed log-probabilities, with the source-map answer as the target, over all questions (diagnostic and control).
    - The model is frozen.
    - Training may use plain PyTorch forward hooks on the underlying Hugging Face module instead of nnsight.
  - `eval_das(m, iv, test_pairs, layer, site) -> pd.DataFrame`: interchange accuracy by qtype, with CIs.

- [ ] **Step 1: Write the failing tests.**
  - `test_q_orthonormal`: QᵀQ ≈ I_k.
  - `test_full_rank_equals_patch`: with k = d on the tiny model, the output equals a full patch at that site.
  - `test_k0_like_identity`: with Q zeroed out via a mask, the output equals the base run.
- [ ] **Step 2: Run the tests and confirm they fail, implement, and confirm they pass.** Run `uv run pytest tests/test_das.py -v`.
- [ ] **Step 3: Run DAS.** Run `uv run python scripts/p4_das.py --model <chosen> --layer <best from P3> --site list --k 2` with 1,000 training pairs and 400 test pairs. If there is time and budget, also k ∈ {4, 8}, though this is cut early if time runs short. Expected: a "P4 DAS" entry with interchange accuracy per question type for k = 2 vs. the 0.7 threshold, and the figure `p4_das.png`.
- [ ] **Step 4: Commit.** `git commit -m "exp: P4 DAS"`

### Task 10: P4b Steering slider

**Files:**
- Create: `src/geomap/steering.py`, `scripts/p4_slider.py`
- Test: `tests/test_steering.py`

**Interfaces:**
- Consumes:
  - `probe_directions` (Task 7)
  - optionally a `SubspaceIntervention` (Task 9)
  - `answer_for`, `score`, `layer_out`
- Produces:
  - `slider(m, scene: Scene, leaf: str, layer: int, dirs: np.ndarray, deltas: np.ndarray, qs: list[Question]) -> pd.DataFrame`
    - `dirs` is `[d, 2]` and maps a coordinate change to an activation change.
    - `deltas` is `[n_steps, 2]`, e.g. a 0.5-unit grid in [-6, 6] along x and then along y.
    - Columns: `step, dx, dy, qid, model_ans, predicted_ans, p_model`. `predicted_ans` comes from `answer_for` with T at its base position + (dx, dy).
  - `slider_agreement(df) -> dict`. Returns the agreement rate and the correlation between predicted and observed flip points (with CIs over scenes).
  - `natural_range(m, scene, leaf, layer, steered_norms, ref_norms) -> float`. Returns the fraction of steered activation norms that fall within the [1st, 99th] percentiles of the reference norms.

- [ ] **Step 1: Write the failing tests.**
  - `test_predicted_flip_points`: in a known layout, the predicted farthest-from-B answer switches to T exactly at the geometric threshold.
  - `test_agreement_math`: on a synthetic dataframe.
  - `test_zero_delta_is_identity`.
- [ ] **Step 2: Run the tests and confirm they fail, implement, and confirm they pass.** Run `uv run pytest tests/test_steering.py -v`.
- [ ] **Step 3: Implement `scripts/p4_slider.py`.**
  - 100 scenes, a sweep along x and then along y, at the best layer, at T's list position.
  - Use the DAS subspace if P4a reached ≥ 0.7; otherwise use the probe directions.
  - **The demo figure** `p4_slider_demo.png`: one scene, with T's x on the horizontal axis, a panel per question showing model probabilities, and dashed lines at the geometric thresholds.
  - An aggregate figure.
  - **Skip with a logged note** if P2 and P4a both found no usable direction (spec branch).
- [ ] **Step 4: Run the slider.** Run `uv run python scripts/p4_slider.py --model <chosen>`. Expected: a "P4 slider" entry with agreement, flip-point correlation and the natural-range fraction.
- [ ] **Step 5: Commit.** `git commit -m "exp: P4 slider"`

### Task 11: Scale check and E1 summary

**Files:**
- Create: `scripts/p5_summary.py`

**Interfaces:**
- Consumes: every `results/p*/` output.
- Produces:
  - `results/figures/fig1_behavior.png` … `fig4_slider.png`, the four core figures named in the spec.
  - A "E1 summary" entry giving the verdict (H-map / H-rel / Mixed) against the preregistration, its main caveats, and comparisons with Tehenan et al. 2025 and An et al. 2026.

- [ ] **Step 1: Scale check, first on the cut list.** Rerun Tasks 5–8 on the next larger model. This means qwen3-8b, or 14B on a rented GPU if qwen3-8b was the main model, using the same seeds via `--model`. Append a "Scale check" entry.
- [ ] **Step 2: Build the figures and summary.** Implement and run `uv run python scripts/p5_summary.py`. Expected: the four figures exist and the summary entry is appended.
- [ ] **Step 3: Commit.** `git commit -m "exp: E1 summary"`
