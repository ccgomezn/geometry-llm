# Study 2 Pilot (triangle ladder, answers only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the triangle scenes, the four-rung question ladder, the baselines and a resumable Colab pilot that
tells us whether Gemma 3 12B can judge "inside the triangle" well enough to run the main study.

**Architecture:** Two new library modules: `triangles.py` for geometry, scenes and ladder questions, and `study2.py`
for baselines, balanced accuracy, the pilot summary and the go rule. They sit on top of Study 1's prompt building
and scoring, which stay unchanged. Then one script, `scripts/s2_pilot.py`, in the P9 pattern (`df_step`, `.done`,
`notify`, `append_entry`), one Colab notebook, and the pilot preregistration.

**Tech Stack:** Python 3.12, numpy, pandas, scikit-learn, torch and nnsight (through `geomap.scoring`), pytest, uv.

**Spec:** `docs/specs/2026-11-13-study2-triangle-dial-design.md` (Step 1 of the spec; Step 2 gets its own plan after
the pilot verdict).

## Global Constraints

- No Study 1 file is moved or renamed. Existing behaviour is unchanged: `QType`, `all_questions`, `battery` and
  `generate_scene` stay as they are.
- No new dependencies.
- Tests need no GPU. The whole suite (`uv run pytest`) stays green.
- The script supports `--smoke --out` and runs end to end on the tiny test model.
- Colab: the same setup cells as `colab/p9_map_answers.ipynb` (pins, TensorFlow fix, Drive symlink, secrets).
- The preregistration is committed before any real (non-smoke) data exists.

## Rulings against the spec text

- **Rungs 3–4 question types live in `triangles.py` as a new enum `TriQType`, not in `QType`.** The spec says
  "alongside the existing ones in `questions.py`". But `all_questions` and `battery` iterate over `QType`, so adding
  members there would change Study 1's question sets and break its reproduction. `Question` objects are still used,
  so `build_prompt` and `score` work unchanged.

## Review Focus

1. **The generator can loop forever** when a triangle has no grid point for the drawn stratum. It must redraw the
   triangle. Test: it terminates with a small bound.
2. **Points on an edge, or tied comparisons, must never reach a question.** "Inside" is never ambiguous, and rung 1–3
   items have a margin of at least 1.
3. **A subset with only one answer class** (for example, rung 3 with no "yes" in a smoke run) must give a NaN
   balanced accuracy, not a crash, and the summary and verdict must handle it.
4. **Balanced accuracy for two-choice rungs needs a meaningful positive class.** Argument order is randomised, and the
   positive class is "the first-named object is the answer".
5. **A rerun after a code change can reuse a stale model-answers parquet.** The file name carries `n` and the seed, so
   a different configuration never reads old answers.

---

### Task 1: Triangle geometry and scene generator

**Files:**
- Create: `src/geomap/triangles.py`
- Test: `tests/test_triangles.py`

**Interfaces:**
- Consumes: `geomap.scenes.Scene`, `Relation`, `TEMPLATES`, `describe`.
- Produces:
  - `Point = tuple[int, int]`
  - `in_triangle(p, a, b, c) -> bool`: strict interior; on an edge → False.
  - `edge_distance(p, a, b, c) -> float`: minimum Euclidean distance from p to the three edge segments.
  - `in_box(p, a, b, c) -> bool`: inclusive axis-aligned bounding box.
  - `stratum(p, a, b, c) -> str`: one of `"inside"`, `"box_hard"` (outside the triangle, inside the box), `"outside_box"`.
  - `@dataclass(frozen=True) class TriScene: scene: Scene; corners: tuple[str, str, str]; point: str; stratum: str`
  - `generate_tri_scene(rng, names_pool, bound=8, min_area=4.0, min_edge_dist=0.5, p_strata=(0.5, 0.35, 0.15)) -> TriScene`

**Generator algorithm** (decided here):
1. Draw 4 distinct names, which become corners A, B, C and point P, in that order.
2. Place A at (0, 0). Draw B and C uniformly from the integer grid [-bound, bound]², all three distinct, and keep the
   triangle only if its area is at least `min_area`.
3. Draw the stratum from `p_strata` in the order (inside, box_hard, outside_box).
4. The eligible P positions are the grid points in [-bound, bound]² that are not a corner, have
   `edge_distance >= min_edge_dist`, and fall in the drawn stratum. Pick one uniformly. If none exist, redraw the
   triangle and keep the stratum.
5. P's parent is a uniformly random corner.
6. Relations, in text order: B relative to A, C relative to B, P relative to its parent. Template ids are random.
7. Remaining Scene fields:
   - `parent`: A → None, B → A, C → B, P → its parent.
   - `hop`: A = 0, B = 1, C = 2, P = hop(parent) + 1.
   - `names`: a random permutation of the 4 names.
   - `seed`: as in `generate_scene`.

- [ ] **Step 1: Write the failing tests**

```python
T = ((0, 0), (4, 0), (0, 4))

def test_in_triangle():
    assert in_triangle((1, 1), *T) and not in_triangle((3, 3), *T)
    assert not in_triangle((2, 0), *T)          # on an edge
    assert not in_triangle((5, 5), *T)

def test_edge_distance():
    assert edge_distance((1, 1), *T) == pytest.approx(1.0)
    assert edge_distance((2, 2), *T) == pytest.approx(0.0)   # on the hypotenuse

def test_stratum():
    assert stratum((1, 1), *T) == "inside"
    assert stratum((3, 3), *T) == "box_hard"
    assert stratum((5, 0), *T) == "outside_box"

def test_generator_invariants():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ts = generate_tri_scene(rng, POOL)
        s, (a, b, c), p = ts.scene, ts.corners, ts.point
        # coordinates follow from the relations (walk the tree from A at the origin)
        assert coords_from_relations(s) == s.coords
        A, B, C, P = (s.coords[n] for n in (a, b, c, p))
        assert abs((B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1])) / 2 >= 4
        assert edge_distance(P, A, B, C) >= 0.5
        assert stratum(P, A, B, C) == ts.stratum
        assert s.parent[p] in ts.corners and s.parent[c] == b and s.parent[b] == a
        assert all(abs(x) <= 8 and abs(y) <= 8 for x, y in s.coords.values())
        assert s.relations[0].subj == b and s.relations[0].obj == a

def test_strata_proportions():
    rng = np.random.default_rng(1)
    st = [generate_tri_scene(rng, POOL).stratum for _ in range(1000)]
    assert 0.45 <= st.count("inside") / 1000 <= 0.55
    assert st.count("box_hard") / 1000 >= 0.30

def test_generator_terminates_small_bound():      # Review Focus 1
    rng = np.random.default_rng(2)
    for _ in range(50):
        generate_tri_scene(rng, POOL, bound=3, min_area=2.0)
```

`POOL` is a list of 8 plain names. `coords_from_relations` is a small test helper that places the root at (0, 0)
and adds each relation's (dx, dy) to its object's position.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_triangles.py -v`
Expected: FAIL with ImportError (`geomap.triangles` does not exist).

- [ ] **Step 3: Implement the Interfaces above in `src/geomap/triangles.py`**

Use the algorithm above. The strict interior test uses the signs of the three cross products; all must be strictly
the same sign.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_triangles.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/geomap/triangles.py tests/test_triangles.py
git commit -m "study2: triangle geometry and scene generator"
```

---

### Task 2: The question ladder

**Files:**
- Modify: `src/geomap/triangles.py`
- Test: `tests/test_triangles.py`

**Interfaces:**
- Consumes: Task 1's `TriScene`, `in_triangle`. From Study 1: `Question`, `QType`, `make_question`, `answer_for`, and
  `build_prompt` (test only).
- Produces:
  - `class TriQType(str, Enum): EXTREME = "extreme"; INSIDE = "inside"`
  - `@dataclass(frozen=True) class LadderQ: question: Question; rung: int; truth: str; positive: str`
  - `ladder(ts: TriScene, rng) -> list[LadderQ]`

**Contents per scene** (P is `ts.point`; corners are named in `ts.corners` order):

| Rung | Questions | Kept when | `positive` |
|---|---|---|---|
| 1 | `MORE_NORTH` and `MORE_EAST` between P and its parent; argument order random | that axis differs by ≥ 1 | `args[0]` |
| 2 | the same two types between P and each non-parent corner | that axis differs by ≥ 1 | `args[0]` |
| 3 | for each of north/east/south/west: `"Is {P} farther {dir} than all of {X}, {Y} and {Z}? Answer yes or no."` Truth is "yes" iff P is strictly beyond every corner on that axis. `Question(TriQType.EXTREME, (P, X, Y, Z), text, ("yes", "no"))` | P's coordinate differs by ≥ 1 from the most extreme corner's on that axis | `"yes"` |
| 4 | exactly one: `"Is {P} inside the triangle formed by {X}, {Y} and {Z}? Answer yes or no."` Truth comes from `in_triangle`. | always | `"yes"` |

Rungs 1–2 use Study 1's `make_question`, and `truth = answer_for(q, coords)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ladder_truths_and_rungs():
    rng = np.random.default_rng(3)
    for _ in range(100):
        ts = generate_tri_scene(rng, POOL)
        items = ladder(ts, rng)
        co, p, par = ts.scene.coords, ts.point, ts.scene.parent[ts.point]
        assert sum(it.rung == 4 for it in items) == 1
        for it in items:
            q = it.question
            if it.rung in (1, 2):
                other = next(a for a in q.args if a != p)
                assert (other == par) == (it.rung == 1) and other in ts.corners
                assert it.truth == answer_for(q, co) and it.positive == q.args[0]
            else:
                assert q.candidates == ("yes", "no") and it.positive == "yes"
            if it.rung == 4:
                assert (it.truth == "yes") == in_triangle(co[p], *(co[c] for c in ts.corners))
            if it.rung == 3:
                ax, sign = {"north": (1, 1), "east": (0, 1), "south": (1, -1), "west": (0, -1)}[q.text.split()[3]]
                beyond = all(sign * (co[p][ax] - co[c][ax]) > 0 for c in ts.corners)
                assert (it.truth == "yes") == beyond

def test_ladder_prompts_build():
    m = load_tiny_for_tests()
    pool = single_token_names(m.tok, NAME_POOLS["people"])
    rng = np.random.default_rng(4)
    for _ in range(5):
        ts = generate_tri_scene(rng, pool)
        for it in ladder(ts, rng):
            spec = build_prompt(ts.scene, it.question, m.tok, m.family)
            assert set(spec.question_pos) == set(it.question.args)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_triangles.py -v`
Expected: the 2 new tests FAIL with ImportError on `ladder`; the 6 from Task 1 pass.

- [ ] **Step 3: Implement `TriQType`, `LadderQ` and `ladder` in `src/geomap/triangles.py` as specified above**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_triangles.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/geomap/triangles.py tests/test_triangles.py
git commit -m "study2: four-rung question ladder"
```

---

### Task 3: Balanced accuracy, baselines, pilot summary and go rule

**Files:**
- Create: `src/geomap/study2.py`
- Test: `tests/test_study2.py`

**Interfaces:**
- Consumes: `TriScene`, `generate_tri_scene`, `in_box` (Tasks 1–2); `geomap.analysis.bootstrap_ci`.
- Produces:
  - `balanced_accuracy(truth, pred, positive) -> float`: the mean of accuracy on positive-class items
    (`truth == positive`) and on the rest. NaN if either class is empty.
  - `ba_ci(truth, pred, positive, groups, n_boot=1000, seed=0) -> tuple[float, float, float]`: a scene-bootstrap
    95% CI. It may reuse `bootstrap_ci` by encoding (correct, is_positive) as one value per item, e.g.
    `correct + 2 * is_positive`. Resamples where a class is missing are dropped (`nanquantile`).
  - `box_rule(tss) -> np.ndarray`: "yes" iff P is in the bounding box.
  - `feature_matrix(tss) -> np.ndarray` of shape (n, 9). Columns: B − A (dx, dy), C − B (dx, dy), P − parent (dx, dy),
    and a one-hot of which corner is P's parent (3 columns). These are the stated offsets only.
  - `fit_feature_rule(train_tss) -> LogisticRegression` (`max_iter=1000`), fitted to rung-4 truth.
    `feature_rule(model, tss) -> np.ndarray` of "yes"/"no".
  - `centroid_score(ts) -> float`: P's distance to the centroid divided by the mean centroid-to-corner distance.
    `fit_centroid_rule(train_tss) -> float`: the threshold that maximises balanced accuracy on train.
    `centroid_rule(tss, thr) -> np.ndarray`.
  - `pilot_summary(df, base_preds, n_boot=1000, seed=0) -> pd.DataFrame`:
    - `df` has one row per question, with columns `scene, rung, stratum, truth, pred, positive, mass`.
    - `base_preds` maps a baseline name to an array of rung-4 predictions in scene order.
    - Output rows have columns `who` (`"model"` or a baseline name), `rung`, `subset`, `n`, `bal_acc`, `lo`, `hi`,
      `acc`, `mean_prob_mass`.
    - The model gets a row with subset `all` for each rung, plus rung-4 rows for subset `hard` (inside + box_hard)
      and for each stratum.
    - Each baseline gets a rung-4 row with subset `hard`.
  - `verdict_pilot(summary, floor=0.65, map_free=("box", "feature")) -> str`, using the model's rung-4 `hard` row:
    - `"go"`: `bal_acc >= floor`, `lo > 0.5`, and `bal_acc` above every map-free baseline's `bal_acc`;
    - `"weak pass"`: `lo > 0.5` and above the map-free baselines, but `bal_acc < floor`;
    - `"fail"`: anything else, including NaN.
    - The centroid rule never affects the verdict.

- [ ] **Step 1: Write the failing tests**

```python
def test_balanced_accuracy():
    assert balanced_accuracy(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"], "yes") == 0.75
    assert np.isnan(balanced_accuracy(["no", "no"], ["no", "yes"], "yes"))      # Review Focus 3

def test_two_choice_positive_class():
    # positive = args[0] per item, so `positive` may be an array
    assert balanced_accuracy(["Ann", "Bo"], ["Ann", "Ann"], np.array(["Ann", "Ann"])) == 0.5

def test_box_rule_at_chance_on_hard():
    rng = np.random.default_rng(5)
    tss = [generate_tri_scene(rng, POOL) for _ in range(500)]
    hard = [t for t in tss if t.stratum in ("inside", "box_hard")]
    truth = ["yes" if t.stratum == "inside" else "no" for t in hard]
    assert balanced_accuracy(truth, box_rule(hard), "yes") == 0.5

def test_feature_matrix():
    rng = np.random.default_rng(6)
    t = generate_tri_scene(rng, POOL)
    X = feature_matrix([t]); co = t.scene.coords; a, b, c = t.corners
    assert X.shape == (1, 9)
    assert tuple(X[0, :2]) == (co[b][0] - co[a][0], co[b][1] - co[a][1])
    assert X[0, 6:].sum() == 1

def test_centroid_rule_beats_chance_on_train():
    rng = np.random.default_rng(7)
    tss = [generate_tri_scene(rng, POOL) for _ in range(400)]
    thr = fit_centroid_rule(tss)
    truth = ["yes" if t.stratum == "inside" else "no" for t in tss]
    assert balanced_accuracy(truth, centroid_rule(tss, thr), "yes") > 0.6

@pytest.mark.parametrize("model_row,base,expected", [
    ((0.72, 0.62), {"box": 0.5, "feature": 0.6, "centroid": 0.9}, "go"),   # centroid ignored
    ((0.60, 0.53), {"box": 0.5, "feature": 0.55}, "weak pass"),
    ((0.70, 0.62), {"box": 0.5, "feature": 0.75}, "fail"),               # below a map-free baseline
    ((0.58, 0.48), {"box": 0.5, "feature": 0.5}, "fail"),
    ((float("nan"), float("nan")), {"box": 0.5, "feature": 0.5}, "fail"),
])
def test_verdict_pilot(model_row, base, expected):
    rows = [dict(who="model", rung=4, subset="hard", bal_acc=model_row[0], lo=model_row[1])]
    rows += [dict(who=k, rung=4, subset="hard", bal_acc=v, lo=v - 0.05) for k, v in base.items()]
    assert verdict_pilot(pd.DataFrame(rows)) == expected

def test_pilot_summary_shape():
    rng = np.random.default_rng(8)
    tss = [generate_tri_scene(rng, POOL) for _ in range(30)]
    rows = []
    for si, t in enumerate(tss):
        for it in ladder(t, rng):
            rows.append(dict(scene=si, rung=it.rung, stratum=t.stratum, truth=it.truth, pred=it.truth,
                             positive=it.positive, mass=0.0))
    s = pilot_summary(pd.DataFrame(rows), {"box": box_rule(tss)}, n_boot=50)
    hard = s[(s.who == "model") & (s.rung == 4) & (s.subset == "hard")].iloc[0]
    assert hard.bal_acc == 1.0
    assert set(s[s.who == "model"].rung) == {1, 2, 3, 4}
    assert {"inside", "box_hard", "outside_box"} <= set(s[(s.who == "model") & (s.rung == 4)].subset)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_study2.py -v`
Expected: FAIL with ImportError (`geomap.study2` does not exist).

- [ ] **Step 3: Implement the Interfaces above in `src/geomap/study2.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_study2.py -v`
Expected: all PASS (11 test cases).

- [ ] **Step 5: Commit**

```bash
git add src/geomap/study2.py tests/test_study2.py
git commit -m "study2: balanced accuracy, baselines, pilot summary and go rule"
```

---

### Task 4: Pilot script

**Files:**
- Create: `scripts/s2_pilot.py`

**Interfaces:**
- Consumes: Tasks 1–3; `geomap.models.load`, `prompts.build_prompt`, `prompts.single_token_names`, `NAME_POOLS`,
  `scoring.score`, `predict`, `candidate_mass`, `checkpoint.df_step`, `notify`, `report.append_entry`, `ROOT`.
- Produces, under `--out` (default `results/s2/pilot`):
  - `{model}_s2_pilot_n{n}_seed702_answers.parquet` (Review Focus 5);
  - `{model}_s2_pilot_summary.csv`;
  - `{model}_s2_pilot.done`, containing the verdict.

**Behaviour:**
- Arguments: `--model gemma3-12b --n 400 --n-train 2000 --batch-size 16 --smoke --out`.
- `--smoke`: tiny model, n = 8, n-train = 60, a temp out dir, and no `append_entry`.
- If the `.done` marker exists, print and exit, as P9 does.
- **Test scenes:** seed 702. Scene i uses the people pool if i is even and the objects pool otherwise, as in Study 1,
  filtered by `single_token_names`. Each scene's ladder is built from the same rng.
- **Training scenes for the baselines:** seed 701, n-train scenes, no model needed.
- **Model answers** (a `df_step`): score every ladder question; `pred = predict(scores)`;
  `mass = exp(candidate_mass(scores))`.
- **Baselines:**
  - `box_rule` on the test scenes;
  - `feature_rule`, fitted on the training scenes;
  - `centroid_rule`, with its threshold fitted on the training scenes.
- Then: `pilot_summary`, `verdict_pilot`, the summary CSV, and a RESULTS.md entry with a markdown table of the
  summary, the verdict and a pointer to `docs/preregistration_s2_pilot.md`. Then the `.done` marker and `notify`.

- [ ] **Step 1: Write `scripts/s2_pilot.py` as specified, following `scripts/p9_map_answers.py`'s structure**

- [ ] **Step 2: Run the smoke test**

Run: `uv run python scripts/s2_pilot.py --smoke 2>&1 | tail -20`
Expected: it exits 0, prints the summary table and a `verdict:` line (any verdict; the tiny model is random), and
writes no RESULTS.md entry (`git status` shows `results/RESULTS.md` unchanged).

- [ ] **Step 3: Run the smoke test again with the same `--out` to check resume**

Run: `D=$(mktemp -d); uv run python scripts/s2_pilot.py --smoke --out $D >/dev/null && uv run python scripts/s2_pilot.py --smoke --out $D`
Expected: the second run prints "already complete".

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q 2>&1 | tail -3`
Expected: all pass (the 138 previous tests plus the new ones).

- [ ] **Step 5: Commit**

```bash
git add scripts/s2_pilot.py
git commit -m "study2: pilot script (answers only, resumable)"
```

---

### Task 5: Preregistration and Colab notebook

**Files:**
- Create: `docs/preregistration_s2_pilot.md`
- Create: `colab/s2_pilot.ipynb`

**Preregistration contents:**
- **Question:** can Gemma 3 12B judge "inside the triangle" beyond map-free shortcuts?
- **Scenes and ladder:** as implemented in Tasks 1–2. Give the generator parameters (bound 8, area ≥ 4, edge
  distance ≥ 0.5, strata 0.5 / 0.35 / 0.15) and the four question templates.
- **Data:** n = 400 test scenes, seed 702; baseline training scenes n = 2000, seed 701.
- **Metric:** balanced accuracy with a scene-bootstrap 95% CI (1,000 resamples), reported per rung. Rung 4 is also
  reported on the hard subset (inside + box_hard) and per stratum.
- **Baselines:** box and feature rules (map-free; these gate the verdict) and the centroid rule (reference only).
- **Go rule** (verbatim from `verdict_pilot`):
  - go = hard balanced accuracy ≥ 0.65, CI lower bound > 0.5, and above both map-free baselines;
  - weak pass = above chance and above the map-free baselines but < 0.65;
  - fail otherwise.
- **What each outcome means:**
  - go → write the main-study plan;
  - weak pass or fail → stop and decide with the author, and any change gets a new preregistration.
- **Sanity check (not a gate):** rungs 1–2 should be roughly at Study 1's levels.

**Notebook:** copy `colab/p9_map_answers.ipynb` cell for cell, with these changes:
- the title and expected time (about 20–40 minutes);
- the step-2 assertion checks `scripts/s2_pilot.py`;
- the step-5 command is `python scripts/s2_pilot.py --model gemma3-12b`;
- step 6 zips `results/s2/pilot` into `s2_pilot_results.zip`;
- the notify tags read `[S2 pilot]`.

- [ ] **Step 1: Write both files**

- [ ] **Step 2: Check the notebook**

Run: `python3 -c "import json;nb=json.load(open('colab/s2_pilot.ipynb'));src=''.join(''.join(c['source']) for c in nb['cells']);assert 's2_pilot.py' in src and 'p9' not in src.lower();print(len(nb['cells']),'cells ok')"`
Expected: `7 cells ok`.

- [ ] **Step 3: Rebuild the bundle and confirm it contains the new code**

Run: `zsh colab/make_bundle.sh && unzip -l colab/geomap_colab.zip | grep -E "s2_pilot.py|triangles.py|study2.py|preregistration_s2_pilot"`
Expected: 4 lines.

- [ ] **Step 4: Commit (this commit is the preregistration timestamp; no real data exists yet)**

```bash
git add docs/preregistration_s2_pilot.md colab/s2_pilot.ipynb
git commit -m "study2: pilot preregistration and Colab notebook"
```

After this, the author uploads `colab/geomap_colab.zip` to `MyDrive/geomap/` and runs `colab/s2_pilot.ipynb` on an
A100.
