import numpy as np
import pandas as pd
import pytest
import torch

from geomap.followups import (chain_items, cv_r2, erase_edit, fit_token_maps, inlp_basis, project, sentence_diffs,
                              writein_rows)
from geomap.models import load_tiny_for_tests
from geomap.patching import make_pairs
from geomap.scoring import score


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def pairs(m):
    return make_pairs(m, 4, seed=1, n_objects=(3,))


def _toy_diffs(n_pairs=60, d=16, seed=0, noise=0.0):
    """dh = A_j @ doff (+ noise) with a known per-token linear map, for 3 tokens per pair."""
    rng = np.random.default_rng(seed)
    A = {j: rng.standard_normal((d, 2)) for j in range(3)}
    recs = []
    for p in range(n_pairs):
        doff = rng.integers(-4, 5, size=2).astype(float)
        for j in range(3):
            recs.append(dict(pair=p, j=j, doff=doff, dh=A[j] @ doff + noise * rng.standard_normal(d)))
    return recs, A


def test_sentence_diffs_shapes_and_offsets(m, pairs):
    recs = sentence_diffs(m, pairs, layer=1)
    base, src, leaf, _ = pairs[0]
    lo, hi = __import__("geomap.prompts", fromlist=["build_prompt"]).build_prompt(base, None, m.tok, m.family).relation_spans[leaf]
    r0 = [r for r in recs if r["pair"] == 0]
    assert len(r0) == hi - lo
    assert sorted(r["j"] for r in r0) == list(range(hi - lo))  # j counts from the sentence end
    rb = next(r for r in base.relations if r.subj == leaf)
    rs = next(r for r in src.relations if r.subj == leaf)
    assert np.allclose(r0[0]["doff"], [rs.dx - rb.dx, rs.dy - rb.dy])
    assert r0[0]["dh"].shape == (m.d_model,)


def test_fit_token_maps_recovers_linear_map():
    recs, A = _toy_diffs()
    Q = np.linalg.qr(np.random.default_rng(1).standard_normal((16, 2)))[0]
    maps = fit_token_maps(project(recs, Q), n_tok=8)
    for j in range(3):
        assert np.allclose(maps[j], Q.T @ A[j], atol=1e-8)
    assert cv_r2(project(recs, Q), n_tok=8) > 0.999


def test_cv_r2_is_low_for_noise():
    recs, _ = _toy_diffs()
    rng = np.random.default_rng(2)
    for r in recs:
        r["dh"] = rng.standard_normal(16)
    Q = np.linalg.qr(rng.standard_normal((16, 2)))[0]
    assert cv_r2(project(recs, Q), n_tok=8) < 0.1


def test_pooled_key_for_far_tokens():
    recs, _ = _toy_diffs()
    for r in recs:
        r["j"] += 10  # every token beyond n_tok
    Q = np.eye(16)[:, :2]
    maps = fit_token_maps(project(recs, Q), n_tok=8)
    assert set(maps) == {8}


def test_writein_rows_adds_mapped_offset_in_subspace():
    d = 16
    Q = np.linalg.qr(np.random.default_rng(3).standard_normal((d, 2)))[0]
    maps = {0: np.array([[1.0, 0], [0, 2.0]]), 1: np.eye(2), 8: np.zeros((2, 2))}
    base_h = torch.randn(3, d)
    rows = [dict(pair=0, pos=[5, 6, 7], src_h=torch.zeros(3, d))]
    out = writein_rows(rows, {0: base_h}, {0: np.array([2.0, -1.0])}, maps, Q, n_tok=8)
    exp = base_h.clone()
    # positions 5,6,7 are j = 2,1,0 from the end; j = 2 is beyond the fitted keys {0, 1} -> pooled key (8)
    for k, j in enumerate([2, 1, 0]):
        W = maps.get(j, maps[8])
        exp[k] += torch.tensor(Q @ (W @ np.array([2.0, -1.0])), dtype=exp.dtype)
    assert torch.allclose(out[0]["src_h"], exp, atol=1e-5)
    assert rows[0]["src_h"].abs().sum() == 0  # input rows untouched


def test_inlp_removes_linear_information():
    rng = np.random.default_rng(4)
    n, d = 400, 20
    X = rng.standard_normal((n, d))
    Y = X[:, :2] @ np.array([[2.0, 0.5], [-1.0, 1.0]]) + 0.05 * rng.standard_normal((n, 2))
    groups = np.repeat(np.arange(n // 4), 4)
    B, hist = inlp_basis(X, Y, groups, max_rounds=10, r2_stop=0.1)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-6)
    assert hist[0] > 0.9 and hist[-1] < 0.1
    Xe = X - (X @ B) @ B.T
    from geomap.probes import GroupRidge
    from geomap.followups import _cv_r2_ridge
    assert _cv_r2_ridge(Xe, Y, groups) < 0.1


def test_erase_edit_empty_basis_is_identity_and_real_basis_changes_scores(m, pairs):
    from geomap.prompts import build_prompt
    base, _, _, qs = pairs[0]
    items = [(build_prompt(base, q, m.tok, m.family), q) for q in qs[:2]]
    clean = score(m, items)
    empty = score(m, items, edit=erase_edit({1: (torch.zeros(m.d_model, 0), torch.zeros(0))}))
    for a, b in zip(clean, empty):
        assert all(abs(a[k] - b[k]) < 1e-4 for k in a)
    B = torch.linalg.qr(torch.randn(m.d_model, 16))[0]
    erased = score(m, items, edit=erase_edit({1: (B, torch.zeros(16))}))
    assert any(abs(a[k] - b[k]) > 1e-4 for a, b in zip(clean, erased) for k in a)


def test_chain_items_flags_composed_questions(m):
    items = chain_items(m, 3, seed=5, exclude_qtypes=("closer_than",))
    assert len(items) == 3
    for scene, mid, child, qs in items:
        assert scene.hop[child] == 2 and scene.parent[child] == mid
        for q, composed in qs:
            assert composed == (child in q.args or child in q.candidates)


def test_groupridge_reports_cv_r2():
    from geomap.probes import GroupRidge
    rng = np.random.default_rng(6)
    X = rng.standard_normal((300, 10))
    Y = X[:, :2] + 0.01 * rng.standard_normal((300, 2))
    g = np.repeat(np.arange(100), 3)
    assert GroupRidge().fit(X, Y, g).cv_r2_ > 0.99
    assert GroupRidge().fit(X, rng.standard_normal((300, 2)), g).cv_r2_ < 0.05


def _erase_df(drop_comp_probe, drop_comp_rand, drop_stated_probe, drop_stated_rand, n=40):
    """Synthetic per-question records: clean accuracy 1, erased accuracy 1 − drop (exact shares over n scenes)."""
    recs = []
    for s in range(n):
        for composed, dp, dr in ((True, drop_comp_probe, drop_comp_rand), (False, drop_stated_probe, drop_stated_rand)):
            recs.append(dict(scene=s, qi=int(composed), composed=composed, config="L21", cond="clean", correct=1.0))
            recs.append(dict(scene=s, qi=int(composed), composed=composed, config="L21", cond="probe",
                             correct=float(s >= dp * n)))
            recs.append(dict(scene=s, qi=int(composed), composed=composed, config="L21", cond="random",
                             correct=float(s >= dr * n)))
    return pd.DataFrame(recs)


def test_erase_summary_and_verdicts():
    from geomap.followups import erase_summary, verdict_erase
    s = erase_summary(_erase_df(0.30, 0.05, 0.05, 0.05), n_boot=50)
    row = s[(s.config == "L21") & s.composed].iloc[0]
    assert abs(row.excess - 0.25) < 1e-9
    assert verdict_erase(s, r2_after=0.05) == "map used"
    s0 = erase_summary(_erase_df(0.06, 0.05, 0.05, 0.05), n_boot=50)
    assert verdict_erase(s0, r2_after=0.05) == "map not used (in this subspace)"
    assert verdict_erase(s0, r2_after=0.30) == "unclear"  # erasure did not remove the readable map
    s1 = erase_summary(_erase_df(0.30, 0.05, 0.30, 0.05), n_boot=50)  # composed no worse hit than stated-only
    assert verdict_erase(s1, r2_after=0.05) == "unclear"


def test_depth_items_groups_by_deepest_object(m):
    from geomap.followups import depth_items
    items = depth_items(m, 4, seed=7, n_objects=(4, 5), exclude_qtypes=("closer_than",), max_q=10)
    assert len(items) == 4
    assert [len(s.names) for s, _ in items] == [4, 5, 4, 5]
    for s, qs in items:
        assert 0 < len(qs) <= 10
        for q, depth in qs:
            assert depth == max(s.hop[n] for n in set(q.args) | set(q.candidates) if n in s.hop)


def test_depth_group_labels():
    from geomap.followups import depth_group
    assert [depth_group(d) for d in (0, 1, 2, 3, 4)] == ["shallow", "shallow", "middle", "deep", "deep"]


def _group_df(drops, n=40):
    """drops: {group: (probe_drop, random_drop)} with clean accuracy 1."""
    recs = []
    for s in range(n):
        for gi, (g, (dp, dr)) in enumerate(drops.items()):
            for cond, d in (("clean", 0.0), ("probe", dp), ("random", dr)):
                recs.append(dict(scene=s, qi=gi, group=g, config="L21", cond=cond, correct=float(s >= d * n)))
    return pd.DataFrame(recs)


def test_verdict_hard():
    from geomap.followups import erase_summary, verdict_hard
    used = erase_summary(_group_df({"shallow": (0.05, 0.05), "middle": (0.1, 0.05), "deep": (0.30, 0.05)}),
                         n_boot=20, group="group")
    assert set(used.group) == {"shallow", "middle", "deep"}
    assert verdict_hard(used, r2_after=0.05, headroom=0.3) == "map used when chaining is hard"
    flat = erase_summary(_group_df({"shallow": (0.05, 0.05), "middle": (0.05, 0.05), "deep": (0.06, 0.05)}),
                         n_boot=20, group="group")
    assert verdict_hard(flat, r2_after=0.05, headroom=0.3) == "still not used"
    assert verdict_hard(flat, r2_after=0.05, headroom=0.10) == "uninformative"
    assert verdict_hard(flat, r2_after=0.20, headroom=0.3) == "unclear"
    both = erase_summary(_group_df({"shallow": (0.30, 0.05), "middle": (0.3, 0.05), "deep": (0.30, 0.05)}),
                         n_boot=20, group="group")
    assert verdict_hard(both, r2_after=0.05, headroom=0.3) == "unclear"  # deep not worse hit than shallow


# ---------- P7 ----------

def _items(m, pairs, n=2):
    from geomap.prompts import build_prompt
    base, _, _, qs = pairs[0]
    return base, [(build_prompt(base, q, m.tok, m.family), q) for q in qs[:n]]


def test_knockout_without_blocks_matches_plain_scoring(m, pairs):
    from geomap.followups import attention_knockout
    _, items = _items(m, pairs)
    a = score(m, items)
    b = score(m, items, batch_ctx=attention_knockout(m, [[] for _ in items], layers=[0, 1]))
    for x, y in zip(a, b):
        assert all(abs(x[k] - y[k]) < 1e-5 for k in x)


def test_knockout_seals_a_span(m, pairs):
    """If no later position may attend to span S (all layers), the words inside S cannot change the answer."""
    from geomap.followups import attention_knockout
    base, items = _items(m, pairs)
    name = base.relations[0].subj
    lo, hi = items[0][0].relation_spans[name]
    blocks = [[(hi, lo, hi)] for _ in items]
    ctx = attention_knockout(m, blocks, layers=[0, 1])
    sealed = score(m, items, batch_ctx=ctx)
    changed = []
    for spec, q in items:
        ids = list(spec.input_ids)
        ids[lo:hi] = [(t + 7) % 1000 + 10 for t in ids[lo:hi]]
        changed.append((type(spec)(**{**spec.__dict__, "input_ids": ids}), q))
    sealed2 = score(m, changed, batch_ctx=ctx)
    for x, y in zip(sealed, sealed2):
        assert all(abs(x[k] - y[k]) < 1e-4 for k in x)
    open_ = score(m, changed)  # without the knockout the change is visible
    assert any(abs(x[k] - y[k]) > 1e-4 for x, y in zip(sealed, open_) for k in x)


def test_p7_spans_and_classes(m):
    from geomap.followups import chain_items, p7_class, p7_spans
    from geomap.prompts import build_prompt
    s, mid, child, qs = chain_items(m, 1, seed=9, exclude_qtypes=("closer_than",))[0]
    root = s.root
    spec = build_prompt(s, qs[0][0], m.tok, m.family)
    sp = p7_spans(spec, child, mid)
    assert sp["own"] == spec.relation_spans[child] and sp["parent"] == spec.relation_spans[mid]
    assert sp["list"][0] == max(e for _, e in spec.relation_spans.values()) and sp["list"][1] == spec.eol_pos + 1
    for q, _ in qs:
        inv = (set(q.args) | set(q.candidates)) & {root, mid, child}  # objects only, never compass words
        exp = "CR" if {child, root} <= inv else "CM" if inv == {child, mid} else "MR"
        assert p7_class(q, root, mid, child) == exp


def test_p7_class_direction_questions_use_objects_not_compass_words():
    from geomap.followups import p7_class
    from geomap.questions import QType, make_question
    names = ("C", "M", "R")
    q = make_question(QType.DIRECTION, ("C", "M"), names)
    assert p7_class(q, "R", "M", "C") == "CM"
    assert p7_class(make_question(QType.DIRECTION, ("C", "R"), names), "R", "M", "C") == "CR"
    assert p7_class(make_question(QType.DIRECTION, ("M", "R"), names), "R", "M", "C") == "MR"


def _ko_df(drops, n=40):
    """drops: {(target, cls): drop} at window 'all'; clean accuracy 1."""
    recs = []
    classes = sorted({c for _, c in drops})
    for s in range(n):
        for qi, c in enumerate(classes):
            recs.append(dict(scene=s, qi=qi, cls=c, target="-", window="-", correct=1.0))
            for (t, cc), d in drops.items():
                if cc == c:
                    recs.append(dict(scene=s, qi=qi, cls=c, target=t, window="all", correct=float(s >= d * n)))
    return pd.DataFrame(recs)


def test_knockout_summary_and_verdict():
    from geomap.followups import knockout_summary, verdict_p7
    base = {("own", "CM"): 0.4, ("parent", "MR"): 0.4, ("list", "CR"): 0.0, ("list", "CM"): 0.0, ("list", "MR"): 0.0,
            ("own", "CR"): 0.3, ("own", "MR"): 0.0, ("parent", "CM"): 0.05}
    s = knockout_summary(_ko_df({**base, ("parent", "CR"): 0.35}), n_boot=20)
    row = s[(s.target == "parent") & (s.cls == "CR") & (s.window == "all")].iloc[0]
    assert abs(row.acc_drop - 0.35) < 1e-9
    assert verdict_p7(s) == ("chaining at question time", "list not read")
    s2 = knockout_summary(_ko_df({**base, ("parent", "CR"): 0.025}), n_boot=20)
    assert verdict_p7(s2)[0] == "composed position available without M's sentence"
    s3 = knockout_summary(_ko_df({**base, ("parent", "CR"): 0.35, ("own", "CM"): 0.05}), n_boot=20)
    assert verdict_p7(s3)[0] == "uninformative"
    s4 = knockout_summary(_ko_df({**base, ("parent", "CR"): 0.35, ("list", "CR"): 0.10}), n_boot=20)
    assert verdict_p7(s4)[1] == "list read"


# ---------- P9 ----------

def test_decoded_coords_recover_linear_map_and_answer_correctly():
    from geomap.followups import decoded_coords
    from geomap.probes import GroupRidge, centered_xy
    from geomap.questions import all_questions, answer_for
    from geomap.scenes import generate_scene
    rng = np.random.default_rng(0)
    names = ["Anna", "Ben", "Carl", "Emma", "Henry", "Jack"]
    scenes = [generate_scene(rng, names, 3) for _ in range(200)]
    d = 16
    W = rng.standard_normal((2, d))
    acts = np.full((len(scenes), 6, d), np.nan)
    for i, s in enumerate(scenes):
        for k, n in enumerate(s.names):
            acts[i, k] = np.asarray(s.coords[n], float) @ W + 0.5  # linear code of the position, plus an offset
    X, Y, meta = centered_xy(acts, scenes, np.arange(150))
    probe = GroupRidge().fit(X, Y, meta.scene.values)
    dec = decoded_coords(probe, acts, scenes, np.arange(150, 200))
    assert len(dec) == 50
    for s, c in zip(scenes[150:], dec):
        assert set(c) == set(s.names)
        for q in all_questions(s):
            assert answer_for(q, c) == answer_for(q, s.coords)


def test_compare_and_verdict_p9():
    from geomap.followups import compare_answers, verdict_p9
    recs = []
    for s in range(40):
        for qi, (cls, mod, mp) in enumerate([("CR", s < 28, s < 36), ("CM", True, True), ("MR", s < 36, s < 36)]):
            recs.append(dict(scene=s, qi=qi, cls=cls, model_correct=float(mod), map_correct=float(mp)))
    df = pd.DataFrame(recs)
    out = compare_answers(df, n_boot=50)
    row = out.set_index("cls").loc["CR"]
    assert abs(row.acc_model - 0.7) < 1e-9 and abs(row.acc_map - 0.9) < 1e-9 and abs(row.map_minus_model - 0.2) < 1e-9
    assert verdict_p9(out) == "the map would have answered better"
    df2 = df.assign(map_correct=df.model_correct)
    assert verdict_p9(compare_answers(df2, n_boot=50)) == "comparable"
    df3 = df.assign(map_correct=np.where(df.cls == "CR", 0.0, df.map_correct))
    assert verdict_p9(compare_answers(df3, n_boot=50)) == "the model answers better"
