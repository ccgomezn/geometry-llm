import numpy as np
import pandas as pd
import pytest
import torch

from geomap.das import SubspaceIntervention, eval_das, pair_rows, row_logprobs, train_das
from geomap.models import load_tiny_for_tests
from geomap.patching import make_pairs, run_pair


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def pairs(m):
    return make_pairs(m, 6, seed=1)


def test_q_orthonormal():
    iv = SubspaceIntervention(64, 2)
    Q = iv.Q
    assert Q.shape == (64, 2)
    assert torch.allclose(Q.T @ Q, torch.eye(2), atol=1e-5)


def test_full_rank_replaces_and_masked_keeps_base():
    torch.manual_seed(0)
    hb, hs = torch.randn(5, 64), torch.randn(5, 64)
    full = SubspaceIntervention(64, 64)
    assert torch.allclose(full(hb, hs), hs, atol=1e-4)
    iv = SubspaceIntervention(64, 2)
    iv.mask.zero_()
    assert torch.allclose(iv(hb, hs), hb)


def test_full_rank_equals_patch(m, pairs):
    from geomap.das import question_scores, _groups
    base, src, leaf, qs = pairs[0]
    full = SubspaceIntervention(m.d_model, m.d_model)
    rows = pair_rows(m, [pairs[0]], layer=1, site="list")
    lp = row_logprobs(m, rows, full, layer=1, grad=False).cpu()
    ref = run_pair(m, base, src, leaf, qs, sites=["list"], layers=[1])
    ref = ref[ref.site == "list"].set_index("qid")
    for g in _groups(rows):
        cands, cl = question_scores(rows, lp[g], g)
        qid = rows[g[0]]["qid"]
        sa = rows[g[0]]["src_ans"]
        assert abs(float(cl[cands.index(sa)]) - ref.loc[qid, "lp_src_ans"]) < 1e-3


def test_training_reduces_loss_and_eval_shape(m, pairs):
    iv = train_das(m, pairs[:4], layer=1, site="list", k=2, epochs=6, lr=1e-2, batch_size=4, seed=0)
    assert len(iv.history) >= 2 and iv.history[-1] <= iv.history[0] + 1e-6
    df = eval_das(m, iv, pairs[4:], layer=1, site="list", n_boot=20)
    assert {"qtype", "acc", "acc_lo", "acc_hi", "n"} <= set(df.columns)
    assert "ALL" in df.qtype.tolist()


def test_eval_headline_counts_only_model_diagnostic(m, pairs):
    from geomap.prompts import build_prompt
    from geomap.scoring import predict, score
    iv = SubspaceIntervention(m.d_model, 2)
    iv.mask.zero_()  # no intervention: the headline must not be inflated by questions whose answer stays put
    df = eval_das(m, iv, pairs, layer=1, site="list", n_boot=20).set_index("qtype")
    assert {"acc_geom", "control_stability"} <= set(df.columns)
    n_diag = 0
    for b, s, l, qs in pairs:
        cb = score(m, [(build_prompt(b, q, m.tok, m.family), q) for q in qs])
        cs = score(m, [(build_prompt(s, q, m.tok, m.family), q) for q in qs])
        n_diag += sum(predict(x) != predict(y) for x, y in zip(cb, cs))
    assert df.loc["ALL", "n"] == n_diag
    if n_diag:
        assert df.loc["ALL", "acc"] == 0.0  # masked intervention never reproduces the source answer


def test_lm_head_sees_only_scored_positions(m, pairs, monkeypatch):
    # memory: full-vocabulary logits at every kept position of 100+ candidate rows OOM'd a 12B model
    rows = pair_rows(m, [pairs[0]], layer=1, site="list")[:6]
    need = sorted({p for r in rows for p in range(r["n"] - 1, len(r["ids"]) - 1)})
    seen = []
    h = m.hf.lm_head.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].shape[1]))
    try:
        row_logprobs(m, rows, SubspaceIntervention(m.d_model, 2), layer=1)
    finally:
        h.remove()
    assert seen == [len(need)]


def test_training_forward_holds_one_question_at_a_time(m, pairs):
    # memory: 8 questions' candidate rows (200+ prompts) in one forward with autograd OOM'd a 12B model
    from geomap.das import _groups
    biggest = max(len(g) for g in _groups(pair_rows(m, pairs[:2], layer=1, site="list")))
    sizes = []
    h = m.hf.lm_head.register_forward_hook(lambda mod, inp, out: sizes.append(inp[0].shape[0]))
    try:
        train_das(m, pairs[:2], layer=1, site="list", k=2, epochs=1, batch_size=8)
    finally:
        h.remove()
    assert max(sizes) <= biggest


def test_mention_site_full_rank_equals_one_token_patch(m, pairs):
    from geomap.das import question_scores, _groups
    from geomap.patching import run_pair_question
    base, src, leaf, qs = pairs[0]
    rows = pair_rows(m, [pairs[0]], layer=2, site="mention")
    assert rows and {r["qid"] for r in rows} == {i for i, q in enumerate(qs) if leaf in q.args}
    lp = row_logprobs(m, rows, SubspaceIntervention(m.d_model, m.d_model), layer=2).cpu()
    ref = run_pair_question(m, base, src, leaf, qs, layers=[2], sites=("mention",))
    ref = ref[ref.site == "mention"].set_index("qid")
    for g in _groups(rows):
        cands, cl = question_scores(rows, lp[g], g)
        r = rows[g[0]]
        assert abs(float(cl[cands.index(r["src_ans"])]) - ref.loc[r["qid"], "lp_src_ans"]) < 1e-3


def test_interpolated_coords_moves_only_the_leaf(pairs):
    from geomap.das import interpolated_coords
    base, src, leaf, _ = pairs[0]
    assert interpolated_coords(base, src, leaf, 0.0) == dict(base.coords)
    assert interpolated_coords(base, src, leaf, 1.0) == {k: tuple(map(float, v)) for k, v in src.coords.items()}
    mid = interpolated_coords(base, src, leaf, 0.5)
    (bx, by), (sx, sy) = base.coords[leaf], src.coords[leaf]
    assert mid[leaf] == ((bx + sx) / 2, (by + sy) / 2)
    assert all(mid[n] == base.coords[n] for n in base.names if n != leaf)


def test_interp_rows_endpoints(m, pairs):
    from geomap.das import interp_rows
    rows = pair_rows(m, pairs[:2], layer=1, site="relation")
    full = SubspaceIntervention(m.d_model, m.d_model)
    r0 = interp_rows(m, rows, pairs[:2], layer=1, site="relation", alpha=0.0)
    r1 = interp_rows(m, rows, pairs[:2], layer=1, site="relation", alpha=1.0)
    assert all(torch.allclose(a["src_h"], b["src_h"]) for a, b in zip(r1, rows))
    clean = row_logprobs(m, rows, None, layer=1).cpu()
    assert torch.allclose(row_logprobs(m, r0, full, layer=1).cpu(), clean, atol=1e-3)


def test_eval_interp_records(m, pairs):
    from geomap.das import eval_interp
    iv = SubspaceIntervention(m.d_model, 2)
    df = eval_interp(m, iv, pairs[:2], layer=1, site="relation", alphas=(0.0, 0.5))
    assert set(df.alpha) == {0.0, 0.5}
    n_q = sum(len(p[3]) for p in pairs[:2])
    assert len(df) == 2 * n_q
    a0 = df[df.alpha == 0.0]
    assert (a0.geo == a0.geo_base).all()


def test_eval_das_accepts_prebuilt_rows(m, pairs):
    torch.manual_seed(0)
    iv = SubspaceIntervention(m.d_model, 2)
    a = eval_das(m, iv, pairs, 1, "relation", n_boot=50)
    b = eval_das(m, iv, pairs, 1, "relation", n_boot=50, rows=pair_rows(m, pairs, 1, "relation"))
    pd.testing.assert_frame_equal(a, b)
