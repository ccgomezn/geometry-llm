import numpy as np
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


def test_logits_to_keep_index_is_on_cpu(m, pairs, monkeypatch):
    # with device_map="auto" the final hidden states can sit on a different GPU than the first parameter;
    # a CPU index tensor is valid for any device, a cuda:0 index is not (crashed P4 on 4 GPUs).
    seen = {}
    real = m.hf.forward

    def spy(*args, **kw):
        if "logits_to_keep" in kw:
            seen["dev"] = kw["logits_to_keep"].device
        return real(*args, **kw)

    rows = pair_rows(m, [pairs[0]], layer=1, site="list")
    monkeypatch.setattr(m.hf, "forward", spy)
    row_logprobs(m, rows[:2], SubspaceIntervention(m.d_model, 2), layer=1)
    assert seen["dev"].type == "cpu"
