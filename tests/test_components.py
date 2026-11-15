import numpy as np
import pandas as pd
import pytest
import torch


def _tiny(kind):
    from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

    from geomap.models import _wrap, load_tiny_for_tests
    if kind == "qwen":
        m = load_tiny_for_tests()
    else:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        cfg = Gemma3TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                               sliding_window=64)
        torch.manual_seed(0)
        hf = Gemma3ForCausalLM(cfg).to(torch.float32)
        with torch.no_grad():   # Gemma's norms start at gain 1 + 0; make the (1 + w) gains matter
            for n, p in hf.named_parameters():
                if "norm" in n:
                    p.normal_(0, 0.5)
        m = _wrap(hf, tok, "tiny")
    return m


def _batch(seed=0, n=5):
    rng = np.random.default_rng(seed)
    ids = [list(rng.integers(10, 5000, int(rng.integers(8, 20)))) for _ in range(n)]
    groups = [[min(j // 4, 3) for j in range(len(x))] for x in ids]
    return ids, groups, list(rng.integers(10, 5000, n)), list(rng.integers(10, 5000, n))


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_contributions_add_up_to_the_answer_logit_difference(kind):
    from geomap.components import component_dla
    m = _tiny(kind)
    ids, groups, c, r = _batch()
    d = component_dla(m, ids, groups, 4, c, r, batch_size=2)
    total = d["emb"] + d["heads"].sum((1, 2)) + d["mlp"].sum(1)
    assert np.allclose(total, d["ld"], atol=1e-3)
    assert np.allclose(d["src"].sum(-1), d["heads"], atol=1e-4)       # each head's pieces by source token
    assert d["z_mean"].shape == (2, 4 * 16) and d["mlp_mean"].shape == (2, 64)


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_ablation_hook_replaces_one_head_at_the_final_position(kind):
    from geomap.components import ablation_ctx, component_dla
    m = _tiny(kind)
    ids, groups, c, r = _batch(1, n=1)
    d = component_dla(m, ids, groups, 4, c, r, batch_size=1)
    finals = [len(ids[0]) - 1]

    def ld(ctx):
        x = torch.tensor([ids[0]], device=m.hf.device)
        with ctx([0]), torch.no_grad():
            lo = m.hf(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, -1].float()
        return float(lo[c[0]] - lo[r[0]])

    own = torch.tensor(d["z_mean"][1, :16])                    # one item: the "mean" is its own value
    same = ablation_ctx(m, finals, {1: {0: own}}, {})
    assert ld(same) == pytest.approx(d["ld"][0], abs=1e-4)
    other = ablation_ctx(m, finals, {1: {0: torch.zeros(16)}}, {})
    assert abs(ld(other) - d["ld"][0]) > 1e-4
    mlp_same = ablation_ctx(m, finals, {}, {0: torch.tensor(d["mlp_mean"][0])})
    assert ld(mlp_same) == pytest.approx(d["ld"][0], abs=1e-4)


def test_design_and_betas():
    from geomap.components import betas, design
    meta = pd.DataFrame(dict(d1=[3, -2, 0, 4, -1, 2], d2=[1, -3, 2, 0, 5, -1], c_first=[True, True, False, False, True, False]))
    F = design(meta)
    assert F.shape == (6, 5) and F[:, 0].tolist() == [1] * 6
    assert F[:, 1].tolist() == [1, 0, 1, 0, 1, 0]                # own_yes = C's own step points the asked way
    y = 2 + 10 * F[:, 1] + 1.5 * F[:, 2] - 0.5 * F[:, 3]
    assert np.allclose(betas(y[:, None], F)[0], [2, 10, 1.5, -0.5, 0], atol=1e-8)


def test_pick_matched_stays_in_layer_and_avoids_the_excluded():
    from geomap.components import pick_matched
    sel = [("head", 3, 1), ("head", 3, 2), ("mlp", 10, -1)]
    rng = np.random.default_rng(0)
    out = pick_matched(sel, n_layers=20, n_heads=6, rng=rng, exclude=set(sel) | {("head", 3, 0)})
    assert [o[:2] for o in out[:2]] == [("head", 3), ("head", 3)] and out[0] != out[1]
    assert all(o not in sel and o != ("head", 3, 0) for o in out)
    assert out[2][0] == "mlp" and abs(out[2][1] - 10) <= 4 and out[2][1] != 10


def test_verdict_components():
    from geomap.components import verdict_components
    assert verdict_components(drop=0.1, acc=0.7, acc_hi=0.75, gain=0.2, gain_lo=0.1) == "ineffective"
    assert verdict_components(drop=0.4, acc=0.72, acc_hi=0.78, gain=0.25, gain_lo=0.1).startswith("a correct path")
    assert verdict_components(drop=0.4, acc=0.5, acc_hi=0.55, gain=0.0, gain_lo=-0.05).startswith("no correct path")
    assert verdict_components(drop=0.4, acc=0.62, acc_hi=0.66, gain=0.1, gain_lo=0.02) == "undecided"
