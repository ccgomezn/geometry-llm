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


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_capture_final_matches_the_attribution_means(kind):
    from geomap.components import capture_final, component_dla
    m = _tiny(kind)
    ids, groups, c, r = _batch(2, n=3)
    cap = capture_final(m, ids, [("head", 1, 2), ("mlp", 0, -1)], batch_size=2)
    assert cap[("head", 1, 2)].shape == (3, 16) and cap[("mlp", 0, -1)].shape == (3, 64)
    d = component_dla(m, ids, groups, 4, c, r, batch_size=3)
    assert np.allclose(cap[("head", 1, 2)].mean(0), d["z_mean"][1, 32:48], atol=1e-4)
    assert np.allclose(cap[("mlp", 0, -1)].mean(0), d["mlp_mean"][0], atol=1e-4)


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_per_item_refs_and_attribution_under_ablation(kind):
    from geomap.components import ablation_ctx, capture_final, component_dla
    m = _tiny(kind)
    ids, groups, c, r = _batch(3, n=4)
    finals = [len(x) - 1 for x in ids]
    sel = [("head", 1, 0), ("head", 1, 3), ("mlp", 1, -1)]
    cap = capture_final(m, ids, sel, batch_size=2)
    clean = component_dla(m, ids, groups, 4, c, r, batch_size=2)
    own = ablation_ctx(m, finals, {1: {0: torch.tensor(cap[("head", 1, 0)]), 3: torch.tensor(cap[("head", 1, 3)])}},
                       {1: torch.tensor(cap[("mlp", 1, -1)])})
    same = component_dla(m, ids, groups, 4, c, r, batch_size=2, batch_ctx=own)        # per-item own values: no change
    assert np.allclose(same["ld"], clean["ld"], atol=1e-3)
    rng = np.random.default_rng(0)
    other = ablation_ctx(m, finals, {1: {0: torch.tensor(rng.normal(size=(4, 16)), dtype=torch.float32)}},
                         {1: torch.tensor(rng.normal(size=(4, 64)), dtype=torch.float32)})
    d = component_dla(m, ids, groups, 4, c, r, batch_size=2, batch_ctx=other)
    assert np.abs(d["ld"] - clean["ld"]).max() > 1e-3
    total = d["emb"] + d["heads"].sum((1, 2)) + d["mlp"].sum(1)                     # attribution sees the ablated run
    assert np.allclose(total, d["ld"], atol=1e-3)
    assert d["rf"].shape == (4,) and np.all(d["rf"] > 0)
    up = lambda x: x["heads"][:, 0] / x["rf"][:, None]                             # layer 0 is upstream: unchanged
    assert np.allclose(up(d), up(clean), atol=1e-3)                                 # up to the final norm's scale


def test_design_truth():
    from geomap.components import design_truth
    meta = pd.DataFrame(dict(d1=[3, -2, 0], d2=[-1, 1, 2], truth=[1, -1, 1], c_first=[True, False, True]))
    F = design_truth(meta)
    assert F.tolist() == [[1, 0, 1, 3, -1, 1], [1, 1, -1, -2, 1, -1], [1, 1, 1, 0, 2, 1]]


def test_repair_groups():
    from geomap.components import repair_groups
    comps = [("emb", -1, -1), ("head", 0, 0), ("head", 0, 1), ("head", 1, 0), ("head", 1, 1), ("head", 2, 0),
             ("mlp", 0, -1), ("mlp", 1, -1), ("mlp", 2, -1)]
    delta = np.array([0, 0.1, 0.2, -5, 1.5, 2.0, 0.3, 0.7, 1.0])
    g = repair_groups(delta, comps, removed=[("head", 1, 0)])
    assert g == pytest.approx(dict(removed=-5, same_layer_heads=1.5, later=2.0 + 0.7 + 1.0, earlier=0.1 + 0.2 + 0.3))


def test_verdict_components2():
    from geomap.components import verdict_components2
    assert verdict_components2(ratio=0.7, acc=0.8, acc_hi=0.85, gain=0.3, gain_lo=0.2) == "ineffective"
    assert verdict_components2(ratio=0.3, acc=0.7, acc_hi=0.75, gain=0.2, gain_lo=0.05).startswith("a correct path")
    assert verdict_components2(ratio=0.3, acc=0.5, acc_hi=0.58, gain=0.1, gain_lo=0.0).startswith("no correct path")
    assert verdict_components2(ratio=0.5, acc=0.62, acc_hi=0.66, gain=0.1, gain_lo=0.02) == "undecided"


def test_pick_matched_mlp_in_a_model_shallower_than_the_radius():
    from geomap.components import pick_matched
    out = pick_matched([("mlp", 1, -1)], n_layers=2, n_heads=4, rng=np.random.default_rng(0), exclude={("mlp", 1, -1)})
    assert out == [("mlp", 0, -1)]


def test_stage2b_components_match_the_stage2_ranking():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("c2", root / "scripts" / "components2_test.py")
    c2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2)
    C = pd.read_csv(root / "results" / "components" / "gemma3-12b_components_components.csv")
    C = C[C.kind != "emb"]
    rank = lambda col, k: [(r.kind, int(r.layer), int(r.head)) for r in C.sort_values(col, ascending=False).head(k).itertuples()]
    assert c2.OWN20 == rank("b_own_yes", 20) and c2.FIRST10 == rank("b_d1", 10)
