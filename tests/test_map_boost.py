import numpy as np
import pytest
import torch

from geomap.axis import CELLS, axis_bulk_items
from geomap.boost import (boost_edit, boost_vec, condition_works, feasible_k, map_directions, matched_random_subspaces,
                          ba_boot, orthonormal, transfer_items, transfer_verdict, verdict_boost)
from geomap.models import layer_out, load_tiny_for_tests
from geomap.prompts import build_prompt
from geomap.questions import QType, answer_for
from geomap.scoring import score
from geomap.study2 import balanced_accuracy


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def items(tiny):
    return axis_bulk_items(tiny, {c: 2 for c in CELLS}, seed=1701)


def test_boost_vec_scales_only_the_subspace_around_the_mean():
    g = torch.Generator().manual_seed(0)
    Q = orthonormal(torch.randn(16, 3, generator=g))
    mu, x = torch.randn(16, generator=g), torch.randn(5, 16, generator=g)
    assert torch.allclose(boost_vec(x, Q, mu, 1.0), x)
    y = boost_vec(x, Q, mu, 4.0)
    inside = lambda v: (v - mu) @ Q
    outside = lambda v: (v - mu) - inside(v) @ Q.T
    assert torch.allclose(inside(y), 4.0 * inside(x), atol=1e-5)
    assert torch.allclose(outside(y), outside(x), atol=1e-5)


def test_map_directions_recover_per_axis_step_directions():
    rng = np.random.default_rng(0)
    n, d = 600, 40
    W = rng.normal(size=(d, 4))                                    # true directions: d1e, d2e, d1n, d2n
    ax = rng.integers(0, 2, n)
    d1, d2 = rng.integers(-5, 6, n).astype(float), rng.integers(-5, 6, n).astype(float)
    X = rng.normal(size=(n, d)) * 0.1
    X += np.where(ax[:, None] == 0, np.outer(d1, W[:, 0]) + np.outer(d2, W[:, 1]),
                  np.outer(d1, W[:, 2]) + np.outer(d2, W[:, 3]))
    D = map_directions(X, d1, d2, ax, np.arange(n) // 3)
    assert D.shape == (d, 4)
    # the ridge coefficient for d1 on east questions predicts d1 there
    e = ax == 0
    pred = (X[e] - X[e].mean(0)) @ D[:, 0]
    assert np.corrcoef(pred, d1[e])[0, 1] > 0.99


def test_random_subspaces_are_matched_in_variance_and_lie_in_the_top_components():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 30)) * np.linspace(5, 0.1, 30)
    Q = orthonormal(torch.as_tensor(rng.normal(size=(30, 4))))
    k = feasible_k(X, Q, k0=10)            # a random 4-d subspace has less variance than any 4 of the top 10
    assert k in (20, 30)                  # 10 · 2 = 20, capped at the 30 components there are
    got = matched_random_subspaces(X, Q, k=k, n=3, tol=0.1, seed=0)
    Xc = torch.as_tensor(X - X.mean(0))
    var = lambda B: float(((Xc @ B) ** 2).sum(1).mean())
    U = torch.linalg.svd(Xc, full_matrices=False).Vh[:k].T
    assert len(got) == 3
    for R in got:
        assert R.shape == (30, 4) and torch.allclose(R.T @ R, torch.eye(4, dtype=R.dtype), atol=1e-6)
        assert abs(var(R) / var(Q) - 1) <= 0.1
        assert torch.allclose(U @ (U.T @ R), R, atol=1e-6)             # inside the top-k span
    again = matched_random_subspaces(X, Q, k=k, n=3, tol=0.1, seed=0)
    assert all(torch.allclose(a, b) for a, b in zip(got, again))


def test_boost_edit_matches_an_independent_forward_hook(tiny, items):
    s, mid, child, qq = items[0]
    qs = [q for q, _ in qq][:2] + [q for q, _ in items[1][3]][:1]
    scenes = [s, s, items[1][0]][:len(qs)]
    specs = [build_prompt(sc, q, tiny.tok, tiny.family) for sc, q in zip(scenes, qs)]
    g = torch.Generator().manual_seed(0)
    Q = orthonormal(torch.randn(tiny.d_model, 2, generator=g))
    mu = torch.randn(tiny.d_model, generator=g)
    pos = [len(sp.input_ids) - 1 for sp in specs]
    layer, alpha = 1, 3.0
    clean = score(tiny, list(zip(specs, qs)))
    same = score(tiny, list(zip(specs, qs)), edit=boost_edit(layer, Q, mu, 1.0, pos))
    assert all(a == pytest.approx(b, abs=1e-5) for a, b in zip(same, clean))
    got = score(tiny, list(zip(specs, qs)), edit=boost_edit(layer, Q, mu, alpha, pos))
    block = tiny.hf.model.layers[layer - 1]
    for i, (sp, q) in enumerate(zip(specs, qs)):
        def hook(mod, inp, out, p=pos[i]):
            h = out[0] if isinstance(out, tuple) else out
            h[0, p] = boost_vec(h[0, p].float(), Q.float().to(h.device), mu.float().to(h.device), alpha).to(h.dtype)
            return out
        hd = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                lp = torch.log_softmax(tiny.hf(torch.tensor([sp.input_ids]).to(tiny.hf.device)).logits[0, -1].float(), -1)
        finally:
            hd.remove()
        for c in q.candidates:
            assert got[i][c] == pytest.approx(float(lp[tiny.tok.encode(c, add_special_tokens=False)[0]]), abs=1e-4)
        assert any(abs(got[i][c] - clean[i][c]) > 1e-6 for c in q.candidates)


def test_ba_boot_point_is_balanced_accuracy():
    rng = np.random.default_rng(0)
    truth = rng.choice([-1, 1], 200)
    preds = {"a": rng.choice([-1, 1], 200), "b": truth.copy()}
    g = np.arange(200) // 4
    out = ba_boot(truth, preds, g, n_boot=50, seed=0)
    assert out["a"][0] == pytest.approx(balanced_accuracy(truth, preds["a"], 1))
    assert out["b"][0] == 1.0 and out["a"][1].shape == (50,) and (out["b"][1] == 1.0).all()


def test_transfer_items_are_other_name_questions_with_geometric_truth(items):
    s, mid, child, _ = items[0]
    tq = transfer_items(s, mid, child)
    assert tq
    for q, kind in tq:
        assert kind in ("C vs M", "M vs R", "farthest", "closest")
        if q.qtype in (QType.MORE_EAST, QType.MORE_NORTH):
            assert set(q.args) != {s.root, child}
        assert q.qtype not in (QType.DIRECTION, QType.CLOSER_THAN)
        assert answer_for(q, s.coords) in q.candidates


EASY = {"own step only": -0.01, "agree": 0.0, "conflict, C larger": -0.02}


def test_condition_works():
    ok = dict(dec_gain=0.15, contrast=0.08, contrast_lo99=0.02, first_lo99=0.01, easy=EASY)
    assert condition_works(ok)
    assert not condition_works({**ok, "dec_gain": 0.09})
    assert not condition_works({**ok, "contrast": 0.04})
    assert not condition_works({**ok, "contrast_lo99": -0.01})
    assert not condition_works({**ok, "first_lo99": 0.0})
    assert not condition_works({**ok, "easy": {**EASY, "agree": -0.06}})


def test_verdict_boost():
    ok = dict(dec_gain=0.15, contrast=0.08, contrast_lo99=0.02, first_lo99=0.01, easy=EASY)
    null = dict(dec_gain=0.0, contrast=0.0, contrast_lo99=-0.03, first_lo99=-0.02, easy=EASY)
    hurt = dict(dec_gain=0.2, contrast=0.1, contrast_lo99=0.05, first_lo99=0.05, easy={**EASY, "agree": -0.3})
    assert verdict_boost([null, ok]) == "the map wins when amplified"
    assert verdict_boost([null, null]) == "no benefit"
    assert verdict_boost([null, hurt]).startswith("partial")


def test_transfer_verdict():
    assert transfer_verdict(overall=0.01, gains={"C vs M": (0.05, 0.01)}) == "transfers"
    assert transfer_verdict(overall=0.0, gains={"C vs M": (0.02, -0.01)}) == "specific"
    assert transfer_verdict(overall=-0.04, gains={"C vs M": (0.05, 0.01)}) == "harmful"


def test_controls_match_a_target_near_the_variance_floor_of_the_top_components():
    """The layer-32 failure on Gemma: S's variance is at the floor of the top 64 components (heavy-tailed spectrum)."""
    rng = np.random.default_rng(2)
    d = 200
    scales = np.logspace(3, -1, d)
    X = rng.normal(size=(3000, d)) * scales
    E = np.eye(d)
    Q = orthonormal(torch.as_tensor(np.stack([E[62], E[63]], 1)))       # exactly the 63rd and 64th components
    k = feasible_k(X, Q, k0=64)
    assert k > 64                                                       # the floor of the top 64 is the target itself
    got = matched_random_subspaces(X, Q, k=k, n=3, tol=0.1, seed=0)
    Xc = X - X.mean(0)
    var = lambda B: float(((Xc @ np.asarray(B, dtype=float)) ** 2).sum(1).mean())
    assert len(got) == 3 and all(abs(var(R) / var(Q) - 1) <= 0.1 for R in got)
    assert not torch.allclose(got[0], got[1])


# ---- dose check (docs/preregistration_map_boost_dose.md) ----

def test_norm_scale_is_one_plus_weight_for_gemma_norms():
    from geomap.boost import norm_scale

    class Gemma3RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.5, -0.25]))

    class LlamaRMSNorm(Gemma3RMSNorm):
        pass

    assert torch.allclose(norm_scale(Gemma3RMSNorm()), torch.tensor([1.5, 0.75]))
    assert torch.allclose(norm_scale(LlamaRMSNorm()), torch.tensor([0.5, -0.25]))


def test_pushing_along_the_answer_direction_raises_that_answer(tiny, items):
    from geomap.boost import answer_directions, push_edit
    s, mid, child, qq = items[0]
    q = qq[0][0]
    spec = build_prompt(s, q, tiny.tok, tiny.family)
    a, b = q.candidates[0], q.candidates[1]
    U = answer_directions(tiny, [(spec, a, b)])
    assert U.shape == (1, tiny.d_model) and torch.allclose(U.norm(dim=1), torch.ones(1), atol=1e-5)
    pos = [len(spec.input_ids) - 1]
    ld = lambda sc: sc[a] - sc[b]
    clean = ld(score(tiny, [(spec, q)])[0])
    moved = [ld(score(tiny, [(spec, q)], edit=push_edit(tiny.n_layers, U, c, pos))[0]) for c in (1.0, 10.0, 100.0)]
    assert clean < moved[0] < moved[1] < moved[2] and moved[2] > 0


def test_push_edit_adds_the_scaled_direction_at_the_final_position(tiny, items):
    from geomap.boost import push_edit
    s, mid, child, qq = items[0]
    qs = [q for q, _ in qq][:2]
    specs = [build_prompt(s, q, tiny.tok, tiny.family) for q in qs]
    pos = [len(sp.input_ids) - 1 for sp in specs]
    g = torch.Generator().manual_seed(1)
    U = torch.nn.functional.normalize(torch.randn(len(qs), tiny.d_model, generator=g), dim=1)
    layer, c = 1, 2.5
    got = score(tiny, list(zip(specs, qs)), edit=push_edit(layer, U, c, pos))
    block = tiny.hf.model.layers[layer - 1]
    for i, (sp, q) in enumerate(zip(specs, qs)):
        def hook(mod, inp, out, p=pos[i], u=U[i]):
            h = out[0] if isinstance(out, tuple) else out
            h[0, p] = h[0, p] + c * u.to(h.device, h.dtype)
            return out
        hd = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                lp = torch.log_softmax(tiny.hf(torch.tensor([sp.input_ids]).to(tiny.hf.device)).logits[0, -1].float(), -1)
        finally:
            hd.remove()
        for cand in q.candidates:
            assert got[i][cand] == pytest.approx(float(lp[tiny.tok.encode(cand, add_special_tokens=False)[0]]), abs=1e-4)


def test_push_moves_the_answer():
    from geomap.boost import push_moves
    assert push_moves(gain=0.25, lo99=0.12)
    assert not push_moves(gain=0.19, lo99=0.15)
    assert not push_moves(gain=0.30, lo99=0.10)


def test_dose_verdict():
    from geomap.boost import dose_verdict
    assert dose_verdict(moves={28: True, 32: False}, works={28: False, 32: False}, new_works=False) \
        == "the map is not read (dose-checked)"
    assert dose_verdict(moves={28: True, 32: True}, works={28: True, 32: False}, new_works=True) \
        == "the map wins at a higher dose"
    assert dose_verdict(moves={28: False, 32: False}, works={28: False, 32: False}, new_works=False) \
        == "still too small to tell"
    # the control moves only in a layer where a condition works, and no new condition works: not "not read"
    assert dose_verdict(moves={28: True, 32: False}, works={28: True, 32: False}, new_works=False) \
        == "still too small to tell"
