import numpy as np
import pytest
import torch

from geomap.cache import cache_prefix
from geomap.checks import (blurred_rates, calibrate_sd, error_overlap, linear_cv, mlp_cv, r2, verdict_nonlinear,
                           verdict_overlap)
from geomap.models import load_tiny_for_tests
from geomap.prompts import NAME_POOLS, build_prompt
from geomap.questions import Question, QType, all_questions, answer_for
from geomap.scenes import generate_scene


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


# ---------- Check A: error overlap ----------

def test_error_overlap_ratio_of_sums_and_ci():
    g = np.repeat(np.arange(50), 4)
    wrong = np.tile([1, 1, 0, 0], 50).astype(float)
    same = np.tile([1, 0, 0, 0], 50).astype(float)          # S = 0.5
    bw = np.full(200, 0.5)
    bs = np.full(200, 0.1)                                   # S_blur = 0.2
    r = error_overlap(wrong, same, bw, bs, g, n_boot=200)
    assert r["S"] == pytest.approx(0.5) and r["S_blur"] == pytest.approx(0.2)
    assert r["effect"] == pytest.approx(0.3) and r["lo"] - 1e-9 <= r["effect"] <= r["hi"] + 1e-9


def test_verdict_overlap():
    assert verdict_overlap(0.12, 0.06) == "the model's mistakes follow the map"
    assert verdict_overlap(0.03, 0.01) == "small overlap"
    assert verdict_overlap(0.08, 0.04) == "the model's mistakes follow the map"  # lo > 0 and effect ≥ 0.05
    assert verdict_overlap(0.049, 0.001) == "small overlap"
    assert verdict_overlap(0.02, -0.03) == "no sign that the mistakes follow the map"


def _items(n_scenes=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for si in range(n_scenes):
        s = generate_scene(rng, NAME_POOLS["people"], 3)
        for q in all_questions(s):
            if q.qtype in (QType.CLOSEST, QType.FARTHEST, QType.MORE_NORTH):
                t = answer_for(q, s.coords)
                out.append((si, q, s.coords, t, t))  # model always right
    return out


def test_blurred_rates_zero_noise_is_never_wrong_and_same_needs_a_wrong_map():
    items = _items()
    w, s = blurred_rates(items, sd=0.0, n_draws=5, seed=0)
    assert np.all(w == 0) and np.all(s == 0)
    w, s = blurred_rates(items, sd=5.0, n_draws=20, seed=0)
    assert w.mean() > 0.1
    assert np.all(s == 0)  # the model is always right, so it never shares a wrong map answer
    flipped = [(si, q, c, t, answer_for(q, {k: (-x, -y) for k, (x, y) in c.items()})) for si, q, c, t, _ in items]
    w2, s2 = blurred_rates(flipped, sd=5.0, n_draws=20, seed=0)
    assert np.allclose(w2, w) and s2.sum() > 0 and np.all(s2 <= w2 + 1e-12)


def test_calibrate_sd_hits_target_accuracy():
    items = _items()
    sd = calibrate_sd(items, target_acc=0.8, n_draws=40, seed=0)
    w, _ = blurred_rates(items, sd=sd, n_draws=40, seed=0)
    assert abs((1 - w.mean()) - 0.8) <= 0.01
    assert calibrate_sd(items, target_acc=1.0, n_draws=10, seed=0) == 0.0


# ---------- Check B: probes after erasure ----------

def test_mlp_beats_linear_on_a_nonlinear_target():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 8))
    Y = np.stack([X[:, 0] * X[:, 1], np.abs(X[:, 2])], 1)
    Y = Y - Y.mean(0)
    g = np.repeat(np.arange(500), 3)
    P_lin = linear_cv(X, Y, g)
    P_mlp = mlp_cv(X, Y, g, hidden=64, max_epochs=200, seed=0)
    assert r2(Y, P_lin) < 0.2 and r2(Y, P_mlp) > 0.5
    assert P_lin.shape == Y.shape and P_mlp.shape == Y.shape


def test_mlp_matches_linear_on_a_linear_target():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1500, 8))
    Y = X[:, :2] @ np.array([[1.0, 0.5], [-0.3, 2.0]]) + 0.1 * rng.standard_normal((1500, 2))
    g = np.repeat(np.arange(500), 3)
    assert r2(Y, mlp_cv(X, Y, g, hidden=64, max_epochs=200, seed=0)) > 0.9


def test_verdict_nonlinear():
    v = lambda **k: verdict_nonlinear(**{**dict(mlp_clean=0.72, lin_clean=0.70, mlp_probe=0.05, mlp_random=0.68), **k})
    assert v() == "the erasure removes the map, linear and nonlinear"
    assert v(mlp_probe=0.40) == "a nonlinear map survives the erasure"
    assert v(mlp_probe=0.20) == "partly"
    assert v(mlp_random=0.40) == "partly"  # random erasure hurts too: not a clean removal
    assert v(mlp_clean=0.60) == "inconclusive (the MLP probe is weaker than the linear one)"


def test_cache_with_erasure(m):
    scenes = [generate_scene(np.random.default_rng(i), NAME_POOLS["people"], 3) for i in range(4)]
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    clean = cache_prefix(m, specs, layers=[1, 2], batch_size=2)["list"]
    none = cache_prefix(m, specs, layers=[1, 2], batch_size=2, erase={1: (torch.zeros(m.d_model, 0), torch.zeros(0))})
    assert torch.equal(torch.nan_to_num(clean), torch.nan_to_num(none["list"]))
    B = torch.linalg.qr(torch.randn(m.d_model, 4, generator=torch.Generator().manual_seed(0)))[0]
    c = torch.full((4,), 0.3)
    er = cache_prefix(m, specs, layers=[1, 2], batch_size=2, erase={1: (B, c)})["list"]
    h1 = er[:, 0, :3].float()
    assert torch.allclose(h1 @ B, c.expand(4, 3, 4), atol=0.05)  # erased at the captured layer (bfloat16 storage)
    assert not torch.allclose(torch.nan_to_num(er[:, 1]).float(), torch.nan_to_num(clean[:, 1]).float())  # downstream


# ---------- closed-loop erasure ----------

def test_erasure_start_spares_early_positions(m):
    from geomap.followups import erase_edit  # noqa: F401  (same operation, used by scoring)
    scenes = [generate_scene(np.random.default_rng(i), NAME_POOLS["people"], 3) for i in range(3)]
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    B = torch.linalg.qr(torch.randn(m.d_model, 4, generator=torch.Generator().manual_seed(0)))[0]
    e = {1: (B, torch.full((4,), 0.3))}
    clean = cache_prefix(m, specs, layers=[2], batch_size=3)["list"]
    beyond = cache_prefix(m, specs, layers=[2], batch_size=3, erase=e, erase_start=10_000)["list"]
    assert torch.equal(torch.nan_to_num(clean), torch.nan_to_num(beyond))  # nothing erased past the prompt
    all_pos = cache_prefix(m, specs, layers=[2], batch_size=3, erase=e)["list"]
    skip_bos = cache_prefix(m, specs, layers=[2], batch_size=3, erase=e, erase_start=1)["list"]
    assert not torch.equal(torch.nan_to_num(all_pos), torch.nan_to_num(skip_bos))


def test_erase_edit_start_matches_cache_erasure(m):
    from geomap.followups import erase_edit
    from geomap.scoring import score
    from geomap.patching import make_pairs
    base, _, _, qs = make_pairs(m, 2, seed=1, n_objects=(3,))[0]
    items = [(build_prompt(base, q, m.tok, m.family), q) for q in qs[:2]]
    B = torch.linalg.qr(torch.randn(m.d_model, 4, generator=torch.Generator().manual_seed(0)))[0]
    e = {1: (B, torch.full((4,), 0.3))}
    a = score(m, items, edit=erase_edit(e))
    b = score(m, items, edit=erase_edit(e, start=1))
    c = score(m, items, edit=erase_edit(e, start=10_000))
    clean = score(m, items)
    assert all(abs(x[k] - y[k]) < 1e-5 for x, y in zip(c, clean) for k in x)
    assert any(abs(x[k] - y[k]) > 1e-5 for x, y in zip(a, b) for k in x)


def test_extend_basis_keeps_old_directions_and_removes_the_rest():
    from geomap.checks import extend_basis
    from geomap.probes import GroupRidge
    rng = np.random.default_rng(0)
    n, d = 900, 30
    g = np.repeat(np.arange(300), 3)
    Y = rng.standard_normal((n, 2))
    Q = np.linalg.qr(rng.standard_normal((d, 6)))[0]
    X = 0.3 * rng.standard_normal((n, d)) + Y @ Q[:, :2].T + 0.5 * Y @ Q[:, 2:4].T + 0.25 * Y[:, ::-1] @ Q[:, 4:6].T
    B_old = Q[:, :1]
    Xe = X - (X @ B_old) @ B_old.T
    B, r2s = extend_basis(B_old, Xe, Y, g, max_steps=5, r2_stop=0.05)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-8)
    assert np.allclose(B[:, :1].T @ B_old, 1, atol=1e-8) or np.linalg.norm(B @ (B.T @ B_old) - B_old) < 1e-8
    assert r2s[0] > 0.5 and GroupRidge().fit(X - (X @ B) @ B.T, Y, g).cv_r2_ < 0.05
    B2, r2s2 = extend_basis(B, X - (X @ B) @ B.T, Y, g, max_steps=5, r2_stop=0.05)
    assert B2.shape == B.shape and r2s2[0] < 0.05  # nothing left: unchanged


def test_verdict_closed_loop():
    from geomap.checks import verdict_closed_loop
    import pandas as pd
    def s(comp, comp_lo, stat):
        return pd.DataFrame([dict(config="c", composed=True, excess=comp, excess_lo=comp_lo),
                             dict(config="c", composed=False, excess=stat, excess_lo=stat - 0.02)])
    ok = dict(max_val_r2=0.05, random_comp_acc=0.70, comp_chance=0.40, mlp21=0.10)
    v = lambda summ, **k: verdict_closed_loop(summ, **{**ok, **k})
    assert v(s(0.15, 0.10, 0.03)) == "map used"
    assert v(s(0.15, 0.10, 0.12)) == "partial use"            # not 0.05 above stated-only, but above it
    assert v(s(0.06, 0.02, 0.01)) == "partial use"
    assert v(s(0.02, -0.01, 0.00)) == "map not used"
    assert v(s(0.02, -0.01, 0.00), mlp21=0.35) == "linear map not used"
    assert v(s(0.05, -0.01, 0.06)) == "unclear"
    assert v(s(0.15, 0.10, 0.03), max_val_r2=0.12) == "the map could not be erased"
    assert v(s(0.15, 0.10, 0.03), random_comp_acc=0.50) == "uninformative (the erasure damages the model in general)"


# ---------- confirmatory axis test ----------

def _aniso(n=600, d=40, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.geomspace(10, 0.1, d)
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    X = (rng.standard_normal((n, d)) * scales) @ Q.T
    return X - X.mean(0), Q


def test_variance_matched_random_matches_the_removed_variance():
    from geomap.checks import variance_matched_random
    X, Q = _aniso()
    var = lambda R: float(((X @ R) ** 2).sum() / len(X))
    B = np.linalg.qr(Q[:, [0, 4, 12]] + 0.2 * Q[:, [3, 16, 31]])[0]    # high-variance target
    R, alpha = variance_matched_random(X, B, seed=0)
    assert R.shape == B.shape and np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert abs(var(R) - var(B)) / var(B) <= 0.02 and 0 < alpha <= 8
    iso = np.linalg.qr(np.random.default_rng(0).standard_normal((40, 3)))[0]
    assert var(iso) < 0.5 * var(B)                                     # isotropic removes much less here
    R0, a0 = variance_matched_random(X, np.zeros((40, 0)), seed=0)
    assert R0.shape == (40, 0)


def test_variance_matched_random_low_target_returns_isotropic():
    from geomap.checks import variance_matched_random
    X, Q = _aniso()
    B = Q[:, -2:]                                                      # the two lowest-variance directions
    R, alpha = variance_matched_random(X, B, seed=0)
    assert alpha == 0 and R.shape == (40, 2)


def test_verdict_axis():
    from geomap.checks import verdict_axis
    ok = dict(max_val_r2=0.05, random_comp_acc=0.70, comp_chance=0.42)
    assert verdict_axis(0.05, 0.02, **ok) == "confirmed: composed axis answers use the map"
    assert verdict_axis(0.02, 0.005, **ok) == "small effect"
    assert verdict_axis(0.05, -0.01, **ok) == "not confirmed"
    assert verdict_axis(0.05, 0.02, **{**ok, "max_val_r2": 0.10}) == "the map could not be erased"
    assert verdict_axis(0.05, 0.02, **{**ok, "random_comp_acc": 0.55}) == "uninformative"
