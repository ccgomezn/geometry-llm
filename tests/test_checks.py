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
