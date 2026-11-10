import numpy as np
import torch

from geomap.probes import (
    centered_xy,
    joint_eol_probe,
    probe_by_layer,
    probe_directions,
    rsa_by_layer,
    scene_split,
)
from geomap.scenes import generate_scene

POOL = [f"N{i}" for i in range(40)]
D = 48


def _synthetic(n_scenes=600, signal=True, offset=True, seed=0):
    """Fake cache: activations = A @ coords + per-scene offset + noise (or pure noise)."""
    rng = np.random.default_rng(seed)
    scenes = [generate_scene(np.random.default_rng(i), POOL, 3 + i % 4) for i in range(n_scenes)]
    A = rng.normal(size=(D, 2))
    lst = torch.full((n_scenes, 2, 6, D), float("nan"))
    eol = torch.zeros((n_scenes, 2, D))
    for i, s in enumerate(scenes):
        off = rng.normal(size=D) * 5 if offset else 0
        for k, name in enumerate(s.names):
            x = A @ np.array(s.coords[name], float) if signal else 0
            for layer in range(2):
                lst[i, layer, k] = torch.tensor(x * (layer == 1) + off + rng.normal(size=D) * 0.3)
    return {"list": lst.half(), "eol": eol.half()}, scenes


def test_split_by_scene():
    tr, te = scene_split(100, 0.2, seed=0)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 100 and len(te) == 20


def test_planted_linear_map_recovered():
    cache, scenes = _synthetic()
    split = scene_split(len(scenes))
    df = probe_by_layer(cache, scenes, split, n_boot=50)
    assert df.loc[df.layer == 1, "r2"].item() > 0.95
    assert df.loc[df.layer == 0, "r2"].item() < 0.1  # layer 0 carries no coordinates


def test_random_targets_r2_near_zero():
    cache, scenes = _synthetic()
    split = scene_split(len(scenes))
    df = probe_by_layer(cache, scenes, split, control="shuffled", n_boot=50)
    assert abs(df.loc[df.layer == 1, "r2"].item()) < 0.05


def test_centering_removes_translation():
    a, scenes = _synthetic(offset=False)
    b, _ = _synthetic(offset=True)
    split = scene_split(len(scenes))
    ra = probe_by_layer(a, scenes, split, n_boot=20).loc[1, "r2"]
    rb = probe_by_layer(b, scenes, split, n_boot=20).loc[1, "r2"]
    assert abs(ra - rb) < 0.02


def test_centered_xy_shapes():
    cache, scenes = _synthetic(50)
    X, Y, meta = centered_xy(cache["list"][:, 1].float().numpy(), scenes, np.arange(50))
    assert X.shape[0] == Y.shape[0] == len(meta) == sum(len(s.names) for s in scenes)
    assert np.allclose(meta.groupby("scene").apply(lambda g: Y[g.index].mean(0)).tolist(), 0, atol=1e-9)


def test_stated_relation_baseline_and_hops():
    cache, scenes = _synthetic()
    split = scene_split(len(scenes))
    df = probe_by_layer(cache, scenes, split, control="stated_relation", n_boot=20)
    assert {"r2_hop1", "r2_hop2", "r2_hop3p"} <= set(df.columns)
    assert df.r2.iloc[0] < 0.95  # own relation alone cannot place multi-hop objects perfectly


def test_probe_directions_inverts_probe():
    cache, scenes = _synthetic()
    from geomap.probes import fit_probe
    X, Y, _ = centered_xy(cache["list"][:, 1].float().numpy(), scenes, np.arange(len(scenes)))
    p = fit_probe(X, Y)
    Dm = probe_directions(p)
    assert Dm.shape == (D, 2)
    assert np.allclose(p.coef_ @ Dm, np.eye(2), atol=1e-6)


def test_rsa_and_joint_eol_run():
    cache, scenes = _synthetic(200)
    r = rsa_by_layer(cache, scenes, n_boot=20)
    assert r.loc[r.layer == 1, "rho"].item() > r.loc[r.layer == 0, "rho"].item()
    four = [i for i, s in enumerate(scenes) if len(s.names) == 4]
    j = joint_eol_probe(cache["eol"][four], [scenes[i] for i in four], scene_split(len(four)), n_boot=20)
    assert set(j.columns) >= {"layer", "r2"}


def test_layer_subset_and_hop2p():
    cache, scenes = _synthetic()
    df = probe_by_layer(cache, scenes, scene_split(len(scenes)), n_boot=20, layers=[1])
    assert df.layer.tolist() == [1] and "r2_hop2p" in df.columns and df.r2_hop2p.item() > 0.9


def test_joint_eol_probe_any_object_count():
    scenes = [generate_scene(np.random.default_rng(i), POOL, 3) for i in range(300)]
    rng = np.random.default_rng(0)
    A = rng.normal(size=(D, 6))
    eol = torch.zeros((300, 1, D))
    for i, s in enumerate(scenes):
        c = np.array([s.coords[n] for n in s.names], float)
        c -= c.mean(0)
        eol[i, 0] = torch.tensor(A @ c.reshape(6) + rng.normal(size=D) * 0.1)
    j = joint_eol_probe(eol, scenes, scene_split(300), n_boot=20)
    assert j.r2.item() > 0.9


def test_probe_does_not_interpolate_noise_with_scene_groups():
    # 3-object scenes, more features than training rows: within-scene LOO would pick a tiny alpha and overfit
    rng = np.random.default_rng(3)
    scenes = [generate_scene(np.random.default_rng(i), POOL, 3) for i in range(80)]
    lst = torch.full((80, 1, 6, 400), float("nan"))
    lst[:, :, :3] = torch.tensor(rng.normal(size=(80, 1, 3, 400)), dtype=torch.float32)
    df = probe_by_layer({"list": lst}, scenes, scene_split(80), n_boot=10)
    assert df.r2.item() < 0.05
    from geomap.probes import fit_probe, ALPHA_GRID
    X, Y, meta = centered_xy(lst[:, 0].numpy(), scenes, np.arange(64))
    p = fit_probe(X, Y, groups=meta["scene"].values)
    assert p.alpha_ > ALPHA_GRID[0] * p.scale_


def test_distance_rows_chain():
    from geomap.probes import distance_rows
    from geomap.scenes import Relation, Scene
    rels = (Relation("M", "R", 0, 2, 0), Relation("C", "M", 3, 0, 1))
    s = Scene(names=("C", "R", "M"), coords={"R": (0, 0), "M": (0, 2), "C": (3, 2)}, relations=rels,
              parent={"R": None, "M": "R", "C": "M"}, hop={"R": 0, "M": 1, "C": 2}, seed=0)
    meta, add = distance_rows([s], [0])
    assert list(meta.name) == ["C", "M"] and list(meta.slot) == [0, 2]  # root excluded, list order kept
    assert np.allclose(meta.dist, [np.hypot(3, 2), 2.0])
    # additive baseline: one-hot of own (dx, dy) and of the parent's (dx, dy), values -5..5 -> 4 * 11 columns
    assert add.shape == (2, 44)
    c = add[0]  # C: own (3, 0), parent M's (0, 2)
    assert c[0 * 11 + 3 + 5] == 1 and c[1 * 11 + 0 + 5] == 1 and c[2 * 11 + 0 + 5] == 1 and c[3 * 11 + 2 + 5] == 1
    assert c.sum() == 4
    m_ = add[1]  # M: own (0, 2), parent is the root: all-zero parent block
    assert m_[22:].sum() == 0 and m_.sum() == 2
