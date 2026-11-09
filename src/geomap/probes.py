"""Coordinate probes with per-scene centring, controls, RSA and the joint end-of-list probe."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from geomap.analysis import bootstrap_ci

ALPHAS = np.logspace(-2, 5, 15)


def scene_split(n_scenes: int, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n_scenes)
    k = int(round(n_scenes * test_frac))
    return np.sort(perm[k:]), np.sort(perm[:k])


def _features(acts, scenes, idx, control):
    """Rows per object (list order); returns raw features, raw coords, meta."""
    X, Y, meta = [], [], []
    for i in idx:
        s = scenes[i]
        for k, name in enumerate(s.names):
            if control == "stated_relation":
                rel = next((r for r in s.relations if r.subj == name), None)
                f = np.zeros(2 + 6)
                if rel is not None:
                    f[:2] = (rel.dx, rel.dy)
                f[2 + s.hop[name]] = 1
                X.append(f)
            else:
                X.append(acts[i, k])
            Y.append(s.coords[name])
            meta.append((i, name, s.hop[name]))
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=float),
            pd.DataFrame(meta, columns=["scene", "name", "hop"]))


def _center(X, Y, meta):
    g = meta["scene"].values
    Xc, Yc = X.astype(np.float64).copy(), Y.copy()
    for sc in np.unique(g):
        r = g == sc
        Xc[r] -= Xc[r].mean(0)
        Yc[r] -= Yc[r].mean(0)
    return Xc, Yc


def centered_xy(acts: np.ndarray, scenes, scene_idx, control: str | None = None, y_override=None):
    """acts: [N, 6, d] at one layer. Subtracts each scene's mean activation and mean coordinate."""
    X, Y, meta = _features(acts, scenes, scene_idx, control)
    if y_override is not None:
        Y = y_override
    Xc, Yc = _center(X, Y, meta)
    return Xc, Yc, meta


def fit_probe(X, Y) -> RidgeCV:
    return RidgeCV(alphas=ALPHAS).fit(X, Y)


def _r2(Y, P) -> float:
    return float(r2_score(Y, P, multioutput="uniform_average")) if len(Y) > 2 else float("nan")


def _r2_ci(Y, P, groups, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        boots.append(_r2(Y[idx], P[idx]))
    return tuple(np.quantile(boots, [0.025, 0.975])) if boots else (np.nan, np.nan)


def _shuffled_targets(scenes, idx, seed):
    """Object-level coordinates permuted across the whole dataset (control task)."""
    _, Y, _ = _features(None, scenes, idx, "stated_relation")
    return Y[np.random.default_rng(seed).permutation(len(Y))]


def probe_by_layer(cache: dict, scenes, split, control: Literal[None, "shuffled", "stated_relation"] = None,
                   nonlinear: bool = False, n_boot: int = 200, seed: int = 0, return_probes: bool = False,
                   layers=None):
    acts_all = cache["list"]
    tr, te = split
    n_layers = acts_all.shape[1] if control != "stated_relation" else 1
    rows, probes = [], {}
    for li in (range(n_layers) if layers is None else layers):
        acts = None if control == "stated_relation" else acts_all[:, li].float().numpy()
        feat = "stated_relation" if control == "stated_relation" else None
        ytr = _shuffled_targets(scenes, tr, seed) if control == "shuffled" else None
        yte = _shuffled_targets(scenes, te, seed + 1) if control == "shuffled" else None
        Xtr, Ytr, _ = centered_xy(acts, scenes, tr, feat, ytr)
        Xte, Yte, mte = centered_xy(acts, scenes, te, feat, yte)
        if nonlinear:
            model = MLPRegressor(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True,
                                 random_state=seed).fit(Xtr, Ytr)
        else:
            model = fit_probe(Xtr, Ytr)
        P = model.predict(Xte)
        lo, hi = _r2_ci(Yte, P, mte["scene"].values, n_boot, seed)
        hop = mte["hop"].values
        rows.append(dict(layer=li, r2=_r2(Yte, P), r2_lo=lo, r2_hi=hi,
                         r2_hop1=_r2(Yte[hop == 1], P[hop == 1]),
                         r2_hop2=_r2(Yte[hop == 2], P[hop == 2]),
                         r2_hop2p=_r2(Yte[hop >= 2], P[hop >= 2]),
                         r2_hop3p=_r2(Yte[hop >= 3], P[hop >= 3])))
        probes[li] = model
    df = pd.DataFrame(rows)
    return (df, probes) if return_probes else df


def joint_eol_probe(cache_eol, scenes4, split, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Predict all objects' centred coordinates (list order, 2k outputs) from the end-of-list activation.
    All scenes must have the same number of objects."""
    tr, te = split
    k = len(scenes4[0].names)
    Y = np.array([[c for n in s.names for c in s.coords[n]] for s in scenes4], dtype=float)
    Y = Y.reshape(len(scenes4), k, 2)
    Y = (Y - Y.mean(1, keepdims=True)).reshape(len(scenes4), 2 * k)
    rows = []
    for li in range(cache_eol.shape[1]):
        X = cache_eol[:, li].float().numpy()
        p = fit_probe(X[tr], Y[tr])
        P = p.predict(X[te])
        lo, hi = _r2_ci(Y[te], P, np.arange(len(te)), n_boot, seed)
        rows.append(dict(layer=li, r2=_r2(Y[te], P), r2_lo=lo, r2_hi=hi))
    return pd.DataFrame(rows)


def rsa_by_layer(cache: dict, scenes, n_boot: int = 200, min_objects: int = 4) -> pd.DataFrame:
    """Per scene: Spearman correlation between activation distances and true distances (centred acts)."""
    acts_all = cache["list"]
    idx = [i for i, s in enumerate(scenes) if len(s.names) >= min_objects]
    rows = []
    for li in range(acts_all.shape[1]):
        rhos = []
        for i in idx:
            n = len(scenes[i].names)
            A = acts_all[i, li, :n].float().numpy()
            C = np.array([scenes[i].coords[nm] for nm in scenes[i].names], float)
            rhos.append(spearmanr(pdist(A), pdist(C)).statistic)
        est, lo, hi = bootstrap_ci(np.nan_to_num(rhos), n_boot=n_boot)
        rows.append(dict(layer=li, rho=est, rho_lo=lo, rho_hi=hi))
    return pd.DataFrame(rows)


def probe_directions(probe) -> np.ndarray:
    """[d, 2] activation directions that move the probe's read-out by one unit in x and in y."""
    return np.linalg.pinv(probe.coef_)
