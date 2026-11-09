"""Shared statistics: bootstrap CIs, chance levels, behavioural error signatures."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from geomap.questions import Question


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, stat: Callable = np.mean,
                 groups=None, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI. With `groups`, whole groups (scenes, pairs) are resampled."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    est = float(stat(values))
    if groups is None:
        idx = rng.integers(0, len(values), (n_boot, len(values)))
        boots = np.array([stat(values[i]) for i in idx])
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(uniq, len(uniq))
            boots[b] = stat(values[np.concatenate([members[g] for g in pick])])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def chance(q: Question) -> float:
    return 1.0 / len(q.candidates)


def _design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    lm = np.log(df["margin"].astype(float).clip(lower=1e-3))
    g = lm.groupby(df["qtype"])
    z = ((lm - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0).values
    hop = df["max_hop"].astype(float).values
    hop = (hop - hop.mean()) / (hop.std() or 1)
    fe = pd.get_dummies(df["qtype"], drop_first=True).astype(float).values
    return np.column_stack([hop, z, fe]), ["hop", "log_margin"]


def error_signature(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Logistic regression of correctness on standardised hop depth and log margin (z-scored within
    question type), with question-type fixed effects. Bootstrap CIs resample scenes."""
    X, names = _design(df)
    y = df["correct"].astype(int).values

    def fit(idx):
        return LogisticRegression(C=1e4, max_iter=2000).fit(X[idx], y[idx]).coef_[0][:2]

    est = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    scenes = df["scene_id"].values
    uniq = np.unique(scenes)
    members = {s: np.flatnonzero(scenes == s) for s in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[s] for s in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(fit(idx))
    boots = np.array(boots)
    out = {}
    for k, name in enumerate(names):
        lo, hi = np.quantile(boots[:, k], [0.025, 0.975])
        out[name] = (float(est[k]), float(lo), float(hi))
    return out


def gate(df: pd.DataFrame, min_acc: float = 0.75, min_excess: float = 0.15) -> dict:
    """P1 model-choice rule: >= 75% on 4-object main-set questions, and accuracy minus chance on
    questions touching objects 2+ hops from the root has a CI lower bound above 0.15."""
    d4 = df[(df["set"] == "main") & (df["n_objects"] == 4)]
    acc4 = bootstrap_ci(d4["correct"].astype(float), groups=d4["scene_id"])
    deep = df[(df["set"] == "main") & (df["max_hop"] >= 2)]
    excess = bootstrap_ci(deep["correct"].astype(float) - deep["chance"], groups=deep["scene_id"])
    return {"pass": bool(acc4[0] >= min_acc and excess[1] > min_excess), "acc4": acc4, "hop2_excess": excess}


def choose_model(results: dict[str, pd.DataFrame], order: list[str]) -> dict:
    """Preregistered P1 rule: the smallest model (in `order`) passing the gate on 4-object scenes; if none, the
    smallest passing on 3-object scenes (fallback). Question types whose accuracy CI includes chance on the
    chosen configuration are excluded from P3/P4 headline metrics."""
    for n in (4, 3):
        for name in order:
            df = results[name].copy()
            if n == 3:  # evaluate the gate on 3-object scenes
                df = df[~((df["set"] == "main") & (df["n_objects"] != 3))].assign(
                    n_objects=lambda d: d["n_objects"].where(d["set"] != "main", 4))
            if gate(df)["pass"]:
                main = results[name][(results[name]["set"] == "main") & (results[name]["n_objects"] == n)]
                excl = []
                for qt, d in main.groupby("qtype"):
                    lo = bootstrap_ci(d["correct"].astype(float), groups=d["scene_id"], n_boot=500)[1]
                    if lo <= d["chance"].mean():
                        excl.append(qt)
                return {"model": name, "n_objects": [n], "exclude": sorted(excl)}
    return {"model": None, "n_objects": [], "exclude": []}
