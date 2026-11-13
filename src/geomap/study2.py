"""Study 2 pilot: balanced accuracy, cheap baselines for "inside the triangle", the pilot summary and the go rule
(docs/preregistration_s2_pilot.md)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geomap.triangles import TriScene, in_box

HARD = ("inside", "box_hard")
MAP_FREE = ("box", "feature", "magnitude")


def balanced_accuracy(truth, pred, positive) -> float:
    """Mean of the accuracy on positive-class items (truth == positive) and on the rest; NaN if a class is empty.
    `positive` is one label or one label per item (two-choice rungs: the first-named object)."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    is_pos = truth == np.asarray(positive)
    correct = truth == pred
    if is_pos.all() or not is_pos.any():
        return float("nan")
    return float((correct[is_pos].mean() + correct[~is_pos].mean()) / 2)


def ba_ci(truth, pred, positive, groups, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Balanced accuracy with a 95% bootstrap CI over whole groups (scenes). Resamples with a missing class are
    dropped."""
    truth, pred, groups = np.asarray(truth), np.asarray(pred), np.asarray(groups)
    positive = np.broadcast_to(np.asarray(positive), truth.shape)
    est = balanced_accuracy(truth, pred, positive)
    if math.isnan(est):
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        boots[b] = balanced_accuracy(truth[idx], pred[idx], positive[idx])
    if np.isnan(boots).all():
        return est, float("nan"), float("nan")
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi)


def _pts(t: TriScene):
    co = t.scene.coords
    return co[t.point], [co[k] for k in t.corners]


def box_rule(tss) -> np.ndarray:
    return np.array(["yes" if in_box(p, *cs) else "no" for p, cs in map(_pts, tss)])


def feature_matrix(tss) -> np.ndarray:
    """Stated offsets only: B − A, C − B, P − its parent, and a one-hot of which corner is P's parent."""
    rows = []
    for t in tss:
        co, (a, b, c), p = t.scene.coords, t.corners, t.point
        par = t.scene.parent[p]
        d = lambda s, o: [co[s][0] - co[o][0], co[s][1] - co[o][1]]
        rows.append(d(b, a) + d(c, b) + d(p, par) + [float(par == k) for k in (a, b, c)])
    return np.asarray(rows, dtype=float)


def _inside_truth(tss) -> np.ndarray:
    return np.array([t.stratum == "inside" for t in tss])


def fit_feature_rule(train_tss) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(feature_matrix(train_tss), _inside_truth(train_tss))


def feature_rule(model: LogisticRegression, tss) -> np.ndarray:
    return np.where(model.predict(feature_matrix(tss)), "yes", "no")


def magnitude_matrix(tss) -> np.ndarray:
    """Size cues from the stated sentences, no shape: |dx|, |dy| of each sentence (B|A, C|B, P|parent), the length of
    P's offset, the three side lengths, the area, and a one-hot of which corner is P's parent (14 columns)."""
    rows = []
    for t in tss:
        co, (a, b, c), p = t.scene.coords, t.corners, t.point
        par = t.scene.parent[p]
        off = lambda s, o: (co[s][0] - co[o][0], co[s][1] - co[o][1])
        ab, bc, pp = off(b, a), off(c, b), off(p, par)
        side = lambda s, o: float(np.hypot(*off(s, o)))
        area = abs(ab[0] * bc[1] - ab[1] * bc[0]) / 2
        rows.append([abs(v) for v in (*ab, *bc, *pp)] + [float(np.hypot(*pp)), side(b, a), side(c, b), side(a, c),
                                                          area] + [float(par == k) for k in (a, b, c)])
    return np.asarray(rows, dtype=float)


def fit_magnitude_rule(train_tss):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        magnitude_matrix(train_tss), _inside_truth(train_tss))


def magnitude_rule(model, tss) -> np.ndarray:
    return np.where(model.predict(magnitude_matrix(tss)), "yes", "no")


def check_answers(df: pd.DataFrame, texts, truths) -> None:
    """Saved answers must belong to the current questions (same count, texts and truths), else ValueError."""
    if len(df) != len(texts) or list(df.text) != list(texts) or list(df.truth) != list(truths):
        raise ValueError("saved answers do not match the current questions; delete the answers file to rerun")


def centroid_score(t: TriScene) -> float:
    """P's distance to the centroid divided by the mean centroid-to-corner distance."""
    p, cs = _pts(t)
    g = np.mean(cs, axis=0)
    return float(np.hypot(*(np.asarray(p) - g)) / np.mean([np.hypot(*(np.asarray(c) - g)) for c in cs]))


def fit_centroid_rule(train_tss) -> float:
    """Threshold ("inside" iff score < threshold) with the best balanced accuracy on the training scenes."""
    sc = np.array([centroid_score(t) for t in train_tss])
    truth = np.where(_inside_truth(train_tss), "yes", "no")
    u = np.unique(sc)
    cands = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    return float(max(cands, key=lambda thr: balanced_accuracy(truth, np.where(sc < thr, "yes", "no"), "yes")))


def centroid_rule(tss, thr: float) -> np.ndarray:
    return np.array(["yes" if centroid_score(t) < thr else "no" for t in tss])


def _row(who, rung, subset, d, pred, n_boot, seed, mass=float("nan")):
    est, lo, hi = ba_ci(d.truth, pred, d.positive, d.scene, n_boot=n_boot, seed=seed)
    return dict(who=who, rung=rung, subset=subset, n=len(d), bal_acc=est, lo=lo, hi=hi,
                acc=float(np.mean(np.asarray(d.truth) == np.asarray(pred))), mean_prob_mass=mass, vs=None)


def ba_diff_ci(truth, pred_a, pred_b, positive, groups, n_boot: int = 1000,
               seed: int = 0) -> tuple[float, float, float]:
    """Balanced accuracy of a minus b on the same items, with a paired 95% bootstrap CI over whole groups."""
    truth, pa, pb, groups = (np.asarray(x) for x in (truth, pred_a, pred_b, groups))
    positive = np.broadcast_to(np.asarray(positive), truth.shape)
    f = lambda i: balanced_accuracy(truth[i], pa[i], positive[i]) - balanced_accuracy(truth[i], pb[i], positive[i])
    est = f(np.arange(len(truth)))
    if math.isnan(est):
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = np.array([f(np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])) for _ in range(n_boot)])
    if np.isnan(boots).all():
        return est, float("nan"), float("nan")
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return est, float(lo), float(hi)


def pilot_summary(df: pd.DataFrame, base_preds: dict, n_boot: int = 1000, seed: int = 0,
                  map_free=MAP_FREE) -> pd.DataFrame:
    """df: one row per question (scene, rung, stratum, truth, pred, positive, mass). base_preds: baseline name ->
    rung-4 predictions indexed by scene number. Adds a row `model_minus_best_map_free`: the model's rung-4 hard
    balanced accuracy minus the best map-free baseline's (named in `vs`), with a paired scene-bootstrap CI."""
    rows = []
    for rung in sorted(df.rung.unique()):
        d = df[df.rung == rung]
        rows.append(_row("model", rung, "all", d, d.pred, n_boot, seed, float(d.mass.mean())))
    d4 = df[df.rung == 4].sort_values("scene")
    hard = d4[d4.stratum.isin(HARD)]
    rows.append(_row("model", 4, "hard", hard, hard.pred, n_boot, seed, float(hard.mass.mean())))
    for st in ("inside", "box_hard", "outside_box"):
        d = d4[d4.stratum == st]
        if len(d):
            rows.append(_row("model", 4, st, d, d.pred, n_boot, seed, float(d.mass.mean())))
    bpred = {name: np.asarray(preds)[hard.scene.values] for name, preds in base_preds.items()}
    for name, p in bpred.items():
        rows.append(_row(name, 4, "hard", hard, p, n_boot, seed))
    free = {r["who"]: r["bal_acc"] for r in rows if r["who"] in map_free and r["subset"] == "hard"}
    if free:
        best = max(free, key=lambda k: -1.0 if math.isnan(free[k]) else free[k])
        est, lo, hi = ba_diff_ci(hard.truth, hard.pred, bpred[best], hard.positive, hard.scene, n_boot, seed)
        rows.append(dict(who="model_minus_best_map_free", rung=4, subset="hard", n=len(hard), bal_acc=est, lo=lo,
                         hi=hi, acc=float("nan"), mean_prob_mass=float("nan"), vs=best))
    return pd.DataFrame(rows)


def verdict_pilot(summary: pd.DataFrame, floor: float = 0.65) -> str:
    """Rung 4, hard subset. go: balanced accuracy >= floor, its CI above 0.5, and the paired CI of (model − best
    map-free baseline) above 0. weak pass: the two CI conditions but below the floor. fail: anything else."""
    h = summary[(summary.rung == 4) & (summary.subset == "hard")]
    m, d = h[h.who == "model"], h[h.who == "model_minus_best_map_free"]
    if m.empty or d.empty:
        return "fail"
    est, lo, dlo = float(m.bal_acc.iloc[0]), float(m.lo.iloc[0]), float(d.lo.iloc[0])
    if any(math.isnan(x) for x in (est, lo, dlo)) or not (lo > 0.5 and dlo > 0):
        return "fail"
    return "go" if est >= floor else "weak pass"
