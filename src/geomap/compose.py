"""Sum-of-parts test (docs/preregistration_compose.md): did the model combine a two-step object's two sentences
into one position, or does it keep each sentence's offset side by side (and only a probe adds them up)?

A position is a sum of the stated offsets, so a linear probe can read it from parts kept side by side. Distance to
the root, |x| and |y| are not sums: no function of one sentence plus a function of the other gives them. The
*parts baseline* is that best case for side-by-side storage: one one-hot code per sentence over its joint (dx, dy)
(any function of the object's own sentence + any function of its parent's), fitted like the probe. Activations that
beat it on held-out scenes carry an interaction of the two sentences: a position the model composed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geomap.probes import GroupRidge

TARGETS = ("dist", "abs_x", "abs_y")
GAP_MIN, NULL_MAX = 0.15, 0.05


def _code(dx: int, dy: int) -> int:
    return (dx + 5) * 11 + (dy + 5)


def part_rows(scenes, idx, hops=(2,)) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows for the non-root objects with the asked hop counts (object-list order): targets relative to the root,
    the parts-baseline features [n, 242] and the true-position reference features (x, y, x², y², xy) [n, 5]."""
    meta, parts, pos = [], [], []
    for i in idx:
        s = scenes[i]
        rel = {r.subj: r for r in s.relations}
        rx, ry = s.coords[s.root]
        for k, name in enumerate(s.names):
            if name == s.root or s.hop[name] not in hops:
                continue
            x, y = s.coords[name][0] - rx, s.coords[name][1] - ry
            f = np.zeros(242, dtype=np.float32)
            own, par = rel[name], rel.get(s.parent[name])
            f[_code(own.dx, own.dy)] = 1
            if par is not None:
                f[121 + _code(par.dx, par.dy)] = 1
            parts.append(f)
            pos.append([x, y, x * x, y * y, x * y])
            meta.append((int(i), k, name, s.hop[name], float(np.hypot(x, y)), abs(x), abs(y)))
    cols = ["scene", "slot", "name", "hop", "dist", "abs_x", "abs_y"]
    return pd.DataFrame(meta, columns=cols), np.asarray(parts, dtype=np.float32), np.asarray(pos, dtype=float)


def _r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_predict(Xtr, ytr, gtr, Xte) -> np.ndarray:
    mx, my = Xtr.mean(0), ytr.mean()
    return GroupRidge().fit(Xtr - mx, (ytr - my)[:, None], gtr).predict(Xte - mx)[:, 0] + my


def site_scores(X, parts, meta, tr, te, target: str = "dist", n_boot: int = 1000, seed: int = 0) -> dict:
    """Held-out R² of a ridge probe on activations X and of the parts baseline, and their difference with a paired
    scene-bootstrap 95% CI (the same resampled test scenes for both)."""
    y, g = meta[target].to_numpy(float), meta.scene.to_numpy()
    tr, te = np.asarray(tr), np.asarray(te)
    p_act = fit_predict(np.asarray(X, dtype=float)[tr], y[tr], g[tr], np.asarray(X, dtype=float)[te])
    p_base = fit_predict(parts[tr].astype(float), y[tr], g[tr], parts[te].astype(float))
    yt, gt = y[te], g[te]
    rng = np.random.default_rng(seed)
    u = np.unique(gt)
    mem = {s: np.flatnonzero(gt == s) for s in u}
    gaps = []
    for _ in range(n_boot):
        i = np.concatenate([mem[s] for s in rng.choice(u, len(u))])
        gaps.append(_r2(yt[i], p_act[i]) - _r2(yt[i], p_base[i]))
    ra, rb = _r2(yt, p_act), _r2(yt, p_base)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return dict(r2_act=ra, r2_base=rb, gap=ra - rb, gap_lo=float(lo), gap_hi=float(hi), n_test=int(te.sum()))


def verdict_compose(s: pd.DataFrame) -> tuple[str, list[str]]:
    """On the primary rows (distance, preregistered layer per site): composed where gap ≥ GAP_MIN with CI lower
    bound > 0; parts only if every site's CI upper bound < NULL_MAX; otherwise undecided."""
    p = s[s.primary & (s.target == "dist")]
    sites = [r.site for r in p.itertuples() if r.gap >= GAP_MIN and r.gap_lo > 0]
    if sites:
        return "composed by the model", sites
    if (p.gap_hi < NULL_MAX).all():
        return "parts only", []
    return "undecided", []


def unseen_pairs(parts: np.ndarray, tr) -> np.ndarray:
    """Rows not in `tr` whose (own sentence, parent sentence) pair of offsets never occurs among the `tr` rows."""
    tr = np.asarray(tr)
    pair = [(int(np.argmax(r[:121])), int(np.argmax(r[121:]))) for r in parts]
    seen = {p for p, t in zip(pair, tr) if t}
    return np.array([(not t) and p not in seen for p, t in zip(pair, tr)])
