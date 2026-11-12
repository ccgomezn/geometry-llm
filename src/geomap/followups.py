"""P5 (docs/preregistration_p5.md): is the P4c 2-D code a genuine encoding, and is the decodable map used?

P5a/b: per token of the moved object's sentence, how the subspace coordinates change with the stated offset
(Δz_j = W_j Δoff), and a write-in edit that sets them from the offset instead of copying them from a source run.
P5c: iterative nullspace projection of the probe-readable coordinates, erased at every token position."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci

from geomap.composition import _chain
from geomap.models import LoadedModel, layer_out
from geomap.patching import prefix_acts
from geomap.probes import GroupRidge
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import Question, all_questions
from geomap.scenes import Scene, generate_scene


# ---------- P5a / P5b ----------

def _rel(scene: Scene, name: str):
    return next(r for r in scene.relations if r.subj == name)


def sentence_diffs(m: LoadedModel, pairs, layer: int) -> list[dict]:
    """One record per (pair, token of the moved object's sentence): dh = h_src − h_base [d] at `layer`, the token's
    index j counted from the sentence end (0 = last token), and doff = change in that sentence's stated (dx, dy)."""
    recs = []
    for p, (base, src, leaf, _) in enumerate(pairs):
        bspec = build_prompt(base, None, m.tok, m.family)
        lo, hi = bspec.relation_spans[leaf]
        hb = prefix_acts(m, bspec, [layer])[layer][lo:hi].double()
        hs = prefix_acts(m, build_prompt(src, None, m.tok, m.family), [layer])[layer][lo:hi].double()
        rb, rs = _rel(base, leaf), _rel(src, leaf)
        doff = np.array([rs.dx - rb.dx, rs.dy - rb.dy], dtype=float)
        dh = (hs - hb).cpu().numpy()
        for k in range(hi - lo):
            recs.append(dict(pair=p, j=hi - lo - 1 - k, doff=doff, dh=dh[k]))
    return recs


def project(recs: list[dict], Q: np.ndarray) -> list[dict]:
    """Adds dz = Qᵀ dh (the change in the 2 subspace coordinates)."""
    return [dict(r, dz=np.asarray(r["dh"]) @ Q) for r in recs]


def _key(j: int, n_tok: int) -> int:
    return j if j < n_tok else n_tok


def _fit(D: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """W (2×2) minimising ||Z − D Wᵀ||: dz ≈ W doff."""
    Wt, *_ = np.linalg.lstsq(D, Z, rcond=None)
    return Wt.T


def fit_token_maps(recs: list[dict], n_tok: int = 8) -> dict[int, np.ndarray]:
    """One 2×2 map per token index j < n_tok from the sentence end, one pooled map (key n_tok) for the rest."""
    by = {}
    for r in recs:
        by.setdefault(_key(r["j"], n_tok), []).append(r)
    return {k: _fit(np.array([r["doff"] for r in v]), np.array([r["dz"] for r in v])) for k, v in by.items()}


def cv_r2(recs: list[dict], n_tok: int = 8, n_folds: int = 5, seed: int = 0) -> float:
    """Pair-grouped cross-validated R² of dz predicted by the per-token maps (pooled over tokens and both axes)."""
    pairs = np.unique([r["pair"] for r in recs])
    fold = {p: k % n_folds for k, p in enumerate(np.random.default_rng(seed).permutation(pairs))}
    Z = np.array([r["dz"] for r in recs])
    P = np.zeros_like(Z)
    for f in range(n_folds):
        tr = [r for r in recs if fold[r["pair"]] != f]
        maps = fit_token_maps(tr, n_tok)
        for i, r in enumerate(recs):
            if fold[r["pair"]] == f:
                W = maps.get(_key(r["j"], n_tok))
                P[i] = W @ r["doff"] if W is not None else 0.0
    return float(1 - ((Z - P) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum())


def writein_rows(rows: list[dict], base_h: dict, doffs: dict, maps: dict, Q: np.ndarray, n_tok: int = 8) -> list[dict]:
    """Copies of P4 rows whose swapped-in states are base + Q W_j Δoff at each sentence token (j from the end).
    With the subspace intervention Q this sets the coordinates to base + W_j Δoff instead of the source run's."""
    Qt = torch.as_tensor(Q, dtype=torch.float64)
    out = []
    for r in rows:
        p, hb = r["pair"], base_h[r["pair"]]
        n = len(r["pos"])
        add = []
        for k in range(n):
            j = n - 1 - k
            W = maps.get(j, maps.get(n_tok))
            add.append(Qt @ torch.as_tensor(W @ doffs[p], dtype=torch.float64))
        out.append(dict(r, src_h=(hb.double() + torch.stack(add)).to(hb.dtype)))
    return out


# ---------- P5c ----------

def _cv_r2_ridge(X, Y, groups, n_folds: int = 3, seed: int = 0) -> float:
    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    fold_of = {g: k % n_folds for k, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    P = np.zeros_like(Y, dtype=float)
    for k in range(n_folds):
        tr, va = f != k, f == k
        P[va] = GroupRidge().fit(X[tr], Y[tr], groups[tr]).predict(X[va])
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def inlp_basis(X, Y, groups, max_rounds: int = 20, r2_stop: float = 0.1):
    """Iterative nullspace projection: fit the ridge probe, add its directions to the basis, project them out, until
    the probe's group-CV R² < r2_stop or max_rounds. Returns (orthonormal basis [d, r], R² before each round + final)."""
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups)
    B = np.zeros((X.shape[1], 0))
    hist = []
    for _ in range(max_rounds):
        probe = GroupRidge().fit(X - (X @ B) @ B.T, Y, groups)  # its group-CV R² is the stopping criterion
        hist.append(probe.cv_r2_)
        if probe.cv_r2_ < r2_stop:
            return B, hist
        W = probe.coef_.T  # [d, 2]
        W = W - B @ (B.T @ W)
        B = np.linalg.qr(np.concatenate([B, W], 1))[0]
    hist.append(GroupRidge().fit(X - (X @ B) @ B.T, Y, groups).cv_r2_)
    return B, hist


def erase_edit(bases: dict):
    """score() edit: at each layer l, h ← h − (h·B − c)Bᵀ at every position (bases[l] = (B [d, r], c = μ·B [r]))."""

    def edit(m: LoadedModel, row_items: list[int]):
        for l in sorted(bases):
            B, c = bases[l]
            if B.shape[1] == 0:
                continue
            h = layer_out(m, l)
            Bd, cd = B.to(h.device, torch.float32), c.to(h.device, torch.float32)
            hf = h[:].float()
            h[:] = (hf - (hf @ Bd - cd) @ Bd.T).to(h.dtype)

    return edit


def chain_items(m: LoadedModel, n: int, seed: int, exclude_qtypes=(), min_margin: float = 1.0,
                min_angle: float = 10.0) -> list[tuple[Scene, str, str, list[tuple[Question, bool]]]]:
    """3-object chains R → M → C with every question that passes the margins, flagged composed when it involves C."""
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    out = []
    while len(out) < n:
        s = generate_scene(rng, pools[len(out) % 2], 3, p_shuffle=0.0)
        mc = _chain(s)
        if mc is None:
            continue
        mid, child = mc
        qs = [q for q in all_questions(s, min_margin, min_angle) if q.qtype.value not in exclude_qtypes]
        if qs:
            out.append((s, mid, child, [(q, child in q.args or child in q.candidates) for q in qs]))
    return out


def erase_summary(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, group: str = "composed") -> pd.DataFrame:
    """Per (config, composed): clean accuracy, drops with the probe basis and the random basis, and the excess drop
    (probe − random) with a scene-bootstrap 95% CI. df: one row per (scene, qi, config, cond) with `correct`."""
    rows = []
    clean = df[df.cond == "clean"].groupby(["scene", "qi"]).correct.first()
    for (cfg, comp), g in df[df.cond != "clean"].groupby(["config", group]):
        w = g.pivot_table(index=["scene", "qi"], columns="cond", values="correct")
        w["clean"] = clean.reindex(w.index)
        w["excess"] = (w.clean - w.probe) - (w.clean - w.random)
        per = w.reset_index()
        est, lo, hi = bootstrap_ci(per.excess, groups=per.scene, n_boot=n_boot, seed=seed)
        rows.append(dict(config=cfg, **{group: bool(comp) if group == "composed" else comp}, n=len(w), clean=w.clean.mean(),
                         drop_probe=(w.clean - w.probe).mean(), drop_random=(w.clean - w.random).mean(),
                         excess=est, excess_lo=lo, excess_hi=hi))
    return pd.DataFrame(rows)


def verdict_erase(summary: pd.DataFrame, r2_after: float, used: float = 0.10, gap: float = 0.05,
                  none: float = 0.03, r2_max: float = 0.1) -> str:
    """docs/preregistration_p5.md, P5c. r2_after: the largest held-out probe R² left after erasure at erased layers."""
    comp = summary[summary.composed].set_index("config").excess
    stat = summary[~summary.composed].set_index("config").excess
    if any(comp[c] >= used and comp[c] - stat.get(c, np.nan) >= gap for c in comp.index):
        return "map used"
    if (comp <= none).all() and r2_after <= r2_max:
        return "map not used (in this subspace)"
    return "unclear"


# ---------- P6 ----------

def depth_group(depth: int) -> str:
    return "shallow" if depth <= 1 else "middle" if depth == 2 else "deep"


def depth_items(m: LoadedModel, n: int, seed: int, n_objects=(4, 5), exclude_qtypes=(), max_q: int = 24,
                min_margin: float = 1.0, min_angle: float = 10.0) -> list[tuple[Scene, list[tuple[Question, int]]]]:
    """Scenes with n_objects alternating; up to max_q margin-passing questions each (random subset), each with its
    depth: the largest hop among the objects it names or offers as candidates."""
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    out = []
    while len(out) < n:
        s = generate_scene(rng, pools[len(out) % 2], n_objects[len(out) % len(n_objects)])
        qs = [q for q in all_questions(s, min_margin, min_angle) if q.qtype.value not in exclude_qtypes]
        if not qs:
            continue
        qs = [qs[i] for i in sorted(rng.permutation(len(qs))[:max_q])]
        out.append((s, [(q, max(s.hop[x] for x in set(q.args) | set(q.candidates) if x in s.hop)) for q in qs]))
    return out


def verdict_hard(summary: pd.DataFrame, r2_after: float, headroom: float, used: float = 0.10, gap: float = 0.05,
                 none: float = 0.03, r2_max: float = 0.1, min_headroom: float = 0.15) -> str:
    """docs/preregistration_p6.md. headroom: clean accuracy on deep questions minus their mean chance."""
    if headroom < min_headroom:
        return "uninformative"
    deep = summary[summary.group == "deep"].set_index("config").excess
    shallow = summary[summary.group == "shallow"].set_index("config").excess
    if any(deep[c] >= used and deep[c] - shallow.get(c, np.nan) >= gap for c in deep.index):
        return "map used when chaining is hard"
    if (deep <= none).all() and r2_after <= r2_max:
        return "still not used"
    return "unclear"
