"""The slider: push one object's position along decoded x/y directions and watch every answer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from geomap.analysis import bootstrap_ci
from geomap.models import LoadedModel, layer_out
from geomap.prompts import build_prompt
from geomap.questions import Question, answer_for
from geomap.scenes import Scene
from geomap.scoring import predict, score


def predicted_answers(scene: Scene, leaf: str, deltas: np.ndarray, q: Question) -> list[str]:
    """Geometric answers with `leaf` moved by each (dx, dy) from its base position (off-grid allowed)."""
    out = []
    bx, by = scene.coords[leaf]
    for dx, dy in deltas:
        c = dict(scene.coords)
        c[leaf] = (bx + float(dx), by + float(dy))
        out.append(answer_for(q, c))
    return out


def steer_edit(meta: list):
    """meta[item] = (layer, positions, vector [d]) added to the residual stream at those positions."""

    def edit(m: LoadedModel, row_items: list[int]):
        ls = sorted({meta[i][0] for i in row_items})
        for l in ls:
            h = layer_out(m, l)
            for j, i in enumerate(row_items):
                if meta[i][0] == l:
                    v = meta[i][2].to(h.device, h.dtype)
                    h[j, meta[i][1], :] = h[j, meta[i][1], :] + v

    return edit


def slider(m: LoadedModel, scene: Scene, leaf: str, layer: int, dirs: np.ndarray, deltas: np.ndarray,
           qs: Sequence[Question], batch_size: int = 32) -> pd.DataFrame:
    dirs_t = torch.tensor(np.asarray(dirs), dtype=torch.float32)
    items, meta, info = [], [], []
    for qi, q in enumerate(qs):
        spec = build_prompt(scene, q, m.tok, m.family)
        pos = [spec.list_pos[leaf]]
        preds = predicted_answers(scene, leaf, deltas, q)
        for st, (dx, dy) in enumerate(deltas):
            items.append((spec, q))
            meta.append((layer, pos, dirs_t @ torch.tensor([float(dx), float(dy)])))
            info.append((st, float(dx), float(dy), qi, q.qtype.value, preds[st]))
    scores = score(m, items, edit=steer_edit(meta), batch_size=batch_size)
    rows = []
    for (st, dx, dy, qi, qt, pred), sc in zip(info, scores):
        lp = np.array(list(sc.values()))
        p = np.exp(lp - lp.max())
        p /= p.sum()
        probs = dict(zip(sc.keys(), p))
        rows.append(dict(step=st, dx=dx, dy=dy, qid=qi, qtype=qt, model_ans=predict(sc), predicted_ans=pred,
                         p_model=float(probs[predict(sc)]), p_pred=float(probs[pred]), probs=probs))
    return pd.DataFrame(rows)


def slider_agreement(df: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Agreement between model and geometry along every sweep, and where the first flip happens.

    df needs: scene, axis, qid, step, dx, dy, predicted_ans, model_ans."""
    agree = (df.model_ans == df.predicted_ans).astype(float)
    est, lo, hi = bootstrap_ci(agree, groups=df.scene, n_boot=n_boot)
    pred_pos, obs_pos, offsets, n_flip, n_obs = [], [], [], 0, 0
    for (_, axis, _), g in df.groupby(["scene", "axis", "qid"]):
        g = g.sort_values("step")
        coord = g.dx.values if axis == "x" else g.dy.values
        pf = [i for i in range(1, len(g)) if g.predicted_ans.iloc[i] != g.predicted_ans.iloc[i - 1]]
        if not pf:
            continue
        n_flip += 1
        of = [i for i in range(1, len(g)) if g.model_ans.iloc[i] != g.model_ans.iloc[i - 1]]
        if not of:
            continue
        n_obs += 1
        mid = lambda i: (coord[i - 1] + coord[i]) / 2  # a flip happens between two grid points
        p0, o0 = mid(pf[0]), mid(of[0])  # first predicted vs first observed flip (no nearest-match leniency)
        pred_pos.append(p0)
        obs_pos.append(o0)
        offsets.append(float(o0 - p0))
    r = pearsonr(pred_pos, obs_pos).statistic if len(pred_pos) >= 3 else float("nan")
    return dict(agreement=est, agreement_lo=lo, agreement_hi=hi, n_sweeps_with_flip=n_flip,
                frac_observed_flip=(n_obs / n_flip) if n_flip else float("nan"), flip_r=float(r),
                flip_offsets=offsets)


def natural_range(steered_norms: np.ndarray, ref_norms: np.ndarray) -> float:
    lo, hi = np.quantile(ref_norms, [0.01, 0.99])
    s = np.asarray(steered_norms)
    return float(((s >= lo) & (s <= hi)).mean())


def das_directions(Q: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directions [d, 2] that move an object by (dx, dy) inside a DAS subspace Q [d, k].

    B [k, 2] maps subspace coordinates of centred activations X to coordinates Y (Y ≈ X Q B). Moving the
    activation by D δ with D = Q pinv(B)ᵀ changes the read-out by δ."""
    B, *_ = np.linalg.lstsq(X @ Q, Y, rcond=None)
    return Q @ np.linalg.pinv(B).T, B
