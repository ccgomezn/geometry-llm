"""E2: move the middle object M of a chain R -> M -> C inside the model and see whether C follows.

See docs/preregistration_e2.md. If a position for C has already been composed downstream of M's sentence, patching
M's sentence after that point moves M but leaves C behind (answers match the "hybrid" scene). If C's position is
rebuilt from the sentences when the question is asked, C follows M (answers match the source scene)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geomap.models import LoadedModel
from geomap.patching import run_pair
from geomap.prompts import NAME_POOLS, aligned, build_prompt, single_token_names
from geomap.questions import Question, _resample_component, all_questions, answer_for, passes_margin
from geomap.scenes import Scene, generate_scene, hybrid, move_object
from geomap.scoring import predict, score


@dataclass(frozen=True)
class ChainPair:
    base: Scene
    src: Scene
    hyb: Scene
    mid: str
    child: str
    qs: tuple[Question, ...]


def _chain(s: Scene) -> tuple[str, str] | None:
    child = next((n for n in s.names if s.hop[n] == 2), None)
    return (s.parent[child], child) if child else None


def make_chain_pairs(m: LoadedModel, n: int, seed: int, exclude_qtypes=(), max_q: int = 8,
                     min_margin: float = 1.0, min_angle: float = 10.0) -> list[ChainPair]:
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    out = []
    while len(out) < n:
        s = generate_scene(rng, pools[len(out) % 2], 3, p_shuffle=0.0)
        mc = _chain(s)
        if mc is None:
            continue
        mid, child = mc
        rel = next(r for r in s.relations if r.subj == mid)
        pair = None
        for _ in range(20):
            if rel.dx and rel.dy:
                dx, dy = _resample_component(rng, rel.dx), _resample_component(rng, rel.dy)
            else:
                k = _resample_component(rng, rel.dx or rel.dy)
                dx, dy = (k, 0) if rng.random() < 0.5 else (0, k)
            if (dx, dy) == (rel.dx, rel.dy):
                continue
            try:
                src = move_object(s, mid, dx, dy)
                hyb = hybrid(s, src, child)
            except ValueError:
                continue
            if aligned(build_prompt(s, None, m.tok, m.family), build_prompt(src, None, m.tok, m.family), mid):
                pair = (src, hyb)
                break
        if pair is None:
            continue
        src, hyb = pair
        qs = [q for q in all_questions(s, min_margin, min_angle) if q.qtype.value not in exclude_qtypes
              and passes_margin(q, src.coords, min_margin, min_angle)
              and passes_margin(q, hyb.coords, min_margin, min_angle)]
        a = lambda q, sc: answer_for(q, sc.coords)
        qs = [q for q in qs if a(q, s) != a(q, src) or a(q, src) != a(q, hyb)]
        if not qs:
            continue
        qs = [qs[i] for i in rng.permutation(len(qs))[:max_q]]
        out.append(ChainPair(s, src, hyb, mid, child, tuple(qs)))
    return out


def run_chain_pair(m: LoadedModel, cp: ChainPair, layers, batch_size: int = 16) -> pd.DataFrame:
    """Clean answers on base ("none"), source and hybrid, plus answers after patching M's sentence ("patched")."""
    df = run_pair(m, cp.base, cp.src, cp.mid, list(cp.qs), ["relation"], layers, batch_size=batch_size)
    df["site"] = df.site.replace({"relation": "patched"})
    hs = score(m, [(build_prompt(cp.hyb, q, m.tok, m.family), q) for q in cp.qs], batch_size=batch_size)
    hy = pd.DataFrame([dict(site="hybrid", layer=-1, qid=i, qtype=q.qtype.value, model_ans=predict(sc))
                       for i, (q, sc) in enumerate(zip(cp.qs, hs))])
    return pd.concat([df, hy], ignore_index=True)


def summarize(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per layer: follow_M (M-questions), follow_C and stay_C (C-questions), with pair-bootstrap 95% CIs."""
    clean = {s: df[df.site == s].set_index(["pair", "qid"]).model_ans for s in ("none", "source", "hybrid")}
    b, s, h = clean["none"], clean["source"], clean["hybrid"].reindex(clean["none"].index)
    is_m = (b != s) & (s == h)
    is_c = s != h
    rng = np.random.default_rng(seed)
    rows = []
    for layer, g in df[df.site == "patched"].groupby("layer"):
        g = g.set_index(["pair", "qid"]).model_ans
        t = pd.DataFrame(dict(ans=g, s=s.reindex(g.index), h=h.reindex(g.index), m=is_m.reindex(g.index),
                              c=is_c.reindex(g.index))).reset_index()
        t["fM"] = (t.ans == t.s) & t.m
        t["fC"] = (t.ans == t.s) & t.c
        t["sC"] = (t.ans == t.h) & t.c
        per = t.groupby("pair")[["fM", "fC", "sC", "m", "c"]].sum()
        row = dict(layer=int(layer), n_M=int(per.m.sum()), n_C=int(per.c.sum()))
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        for col, den in (("fM", "m"), ("fC", "c"), ("sC", "c")):
            name = {"fM": "follow_M", "fC": "follow_C", "sC": "stay_C"}[col]
            num, d = per[col].to_numpy(float), per[den].to_numpy(float)
            row[name] = num.sum() / d.sum() if d.sum() else np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = num[idx].sum(1) / d[idx].sum(1)
            row[name + "_lo"], row[name + "_hi"] = np.nanpercentile(boot, [2.5, 97.5]) if d.sum() else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def verdict(summary: pd.DataFrame, effective: float = 0.8) -> tuple[str, int | None]:
    """Preregistered rule (docs/preregistration_e2.md), over layers where the patch moves M (follow_M >= 0.8)."""
    eff = summary[summary.follow_M >= effective].sort_values("layer")
    hit = eff[(eff.follow_C <= 0.3) & (eff.stay_C >= 0.5)]
    if len(hit):
        return "eager", int(hit.layer.iloc[0])
    if len(eff) and (eff.follow_C >= eff.follow_M - 0.15).all():
        return "lazy", None
    return "mixed", None
