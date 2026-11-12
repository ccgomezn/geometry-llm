"""E3: is a *composed* position carried where no stated text differs?

Chain R -> M -> C. The source scene changes only M's sentence, so M moves and C moves with it, while C's own
sentence is word-for-word the same. Copying C's sentence from the source run into the base run therefore transfers
only what C's sentence has read from M's: C's composed absolute position. If the answers then treat C as sitting at
its source position, with M left in place (the *target* layout), that composed position is carried there and used.
See docs/preregistration_e3.md."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geomap.composition import _chain
from geomap.models import LoadedModel
from geomap.patching import multi_patch_edit, prefix_acts, run_pair
from geomap.prompts import NAME_POOLS, aligned, build_prompt, single_token_names
from geomap.questions import Question, _resample_component, all_questions, answer_for, passes_margin
from geomap.scenes import Scene, generate_scene, hybrid, move_object
from geomap.scoring import predict, score


@dataclass(frozen=True)
class TransferPair:
    base: Scene
    src: Scene
    tgt: Scene
    mid: str
    child: str
    qs: tuple[Question, ...]


def target_scene(base: Scene, src: Scene, child: str) -> Scene:
    """`base` with the leaf `child` at its position in `src` (re-expressed relative to its unmoved parent)."""
    return hybrid(src, base, child)


def make_transfer_pairs(m: LoadedModel, n: int, seed: int, exclude_qtypes=(), n_target_q: int = 6,
                        n_control_q: int = 2, min_margin: float = 1.0, min_angle: float = 10.0) -> list[TransferPair]:
    """Target questions: the geometric answer differs between base and target. Controls: it does not."""
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
        found = None
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
                tgt = target_scene(s, src, child)
            except ValueError:
                continue
            if aligned(build_prompt(s, None, m.tok, m.family), build_prompt(src, None, m.tok, m.family), mid):
                found = (src, tgt)
                break
        if found is None:
            continue
        src, tgt = found
        qs = [q for q in all_questions(s, min_margin, min_angle) if q.qtype.value not in exclude_qtypes
              and passes_margin(q, tgt.coords, min_margin, min_angle)]
        tq = [q for q in qs if answer_for(q, tgt.coords) != answer_for(q, s.coords)]
        cq = [q for q in qs if answer_for(q, tgt.coords) == answer_for(q, s.coords)]
        if not tq:
            continue
        pick = lambda lst, k: [lst[i] for i in rng.permutation(len(lst))[:k]]
        out.append(TransferPair(s, src, tgt, mid, child, tuple(pick(tq, n_target_q) + pick(cq, n_control_q))))
    return out


def run_transfer_pair(m: LoadedModel, tp: TransferPair, layers, batch_size: int = 16) -> pd.DataFrame:
    """Clean answers on base ("none"), source and target, plus answers after copying C's sentence ("patched")."""
    df = run_pair(m, tp.base, tp.src, tp.child, list(tp.qs), ["relation"], layers, batch_size=batch_size)
    df["site"] = df.site.replace({"relation": "patched"})
    ts = score(m, [(build_prompt(tp.tgt, q, m.tok, m.family), q) for q in tp.qs], batch_size=batch_size)
    tg = pd.DataFrame([dict(site="target", layer=-1, qid=i, qtype=q.qtype.value, model_ans=predict(sc))
                       for i, (q, sc) in enumerate(zip(tp.qs, ts))])
    return pd.concat([df, tg], ignore_index=True)


def summarize_transfer(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per layer, with pair-bootstrap 95% CIs (model-relative):
    follow_T: patched answer = the model's target answer, on questions where its target and base answers differ;
    follow_S: patched answer = its source answer, on those questions where source and target answers differ too;
    stable: patched answer = its base answer, on questions where its target and base answers agree."""
    clean = {s: df[df.site == s].set_index(["pair", "qid"]).model_ans for s in ("none", "source", "target")}
    b, s, t = clean["none"], clean["source"].reindex(clean["none"].index), clean["target"].reindex(clean["none"].index)
    is_t, is_c = t != b, t == b
    rng = np.random.default_rng(seed)
    rows = []
    for layer, g in df[df.site == "patched"].groupby("layer"):
        g = g.set_index(["pair", "qid"]).model_ans
        x = pd.DataFrame(dict(ans=g, b=b.reindex(g.index), s=s.reindex(g.index), t=t.reindex(g.index),
                              qt=is_t.reindex(g.index), qc=is_c.reindex(g.index))).reset_index()
        x["qs"] = x.qt & (x.s != x.t)
        x["fT"] = (x.ans == x.t) & x.qt
        x["fS"] = (x.ans == x.s) & x.qs
        x["st"] = (x.ans == x.b) & x.qc
        per = x.groupby("pair")[["fT", "fS", "st", "qt", "qs", "qc"]].sum()
        row = dict(layer=int(layer), n_T=int(per.qt.sum()), n_ctrl=int(per.qc.sum()))
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        for name, num, den in (("follow_T", "fT", "qt"), ("follow_S", "fS", "qs"), ("stable", "st", "qc")):
            nu, de = per[num].to_numpy(float), per[den].to_numpy(float)
            row[name] = nu.sum() / de.sum() if de.sum() else np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = nu[idx].sum(1) / de[idx].sum(1)
            row[name + "_lo"], row[name + "_hi"] = np.nanpercentile(boot, [2.5, 97.5]) if de.sum() else (np.nan, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def verdict_transfer(summary: pd.DataFrame, follow: float = 0.5, stable: float = 0.9, none: float = 0.1):
    """Decision rule of docs/preregistration_e3.md, over layers >= 1 (at layer 0 C's sentence is the same text):
    carried: at some layer follow_T >= 0.5 while control answers stay put (stable >= 0.9); first such layer returned.
    not carried: follow_T <= 0.1 at every layer. Otherwise: partial."""
    s = summary[summary.layer >= 1].sort_values("layer")
    hit = s[(s.follow_T >= follow) & (s.stable >= stable)]
    if len(hit):
        return "carried", int(hit.layer.iloc[0])
    if (s.follow_T <= none).all():
        return "not carried", None
    return "partial", None


def run_transfer_multi(m: LoadedModel, tp: TransferPair, starts, batch_size: int = 16) -> pd.DataFrame:
    """P8: clean answers on base ("none"), source and target, plus answers after copying C's sentence from the source
    run at every layer output from L to the last ("patched", layer = L), for each L in `starts`."""
    bpre = build_prompt(tp.base, None, m.tok, m.family)
    acts = prefix_acts(m, build_prompt(tp.src, None, m.tok, m.family), range(m.n_layers + 1))
    pos = list(range(*bpre.relation_spans[tp.child]))
    items, meta, info = [], [], []
    for qi, q in enumerate(tp.qs):
        bspec = build_prompt(tp.base, q, m.tok, m.family)
        for site, scene in (("none", tp.base), ("source", tp.src), ("target", tp.tgt)):
            items.append((bspec if site == "none" else build_prompt(scene, q, m.tok, m.family), q))
            meta.append(None)
            info.append((site, -1, qi))
        for L in starts:
            items.append((bspec, q))
            meta.append([(l, pos, acts[l][pos]) for l in range(L, m.n_layers + 1)])
            info.append(("patched", L, qi))
    scores = score(m, items, edit=multi_patch_edit(meta), batch_size=batch_size)
    return pd.DataFrame([dict(site=site, layer=L, qid=qi, qtype=tp.qs[qi].qtype.value, model_ans=predict(sc))
                         for (site, L, qi), sc in zip(info, scores)])


def verdict_multi(summary: pd.DataFrame, follow: float = 0.5, stable: float = 0.9, none: float = 0.1):
    """docs/preregistration_p8.md: carried if some start L has follow_T >= 0.5 with stable >= 0.9 (returns the latest
    such L); not carried if follow_T <= 0.1 at every L; otherwise partial."""
    hit = summary[(summary.follow_T >= follow) & (summary.stable >= stable)]
    if len(hit):
        return "carried by the dependent's sentence", int(hit.layer.max())
    if (summary.follow_T <= none).all():
        return "not carried", None
    return "partial", None
