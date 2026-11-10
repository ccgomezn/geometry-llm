"""Interchange interventions: move one object inside the model and check that every answer follows."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci
from geomap.models import LoadedModel, layer_out
from geomap.prompts import NAME_POOLS, PromptSpec, build_prompt, make_pair_aligned, single_token_names
from geomap.questions import Question, all_questions, answer_for, split_questions
from geomap.scenes import Scene, generate_scene
from geomap.scoring import predict, score

Site = Literal["list", "relation", "eol", "all", "all_but_readout", "sent_end", "sent_rest"]


def site_positions(spec: PromptSpec, site: Site, leaf: str) -> list[int]:
    if site == "list":
        return [spec.list_pos[leaf]]
    if site == "relation":
        return list(range(*spec.relation_spans[leaf]))
    if site == "sent_end":  # last token of the object's own sentence
        return [spec.relation_spans[leaf][1] - 1]
    if site == "sent_rest":  # the rest of that sentence
        lo, hi = spec.relation_spans[leaf]
        return list(range(lo, hi - 1))
    if site == "eol":
        return [spec.eol_pos]
    if site == "all":
        return list(range(spec.prefix_len))
    if site == "all_but_readout":  # necessity test: everything except every list position and the end of the list
        readout = set(spec.list_pos.values()) | {spec.eol_pos}
        return [p for p in range(spec.prefix_len) if p not in readout]
    raise ValueError(site)


def prefix_acts(m: LoadedModel, spec: PromptSpec, layers: Sequence[int]) -> dict[int, torch.Tensor]:
    ids = torch.tensor([spec.input_ids[: spec.prefix_len]])
    saved = {}
    with torch.no_grad(), m.lm.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}):
        for l in sorted(layers):
            saved[l] = layer_out(m, l)[0].detach().cpu().save()
    return {l: v for l, v in saved.items()}


def patch_edit(meta: list):
    """meta[item] = (layer, positions, source activations [n_pos, d]) or None. Handles mixed layers per batch."""

    def edit(m: LoadedModel, row_items: list[int]):
        ls = sorted({meta[i][0] for i in row_items if meta[i] is not None})
        for l in ls:
            h = layer_out(m, l)
            for j, i in enumerate(row_items):
                if meta[i] is not None and meta[i][0] == l:
                    h[j, meta[i][1], :] = meta[i][2].to(h.device, h.dtype)

    return edit


def run_pair(m: LoadedModel, base: Scene, src: Scene, leaf: str, qs: Sequence[Question],
             sites: Sequence[Site], layers: Sequence[int], batch_size: int = 16) -> pd.DataFrame:
    base_pre = build_prompt(base, None, m.tok, m.family)
    src_pre = build_prompt(src, None, m.tok, m.family)
    acts = prefix_acts(m, src_pre, layers)
    items, meta, info = [], [], []
    for qi, q in enumerate(qs):
        bspec = build_prompt(base, q, m.tok, m.family)
        items.append((bspec, q)); meta.append(None); info.append(("none", -1, qi))
        items.append((build_prompt(src, q, m.tok, m.family), q)); meta.append(None); info.append(("source", -1, qi))
        for site in sites:
            pos = site_positions(base_pre, site, leaf)
            for l in layers:
                items.append((bspec, q))
                meta.append((l, pos, acts[l][pos]))
                info.append((site, l, qi))
    scores = score(m, items, edit=patch_edit(meta), batch_size=batch_size)
    rows = []
    for (site, l, qi), sc in zip(info, scores):
        q = qs[qi]
        b, s = answer_for(q, base.coords), answer_for(q, src.coords)
        rows.append(dict(site=site, layer=l, qid=qi, qtype=q.qtype.value, diagnostic=b != s,
                         base_ans=b, src_ans=s, model_ans=predict(sc), lp_base_ans=sc[b], lp_src_ans=sc[s]))
    return pd.DataFrame(rows)


def consistency(df_all: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Per (site, layer).

    Headline (model-relative, ceiling 1): joint_iia = share of pairs whose patched answers equal the model's own
    clean-source answers on every question, over pairs where at least one clean answer differs between base
    and source; iia_rate = the same per question, on those differing questions.
    Geometric (secondary): flip rate on diagnostic questions, control stability, joint consistency per pair."""
    d = df_all[~df_all.site.isin(["none", "source"])].copy()
    d["follows_src"] = d.model_ans == d.src_ans
    d["stays_base"] = d.model_ans == d.base_ans
    # the model's own clean answers (targets of the model-relative interchange metric, ceiling 1)
    clean_b = df_all[df_all.site == "none"].set_index(["pair", "qid"]).model_ans
    clean_s = df_all[df_all.site == "source"].set_index(["pair", "qid"]).model_ans
    have = len(clean_b) > 0 and len(clean_s) > 0
    if have:
        key = list(zip(d.pair, d.qid))
        d["m_base"] = clean_b.reindex(key).values
        d["m_src"] = clean_s.reindex(key).values
        d["m_diag"] = d.m_src != d.m_base
        d["iia"] = d.model_ans == d.m_src
    out = []
    for (site, layer), g in d.groupby(["site", "layer"]):
        per_pair = []
        for p, gp in g.groupby("pair"):
            dg, cg = gp[gp.diagnostic], gp[~gp.diagnostic]
            if len(dg) == 0:
                continue
            per_pair.append((p, float(dg.follows_src.all() and cg.stays_base.all())))
        n_pairs = len(per_pair)
        n_drop = g.pair.nunique() - n_pairs
        diag, ctrl = g[g.diagnostic], g[~g.diagnostic]
        row = dict(site=site, layer=layer, n_pairs=n_pairs, n_dropped=n_drop)
        if n_pairs:
            jc = bootstrap_ci([v for _, v in per_pair], n_boot=n_boot)
            fr = bootstrap_ci(diag.follows_src.astype(float), groups=diag.pair, n_boot=n_boot)
            row.update(joint_consistency=jc[0], jc_lo=jc[1], jc_hi=jc[2],
                       flip_rate=fr[0], flip_lo=fr[1], flip_hi=fr[2])
        row["control_stability"] = float(ctrl.stays_base.mean()) if len(ctrl) else np.nan
        if have:
            jp = [(p, float(gp.iia.all())) for p, gp in g.groupby("pair") if gp.m_diag.any()]
            row["n_pairs_iia"] = len(jp)
            if jp:
                ji = bootstrap_ci([v for _, v in jp], n_boot=n_boot)
                md = g[g.m_diag]
                row.update(joint_iia=ji[0], jiia_lo=ji[1], jiia_hi=ji[2], iia_rate=float(md.iia.mean()))
                for qt, gq in md.groupby("qtype"):
                    row[f"iia_{qt}"] = float(gq.iia.mean())
        for qt, gq in diag.groupby("qtype"):
            row[f"flip_{qt}"] = float(gq.follows_src.mean())
        out.append(row)
    return pd.DataFrame(out)


def make_pairs(m, n, seed, max_diag=6, max_ctrl=4, n_objects=(4, 5), exclude_qtypes=()):
    """Aligned (base, source, leaf, questions) pairs; diagnostic questions spread across types, plus controls."""
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    pairs = []
    while len(pairs) < n:
        s = generate_scene(rng, pools[len(pairs) % 2], n_objects[len(pairs) % len(n_objects)])
        try:
            base, src, leaf = make_pair_aligned(s, rng, m.tok, m.family)
        except ValueError:
            continue
        qs_all = [q for q in all_questions(base) if q.qtype.value not in exclude_qtypes]
        diag, ctrl = split_questions(qs_all, base.coords, src.coords)
        if not diag:
            continue
        # spread diagnostic questions across question types
        by_t = {}
        for q in rng.permutation(len(diag)):
            by_t.setdefault(diag[q].qtype, []).append(diag[q])
        chosen = []
        while len(chosen) < max_diag and any(by_t.values()):
            for t in list(by_t):
                if by_t[t] and len(chosen) < max_diag:
                    chosen.append(by_t[t].pop())
        chosen += [ctrl[i] for i in rng.permutation(len(ctrl))[:max_ctrl]]
        pairs.append((base, src, leaf, chosen))
    return pairs
