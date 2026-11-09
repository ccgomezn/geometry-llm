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

Site = Literal["list", "relation", "eol", "all"]


def site_positions(spec: PromptSpec, site: Site, leaf: str) -> list[int]:
    if site == "list":
        return [spec.list_pos[leaf]]
    if site == "relation":
        return list(range(*spec.relation_spans[leaf]))
    if site == "eol":
        return [spec.eol_pos]
    if site == "all":
        return list(range(spec.prefix_len))
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
    """Per (site, layer): flip rate on diagnostic questions, control stability, joint consistency per pair."""
    d = df_all[~df_all.site.isin(["none", "source"])].copy()
    d["follows_src"] = d.model_ans == d.src_ans
    d["stays_base"] = d.model_ans == d.base_ans
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
        for qt, gq in diag.groupby("qtype"):
            row[f"flip_{qt}"] = float(gq.follows_src.mean())
        out.append(row)
    return pd.DataFrame(out)


def make_pairs(m, n, seed, max_diag=6, max_ctrl=4):
    """Aligned (base, source, leaf, questions) pairs; diagnostic questions spread across types, plus controls."""
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    pairs = []
    while len(pairs) < n:
        s = generate_scene(rng, pools[len(pairs) % 2], 4 + len(pairs) % 2)
        try:
            base, src, leaf = make_pair_aligned(s, rng, m.tok, m.family)
        except ValueError:
            continue
        diag, ctrl = split_questions(all_questions(base), base.coords, src.coords)
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
