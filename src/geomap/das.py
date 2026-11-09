"""Distributed Alignment Search: learn a k-dim subspace whose swap moves an object's position."""

from __future__ import annotations

import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

from geomap.analysis import bootstrap_ci
from geomap.models import LoadedModel
from geomap.patching import Site, prefix_acts, site_positions
from geomap.prompts import build_prompt
from geomap.questions import answer_for
from geomap.scoring import candidate_ids


class SubspaceIntervention(nn.Module):
    """h' = h_base + Q Qᵀ (h_src − h_base), with Q (d × k) kept orthonormal."""

    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.lin = nn.utils.parametrizations.orthogonal(nn.Linear(d_model, k, bias=False))
        self.register_buffer("mask", torch.ones(k))
        self.history: list[float] = []

    @property
    def Q(self) -> torch.Tensor:
        return self.lin.weight.T * self.mask

    def forward(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        Q = self.Q
        return h_base + (h_src - h_base) @ Q @ Q.T


def _module(m: LoadedModel, layer: int):
    if layer == 0:
        return functools.reduce(getattr, m.embed_path.split("."), m.hf)
    return functools.reduce(getattr, m.layers_path.split("."), m.hf)[layer - 1]


def pair_rows(m: LoadedModel, pairs, layer: int, site: Site) -> list[dict]:
    """One row per (pair, question, candidate), carrying the source activations to swap in."""
    rows = []
    for p, (base, src, leaf, qs) in enumerate(pairs):
        bpre = build_prompt(base, None, m.tok, m.family)
        pos = site_positions(bpre, site, leaf)
        src_h = prefix_acts(m, build_prompt(src, None, m.tok, m.family), [layer])[layer][pos].float()
        for qi, q in enumerate(qs):
            spec = build_prompt(base, q, m.tok, m.family)
            sa, ba = answer_for(q, src.coords), answer_for(q, base.coords)
            for c in q.candidates:
                cid = candidate_ids(m.tok, c)
                rows.append(dict(pair=p, qid=qi, qtype=q.qtype.value, cand=c, src_ans=sa, base_ans=ba,
                                 ids=spec.input_ids + cid, n=len(spec.input_ids), pos=pos, src_h=src_h))
    return rows


def _forward(m: LoadedModel, batch: list[dict], iv: nn.Module | None, layer: int) -> torch.Tensor:
    dev = next(m.hf.parameters()).device
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    B, T = len(batch), max(len(r["ids"]) for r in batch)
    ids = torch.full((B, T), pad, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    for j, r in enumerate(batch):
        ids[j, :len(r["ids"])] = torch.tensor(r["ids"])
        am[j, :len(r["ids"])] = 1
    need = sorted({p for r in batch for p in range(r["n"] - 1, len(r["ids"]) - 1)})
    where = {p: i for i, p in enumerate(need)}

    def hook(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).clone()
        for j, r in enumerate(batch):
            pos = torch.tensor(r["pos"], device=h.device)
            h[j, pos] = iv(h[j, pos].float(), r["src_h"].to(h.device)).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    if iv is not None:
        iv.to(dev)
    handle = _module(m, layer).register_forward_hook(hook) if iv is not None else None
    try:
        logits = m.hf(input_ids=ids.to(dev), attention_mask=am.to(dev),
                      logits_to_keep=torch.tensor(need, device=dev)).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    lp = logits.log_softmax(-1)
    out = []
    for j, r in enumerate(batch):
        toks = r["ids"][r["n"]:]
        out.append(sum(lp[j, where[r["n"] - 1 + t], tok] for t, tok in enumerate(toks)))
    return torch.stack(out)


def row_logprobs(m: LoadedModel, rows: list[dict], iv: nn.Module | None, layer: int, grad: bool = False,
                 batch_size: int = 16) -> torch.Tensor:
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for b in range(0, len(rows), batch_size):
            outs.append(_forward(m, rows[b:b + batch_size], iv, layer))
    return torch.cat(outs)


def _groups(rows):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[(r["pair"], r["qid"])].append(i)
    return list(g.values())


def train_das(m: LoadedModel, train_pairs, layer: int, site: Site = "list", k: int = 2, epochs: int = 5,
              lr: float = 1e-3, batch_size: int = 8, seed: int = 0) -> SubspaceIntervention:
    """Cross-entropy over each question's candidates toward the source-map answer; the model is frozen."""
    torch.manual_seed(seed)
    m.hf.requires_grad_(False)
    dev = next(m.hf.parameters()).device
    iv = SubspaceIntervention(m.d_model, k).to(dev)
    opt = torch.optim.Adam(iv.parameters(), lr=lr)
    rows = pair_rows(m, train_pairs, layer, site)
    groups = _groups(rows)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        losses = []
        perm = rng.permutation(len(groups))
        for b in range(0, len(groups), batch_size):
            gb = [groups[i] for i in perm[b:b + batch_size]]
            idx = [i for g in gb for i in g]
            lp = row_logprobs(m, [rows[i] for i in idx], iv, layer, grad=True, batch_size=len(idx))
            loss, off = 0.0, 0
            for g in gb:
                cand_lp = lp[off:off + len(g)]
                target = next(t for t, i in enumerate(g) if rows[i]["cand"] == rows[i]["src_ans"])
                loss = loss + nn.functional.cross_entropy(cand_lp[None], torch.tensor([target], device=lp.device))
                off += len(g)
            loss = loss / len(gb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        iv.history.append(float(np.mean(losses)))
    return iv


def eval_das(m: LoadedModel, iv: SubspaceIntervention, test_pairs, layer: int, site: Site,
             n_boot: int = 1000) -> pd.DataFrame:
    """Interchange accuracy on diagnostic questions (the source map changes the answer); also reports
    accuracy over all questions and stability on control questions."""
    rows = pair_rows(m, test_pairs, layer, site)
    lp = row_logprobs(m, rows, iv, layer).cpu().numpy()
    recs = []
    for g in _groups(rows):
        best = max(g, key=lambda i: lp[i])
        r = rows[g[0]]
        recs.append(dict(pair=r["pair"], qtype=r["qtype"], correct=float(rows[best]["cand"] == r["src_ans"]),
                         diagnostic=r["src_ans"] != r["base_ans"]))
    df = pd.DataFrame(recs)
    out = []
    for qt, d in list(df.groupby("qtype")) + [("ALL", df)]:
        dd = d[d.diagnostic]  # headline: only questions whose answer should change
        if len(dd):
            est, lo, hi = bootstrap_ci(dd.correct, groups=dd.pair, n_boot=n_boot)
        else:
            est = lo = hi = np.nan
        cc = d[~d.diagnostic]
        out.append(dict(qtype=qt, acc=est, acc_lo=lo, acc_hi=hi, n=len(dd),
                        acc_all=d.correct.mean(), control_stability=cc.correct.mean() if len(cc) else np.nan))
    return pd.DataFrame(out)
