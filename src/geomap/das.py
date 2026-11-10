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
from geomap.scoring import candidate_sequences, predict, score


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
    """One row per (pair, question, candidate token sequence), carrying the source activations to swap in.

    Each row also records the model's own answers on the clean base and clean source prompts: the target of an
    interchange intervention is what the model says when the object really is elsewhere (ceiling = 1)."""
    rows = []
    for p, (base, src, leaf, qs) in enumerate(pairs):
        bpre = build_prompt(base, None, m.tok, m.family)
        pos = site_positions(bpre, site, leaf)
        src_h = prefix_acts(m, build_prompt(src, None, m.tok, m.family), [layer])[layer][pos].float()
        bspecs = [build_prompt(base, q, m.tok, m.family) for q in qs]
        clean = score(m, [(sp, q) for sp, q in zip(bspecs, qs)] +
                      [(build_prompt(src, q, m.tok, m.family), q) for q in qs])
        for qi, (q, spec) in enumerate(zip(qs, bspecs)):
            sa, ba = answer_for(q, src.coords), answer_for(q, base.coords)
            mb, ms = predict(clean[qi]), predict(clean[len(qs) + qi])
            for c, seqs in candidate_sequences(m.tok, m.family, q.candidates).items():
                for t in seqs:
                    rows.append(dict(pair=p, qid=qi, qtype=q.qtype.value, cand=c, src_ans=sa, base_ans=ba,
                                     model_base=mb, model_src=ms, ids=spec.input_ids + t, n=len(spec.input_ids),
                                     pos=pos, src_h=src_h))
    return rows


def question_scores(rows: list[dict], lp_q: torch.Tensor, idx: list[int]) -> tuple[list[str], torch.Tensor]:
    """Candidate log-probabilities for one question (log-sum-exp over each candidate's sequences).

    `idx` are the question's row indices into `rows`; `lp_q[k]` is the log-probability of row idx[k]."""
    assert len(lp_q) == len(idx)
    cands = list(dict.fromkeys(rows[i]["cand"] for i in idx))
    vals = [torch.logsumexp(torch.stack([lp_q[k] for k, i in enumerate(idx) if rows[i]["cand"] == c]), 0)
            for c in cands]
    return cands, torch.stack(vals)


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
            ivdev = iv.mask.device
            h[j, pos] = iv(h[j, pos].float().to(ivdev), r["src_h"].to(ivdev)).to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    if iv is not None:
        iv.to(dev)
    handle = _module(m, layer).register_forward_hook(hook) if iv is not None else None
    try:
        # Run the decoder, then apply the output head only at the scored positions: full-vocabulary logits at
        # every position of 100+ candidate rows do not fit. The index is built on the hidden states' device
        # (with device_map="auto" that is the last GPU, not the first).
        hs = m.hf.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
        logits = m.hf.lm_head(hs[:, torch.tensor(need, device=hs.device)])
        cfg = getattr(m.hf.config, "text_config", m.hf.config)
        cap = getattr(cfg, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
    finally:
        if handle is not None:
            handle.remove()
    lp = logits.float().log_softmax(-1)
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
    """Cross-entropy over each question's candidates toward the model's own clean-source answer; frozen model."""
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
            # one question per forward/backward, gradients accumulated: activations kept for backward through
            # the upper layers of a 12B model fit for one question's ~30 candidate rows, not for 8 questions'
            opt.zero_grad()
            loss_sum = 0.0
            for g in gb:
                lp = row_logprobs(m, [rows[i] for i in g], iv, layer, grad=True, batch_size=len(g))
                cands, cl = question_scores(rows, lp, g)
                target = cands.index(rows[g[0]]["model_src"])
                loss = nn.functional.cross_entropy(cl[None], torch.tensor([target], device=cl.device)) / len(gb)
                loss.backward()
                loss_sum += float(loss)
            opt.step()
            losses.append(loss_sum)
        iv.history.append(float(np.mean(losses)))
    return iv


def eval_das(m: LoadedModel, iv: SubspaceIntervention, test_pairs, layer: int, site: Site,
             n_boot: int = 1000) -> pd.DataFrame:
    """Interchange accuracy: on questions where the model's clean base and clean source answers differ, the share
    whose answer after swapping the subspace equals the clean-source answer (ceiling 1). Also reports the
    geometric version (source-map answers on geometrically diagnostic questions) and control stability."""
    rows = pair_rows(m, test_pairs, layer, site)
    lp = row_logprobs(m, rows, iv, layer).cpu()
    recs = []
    for g in _groups(rows):
        cands, cl = question_scores(rows, lp[g], g)
        pred = cands[int(cl.argmax())]
        r = rows[g[0]]
        recs.append(dict(pair=r["pair"], qtype=r["qtype"], correct=float(pred == r["model_src"]),
                         diagnostic=r["model_src"] != r["model_base"],
                         correct_geom=float(pred == r["src_ans"]), diagnostic_geom=r["src_ans"] != r["base_ans"]))
    df = pd.DataFrame(recs)
    out = []
    for qt, d in list(df.groupby("qtype")) + [("ALL", df)]:
        dd = d[d.diagnostic]
        if len(dd):
            est, lo, hi = bootstrap_ci(dd.correct, groups=dd.pair, n_boot=n_boot)
        else:
            est = lo = hi = np.nan
        cc, dg = d[~d.diagnostic], d[d.diagnostic_geom]
        out.append(dict(qtype=qt, acc=est, acc_lo=lo, acc_hi=hi, n=len(dd),
                        acc_geom=dg.correct_geom.mean() if len(dg) else np.nan,
                        control_stability=cc.correct.mean() if len(cc) else np.nan))
    return pd.DataFrame(out)
