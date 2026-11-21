"""Blind reply (docs/preregistration_blind_reply.md): greedy generation in which chosen prompt spans are hidden (their
keys masked in every layer) from the prompt positions, from the reply tokens, or from both. Positions are always
numbered as in the unmasked prompt (explicit position ids), so hiding a span does not shift later positions."""

from __future__ import annotations

import math

import torch

from geomap.cot_axis import cot_axis_prompt
from geomap.prompts import build_prompt


def blind_spans(s, q, mid: str, child: str, tok, family: str):
    """Step-by-step prompt ids, M's sentence span and the object-list span (first token after C's sentence to the end
    of the list), as [start, end) token ranges."""
    ids = tok(cot_axis_prompt(s, q, tok, family), add_special_tokens=False).input_ids
    bp = build_prompt(s, q, tok, family)
    end = bp.eol_pos + 1
    if bp.input_ids[:end] != ids[:end]:
        raise ValueError("step-by-step prompt does not share the one-word prompt's tokens up to the list")
    return ids, tuple(bp.relation_spans[mid]), (bp.relation_spans[child][1], end)


def _mask(am: torch.Tensor, offsets: list[int], spans: list[list[tuple[int, int]]]) -> torch.Tensor:
    m = am.clone()
    for b, (off, ss) in enumerate(zip(offsets, spans)):
        for a, e in ss:
            m[b, off + a:off + e] = 0
    return m


def greedy_masked(m, seqs: list[list[int]], prompt_block, reply_block, max_new_tokens: int,
                  after_prefill=None) -> list[list[int]]:
    """Greedy replies. prompt_block[b] / reply_block[b]: spans (in unpadded prompt coordinates) whose keys are hidden
    from the prompt positions / from the reply tokens. after_prefill(cache, offsets) is a test hook."""
    tok = m.tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = m.hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | ({tok.eos_token_id} if tok.eos_token_id else set())
    B, T = len(seqs), max(map(len, seqs))
    ids = torch.full((B, T), pad, dtype=torch.long)
    am = torch.zeros_like(ids)
    for j, x in enumerate(seqs):
        ids[j, T - len(x):] = torch.tensor(x)
        am[j, T - len(x):] = 1
    offs = [T - len(x) for x in seqs]
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    dev = m.hf.device
    with torch.no_grad():
        out = m.hf(input_ids=ids.to(dev), attention_mask=_mask(am, offs, prompt_block).to(dev),
                   position_ids=pos.to(dev), use_cache=True)
        cache = out.past_key_values
        if after_prefill is not None:
            after_prefill(cache, offs)
        gmask = _mask(am, offs, reply_block).to(dev)
        nxt = out.logits[:, -1].argmax(-1)
        last = pos[:, -1].to(dev)
        replies, done = [[] for _ in range(B)], [False] * B
        for t in range(max_new_tokens):
            for b in range(B):
                if not done[b]:
                    v = int(nxt[b])
                    if v in eos:
                        done[b] = True
                    else:
                        replies[b].append(v)
            if all(done) or t == max_new_tokens - 1:
                break
            gmask = torch.cat([gmask, torch.ones(B, 1, dtype=gmask.dtype, device=dev)], dim=1)
            out = m.hf(input_ids=nxt[:, None], attention_mask=gmask, position_ids=(last + 1 + t)[:, None],
                       past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
    return replies


def recovery(blind: float, floor: float, normal: float) -> float:
    """Share of the gap between the floor (no access to M's step) and normal recovered with the reply blind to M's
    sentence."""
    return (blind - floor) / (normal - floor) if normal != floor else float("nan")


def verdict_blind(rec: tuple, normal: tuple, floor: tuple, blind: tuple, control: float) -> str:
    """rec, normal, floor, blind: (estimate, lo, hi); control: the list control's balanced accuracy."""
    vals = [*rec, *normal, *floor, *blind, control]
    if any(isinstance(x, float) and math.isnan(x) for x in vals):
        return "uninformative"
    if normal[0] < 0.75 or not floor[2] < normal[1] or control < normal[0] - 0.15:
        return "uninformative"
    if rec[1] >= 0.50 and blind[1] > 0.5:
        return "the processed state is usable"
    if rec[2] <= 0.20:
        return "the processed state is not usable by the reply"
    return "partial"
