"""Where does written reasoning get M's step (docs/preregistration_readback.md)? Prompts whose M sentence comes from one
version of the scene and whose downstream part (every position after M's sentence) comes from the other: the output
of every decoder layer at the downstream positions is overwritten, during the prompt pass, by the other version's.

Both the recording pass and the patched pass go through `generate`, so that positions are numbered the same way."""

from __future__ import annotations

import functools
from contextlib import contextmanager

import torch

from geomap.cot_axis import cot_axis_prompt
from geomap.prompts import build_prompt


def mix_spec(s, sf, q, mid: str, tok, family: str):
    """Token ids of the step-by-step prompt for the original scene s and the flipped scene sf, and M's sentence span
    [a, b). None unless the two prompts have the same length and differ only inside M's sentence."""
    ids_o = tok(cot_axis_prompt(s, q, tok, family), add_special_tokens=False).input_ids
    ids_f = tok(cot_axis_prompt(sf, q, tok, family), add_special_tokens=False).input_ids
    if len(ids_o) != len(ids_f):
        return None
    bo, bf = build_prompt(s, q, tok, family), build_prompt(sf, q, tok, family)
    a, b = bo.relation_spans[mid]
    if bf.relation_spans != bo.relation_spans or bo.input_ids[:b] != ids_o[:b] or bf.input_ids[:b] != ids_f[:b]:
        return None
    diff = [i for i, (x, y) in enumerate(zip(ids_o, ids_f)) if x != y]
    if not diff or not all(a <= i < b for i in diff):
        return None
    return ids_o, ids_f, a, b


def pad_left(seqs: list[list[int]], tok, starts: list[int]):
    """Left-padded ids, attention mask, and each sequence's downstream start in padded coordinates."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    am = torch.zeros_like(ids)
    for j, x in enumerate(seqs):
        ids[j, T - len(x):] = torch.tensor(x)
        am[j, T - len(x):] = 1
    return ids, am, [T - len(x) + s for x, s in zip(seqs, starts)]


def _layers(m):
    return functools.reduce(getattr, m.layers_path.split("."), m.hf)


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _gen(m, ids, am, **kw):
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    with torch.no_grad():
        return m.hf.generate(input_ids=ids.to(m.hf.device), attention_mask=am.to(m.hf.device), do_sample=False,
                             pad_token_id=pad, **kw)


def record_layers(m, ids, am) -> list[torch.Tensor]:
    """Output of every decoder layer over the whole prompt ([B, T, d] per layer), from the prompt pass of generate."""
    T, rec, hooks = ids.shape[1], {}, []
    for li, layer in enumerate(_layers(m)):
        def hook(mod, inp, out, li=li):
            h = _hidden(out)
            if h.shape[1] == T:
                rec[li] = h.detach().clone()
        hooks.append(layer.register_forward_hook(hook))
    try:
        _gen(m, ids, am, max_new_tokens=1)
    finally:
        for h in hooks:
            h.remove()
    return [rec[i] for i in range(len(rec))]


@contextmanager
def patched(m, rec: list[torch.Tensor], starts: list[int], T: int):
    """While active, the prompt pass (sequence length T) gets rec's layer outputs at positions >= each start."""
    hooks = []
    for li, layer in enumerate(_layers(m)):
        def hook(mod, inp, out, li=li):
            h = _hidden(out)
            if h.shape[1] == T:
                for b, st in enumerate(starts):
                    h[b, st:] = rec[li][b, st:].to(h.device, h.dtype)
        hooks.append(layer.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def last_logits(m, ids, am) -> torch.Tensor:
    """Logits of the next token after the prompt, from generate's prompt pass."""
    out = _gen(m, ids, am, max_new_tokens=1, output_logits=True, return_dict_in_generate=True)
    return out.logits[0].float().cpu()


def readback_index(f_text: float, acc: float) -> float:
    """1 if the mixed replies follow M's sentence as reliably as the model answers at all, 0 if they follow the
    downstream state as reliably."""
    return (f_text - (1 - acc)) / (2 * acc - 1)


def verdict_readback(index: tuple, acc: float, acc_original: float, acc_flipped: float) -> str:
    import math
    if any(math.isnan(x) for x in (acc, acc_original, acc_flipped)) or acc < 0.75 \
            or min(acc_original, acc_flipped) < 0.70:
        return "uninformative"
    if index[1] >= 0.70:
        return "the reasoning re-reads the sentence"
    if index[2] <= 0.30:
        return "the reasoning reads the processed state"
    return "both"
