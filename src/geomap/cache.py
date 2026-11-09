"""Cache residual-stream activations at the readout positions, from the shared prompt prefix only."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from geomap.models import LoadedModel, layer_out
from geomap.prompts import PromptSpec

MAX_OBJS = 6


def cache_prefix(m: LoadedModel, specs: Sequence[PromptSpec], layers: Sequence[int],
                 batch_size: int = 16) -> dict[str, torch.Tensor]:
    """Returns {"list": [N, L, 6, d] bfloat16 (NaN-padded, slots in object-list order), "eol": [N, L, d]}.

    bfloat16, not float16: Gemma 3 residual-stream values exceed float16's range. Raises if any cached value
    is not finite."""
    layers = sorted(layers)
    N, L, d = len(specs), len(layers), m.d_model
    out_list = torch.full((N, L, MAX_OBJS, d), float("nan"), dtype=torch.bfloat16)
    out_eol = torch.empty((N, L, d), dtype=torch.bfloat16)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    for b0 in range(0, N, batch_size):
        batch = specs[b0:b0 + batch_size]
        B = len(batch)
        T = max(len(s.input_ids) for s in batch)
        ids = torch.full((B, T), pad, dtype=torch.long)
        am = torch.zeros((B, T), dtype=torch.long)
        pos = torch.zeros((B, MAX_OBJS), dtype=torch.long)
        n_obj = []
        for j, s in enumerate(batch):
            ids[j, :len(s.input_ids)] = torch.tensor(s.input_ids)
            am[j, :len(s.input_ids)] = 1
            p = list(s.list_pos.values())
            pos[j, :len(p)] = torch.tensor(p)
            n_obj.append(len(p))
        eol = torch.tensor([s.eol_pos for s in batch])
        rows = torch.arange(B)
        saved = []
        with torch.no_grad(), m.lm.trace({"input_ids": ids, "attention_mask": am}):
            for l in layers:
                h = layer_out(m, l)
                dev = h.device
                saved.append((h[rows.to(dev)[:, None], pos.to(dev)].to(torch.bfloat16).cpu().save(),
                              h[rows.to(dev), eol.to(dev)].to(torch.bfloat16).cpu().save()))
        for k, (hl, he) in enumerate(saved):
            if not (torch.isfinite(hl).all() and torch.isfinite(he).all()):
                raise ValueError(f"non-finite activations at layer {layers[k]}")
            for j in range(B):
                out_list[b0 + j, k, :n_obj[j]] = hl[j, :n_obj[j]]
            out_eol[b0:b0 + B, k] = he
    return {"list": out_list, "eol": out_eol}


def save_cache(path, cache: dict, scenes, layers: Sequence[int]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache": cache, "scenes": list(scenes), "layers": list(layers)}, path)


def load_cache(path):
    obj = torch.load(path, weights_only=False)
    return obj["cache"], obj["scenes"], obj["layers"]
