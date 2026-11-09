"""Score closed-set answers by the summed log-probability of each candidate continuation."""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from geomap.models import LoadedModel
from geomap.prompts import PromptSpec
from geomap.questions import Question

Edit = Callable[[LoadedModel, list[int]], None]


def candidate_ids(tok, cand: str) -> list[int]:
    """Token ids of a candidate answer as it appears right after the assistant prefix."""
    return tok.encode(cand, add_special_tokens=False)


def _rows(m: LoadedModel, items: Sequence[tuple[PromptSpec, Question]]):
    rows = []  # (item index, candidate, ids, n_prompt)
    for i, (spec, q) in enumerate(items):
        for c in q.candidates:
            cid = candidate_ids(m.tok, c)
            rows.append((i, c, spec.input_ids + cid, len(spec.input_ids)))
    return rows


def score(m: LoadedModel, items: Sequence[tuple[PromptSpec, Question]], edit: Edit | None = None,
          batch_size: int = 16) -> list[dict[str, float]]:
    out: list[dict[str, float]] = [dict() for _ in items]
    rows = _rows(m, items)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    for b0 in range(0, len(rows), batch_size):
        batch = rows[b0:b0 + batch_size]
        T = max(len(r[2]) for r in batch)
        K = max(len(r[2]) - r[3] for r in batch)
        ids = torch.full((len(batch), T), pad, dtype=torch.long)
        am = torch.zeros((len(batch), T), dtype=torch.long)
        P = torch.zeros((len(batch), K), dtype=torch.long)  # positions whose logits predict each candidate token
        C = torch.zeros((len(batch), K), dtype=torch.long)
        M = torch.zeros((len(batch), K))
        for j, (_, _, r, n) in enumerate(batch):  # right padding keeps prompt positions aligned
            ids[j, :len(r)] = torch.tensor(r)
            am[j, :len(r)] = 1
            k = len(r) - n
            P[j, :k] = torch.arange(n - 1, n - 1 + k)
            C[j, :k] = torch.tensor(r[n:])
            M[j, :k] = 1
        row_items = [r[0] for r in batch]
        with torch.no_grad(), m.lm.trace({"input_ids": ids, "attention_mask": am}):
            if edit is not None:
                edit(m, row_items)
            logits = m.lm.output.logits
            sel = logits[torch.arange(len(batch), device=logits.device)[:, None], P.to(logits.device)]
            lp = sel.float().log_softmax(-1).gather(-1, C.to(logits.device)[..., None]).squeeze(-1)
            total = (lp * M.to(logits.device)).sum(-1).cpu().save()
        for j, (i, c, _, _) in enumerate(batch):
            out[i][c] = float(total[j])
    return out


def predict(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)
