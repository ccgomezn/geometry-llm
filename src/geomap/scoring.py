"""Score closed-set answers by the summed log-probability of each candidate continuation."""

from __future__ import annotations

import re
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
    """One row per distinct answer prefix (a trie over candidate tokens): the row's last position predicts
    the next token of every candidate sharing that prefix. Single-token answers need one row per question."""
    rows, cand_tok = [], []  # rows: (item, prefix, ids, next tokens needed)
    for i, (spec, q) in enumerate(items):
        ct = {c: candidate_ids(m.tok, c) for c in q.candidates}
        cand_tok.append(ct)
        need: dict[tuple, set] = {}
        for toks in ct.values():
            for k in range(len(toks)):
                need.setdefault(tuple(toks[:k]), set()).add(toks[k])
        for pre, nxt in need.items():
            rows.append((i, pre, spec.input_ids + list(pre), sorted(nxt)))
    return rows, cand_tok


def score(m: LoadedModel, items: Sequence[tuple[PromptSpec, Question]], edit: Edit | None = None,
          batch_size: int = 16) -> list[dict[str, float]]:
    """Summed log-probability of each candidate. Rows are right-padded so prompt positions align for edits."""
    rows, cand_tok = _rows(m, items)
    next_lp: list[dict[tuple, dict[int, float]]] = [dict() for _ in items]
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    for b0 in range(0, len(rows), batch_size):
        batch = rows[b0:b0 + batch_size]
        B, T = len(batch), max(len(r[2]) for r in batch)
        K = max(len(r[3]) for r in batch)
        ids = torch.full((B, T), pad, dtype=torch.long)
        am = torch.zeros((B, T), dtype=torch.long)
        last = torch.zeros(B, dtype=torch.long)
        C = torch.zeros((B, K), dtype=torch.long)
        for j, (_, _, r, nxt) in enumerate(batch):
            ids[j, :len(r)] = torch.tensor(r)
            am[j, :len(r)] = 1
            last[j] = len(r) - 1
            C[j, :len(nxt)] = torch.tensor(nxt)
        row_items = [r[0] for r in batch]
        with torch.no_grad(), m.lm.trace({"input_ids": ids, "attention_mask": am}):
            if edit is not None:
                edit(m, row_items)
            logits = m.lm.output.logits
            sel = logits[torch.arange(B, device=logits.device), last.to(logits.device)]
            lp = sel.float().log_softmax(-1).gather(-1, C.to(logits.device)).cpu().save()
        for j, (i, pre, _, nxt) in enumerate(batch):
            next_lp[i][pre] = {t: float(lp[j, k]) for k, t in enumerate(nxt)}
    out = []
    for i, ct in enumerate(cand_tok):
        out.append({c: sum(next_lp[i][tuple(toks[:k])][toks[k]] for k in range(len(toks))) for c, toks in ct.items()})
    return out


def predict(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def parse_cot_answer(text: str, candidates) -> str | None:
    """Final answer after </think>: the candidate whose last whole-word match ends latest (longer wins ties)."""
    if "</think>" not in text:
        return None
    tail = text.split("</think>")[-1].lower()
    best, best_key = None, (-1, 0)
    for c in candidates:
        for mt in re.finditer(r"\b" + re.escape(c.lower()) + r"\b", tail):
            if (mt.end(), len(c)) > best_key:
                best, best_key = c, (mt.end(), len(c))
    return best
