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


_EOT = {"qwen3": "<|im_end|>", "llama3": "<|eot_id|>", "gemma3": "<end_of_turn>"}


def terminator_ids(tok, family: str) -> list[int]:
    """Tokens that can end a one-word answer: a newline or the chat end-of-turn token."""
    out = []
    nl = tok.encode("\n", add_special_tokens=False)
    if len(nl) == 1:
        out.append(nl[0])
    eot = tok.convert_tokens_to_ids(_EOT.get(family, ""))
    if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id:
        out.append(eot)
    return out


def candidate_sequences(tok, family: str, candidates: Sequence[str]) -> dict[str, list[list[int]]]:
    """Token sequences whose total probability is a candidate's score.

    Each candidate is scored as itself and, if lowercase, capitalised ("west" and "West"). A variant that is a
    token-prefix of another candidate's variant ("North" of "Northwest") must be followed by a terminator, so
    the shorter answer does not also count every longer answer that starts with it."""
    variants = {c: list(dict.fromkeys([c] + ([c[0].upper() + c[1:]] if c[:1].islower() else []))) for c in candidates}
    toks = {c: [candidate_ids(tok, v) for v in vs] for c, vs in variants.items()}
    every = [(c, t) for c, ts in toks.items() for t in ts]
    terms = terminator_ids(tok, family)
    out = {}
    for c, ts in toks.items():
        seqs = []
        for t in ts:
            is_prefix = any(len(u) > len(t) and u[:len(t)] == t for c2, u in every if c2 != c)
            seqs += [t + [x] for x in terms] if is_prefix else [t]
        out[c] = seqs
    return out


def _rows(m: LoadedModel, items: Sequence[tuple[PromptSpec, Question]]):
    """One row per distinct answer prefix (a trie over candidate token sequences): the row's last position
    predicts the next token of every sequence sharing that prefix."""
    rows, cand_seqs = [], []  # rows: (item, prefix, ids, next tokens needed)
    for i, (spec, q) in enumerate(items):
        cs = candidate_sequences(m.tok, m.family, q.candidates)
        cand_seqs.append(cs)
        need: dict[tuple, set] = {}
        for seqs in cs.values():
            for toks in seqs:
                for k in range(len(toks)):
                    need.setdefault(tuple(toks[:k]), set()).add(toks[k])
        for pre, nxt in need.items():
            rows.append((i, pre, spec.input_ids + list(pre), sorted(nxt)))
    return rows, cand_seqs


def score(m: LoadedModel, items: Sequence[tuple[PromptSpec, Question]], edit: Edit | None = None,
          batch_size: int = 16) -> list[dict[str, float]]:
    """Log-probability of each candidate (summed over tokens, log-sum-exp over its variants; see
    candidate_sequences). Rows are right-padded so prompt positions align for edits."""
    rows, cand_seqs = _rows(m, items)
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
    for i, cs in enumerate(cand_seqs):
        d = {}
        for c, seqs in cs.items():
            parts = [sum(next_lp[i][tuple(t[:k])][t[k]] for k in range(len(t))) for t in seqs]
            d[c] = float(torch.logsumexp(torch.tensor(parts, dtype=torch.float64), 0))
        out.append(d)
    return out


def candidate_mass(scores: dict[str, float]) -> float:
    """Log of the total probability the model puts on the candidate answers (a validity check)."""
    return float(torch.logsumexp(torch.tensor(list(scores.values()), dtype=torch.float64), 0))


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
