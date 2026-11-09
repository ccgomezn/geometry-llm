"""Chat prompts with token positions for the per-object readout spots, end of list and relation spans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geomap.questions import Question, make_pair
from geomap.scenes import Scene, relation_text

# Single-token (with and without a leading space) in the Qwen3, Llama 3.2 and Gemma 3 tokenizers,
# with names that double as common English words (Will, Hope, Max, ...) removed.
NAME_POOLS: dict[str, list[str]] = {
    "people": ["Anna", "Ben", "Carl", "Emma", "Henry", "Jack", "Kate", "Leo", "Paul", "Sam", "Adam",
               "Chris", "David", "Fred", "George", "Jane", "Kevin", "Laura", "Peter", "Rachel", "Tom",
               "Alice", "Bob", "Dan", "Eric", "Greg", "Steve", "Amy", "Brian", "Kim", "Neil", "Rick",
               "Sean", "Alan", "Beth", "Cole", "Jake"],
    "objects": ["Chair", "Table", "Clock", "Bell", "Rock", "Tree", "Box", "Book", "Ring", "Coin", "Shell",
                "Stone", "Car", "Ball", "Hat", "Pen", "Bag", "Desk", "Door", "Glass", "Jar", "Knife", "Leaf",
                "Mask", "Pipe", "Wheel", "Brush", "Flag", "Mirror", "Plate", "Tower", "Basket", "Bucket",
                "Cart", "Gate", "Pot", "Anchor", "Arrow", "Banner"],
}

FAMILIES = {"qwen3", "gemma3", "llama3"}
_TEMPLATE_KWARGS = {"qwen3": {"enable_thinking": False}, "gemma3": {}, "llama3": {}}
ANSWER_SUFFIX = " Answer with one word."


@dataclass
class PromptSpec:
    input_ids: list[int]
    list_pos: dict[str, int]
    eol_pos: int
    relation_spans: dict[str, tuple[int, int]]  # subject name -> [start, end) token range
    prefix_len: int


def single_token_names(tok, pool: list[str]) -> list[str]:
    return [n for n in pool
            if len(tok.encode(" " + n, add_special_tokens=False)) == 1
            and len(tok.encode(n, add_special_tokens=False)) == 1]


def render_chat(tok, family: str, content: str, thinking: bool = False) -> str:
    kwargs = dict(_TEMPLATE_KWARGS[family])
    if family == "qwen3":
        kwargs["enable_thinking"] = thinking
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, **kwargs)


def build_prompt(scene: Scene, question: Question | None, tok, family: str, thinking: bool = False) -> PromptSpec:
    if family not in FAMILIES:
        raise ValueError(family)
    # content pieces, tracking character spans inside the content string
    rel_spans_chars: dict[str, tuple[int, int]] = {}
    content = ""
    for i, r in enumerate(scene.relations):
        if i:
            content += " "
        start = len(content)
        content += relation_text(r)
        rel_spans_chars[r.subj] = (start, len(content))
    content += "\n\nThe objects are: "
    name_chars: dict[str, int] = {}
    for i, n in enumerate(scene.names):
        if i:
            content += ", "
        name_chars[n] = len(content)
        content += n
    eol_char = len(content)
    content += "."
    qtext = question.text if question is not None else f"Which object is farthest from {scene.names[0]}?"
    content += "\n\n" + qtext + ANSWER_SUFFIX

    text = render_chat(tok, family, content, thinking)
    c0 = text.index(content)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    def tok_at(ch: int) -> int:
        ch += c0
        for i, (s, e) in enumerate(offs):
            if s <= ch < e:
                return i
        raise ValueError(f"no token covers char {ch}")

    list_pos = {n: tok_at(c) for n, c in name_chars.items()}
    eol_pos = tok_at(eol_char)
    spans = {n: (tok_at(a), tok_at(b - 1) + 1) for n, (a, b) in rel_spans_chars.items()}
    prefix_len = eol_pos + 1
    return PromptSpec(input_ids=list(ids) if question is not None else list(ids[:prefix_len]),
                      list_pos=list_pos, eol_pos=eol_pos, relation_spans=spans, prefix_len=prefix_len)


def aligned(base: PromptSpec, src: PromptSpec, leaf: str) -> bool:
    if len(base.input_ids) != len(src.input_ids) or base.relation_spans != src.relation_spans:
        return False
    lo, hi = base.relation_spans[leaf]
    return base.input_ids[:lo] == src.input_ids[:lo] and base.input_ids[hi:] == src.input_ids[hi:]


def make_pair_aligned(scene: Scene, rng: np.random.Generator, tok, family: str,
                      max_tries: int = 50) -> tuple[Scene, Scene, str]:
    for _ in range(max_tries):
        base, src, leaf = make_pair(scene, rng)
        if aligned(build_prompt(base, None, tok, family), build_prompt(src, None, tok, family), leaf):
            return base, src, leaf
    raise ValueError("could not build an aligned pair")
