"""Connect the map to the answers (docs/design_map_connect.md): data splits, a small LoRA, and a general-knowledge
check. Development phase only; nothing here touches the confirmation-only sets."""

from __future__ import annotations

import functools
import math
import zlib

import numpy as np
import torch

from geomap.axis import CELLS, axis_cell
from geomap.composition import _chain
from geomap.flow import axis_of, steps
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import QType, all_questions, answer_for
from geomap.scenes import generate_scene

SEED_TRAIN, SEED_DEV, SEED_DEV_NAMES = 3101, 3201, 3301      # disjoint from every earlier study (≤ 1701)
TRAIN_KINDS = ("C-R", "C-M", "M-R")
DEV_TRANSFER = (QType.FARTHEST, QType.CLOSEST)


# ---- data ----

def name_pools(tok, held_out: bool) -> list[list[str]]:
    """People and object names (single-token), split by alternate index: training names (even) and held-out names
    (odd), so the dev 'new names' set uses names never seen in training."""
    return [single_token_names(tok, NAME_POOLS[k])[int(held_out)::2] for k in ("people", "objects")]


def pair_kind(s, mid: str, child: str, q) -> str | None:
    """C-R (composed), C-M and M-R (one stated step) for farther-east / north questions; None otherwise."""
    if q.qtype not in (QType.MORE_EAST, QType.MORE_NORTH):
        return None
    return {frozenset({child, s.root}): "C-R", frozenset({child, mid}): "C-M",
            frozenset({mid, s.root}): "M-R"}.get(frozenset(q.args))


def make_set(tok, seed: int, cr_sizes: dict, one_step_per_scene: int = 2, transfer_per_scene: int = 0,
             held_out_names: bool = False) -> list[dict]:
    """3-object chains R → M → C. Per scene: C–R axis questions (until each cell holds cr_sizes[cell]), up to
    one_step_per_scene one-step comparisons (C–M, M–R), and up to transfer_per_scene farthest / closest questions.
    Rows carry the scene and question objects (for prompts) and their labels (from geometry)."""
    rng = np.random.default_rng(seed)
    pools = name_pools(tok, held_out_names)
    left, rows, k, si = dict(cr_sizes), [], 0, -1
    while any(v > 0 for v in left.values()):
        s = generate_scene(rng, pools[k % 2], 3, p_shuffle=0.0)
        k += 1
        mc = _chain(s)
        if mc is None:
            continue
        mid, child = mc
        qs = all_questions(s)
        keep = []
        for q in qs:
            c = axis_cell(s, mid, child, q)
            if c is not None and left.get(c, 0) > 0:
                keep.append((q, "C-R", c))
                left[c] -= 1
        if not keep:
            continue
        one = [q for q in qs if pair_kind(s, mid, child, q) in ("C-M", "M-R")]
        for i in rng.permutation(len(one))[:one_step_per_scene]:
            keep.append((one[i], pair_kind(s, mid, child, one[i]), ""))
        tr = [q for q in qs if q.qtype in DEV_TRANSFER]
        for i in rng.permutation(len(tr))[:transfer_per_scene]:
            keep.append((tr[i], tr[i].qtype.value, ""))
        si += 1
        for q, kind, cell in keep:
            r = dict(scene=si, scene_seed=s.seed, kind=kind, cell=cell, question=q.text, answer=answer_for(q, s.coords),
                     child=child, root=s.root, mid=mid, s=s, q=q)
            if kind == "C-R":
                d1, d2, truth = steps(s, mid, child, q)
                r.update(axis=axis_of(q), d1=d1, d2=d2, truth=truth)
            rows.append(r)
    return rows


def training_examples(tok, family: str, rows) -> list[tuple[list[int], int]]:
    """(prompt ids, answer's first token) per row, chat format."""
    from geomap.scoring import candidate_ids
    out = []
    for r in rows:
        sp = build_prompt(r["s"], r["q"], tok, family)
        out.append((sp.input_ids, candidate_ids(tok, r["answer"], sp.lead)[0]))
    return out


# ---- LoRA ----

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(torch.nn.Module):
    """base(x) + (α/r) · B A x, with A, B trainable (fp32) and base frozen. B starts at 0, so training starts from the
    unchanged model."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float, seed: int):
        super().__init__()
        self.base, self.scale = base, alpha / r
        g = torch.Generator().manual_seed(seed)
        dev = base.weight.device
        self.A = torch.nn.Parameter((torch.randn(r, base.in_features, generator=g) / math.sqrt(base.in_features)).to(dev))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev))

    def forward(self, x):
        return self.base(x) + (self.scale * ((x.float() @ self.A.T) @ self.B.T)).to(x.dtype)


def _decoder(hf, layers_path: str):
    return functools.reduce(getattr, layers_path.split("."), hf)


def add_lora(hf, layers_path: str, from_layer: int, r: int = 16, alpha: float = 32.0, targets=TARGETS,
             seed: int = 0) -> dict[str, LoRALinear]:
    """Freeze the model and wrap the target linears of decoder blocks with index ≥ from_layer (0-based; block l
    writes the residual stream at layer l + 1) in LoRA. Returns {qualified name: module}."""
    for p in hf.parameters():
        p.requires_grad_(False)
    dec, out = _decoder(hf, layers_path), {}
    for l in range(from_layer, len(dec)):
        for name, mod in list(dec[l].named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in targets and isinstance(mod, torch.nn.Linear):
                parent = dec[l].get_submodule(name.rsplit(".", 1)[0]) if "." in name else dec[l]
                lora = LoRALinear(mod, r, alpha, seed=zlib.crc32(f"{seed}|{l}|{name}".encode()))
                setattr(parent, leaf, lora)
                out[f"{l}.{name}"] = lora
    return out


def lora_state(loras: dict) -> dict:
    return {k: (v.A.detach().cpu(), v.B.detach().cpu(), v.scale) for k, v in loras.items()}


def merge_lora(hf, layers_path: str, state: dict) -> None:
    """Fold a saved adapter into the base weights in place (W += scale · B A); the model keeps plain Linear layers,
    so hooks and nnsight paths are unchanged."""
    dec = _decoder(hf, layers_path)
    with torch.no_grad():
        for key, (A, B, scale) in state.items():
            l, name = key.split(".", 1)
            mod = dec[int(l)].get_submodule(name)
            if isinstance(mod, LoRALinear):
                raise ValueError("merge into a model without live LoRA wrappers")
            W = mod.weight
            W += (scale * (B.float() @ A.float())).to(W.device, W.dtype)


def attach_lora(hf, layers_path: str, state: dict) -> None:
    """Wrap exactly the linears named in a saved adapter and load its weights, for models whose weights cannot absorb
    the adapter (8-bit). The model is frozen; outputs equal the trained model's."""
    for p in hf.parameters():
        p.requires_grad_(False)
    dec = _decoder(hf, layers_path)
    for key, (A, B, scale) in state.items():
        l, name = key.split(".", 1)
        blk = dec[int(l)]
        mod = blk.get_submodule(name)
        parent = blk.get_submodule(name.rsplit(".", 1)[0]) if "." in name else blk
        w = LoRALinear(mod, A.shape[0], scale * A.shape[0], seed=0)
        with torch.no_grad():
            w.A.copy_(A.to(w.A.device))
            w.B.copy_(B.to(w.B.device))
        w.A.requires_grad_(False)
        w.B.requires_grad_(False)
        setattr(parent, name.rsplit(".", 1)[-1], w)


def unwrap_lora(hf, layers_path: str) -> None:
    """Replace live LoRA wrappers by their (unchanged) base linears."""
    dec = _decoder(hf, layers_path)
    for blk in dec:
        for name, mod in list(blk.named_modules()):
            if isinstance(mod, LoRALinear):
                parent = blk.get_submodule(name.rsplit(".", 1)[0]) if "." in name else blk
                setattr(parent, name.rsplit(".", 1)[-1], mod.base)


def last_token_logits(hf, batch: list[list[int]], pad: int) -> torch.Tensor:
    """Logits at each sequence's last token (right-padded batch)."""
    B, T = len(batch), max(map(len, batch))
    ids = torch.full((B, T), pad, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    for j, x in enumerate(batch):
        ids[j, :len(x)] = torch.tensor(x)
        am[j, :len(x)] = 1
    dev = next(hf.parameters()).device
    out = hf(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits
    last = torch.tensor([len(x) - 1 for x in batch], device=out.device)
    return out[torch.arange(B, device=out.device), last]


# ---- general check (damage) ----

_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Germany": "Berlin", "Spain": "Madrid",
             "Egypt": "Cairo", "Russia": "Moscow", "China": "Beijing", "Greece": "Athens", "Ireland": "Dublin",
             "Norway": "Oslo", "Austria": "Vienna", "Portugal": "Lisbon", "Poland": "Warsaw", "Kenya": "Nairobi",
             "Peru": "Lima", "Cuba": "Havana", "Thailand": "Bangkok", "Canada": "Ottawa", "Sweden": "Stockholm"}


def general_items() -> list[tuple[str, str]]:
    """Non-spatial one-word questions with unambiguous answers: single-digit sums, the next number, the next letter,
    and well-known capitals."""
    out = [(f"What is {a} plus {b}?", str(a + b)) for a in range(1, 9) for b in range(1, 9) if a + b < 10]
    out += [(f"Which number comes right after {n}?", str(n + 1)) for n in range(1, 9)]
    out += [(f"Which letter comes right after {c} in the alphabet?", chr(ord(c) + 1)) for c in "BCDFGHKLMPRST"]
    out += [(f"What is the capital of {k}?", v) for k, v in _CAPITALS.items()]
    return out


def general_prompts(tok, family: str) -> list[tuple[list[int], int]]:
    from geomap.prompts import ANSWER_SUFFIX, render_chat
    out = []
    for q, a in general_items():
        ids = tok(render_chat(tok, family, q + ANSWER_SUFFIX), add_special_tokens=False).input_ids
        out.append((ids, tok.encode(a, add_special_tokens=False)[0]))
    return out


@torch.no_grad()
def first_token_accuracy(hf, examples, pad: int, batch_size: int = 16) -> float:
    hit = []
    for b0 in range(0, len(examples), batch_size):
        ex = examples[b0:b0 + batch_size]
        lg = last_token_logits(hf, [x for x, _ in ex], pad)
        hit += (lg.argmax(-1).cpu() == torch.tensor([t for _, t in ex])).tolist()
    return float(np.mean(hit))
