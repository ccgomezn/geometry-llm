"""Helpers for the interactive demo: move an object by editing the learned 2-D code, read the model's map back."""

from __future__ import annotations

import numpy as np
import torch

from geomap.das import _module
from geomap.models import LoadedModel
from geomap.prompts import PromptSpec, aligned, build_prompt
from geomap.questions import Question, QType, make_question
from geomap.scenes import Scene, move_leaf


def demo_questions(scene: Scene, leaf: str) -> list[Question]:
    """Six fixed questions about the moved leaf (its parent P and the third object O)."""
    p = scene.parent[leaf]
    o = next(n for n in scene.names if n not in (leaf, p))
    spec = [(QType.DIRECTION, (p, leaf)), (QType.DIRECTION, (o, leaf)), (QType.MORE_NORTH, (leaf, o)),
            (QType.MORE_EAST, (leaf, o)), (QType.CLOSEST, (leaf,)), (QType.FARTHEST, (o,))]
    return [make_question(t, a, scene.names) for t, a in spec]


def leaf_targets(m: LoadedModel, scene: Scene, leaf: str, max_n: int, rng: np.random.Generator,
                 span: int = 5) -> list[Scene]:
    """Scenes where only the leaf's sentence changes (token-aligned with the original), at distinct positions."""
    bp = build_prompt(scene, None, m.tok, m.family)
    out = []
    for dx in range(-span, span + 1):
        for dy in range(-span, span + 1):
            try:
                t = move_leaf(scene, leaf, dx, dy)
            except ValueError:
                continue
            if t.coords[leaf] == scene.coords[leaf]:
                continue
            if aligned(bp, build_prompt(t, None, m.tok, m.family), leaf):
                out.append(t)
    return [out[i] for i in sorted(rng.permutation(len(out))[:max_n])]


def readout(m: LoadedModel, spec: PromptSpec, iv, edit_layer: int, edit_pos, target_h, read_layer: int,
            read_pos) -> torch.Tensor:
    """Run the description (prefix) once; optionally set the subspace at `edit_pos` (layer `edit_layer`) toward
    `target_h` with intervention `iv`; return the residual stream at `read_pos` after layer `read_layer`."""
    dev = next(m.hf.parameters()).device
    ids = torch.tensor([spec.input_ids[:spec.prefix_len]], device=dev)
    got = {}

    def edit(mod, inp, out):
        h = (out[0] if isinstance(out, tuple) else out).clone()
        pos = torch.tensor(edit_pos, device=h.device)
        ivd = iv.mask.device
        h[0, pos] = iv(h[0, pos].float().to(ivd), target_h.float().to(ivd)).to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def read(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        got["h"] = h[0, torch.tensor(read_pos, device=h.device)].float().cpu()

    handles = []
    if iv is not None:
        iv.to(dev)
        handles.append(_module(m, edit_layer).register_forward_hook(edit))
    handles.append(_module(m, read_layer).register_forward_hook(read))
    try:
        with torch.no_grad():
            m.hf.model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        for h in handles:
            h.remove()
    return got["h"]


def anchored_positions(decoded: dict[str, np.ndarray], root: str, root_true) -> dict[str, tuple[float, float]]:
    """Probe outputs are centred per scene, so they fix positions up to a shift. Pin the decoded root to its true
    position (the root never moves in the demo) and place every object relative to it."""
    r = decoded[root]
    return {n: tuple(float(v) for v in np.asarray(root_true, float) + (d - r)) for n, d in decoded.items()}
