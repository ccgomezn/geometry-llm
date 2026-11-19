"""Stage 2 (docs/preregistration_components.md): which components write the answer at the final position, and how do
they weigh the two steps?

Direct logit attribution at the final position: the residual stream there is the embedding plus every layer's
attention and MLP additions, and the answer's logit difference logit(C) − logit(R) is linear in that sum once the final
norm's scale is fixed. Each attention addition splits exactly into heads (slices of the output projection; Gemma's
post-attention norm is a per-item scale) and each head's part splits exactly into the source tokens it attends to."""

from __future__ import annotations

import functools
from contextlib import contextmanager, nullcontext
from typing import Sequence

import numpy as np
import torch

GROUPS = ("chat", "M_sentence", "C_sentence", "list", "Q_C", "Q_R", "Q_other", "final")
FEATURES = ("own_yes", "d1", "d2", "c_first")


def _gemma(m) -> bool:
    return "gemma" in type(m.hf).__name__.lower()


def _layers(m):
    return functools.reduce(getattr, m.layers_path.split("."), m.hf)


def _final_norm(m):
    return functools.reduce(getattr, (m.layers_path.rsplit(".", 1)[0] + ".norm").split("."), m.hf)


def _embed(m):
    return functools.reduce(getattr, m.embed_path.split("."), m.hf)


def _eps(norm) -> float:
    return float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))


def _gain(norm, gemma: bool) -> torch.Tensor:
    w = norm.weight.float()
    return 1 + w if gemma else w


def _rinv(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


def mlp_out_module(layer, gemma: bool):
    """The module whose output is the MLP's addition to the residual stream."""
    return layer.post_feedforward_layernorm if gemma else layer.mlp


def head_shape(m) -> tuple[int, int, int]:
    a = _layers(m)[0].self_attn
    return a.config.num_attention_heads, a.head_dim, a.num_key_value_groups


def group_ids(spec, mid: str, child: str, root: str) -> list[int]:
    """GROUPS index of every prompt position."""
    T = len(spec.input_ids)
    g = [0] * T
    for name, k in ((mid, 1), (child, 2)):
        for p in range(*spec.relation_spans[name]):
            g[p] = k
    for p in range(max(e for _, e in spec.relation_spans.values()), spec.eol_pos + 1):
        g[p] = 3
    for p in range(spec.prefix_len, T - 1):
        g[p] = 6
    g[spec.question_pos[child]], g[spec.question_pos[root]], g[T - 1] = 4, 5, 7
    return g


@torch.no_grad()
def component_dla(m, ids_list: Sequence[Sequence[int]], groups_list: Sequence[Sequence[int]], n_groups: int,
                  c_ids: Sequence[int], r_ids: Sequence[int], batch_size: int = 16, batch_ctx=None) -> dict:
    """Direct contributions to logit(c) − logit(r) at each prompt's last position (right-padded batches, eager
    attention). Returns emb [N], heads [N, L, H], mlp [N, L], src [N, L, H, G] (each head's part by source-token
    group), ld [N] (the model's own logit difference, for the check), rf [N] (the final norm's 1/rms), and the mean
    over prompts of each layer's o_proj input (z_mean [L, H·hd]) and MLP addition (mlp_mean [L, d]) at the last
    position (for mean ablation). With batch_ctx (e.g. ablation_ctx, whose hooks run first), the attribution is of the
    intervened run."""
    gemma, layers = _gemma(m), _layers(m)
    L, (H, hd, n_rep) = len(layers), head_shape(m)
    m.hf.set_attn_implementation("eager")
    for layer in layers:
        assert layer.self_attn.o_proj.bias is None
    N = len(ids_list)
    out = dict(emb=np.zeros(N), heads=np.zeros((N, L, H)), mlp=np.zeros((N, L)), src=np.zeros((N, L, H, n_groups)),
               ld=np.zeros(N), rf=np.zeros(N))
    z_sum, m_sum = None, None
    st: dict = {}

    def at_f(t):
        return t[torch.arange(t.shape[0], device=t.device), st["f"].to(t.device)]

    hooks = [_embed(m).register_forward_hook(lambda mod, a, o: st.__setitem__("e", at_f(o).float())),
             _final_norm(m).register_forward_pre_hook(lambda mod, a: st.__setitem__("r", at_f(a[0]).float()))]
    for l, layer in enumerate(layers):
        at = layer.self_attn
        hooks += [
            at.o_proj.register_forward_pre_hook(lambda mod, a, l=l: st.__setitem__(("z", l), at_f(a[0]).float())),
            at.o_proj.register_forward_hook(lambda mod, a, o, l=l: st.__setitem__(("o", l), at_f(o).float())),
            at.v_proj.register_forward_hook(lambda mod, a, o, l=l: st.__setitem__(("v", l), o.float())),
            at.register_forward_hook(lambda mod, a, o, l=l: st.__setitem__(("a", l), o[1][
                torch.arange(o[1].shape[0], device=o[1].device)[:, None],
                torch.arange(o[1].shape[1], device=o[1].device)[None, :], st["f"].to(o[1].device)[:, None]].float())),
            mlp_out_module(layer, gemma).register_forward_hook(
                lambda mod, a, o, l=l: st.__setitem__(("m", l), at_f(o).float())),
        ]
    fn = _final_norm(m)
    W = m.hf.get_output_embeddings().weight
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    dev = W.device
    try:
        for b0 in range(0, N, batch_size):
            batch = list(ids_list[b0:b0 + batch_size])
            B, T = len(batch), max(len(x) for x in batch)
            ids = torch.full((B, T), pad, dtype=torch.long)
            am = torch.zeros((B, T), dtype=torch.long)
            gid = torch.zeros((B, T), dtype=torch.long)
            for j, x in enumerate(batch):
                ids[j, :len(x)], am[j, :len(x)] = torch.tensor(list(x)), 1
                gid[j, :len(x)] = torch.tensor(list(groups_list[b0 + j]))
            st["f"] = am.sum(1) - 1
            with batch_ctx(list(range(b0, b0 + B))) if batch_ctx else nullcontext():
                logits = m.hf(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits
            c = torch.tensor(list(c_ids[b0:b0 + B]), device=dev)
            r = torch.tensor(list(r_ids[b0:b0 + B]), device=dev)
            lf = at_f(logits).float()
            out["ld"][b0:b0 + B] = (lf.gather(1, c[:, None]) - lf.gather(1, r[:, None]))[:, 0].cpu().numpy()
            rr = st["r"].to(dev)
            out["rf"][b0:b0 + B] = _rinv(rr, _eps(fn))[:, 0].cpu().numpy()
            u = _gain(fn, gemma).to(dev) * _rinv(rr, _eps(fn)) * (W[c].float() - W[r].float())      # [B, d]
            out["emb"][b0:b0 + B] = (st["e"].to(dev) * u).sum(-1).cpu().numpy()
            for l, layer in enumerate(layers):
                at = layer.self_attn
                ua = u
                if gemma:
                    pn = layer.post_attention_layernorm
                    ua = u * _gain(pn, True).to(dev) * _rinv(st[("o", l)].to(dev), _eps(pn))
                w = (ua @ at.o_proj.weight.float().to(dev)).view(B, H, hd)                             # [B, H, hd]
                z = st[("z", l)].to(dev).view(B, H, hd)
                out["heads"][b0:b0 + B, l] = (z * w).sum(-1).cpu().numpy()
                v = st[("v", l)].to(dev).view(B, T, H // n_rep, hd).repeat_interleave(n_rep, dim=2)     # [B, T, H, hd]
                contrib = st[("a", l)].to(dev)[:, :, :T] * torch.einsum("bthd,bhd->bht", v, w)         # [B, H, T]
                g = gid.to(dev)[:, None, :].expand(B, H, T)
                out["src"][b0:b0 + B, l] = torch.zeros(B, H, n_groups, device=dev).scatter_add_(2, g, contrib).cpu().numpy()
                out["mlp"][b0:b0 + B, l] = (st[("m", l)].to(dev) * u).sum(-1).cpu().numpy()
            zs = torch.stack([st[("z", l)].cpu() for l in range(L)]).sum(1)
            ms = torch.stack([st[("m", l)].cpu() for l in range(L)]).sum(1)
            z_sum = zs if z_sum is None else z_sum + zs
            m_sum = ms if m_sum is None else m_sum + ms
    finally:
        for h in hooks:
            h.remove()
    out["z_mean"] = (z_sum / N).numpy()
    out["mlp_mean"] = (m_sum / N).numpy()
    return out


def ablation_ctx(m, finals: Sequence[int], heads: dict, mlps: dict, alpha: float | None = None):
    """batch_ctx for scoring.score(): at item i's position finals[i], replace head h's slice of the o_proj input in
    layer l by heads[l][h], and the MLP's residual addition in layer l by mlps[l]: one vector for all items, or one row
    per item ([n_items, ·]). With alpha, scale instead around those values: v + alpha·(current − v). The hooks run
    before any other hook on the same module (so component_dla records the intervened values)."""
    gemma, layers = _gemma(m), _layers(m)
    _, hd, _ = head_shape(m)

    def new(cur, ref, rows):
        ref = (ref[rows] if ref.dim() == 2 else ref.expand_as(cur)).to(cur.device, cur.dtype)
        return ref if alpha is None else ref + alpha * (cur - ref)

    @contextmanager
    def ctx(row_items: list[int]):
        f = torch.tensor([finals[i] for i in row_items])
        rows = torch.tensor(list(row_items))
        hs = []

        def pre(mod, args, hv):
            z = args[0].clone()
            b, ff = torch.arange(z.shape[0], device=z.device), f.to(z.device)
            for h, ref in hv.items():
                sl = slice(h * hd, (h + 1) * hd)
                z[b, ff, sl] = new(z[b, ff, sl], ref, rows)
            return (z,) + tuple(args[1:])

        def post(mod, args, o, ref):
            o = o.clone()
            b, ff = torch.arange(o.shape[0], device=o.device), f.to(o.device)
            o[b, ff] = new(o[b, ff], ref, rows)
            return o

        for l, hv in heads.items():
            hs.append(layers[l].self_attn.o_proj.register_forward_pre_hook(functools.partial(pre, hv=hv), prepend=True))
        for l, ref in mlps.items():
            hs.append(mlp_out_module(layers[l], gemma).register_forward_hook(functools.partial(post, ref=ref),
                                                                             prepend=True))
        try:
            yield
        finally:
            for h in hs:
                h.remove()

    return ctx


@torch.no_grad()
def capture_final(m, ids_list: Sequence[Sequence[int]], sel, batch_size: int = 16) -> dict:
    """Clean values at each prompt's last position of the selected components [(kind, layer, head)]: a head's slice of
    its layer's o_proj input [N, hd], an MLP's residual addition [N, d]. Right-padded batches."""
    gemma, layers = _gemma(m), _layers(m)
    _, hd, _ = head_shape(m)
    st: dict = {}
    at_f = lambda t: t[torch.arange(t.shape[0], device=t.device), st["f"].to(t.device)].float().cpu()
    hooks = []
    for l in sorted({c[1] for c in sel if c[0] == "head"}):
        hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(
            lambda mod, a, l=l: st.__setitem__(("head", l), at_f(a[0]))))
    for l in sorted({c[1] for c in sel if c[0] == "mlp"}):
        hooks.append(mlp_out_module(layers[l], gemma).register_forward_hook(
            lambda mod, a, o, l=l: st.__setitem__(("mlp", l), at_f(o))))
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    dev = m.hf.get_output_embeddings().weight.device
    parts: dict = {c: [] for c in sel}
    try:
        for b0 in range(0, len(ids_list), batch_size):
            batch = list(ids_list[b0:b0 + batch_size])
            B, T = len(batch), max(len(x) for x in batch)
            ids = torch.full((B, T), pad, dtype=torch.long)
            am = torch.zeros((B, T), dtype=torch.long)
            for j, x in enumerate(batch):
                ids[j, :len(x)], am[j, :len(x)] = torch.tensor(list(x)), 1
            st["f"] = am.sum(1) - 1
            m.hf(input_ids=ids.to(dev), attention_mask=am.to(dev))
            for c in sel:
                kind, l, h = c
                parts[c].append(st[(kind, l)][:, h * hd:(h + 1) * hd] if kind == "head" else st[(kind, l)])
    finally:
        for h in hooks:
            h.remove()
    return {c: torch.cat(v).numpy() for c, v in parts.items()}


def design(meta) -> np.ndarray:
    """[1, own_yes, d1, d2, c_first] per question, C frame: own_yes = 1 when C's own step points the asked way."""
    d1, d2 = meta.d1.to_numpy(float), meta.d2.to_numpy(float)
    return np.c_[np.ones(len(meta)), (d2 > 0).astype(float), d1, d2, np.where(meta.c_first.to_numpy(), 1.0, -1.0)]


def design_truth(meta) -> np.ndarray:
    """[1, own_yes, truth (±1), d1, d2, c_first] per question: design() with the correct answer added, so that a
    read-out turning towards the truth is not counted as shortcut."""
    F = design(meta)
    return np.c_[F[:, :2], meta.truth.to_numpy(float), F[:, 2:]]


def betas(Y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """OLS coefficients [k, p] of each column of Y [n, k] on the design F [n, p]."""
    return np.linalg.lstsq(F, Y, rcond=None)[0].T


def pick_matched(selected, n_layers: int, n_heads: int, rng, exclude: set) -> list:
    """A random stand-in for each selected component: a head of the same layer, or an MLP within 4 layers, not in
    `exclude` and not already picked (widening to nearby layers, 2 at a time, if none is left)."""
    out, used = [], set(exclude)
    for kind, l, h in selected:
        radius, pool = (0 if kind == "head" else 4), []
        while not pool and radius <= n_layers + 4:
            lays = range(max(0, l - radius), min(n_layers, l + radius + 1))
            cand = ([("head", k, j) for k in lays for j in range(n_heads)] if kind == "head"
                    else [("mlp", k, -1) for k in lays])
            pool = [c for c in cand if c not in used]
            radius += 2
        pick = pool[int(rng.integers(len(pool)))]
        out.append(pick)
        used.add(pick)
    return out


def verdict_components(drop: float, acc: float, acc_hi: float, gain: float, gain_lo: float,
                       min_drop: float = 0.20, good: float = 0.65, min_gain: float = 0.15, none_max: float = 0.60) -> str:
    """drop: shortcut-right accuracy lost beyond the matched control; acc (CI upper acc_hi): accuracy on the decisive
    questions with the own-direction writers ablated; gain (CI lower gain_lo): that accuracy minus the control's."""
    if drop < min_drop:
        return "ineffective"
    if acc >= good and gain >= min_gain and gain_lo > 0:
        return "a correct path underneath the shortcut"
    if acc_hi < none_max:
        return "no correct path underneath the shortcut"
    return "undecided"


def repair_groups(delta: np.ndarray, comps: Sequence, removed: Sequence) -> dict:
    """Sum a per-component change (e.g. in the own_yes coefficient) by where the component sits relative to the
    removed ones: removed; other heads in a layer with a removed head; later (after the first removed component in
    the residual order: layer l's heads, then its MLP); earlier (the embedding counts as earliest)."""
    rem = set(removed)
    pos = lambda c: -1 if c[0] == "emb" else 2 * c[1] + (c[0] == "mlp")
    p0 = min(pos(c) for c in rem)
    head_layers = {c[1] for c in rem if c[0] == "head"}
    out = dict(removed=0.0, same_layer_heads=0.0, later=0.0, earlier=0.0)
    for c, x in zip(comps, delta):
        g = ("removed" if c in rem else "same_layer_heads" if c[0] == "head" and c[1] in head_layers
             else "later" if pos(c) > p0 else "earlier")
        out[g] += float(x)
    return out


def verdict_components2(ratio: float, acc: float, acc_hi: float, gain: float, gain_lo: float, max_ratio: float = 0.5,
                        good: float = 0.65, min_gain: float = 0.15, none_max: float = 0.60) -> str:
    """ratio: shortcut (own_yes coefficient) with the components removed ÷ the matched control's; acc (CI upper
    acc_hi): accuracy on *conflict, M larger* with them removed; gain (CI lower gain_lo): that minus the control's."""
    if ratio > max_ratio:
        return "ineffective"
    if acc >= good and gain >= min_gain and gain_lo > 0:
        return "a correct path underneath the shortcut"
    if acc_hi < none_max:
        return "no correct path underneath the shortcut"
    return "undecided"
