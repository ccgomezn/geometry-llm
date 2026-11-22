"""Map boost (docs/preregistration_map_boost.md): locate the map's directions at the answer position, amplify them
there, and judge the effect against variance-matched random subspaces.

At one layer, at the final prompt position only: h ← h + (α − 1) · P(h − μ), P the orthogonal projection onto the
subspace S spanned by the probe directions (or a random control subspace)."""

from __future__ import annotations

import numpy as np
import torch

from geomap.models import layer_out
from geomap.probes import GroupRidge
from geomap.questions import QType, all_questions

EASY = ("own step only", "agree", "conflict, C larger")
DEC, FIRST = "conflict, M larger", "first step only"
MIN_GAIN, MIN_CONTRAST, MAX_EASY_DROP = 0.10, 0.05, 0.05
TRANSFER_DROP, TRANSFER_GAIN = 0.03, 0.03


# ---- locate ----

def map_directions(X, d1, d2, ax, groups) -> np.ndarray:
    """[d, 4] ridge coefficients for d1-east, d2-east, d1-north, d2-north (each fitted on its axis's questions, on
    centred activations, penalty chosen by scene-grouped CV as in the flow test). Uses the steps only, no answers."""
    X, d1, d2, ax, groups = map(np.asarray, (X, d1, d2, ax, groups))
    cols = []
    for a in (0, 1):
        r = ax == a
        Xc = X[r].astype(np.float64) - X[r].mean(0)
        for y in (d1, d2):
            yr = y[r].astype(np.float64)
            cols.append(GroupRidge().fit(Xc, (yr - yr.mean())[:, None], groups[r]).coef_[0])
    return np.stack(cols, 1)


def orthonormal(W) -> torch.Tensor:
    return torch.linalg.qr(torch.as_tensor(W)).Q


# ---- amplify ----

def boost_vec(x: torch.Tensor, Q: torch.Tensor, mu: torch.Tensor, alpha: float) -> torch.Tensor:
    return x + (alpha - 1) * ((x - mu) @ Q) @ Q.T


def boost_edit(layer: int, Q: torch.Tensor, mu: torch.Tensor, alpha: float, pos_of_item):
    """An edit for scoring.score: at `layer`, boost the residual at each item's final prompt position."""
    def edit(m, row_items):
        h = layer_out(m, layer)
        dev = h.device
        rows = torch.arange(len(row_items), device=dev)
        pos = torch.tensor([pos_of_item[i] for i in row_items], device=dev)
        x = h[rows, pos].float()
        h[rows, pos] = boost_vec(x, Q.to(dev, torch.float32), mu.to(dev, torch.float32), alpha).to(h.dtype)
    return edit


# ---- control ----

def _pcs(X, k):
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc, Vt[:k].T, (s[:k] ** 2) / len(Xc)


def _variance(Xc, B) -> float:
    return float(((Xc @ np.asarray(B, dtype=np.float64)) ** 2).sum(1).mean())


def feasible_k(X, Q, k0: int = 64) -> int:
    """Smallest k = k0 · 2^j for which a rank-r subspace of the top-k principal components can match S's variance
    (the top components can always reach it from above; low-variance probe directions may need more components)."""
    Xc = np.asarray(X, dtype=np.float64) - np.asarray(X, dtype=np.float64).mean(0)
    target, r = _variance(Xc, Q), Q.shape[1]
    k = k0
    while True:
        lam = _pcs(X, min(k, min(Xc.shape)))[2]
        if lam[-r:].sum() <= target * (1 + 0.1) or k >= min(Xc.shape):
            return min(k, min(Xc.shape))
        k *= 2


def matched_random_subspaces(X, Q, k: int = 64, n: int = 3, tol: float = 0.1, seed: int = 0,
                             max_tries: int = 200000) -> list[torch.Tensor]:
    """n random rank-r orthonormal subspaces inside the span of the top-k principal components of X, each with total
    projected variance within ±tol of S's. Draws are rotations of Gaussian matrices whose rows are tilted by a random
    power of the component variances (so low- and high-variance targets are both reachable); a draw is kept only if
    it matches."""
    Xc, U, lam = _pcs(X, k)
    r = Q.shape[1]
    target = _variance(Xc, Q.detach().cpu().numpy() if isinstance(Q, torch.Tensor) else Q)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(max_tries):
        gamma = rng.uniform(-4, 4)
        G = rng.normal(size=(len(lam), r)) * ((lam / lam.max()) ** (gamma / 2))[:, None]
        Qg = np.linalg.qr(G)[0]
        v = float((lam[:, None] * Qg ** 2).sum())
        if abs(v / target - 1) <= tol:
            out.append(torch.as_tensor(U @ Qg, dtype=Q.dtype if isinstance(Q, torch.Tensor) else torch.float64))
            if len(out) == n:
                return out
    raise ValueError(f"no matched random subspace in the top {k} components (target variance {target:.3g})")


# ---- measure ----

def ba_boot(truth, preds: dict, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Balanced accuracy (mean recall over the classes present) of each prediction vector, with paired scene-bootstrap
    samples (the same resampled scenes for every vector). Returns {name: (point, samples[n_boot])}."""
    truth, groups = np.asarray(truth), np.asarray(groups)
    u, gi = np.unique(groups, return_inverse=True)
    G = len(u)
    classes = [c for c in (1, -1) if (truth == c).any()]
    n_c = {c: np.bincount(gi, weights=(truth == c).astype(float), minlength=G) for c in classes}
    W = np.stack([np.bincount(rng_choice, minlength=G) for rng_choice in
                  np.random.default_rng(seed).integers(0, G, (n_boot, G))]) if n_boot else np.zeros((0, G))
    out = {}
    for name, p in preds.items():
        hit = np.asarray(p) == truth
        c_c = {c: np.bincount(gi, weights=((truth == c) & hit).astype(float), minlength=G) for c in classes}
        point = float(np.mean([c_c[c].sum() / n_c[c].sum() for c in classes]))
        with np.errstate(invalid="ignore", divide="ignore"):
            samp = np.nanmean(np.stack([(W @ c_c[c]) / (W @ n_c[c]) for c in classes]), 0) if n_boot else np.array([])
        out[name] = (point, samp)
    return out


def transfer_items(s, mid: str, child: str) -> list[tuple]:
    """Other one-word name questions on the scene: farther east / north between C and M or M and R, farthest and
    closest object. Direction (compass-word) and yes/no questions are left out."""
    out = []
    for q in all_questions(s):
        if q.qtype in (QType.MORE_EAST, QType.MORE_NORTH):
            a = set(q.args)
            kind = "C vs M" if a == {child, mid} else "M vs R" if a == {mid, s.root} else None
        else:
            kind = {QType.FARTHEST: "farthest", QType.CLOSEST: "closest"}.get(q.qtype)
        if kind:
            out.append((q, kind))
    return out


# ---- criteria ----

def condition_works(c: dict) -> bool:
    """c: dec_gain, contrast (decisive gain minus the control's), contrast_lo99, first_lo99 (lower 99% bound of the
    first-step-only gain), easy {kind: gain}."""
    return (c["dec_gain"] >= MIN_GAIN and c["contrast"] >= MIN_CONTRAST and c["contrast_lo99"] > 0
            and c["first_lo99"] > 0 and all(v >= -MAX_EASY_DROP for v in c["easy"].values()))


def verdict_boost(conds: list[dict]) -> str:
    if any(condition_works(c) for c in conds):
        return "the map wins when amplified"
    if all(c["contrast_lo99"] <= 0 for c in conds):
        return "no benefit"
    return "partial (the decisive contrast is above 0 in some condition, but no condition meets every criterion)"


def transfer_verdict(overall: float, gains: dict) -> str:
    """overall: change in transfer accuracy; gains: {kind: (gain, lower 95% bound of its contrast with control)}."""
    if overall < -TRANSFER_DROP:
        return "harmful"
    if any(g >= TRANSFER_GAIN and lo > 0 for g, lo in gains.values()):
        return "transfers"
    return "specific"
