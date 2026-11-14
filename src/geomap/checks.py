"""Two checks on whether the model uses its map (docs/preregistration_map_use_checks.md).

A. Error overlap: where the probe-decoded map answers wrongly, does the model give the same wrong answer more often
   than with a blurred true map that errs as often, on the same kinds of question?
B. Nonlinear probe after erasure: can an MLP still read positions once P5c's INLP subspace is erased?"""

from __future__ import annotations

import copy

import numpy as np
import torch

from geomap.probes import GroupRidge
from geomap.questions import answer_for


# ---------- Check A ----------

def blurred_rates(items, sd: float, n_draws: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """items: (scene_idx, question, true coords, truth, model_answer). Every draw adds N(0, sd²) noise to each object's
    true position (one map per scene and draw, shared by its questions; common random numbers across sd values).
    Returns per question: P(blurred map wrong), P(blurred map wrong and equal to the model's answer)."""
    rng = np.random.default_rng(seed)
    z = {}
    for si, _, coords, _, _ in items:
        if si not in z:
            z[si] = {n: rng.standard_normal((n_draws, 2)) for n in sorted(coords)}
    wrong, same = np.zeros(len(items)), np.zeros(len(items))
    for k, (si, q, coords, truth, model) in enumerate(items):
        for d in range(n_draws):
            blur = {n: (x + sd * z[si][n][d, 0], y + sd * z[si][n][d, 1]) for n, (x, y) in coords.items()}
            a = answer_for(q, blur)
            if a != truth:
                wrong[k] += 1
                same[k] += a == model
    return wrong / n_draws, same / n_draws


def calibrate_sd(items, target_acc: float, n_draws: int, seed: int = 0, tol: float = 0.01, hi: float = 64.0,
                 max_iter: int = 40) -> float:
    """Noise sd at which the blurred map's mean accuracy on `items` equals target_acc (within tol), by bisection."""
    if target_acc >= 1 - tol:
        return 0.0
    acc = lambda sd: 1 - blurred_rates(items, sd, n_draws, seed)[0].mean()
    if acc(hi) > target_acc:
        return hi
    lo = 0.0
    mid = hi / 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        a = acc(mid)
        if abs(a - target_acc) <= tol:
            return mid
        lo, hi = (mid, hi) if a > target_acc else (lo, mid)
    return mid


def error_overlap(wrong, same, blur_wrong, blur_same, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """S = Σsame/Σwrong for the decoded map, S_blur likewise with expected counts; effect = S − S_blur with a
    scene-bootstrap 95% CI (ratios of sums recomputed per resample)."""
    cols = np.stack([np.asarray(a, float) for a in (wrong, same, blur_wrong, blur_same)], 1)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(uniq), 4))
    np.add.at(sums, inv, cols)

    def stats(t):
        S, Sb = t[1] / t[0], t[3] / t[2]
        return S, Sb, S - Sb

    S, Sb, eff = stats(sums.sum(0))
    rng = np.random.default_rng(seed)
    boots = [stats(sums[rng.integers(0, len(uniq), len(uniq))].sum(0))[2] for _ in range(n_boot)]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return dict(S=float(S), S_blur=float(Sb), effect=float(eff), lo=float(lo), hi=float(hi),
                n_map_wrong=float(cols[:, 0].sum()))


def verdict_overlap(effect: float, lo: float, min_effect: float = 0.05) -> str:
    if lo > 0 and effect >= min_effect:
        return "the model's mistakes follow the map"
    if lo > 0:
        return "small overlap"
    return "no sign that the mistakes follow the map"


# ---------- Check B ----------

def r2(Y, P) -> float:
    """R² pooled over output columns."""
    Y, P = np.asarray(Y, float), np.asarray(P, float)
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def _folds(groups, n_folds: int, seed: int) -> np.ndarray:
    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    fold_of = {g: k % n_folds for k, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def linear_cv(X, Y, groups, n_folds: int = 3, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of the Study 1 ridge probe (scene-grouped folds)."""
    X, Y, groups = np.asarray(X, np.float64), np.asarray(Y, np.float64), np.asarray(groups)
    f = _folds(groups, n_folds, seed)
    P = np.zeros_like(Y)
    for k in range(n_folds):
        tr, va = f != k, f == k
        P[va] = GroupRidge().fit(X[tr], Y[tr], groups[tr]).predict(X[va])
    return P


def _fit_mlp(Xtr, Ytr, Xva, Yva, hidden, dropout, lr, wd, batch, max_epochs, patience, seed, device):
    torch.manual_seed(seed)
    d, o = Xtr.shape[1], Ytr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, o)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(batch):
            idx = idx.to(device)
            loss = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xva) - Yva) ** 2).mean())
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def mlp_cv(X, Y, groups, n_folds: int = 3, seed: int = 0, hidden: int = 512, dropout: float = 0.1, lr: float = 1e-3,
           wd: float = 1e-2, batch: int = 256, max_epochs: int = 300, patience: int = 20, val_frac: float = 0.1,
           device: str | None = None) -> np.ndarray:
    """Out-of-fold predictions of a 2-hidden-layer MLP probe. Inputs and targets are standardised with the training
    fold's statistics; early stopping on `val_frac` of the training fold's groups; predictions from the best epoch."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, groups = np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(groups)
    f = _folds(groups, n_folds, seed)
    P = np.zeros_like(Y, dtype=np.float64)
    for k in range(n_folds):
        tr_all, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        gtr = np.unique(groups[tr_all])
        stop_groups = np.random.default_rng(seed + k).choice(gtr, max(1, int(round(val_frac * len(gtr)))),
                                                              replace=False)
        is_stop = np.isin(groups[tr_all], stop_groups)
        tr, st = tr_all[~is_stop], tr_all[is_stop]
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
        my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        T = lambda A, mu, s: torch.as_tensor((A - mu) / s, dtype=torch.float32, device=device)
        net = _fit_mlp(T(X[tr], mx, sx), T(Y[tr], my, sy), T(X[st], mx, sx), T(Y[st], my, sy), hidden, dropout, lr,
                       wd, batch, max_epochs, 20 if patience is None else patience, seed + k, device)
        with torch.no_grad():
            P[va] = net(T(X[va], mx, sx)).double().cpu().numpy() * sy + my
    return P


def verdict_nonlinear(mlp_clean: float, lin_clean: float, mlp_probe: float, mlp_random: float,
                      survive: float = 0.30, gone: float = 0.10, intact: float = 0.50, slack: float = 0.05) -> str:
    """docs/preregistration_map_use_checks.md, check B (layer 21)."""
    if mlp_clean < lin_clean - slack:
        return "inconclusive (the MLP probe is weaker than the linear one)"
    if mlp_probe >= survive:
        return "a nonlinear map survives the erasure"
    if mlp_probe < gone and mlp_random >= intact:
        return "the erasure removes the map, linear and nonlinear"
    return "partly"


# ---------- closed-loop erasure (docs/preregistration_erase_closed_loop.md) ----------

def extend_basis(B_old, X, Y, groups, max_steps: int = 5, r2_stop: float = 0.05):
    """X: scene-centred activations captured with B_old already erased. Runs INLP on X (up to max_steps, stopping at
    group-CV R² < r2_stop) and returns (orthonormal basis spanning B_old and the new directions, R² before each step)."""
    from geomap.followups import inlp_basis
    B_old = np.asarray(B_old, np.float64)
    B_new, hist = inlp_basis(X, Y, groups, max_rounds=max_steps, r2_stop=r2_stop)
    if B_new.shape[1] == 0:
        return B_old, hist
    W = B_new - B_old @ (B_old.T @ B_new)
    W = np.linalg.qr(W)[0]
    keep = np.linalg.norm(W - B_old @ (B_old.T @ W), axis=0) > 1e-6
    return np.concatenate([B_old, W[:, keep]], 1), hist


def verdict_closed_loop(summary, max_val_r2: float, random_comp_acc: float, comp_chance: float, mlp21: float,
                        r2_max: float = 0.10, headroom: float = 0.15, used: float = 0.10, gap: float = 0.05,
                        none: float = 0.03, mlp_max: float = 0.30) -> str:
    """docs/preregistration_erase_closed_loop.md. summary: erase_summary rows for the closed-loop configuration."""
    if max_val_r2 >= r2_max:
        return "the map could not be erased"
    if random_comp_acc < comp_chance + headroom:
        return "uninformative (the erasure damages the model in general)"
    c = summary[summary.composed].iloc[0]
    st = summary[~summary.composed].iloc[0]
    if c.excess >= used and c.excess - st.excess >= gap:
        return "map used"
    if c.excess_lo > 0 and c.excess > st.excess:
        return "partial use"
    if c.excess <= none:
        return "linear map not used" if mlp21 >= mlp_max else "map not used"
    return "unclear"


# ---------- confirmatory axis test (docs/preregistration_erase_axis_confirm.md) ----------

def variance_matched_random(X, B, seed: int, alpha_max: float = 8.0, tol: float = 0.02, max_iter: int = 60,
                            alpha_min: float = 0.0):
    """Random orthonormal basis R with B's rank that removes as much variance as B: tr(RᵀΣR) = tr(BᵀΣB) within tol
    (relative), Σ = XᵀX / n (X centred). Directions w ~ (Σ + floor)^(α/2) z: α = 0 is isotropic, α > 0 leans towards
    the top principal directions, α < 0 towards the low-variance ones; the null space of X gets the smallest positive
    eigenvalue. α by bisection in [alpha_min, alpha_max]. Returns (R, α). With alpha_min = 0 (the confirmatory test's
    setting) a target below isotropic returns isotropic directions; the fair control uses alpha_min < 0."""
    X, B = np.asarray(X, np.float64), np.asarray(B, np.float64)
    d, r = B.shape
    if r == 0:
        return np.zeros((d, 0)), 0.0
    n = len(X)
    if n >= d:
        lam, V = np.linalg.eigh(X.T @ X / n)
    else:
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        lam, V = S ** 2 / n, Vt.T
    pos = lam > lam.max() * 1e-12
    V, lam_raw = V[:, pos], lam[pos]
    lam = lam_raw / lam_raw.max()
    floor = lam.min()
    Z = np.random.default_rng(seed).standard_normal((d, r))
    ZV = V.T @ Z
    Znull = Z - V @ ZV
    var = lambda M: float((((V.T @ M) ** 2) * lam_raw[:, None]).sum())
    target = var(B)

    def basis(alpha):
        W = V @ (lam[:, None] ** (alpha / 2) * ZV) + floor ** (alpha / 2) * Znull
        return np.linalg.qr(W)[0]

    lo, hi = alpha_min, alpha_max
    R = basis(0.0)
    v0 = var(R)
    if abs(v0 - target) <= tol * target:
        return R, 0.0
    if alpha_min >= 0 and v0 > target:
        return R, 0.0
    if var(basis(hi)) < target:
        return basis(hi), hi
    if var(basis(lo)) > target:
        return basis(lo), lo
    mid = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        R = basis(mid)
        v = var(R)
        if abs(v - target) <= tol * target:
            return R, mid
        lo, hi = (mid, hi) if v < target else (lo, mid)
    return R, mid


def verdict_axis(excess: float, lo: float, max_val_r2: float, random_comp_acc: float, comp_chance: float,
                 r2_max: float = 0.10, headroom: float = 0.15, min_effect: float = 0.03) -> str:
    """docs/preregistration_erase_axis_confirm.md. excess, lo: CR-axis excess drop vs the matched random control."""
    if max_val_r2 >= r2_max:
        return "the map could not be erased"
    if random_comp_acc < comp_chance + headroom:
        return "uninformative"
    if lo > 0 and excess >= min_effect:
        return "confirmed: composed axis answers use the map"
    if lo > 0:
        return "small effect"
    return "not confirmed"


def verdict_overall(excess: float, lo: float, random_comp_acc: float, comp_chance: float, used: float = 0.10,
                    none: float = 0.03, headroom: float = 0.15) -> str:
    """Primary B of docs/preregistration_erase_fair_control.md (all composed questions)."""
    if random_comp_acc < comp_chance + headroom:
        return "uninformative"
    if excess >= used and lo > 0:
        return "map used"
    if lo > 0:
        return "partial use"
    if excess <= none:
        return "map not used beyond generic damage"
    return "unclear"
