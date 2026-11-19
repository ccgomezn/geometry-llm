"""Figures for the short paper (paper/short/main.tex). Reads committed result files only.
Run: uv run python paper/short/make_figs.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geomap.components import betas, design_truth

ROOT = Path(__file__).resolve().parents[2]
FLOW, COMP, OUT = ROOT / "results/flow", ROOT / "results/components", Path(__file__).resolve().parent / "figures"
MODELS = [("gemma3-12b", "Gemma 3 12B", 48, "#2a78d6"), ("gemma3-27b", "Gemma 3 27B", 62, "#eb6834")]
CELLS = ["own step only", "agree", "first step only", "conflict, C larger", "conflict, M larger"]
CELL_LABELS = ["own step\nonly", "agree", "first step\nonly", "conflict,\nC larger", "conflict,\nM larger"]
DEC = "conflict, M larger"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
OWN20 = {"gemma3-12b": ["L44H10", "L47H2", "L45H1", "L41H0", "L41H11", "L42H0", "L41H5", "L41H3", "L35H8", "L47H10",
                        "L47H8", "L35H3", "L41H4", "MLP40", "L35H12", "L46H4", "L47H7", "MLP43", "L41H1", "L32H7"]}
N_BOOT = 2000

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.family": "serif"})


def boot_ci(x: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean, resampling scenes."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    means = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        means.append(x[pick].mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def fig1():
    """(a) accuracy by cell, both models, the same 2,400 questions; (b) on *conflict, M larger*: the truth probe at the
    final position vs the model's own read-out (logit lens) by depth."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1]))
    w = 0.38
    for i, (m, name, n, c) in enumerate(MODELS):
        it = pd.read_csv(FLOW / f"{m}_flow_seed1601_items.csv")
        acc, lo, hi = [], [], []
        for cell in CELLS:
            d = it[it.cell == cell]
            x = d.correct.astype(float).to_numpy()
            acc.append(x.mean())
            l, h = boot_ci(x, d.scene.to_numpy())
            lo.append(x.mean() - l)
            hi.append(h - x.mean())
        xs = np.arange(len(CELLS)) + (i - 0.5) * w
        a.bar(xs, acc, w - 0.04, color=c, label=name, zorder=2)
        a.errorbar(xs, acc, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.7, capsize=0, zorder=3)
        a.text(xs[-1], acc[-1] + hi[-1] + 0.03, f"{acc[-1]:.2f}", ha="center", fontsize=7, color=INK)
    a.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    a.set_xticks(np.arange(len(CELLS)), CELL_LABELS)
    a.set_ylim(0, 1.05)
    a.set_ylabel("accuracy")
    a.set_title("(a) Accuracy by how the two steps combine", loc="left")
    style(a)

    for m, name, n, c in MODELS:
        pr = pd.read_csv(FLOW / f"{m}_flow_seed1601_probes.csv")
        pr = pr[pr.site == "final"]
        ln = pd.read_csv(FLOW / f"{m}_flow_seed1601_lens.csv")
        ln = ln[ln.cell == DEC]
        b.plot(pr.layer / n, pr.truth_conflict_m, "-", color=c, lw=2, label=f"{name}: linear probe")
        b.plot(ln.layer / n, ln.acc, "--", color=c, lw=2, label=f"{name}: model (logit lens)")
    b.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    b.set_xlim(0, 1)
    b.set_ylim(0, 1.05)
    b.set_xlabel("layer (share of depth)")
    b.set_ylabel("accuracy on conflict, M larger")
    b.set_title("(b) Final position: decodable vs answered", loc="left")
    b.text(0.97, 0.66, "linear probe", ha="right", fontsize=7, color=INK)
    b.text(0.97, 0.12, "model's read-out", ha="right", fontsize=7, color=INK)
    style(b)
    fig.legend(*a.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig1_behaviour_probe.pdf")
    fig.savefig(OUT / "fig1_behaviour_probe.png", dpi=200)


def stage2b_acc(m: str) -> dict:
    """Accuracy on *conflict, M larger* (Stage 2b set) per condition, with scene-bootstrap CIs; the linear prediction
    as in scripts/components2_test.py (clean logit difference minus the clean shortcut term)."""
    T = pd.read_csv(COMP / f"{m}_components2_test.csv")
    F = design_truth(T)
    own_yes = F[:, 1]
    b = betas(T.ld_clean.to_numpy()[:, None], F)[0]
    lin = T.ld_clean - b[1] * own_yes
    ok = lambda ld: (np.sign(ld) == T.truth).astype(float)
    cond = {"clean": ok(T.ld_clean),
            "random 20 removed": np.mean([ok(T[f"ld_twinmean_rand20_{i}"]) for i in range(3)], axis=0),
            "top 20 removed": ok(T.ld_twinmean_top20),
            "linear prediction": ok(lin)}
    dec = (T.cell == DEC).to_numpy()
    out = {}
    for k, v in cond.items():
        x = np.asarray(v)[dec]
        out[k] = (x.mean(), *boot_ci(x, T.scene.to_numpy()[dec]))
    return out


def fig2():
    """(a) 12B: shortcut coefficient of each component, clean vs top 20 removed (writers and brakes); (b) both models:
    accuracy on *conflict, M larger* with the shortcut writers removed."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    R = pd.read_csv(COMP / "gemma3-12b_components2_repair_top20.csv")
    rem = R[R.component.isin(OWN20["gemma3-12b"])].nlargest(8, "b_own_clean")
    brk = R[~R.component.isin(OWN20["gemma3-12b"]) & (R.b_own_clean < 0)].nlargest(6, "change")
    rows = pd.concat([rem.assign(role="writer (removed)"), brk.assign(role="brake (not removed)")])
    ys = np.arange(len(rows))[::-1]
    for y, (_, r) in zip(ys, rows.iterrows()):
        c = MODELS[0][3]
        a.annotate("", xy=(r.b_own_removed, y), xytext=(r.b_own_clean, y),
                   arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, shrinkA=3, shrinkB=3, mutation_scale=7))
        a.plot(r.b_own_clean, y, "o", ms=5, color=c, zorder=3)
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_yticks(ys, rows.component)
    a.set_xlabel("shortcut coefficient (logits); dot = clean, arrow = top 20 removed")
    a.set_title("(a) 12B: writers removed, brakes let go", loc="left")
    a.axhline(len(brk) - 0.5, color=GRID, lw=0.8)
    a.text(7.6, len(brk) - 0.4, "writers (removed) ↑", color=INK, fontsize=7, ha="right", va="bottom")
    a.text(7.6, len(brk) - 0.6, "brakes (not removed) ↓", color=INK, fontsize=7, ha="right", va="top")
    a.set_ylim(-1.3, len(rows) - 0.1)
    a.set_xlim(-7.3, 7.7)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="x", color=GRID, lw=0.5)
    a.set_axisbelow(True)

    conds = ["clean", "random 20 removed", "top 20 removed", "linear prediction"]
    w = 0.38
    for i, (m, name, n, c) in enumerate(MODELS):
        S = stage2b_acc(m)
        xs = np.arange(len(conds)) + (i - 0.5) * w
        v = np.array([S[k][0] for k in conds])
        lo = v - np.array([S[k][1] for k in conds])
        hi = np.array([S[k][2] for k in conds]) - v
        b.bar(xs, v, w - 0.04, color=c, label=name, zorder=2)
        b.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.7, zorder=3)
        for x, val, h in zip(xs, v, hi):
            b.text(x, val + h + 0.02, f"{val:.2f}", ha="center", fontsize=6.5, color=INK)
    b.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    b.axhline(0.65, color=INK, lw=0.8, ls="--")
    b.text(-0.45, 0.665, "preregistered bar for a correct path (0.65)", fontsize=6.5, color=INK)
    b.set_xticks(np.arange(len(conds)), ["clean", "random 20\nremoved", "top 20\nremoved", "linear\nprediction"])
    b.set_ylim(0, 0.85)
    b.set_ylabel("accuracy on conflict, M larger")
    b.set_title("(b) Without the shortcut writers", loc="left")
    b.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.02))
    style(b)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig2_mechanism.pdf")
    fig.savefig(OUT / "fig2_mechanism.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    print(OUT)
