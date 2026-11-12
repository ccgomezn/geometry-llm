"""Four reader-facing charts for paper/blog_post.md, one per finding, from the committed result files.
Plain-language labels; the research figures in results/figures keep the technical detail."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.report import ROOT

OUT = ROOT / "paper" / "figures"
BLUE, ORANGE, GREEN, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8c939d", "#d5d9de"
INK, MUTED = "#1f2328", "#5f6670"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": LIGHT,
                     "axes.spines.top": False, "axes.spines.right": False})


def title(fig, main, sub):
    fig.text(0.02, 0.97, main, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.89, sub, fontsize=11, color=MUTED, va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(OUT / name)


def pct(v):
    return f"{v:.0%}"


def fig1():
    """Worked-out positions are readable from activations; the stated sentences alone don't give them."""
    real = pd.read_csv(ROOT / "results/p2/gemma3-12b_real.csv").set_index("layer").loc[21]
    base = pd.read_csv(ROOT / "results/p2/gemma3-12b_stated_relation.csv").iloc[0]  # layer-independent
    dist = pd.read_csv(ROOT / "results/p2/gemma3-12b_distance.csv").set_index("layer").loc[21]
    groups = ["Position of objects\nit had to work out", "Distance to the\nfirst object"]
    model = [real.r2_hop2, dist.hop2]
    stated = [base.r2_hop2, -0.05]  # best additive baseline for distance (RESULTS.md, P2d)
    fig, ax = plt.subplots(figsize=(7, 3.9))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.72, bottom=0.16)
    y = [1, 0]
    h = 0.34
    for i, (m, s) in enumerate(zip(model, stated)):
        ax.barh(y[i] + h / 2 + 0.02, m, h, color=BLUE, label="Read from the model's insides" if i == 0 else None)
        ax.barh(y[i] - h / 2 - 0.02, max(s, 0.004), h, color=GRAY, label="Best guess from the sentences alone" if i == 0 else None)
        ax.text(m + 0.015, y[i] + h / 2 + 0.02, f"{m:.2f}", va="center", fontsize=12)
        ax.text(max(s, 0) + 0.015, y[i] - h / 2 - 0.02, f"{s:.2f}" if s > 0 else "none", va="center", fontsize=12)
    ax.set_yticks(y, groups)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("how well it can be read out (0 = not at all, 1 = perfectly)")
    ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(-0.36, 1.0), ncol=2)
    title(fig, "The model works out positions nobody stated",
          "Gemma 3 12B, 600 scenes it had never seen")
    save(fig, "blog_1_worked_out.png")


def fig2():
    """The learned 2-number edit vs 2 random numbers, both models, against the preregistered bar."""
    rows = []
    for model, f in [("Gemma 3 12B", "gemma3-12b_das_relation_L14_confirm.csv"),
                     ("Gemma 3 27B", "gemma3-27b_das_relation_L18_confirm.csv")]:
        d = pd.read_csv(ROOT / "results/p4" / f)
        d = d[d.qtype == "ALL"].set_index("method").acc
        rows.append((model, d["DAS k=2"], d["random k=2"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.74, bottom=0.12)
    w = 0.34
    for i, (model, das, rnd) in enumerate(rows):
        ax.bar(i - w / 2 - 0.02, das, w, color=BLUE, label="Edit the 2 numbers we found" if i == 0 else None)
        ax.bar(i + w / 2 + 0.02, max(rnd, 0.006), w, color=GRAY, label="Edit 2 random numbers" if i == 0 else None)
        ax.text(i - w / 2 - 0.02, das + 0.02, pct(das), ha="center", fontsize=13, fontweight="bold")
        ax.text(i + w / 2 + 0.02, rnd + 0.02, pct(rnd), ha="center", fontsize=13)
    ax.axhline(0.7, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(1.45, 0.715, "pass mark,\nfixed in advance", ha="left", va="bottom", fontsize=10, color=MUTED)
    ax.set_xticks([0, 1], [r[0] for r in rows])
    ax.tick_params(axis="x", length=0, labelcolor=INK, labelsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 2.05)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, 1.0), ncol=2)
    title(fig, "Two numbers inside the model move the object",
          "Share of answers that follow the edit; 300 new scenes per model")
    save(fig, "blog_2_two_numbers.png")


def fig3():
    """Halfway setting: what the model answers on questions whose in-between answer is new."""
    rows = []
    for model, f in [("Gemma 3 12B", "gemma3-12b_interp_relation_L14_confirm.csv"),
                     ("Gemma 3 27B", "gemma3-27b_interp_relation_L18_confirm.csv")]:
        r = pd.read_csv(ROOT / "results/p4" / f).set_index("alpha").loc[0.5]
        rows.append((model, r.novel, r.novel_base, r.novel_src))
    cats = [("In-between answer", BLUE), ("Original layout", ORANGE), ("Moved layout", GREEN), ("Something else", LIGHT)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.subplots_adjust(left=0.19, right=0.95, top=0.66, bottom=0.17)
    for i, (model, nov, base, src) in enumerate(rows):
        vals = [nov, base, src, max(0.0, 1 - nov - base - src)]
        left, yy = 0.0, 1 - i
        for (lab, c), v in zip(cats, vals):
            ax.barh(yy, v, 0.56, left=left, color=c, edgecolor="white", linewidth=2, label=lab if i == 0 else None)
            if v >= 0.05:
                ax.text(left + v / 2, yy, pct(v), ha="center", va="center", fontsize=11,
                        color="white" if c in (BLUE, ORANGE, GREEN) else INK)
            left += v
    ax.axvline(0.141, color=INK, lw=1, ls=(0, (4, 3)), ymin=0.03, ymax=0.97)
    ax.text(0.141, -0.55, "guessing would give 14% in-between", fontsize=10, color=MUTED, ha="left")
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.65, 1.45)
    ax.xaxis.set_major_formatter(lambda v, _: pct(v))
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(-0.2, 1.0), ncol=4, handlelength=1.2, columnspacing=1.2)
    title(fig, "Set halfway, the model snaps to one layout",
          "Questions whose answer at the halfway position differs from both layouts")
    save(fig, "blog_3_switch.png")


def fig4():
    """Chain test: copying Cy's sentence (which carries its worked-out position) barely moves the answers."""
    s = pd.read_csv(ROOT / "results/e3/gemma3-12b_e3_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 3.9))
    fig.subplots_adjust(left=0.11, right=0.78, top=0.76, bottom=0.15)
    for col, c, lab in [("stable", GRAY, "other answers\nstay the same"), ("follow_T", BLUE, "answers put Cy\nat its new spot")]:
        ax.fill_between(s.layer, s[col + "_lo"], s[col + "_hi"], color=c, alpha=0.18, lw=0)
        ax.plot(s.layer, s[col], color=c, lw=2)
        ax.text(s.layer.iloc[-1] + 1.5, max(s[col].iloc[-1], 0.09), lab, va="center", fontsize=11, color=INK)
    peak = s.loc[s[s.layer >= 1].follow_T.idxmax()]
    ax.annotate(f"best: {pct(peak.follow_T)}", (peak.layer, peak.follow_T), (peak.layer, 0.28), ha="center",
                fontsize=10, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0, 40)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_xlabel("layer where Cy's sentence is copied (of 48)")
    title(fig, "The worked-out position isn't what the answers use",
          "Gemma 3 12B, 150 chains where only Ana's sentence changed")
    save(fig, "blog_4_not_used.png")


def fig5():
    """Part 2: the erasure removed the readable map at every layer."""
    f = pd.read_csv(ROOT / "results/p5/gemma3-12b_p5c_inlp.csv").sort_values("layer")
    fig, ax = plt.subplots(figsize=(7, 3.9))
    fig.subplots_adjust(left=0.11, right=0.80, top=0.76, bottom=0.15)
    for col, c, lab in [("r2_before", BLUE, "before\nerasing"), ("r2_after", GRAY, "after\nerasing")]:
        ax.plot(f.layer, f[col].clip(lower=0), color=c, lw=2, marker="o", ms=4)
        ax.text(f.layer.iloc[-1] + 0.5, max(f[col].iloc[-1], 0.04), lab, va="center", fontsize=11, color=INK)
    ax.set_ylim(0, 1)
    ax.set_xlim(12, 30)
    ax.set_xlabel("layer (of 48)")
    ax.set_ylabel("how well the map can be read")
    title(fig, "We deleted the map",
          "Gemma 3 12B; readability of each object's (x, y), before and after erasing it")
    save(fig, "blog_5_deleted.png")


def fig6():
    """Part 2: erasing the map costs no more accuracy than erasing random directions."""
    s = pd.read_csv(ROOT / "results/p5/gemma3-12b_p5c_summary.csv")
    s = s[s.composed]
    order = [("L14", "layer 14"), ("L21", "layer 21"), ("L28", "layer 28"), ("window 12-30", "layers\n12–30")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.74, bottom=0.17)
    w = 0.34
    for i, (cfg, lab) in enumerate(order):
        r = s[s.config == cfg].iloc[0]
        for dx, v, c, name in [(-w / 2 - 0.02, r.drop_probe, BLUE, "Erase the map"),
                               (w / 2 + 0.02, r.drop_random, GRAY, "Erase random directions (same size)")]:
            ax.bar(i + dx, max(v, 0.0015), w, color=c, label=name if i == 0 else None)
            ax.text(i + dx, max(v, 0) + 0.004, pct(v) if abs(v) >= 0.005 else "0%", ha="center", fontsize=11)
    ax.set_xticks(range(len(order)), [o[1] for o in order])
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.set_ylim(0, 0.15)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_xlabel("where it was erased")
    ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, 1.0), ncol=2)
    title(fig, "...and the answers didn't notice",
          "Accuracy lost on questions about worked-out positions (300 new scenes)")
    save(fig, "blog_6_nothing_happened.png")


def fig7():
    """Part 3: answering from the model's own (unused) map vs the model's answers, layer 21."""
    s = pd.read_csv(ROOT / "results/p9/gemma3-12b_p9_summary.csv")
    s = s[s.layer == 21].set_index("cls")
    order = [("CR", "Cy vs Bo\n(needs the chain)"), ("CM", "Cy vs Ana\n(stated directly)"),
             ("MR", "Ana vs Bo\n(stated directly)")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.74, bottom=0.17)
    w = 0.34
    for i, (c, lab) in enumerate(order):
        r = s.loc[c]
        for dx, v, col, name in [(-w / 2 - 0.02, r.acc_model, GRAY, "The model's own answers"),
                                 (w / 2 + 0.02, r.acc_map, BLUE, "Answers read off its unused map")]:
            ax.bar(i + dx, v, w, color=col, label=name if i == 0 else None)
            ax.text(i + dx, v + 0.015, pct(v), ha="center", fontsize=12,
                    fontweight="bold" if c == "CR" else "normal")
    ax.set_xticks(range(len(order)), [o[1] for o in order])
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.legend(loc="lower left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, 1.0), ncol=2)
    title(fig, "The unused map answers better",
          "Accuracy by question type; Gemma 3 12B, 300 new scenes, 5,350 questions")
    save(fig, "blog_7_map_beats_model.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
