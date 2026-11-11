"""Figure 3 of paper/draft.md: (a) DAS interchange accuracy by question type and site/model; (b) the snap curve.
Reads committed result files only (results/p4, results/free). Run: uv run python paper/make_paper_figs.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P4, FREE, OUT = ROOT / "results/p4", ROOT / "results/free", ROOT / "paper/figures"
QT = ["more_north", "more_east", "direction", "closest", "farthest", "ALL"]
QL = ["further N", "further E", "direction", "closest", "farthest", "all"]
RUNS = [("12B, list position, L18 (prereg.)", "gemma3-12b_das.csv", "#8c939d"),
        ("12B, sentence, L14 (exploratory)", "gemma3-12b_das_relation_L14.csv", "#9cc3ef"),
        ("12B, sentence, L14 (prereg. confirm)", "gemma3-12b_das_relation_L14_confirm.csv", "#2a78d6"),
        ("27B, sentence, L18 (prereg. confirm)", "gemma3-27b_das_relation_L18_confirm.csv", "#eb6834")]


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    w = 0.2
    for i, (lab, f, c) in enumerate(RUNS):
        d = pd.read_csv(P4 / f)
        d = d[d.method == "DAS k=2"].set_index("qtype").reindex(QT)
        x = np.arange(len(QT)) + (i - 1.5) * w
        a.bar(x, d.acc, w * 0.92, color=c, label=lab)
        a.errorbar(x, d.acc, yerr=[d.acc - d.acc_lo, d.acc_hi - d.acc], fmt="none", ecolor="#333", lw=0.8)
    a.axhline(0.7, color="#333", lw=0.8, ls="--")
    a.set_xticks(np.arange(len(QT)), QL)
    a.set_ylim(0, 1.22)
    a.set_yticks(np.linspace(0, 1, 6))
    a.set_ylabel("interchange accuracy (k = 2)")
    a.set_title("(a) Swapping a learned 2-D subspace; random 2-D: ≤ 0.04 everywhere", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    for model, c in [("gemma3-12b", "#2a78d6"), ("gemma3-27b", "#eb6834")]:
        s = pd.read_csv(FREE / f"{model}_snap.csv")
        b.plot(s.alpha, s.takes_moved, "-o", color=c, ms=4, label=f"{model}: moved-layout answer")
        b.plot(s.alpha, s.other, ":o", color=c, ms=3, label=f"{model}: neither layout")
    b.set_xlabel("α (subspace set to base + α·(source − base))")
    b.set_ylabel("share of questions")
    b.set_ylim(0, 1)
    b.set_title("(b) Snap curve (answers differ between the two layouts)", fontsize=10)
    b.legend(fontsize=7.5, frameon=False, loc="center right")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "paper_fig3_code.png", dpi=180)
    print(OUT / "paper_fig3_code.png")


if __name__ == "__main__":
    main()
