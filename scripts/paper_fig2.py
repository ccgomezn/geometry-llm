"""Paper figure 2: (a) probes by layer, (b) patching by layer and site, (c) DAS vs random subspace.

Each panel is drawn only if its result files exist."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.report import ROOT

R = ROOT / "results"
UNREAD, READ = "#b4761d", "#2b57c9"  # list position (decodable, unused) / relation sentence (used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    a = ap.parse_args()
    M = a.model
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axs[0]
    real = pd.read_csv(R / "p2" / f"{M}_real.csv")
    ax.fill_between(real.layer, real.r2_lo, real.r2_hi, alpha=0.25, color=UNREAD)
    ax.plot(real.layer, real.r2, color=UNREAD, label="probe at list position (all objects)")
    if "r2_hop2p" in real:
        ax.plot(real.layer, real.r2_hop2p, color=UNREAD, ls="--", lw=1, label="same, objects two steps away")
    for f, lab, c in [("shuffled", "shuffled labels", "gray"), ("rand", "random-init model", "k")]:
        p = R / "p2" / f"{M}_{f}.csv"
        if p.exists():
            d = pd.read_csv(p)
            ax.plot(d.layer, d.r2, color=c, lw=0.8, ls=":", label=lab)
    st = R / "p2" / f"{M}_stated_relation.csv"
    if st.exists():
        v = pd.read_csv(st).r2_hop2p.item()
        ax.axhline(v, color="C3", lw=0.8, ls="-.", label=f"stated-sentence baseline, 2+ steps ({v:.2f})")
    ax.set_ylim(-0.05, 1)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out R² of (x, y)")
    ax.set_title("(a) Coordinates are decodable from layer 12 on", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    ax = axs[1]
    p3 = R / "p3" / f"{M}_sweep_consistency.csv"
    if p3.exists():
        c = pd.read_csv(p3)
        for s, lab, col in [("list", "object's list position", UNREAD), ("relation", "its relation sentence", READ),
                            ("eol", "end of object list", "#7a8b91"), ("all", "whole description (control)", "#b8c2c6")]:
            d = c[c.site == s].sort_values("layer")
            ax.plot(d.layer, d.joint_iia, marker="o", ms=3, color=col, label=lab)
            if "jiia_lo" in d:
                ax.fill_between(d.layer, d.jiia_lo, d.jiia_hi, color=col, alpha=0.15)
        ax.legend(fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer patched")
    ax.set_ylabel("joint agreement (all answers follow)")
    ax.set_title("(b) Moving one object inside the model", fontsize=10)

    ax = axs[2]
    p4 = R / "p4" / f"{M}_das.csv"
    if p4.exists():
        d = pd.read_csv(p4)
        qts = [q for q in d.qtype.unique() if q != "ALL"] + ["ALL"]
        meths = list(dict.fromkeys(d.method))
        w = 0.8 / len(meths)
        for i, meth in enumerate(meths):
            dm = d[d.method == meth].set_index("qtype").reindex(qts)
            ax.bar([j + i * w for j in range(len(qts))], dm.acc, w, label=meth,
                   yerr=[dm.acc - dm.acc_lo, dm.acc_hi - dm.acc], capsize=2)
        ax.axhline(0.7, color="g", ls="-.", lw=0.8)
        ax.set_xticks([j + 0.4 - w / 2 for j in range(len(qts))], [q.replace("_", " ") for q in qts], rotation=25,
                      fontsize=8)
        ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("interchange accuracy")
    ax.set_title("(c) Swapping only a learned subspace", fontsize=10)

    fig.tight_layout()
    out = R / "figures" / "paper_fig2_mechanism.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
