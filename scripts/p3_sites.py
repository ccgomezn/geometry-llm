"""P3c (exploratory): narrow the used site. Copy only the LAST token of the moved object's sentence, or only the
rest of that sentence, and compare with copying the whole sentence. Uses the P3 sweep's pairs."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.models import load
from geomap.patching import consistency, make_pairs, run_pair
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p3"
FIG = ROOT / "results" / "figures"
LABEL = {"relation": "whole sentence", "sent_end": "last token of the sentence only", "sent_rest": "rest of the sentence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sites", nargs="+", default=["relation", "sent_end", "sent_rest"])
    ap.add_argument("--max-layer", type=int, default=32)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--n-objects", type=int, nargs="+", default=[3])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    m = load(a.model)
    layers = list(range(0, a.max_layer + 1, a.layer_step))
    pairs = make_pairs(m, a.n, seed=3, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
    dfs, t0 = [], time.time()
    for p, (base, src, leaf, qs) in enumerate(pairs):
        d = run_pair(m, base, src, leaf, qs, a.sites, layers, batch_size=a.batch_size)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[sites] pair {p}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(OUT / f"{a.model}_sites.parquet")
    c = consistency(df, n_boot=1000)
    c.to_csv(OUT / f"{a.model}_sites_consistency.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for s, col in zip(a.sites, ["#2b57c9", "#c0392b", "#7a8b91"]):
        d = c[c.site == s].sort_values("layer")
        ax.plot(d.layer, d.joint_iia, marker="o", ms=3, color=col, label=LABEL.get(s, s))
        ax.fill_between(d.layer, d.jiia_lo, d.jiia_hi, color=col, alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer patched")
    ax.set_ylabel("joint agreement")
    ax.set_title(f"Which part of the moved object's sentence carries its position? ({a.model})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p3_sites.png", dpi=170)
    plt.close(fig)

    piv = c.pivot(index="layer", columns="site", values="joint_iia")
    lines = [f"Model **{a.model}**, {len(pairs)} pairs (the P3 sweep's pairs). Exploratory. Joint agreement (all answers "
             "equal the model's answers on the moved scene) after copying one part of the moved object's sentence.", "",
             "| layer | " + " | ".join(LABEL.get(s, s) for s in a.sites) + " |", "|---|" + "---|" * len(a.sites)]
    lines += [f"| {l} | " + " | ".join(f"{piv.loc[l, s]:.2f}" for s in a.sites) + " |" for l in piv.index]
    append_entry(f"P3c which tokens carry the position ({a.model}) — exploratory", "\n".join(lines), [FIG / "p3_sites.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
