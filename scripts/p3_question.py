"""P3q (exploratory): when does the moved object's position arrive at the question? Copy the question tokens (or only
the final position) from the run where the object sits elsewhere, layer by layer. Same pairs as the P3 sweep; the
sentence-patch curve from the sweep is overlaid: together they show the hand-off from description to question."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.models import load
from geomap.patching import consistency, make_pairs, run_pair_question
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p3"
FIG = ROOT / "results" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--n-objects", type=int, nargs="+", default=[3])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    m = load(a.model)
    layers = sorted(set(range(0, m.n_layers + 1, a.layer_step)) | {m.n_layers})
    pairs = make_pairs(m, a.n, seed=3, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
    dfs, t0 = [], time.time()
    for p, (base, src, leaf, qs) in enumerate(pairs):
        d = run_pair_question(m, base, src, leaf, qs, layers, batch_size=a.batch_size)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[question] pair {p}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(OUT / f"{a.model}_question.parquet")
    c = consistency(df, n_boot=1000)
    c.to_csv(OUT / f"{a.model}_question_consistency.csv", index=False)
    sw = pd.read_csv(OUT / f"{a.model}_sweep_consistency.csv")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for d, lab, col in [(sw[sw.site == "relation"], "moved object's sentence (P3 sweep)", "#2b57c9"),
                        (c[c.site == "question"], "question tokens", "#c0392b"),
                        (c[c.site == "question_last"], "final position only", "#7a8b91")]:
        d = d.sort_values("layer")
        ax.plot(d.layer, d.joint_iia, marker="o", ms=3, color=col, label=lab)
        ax.fill_between(d.layer, d.jiia_lo, d.jiia_hi, color=col, alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer patched")
    ax.set_ylabel("joint agreement")
    ax.set_title(f"Hand-off: from the description to the question ({a.model})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p3_handoff.png", dpi=170)
    plt.close(fig)

    piv = c.pivot(index="layer", columns="site", values="joint_iia")
    rel = sw[sw.site == "relation"].set_index("layer").joint_iia
    lines = [f"Model **{a.model}**, {len(pairs)} pairs (the P3 sweep's pairs). Exploratory. Joint agreement after copying "
             "the question tokens (or only the final position) from the run where the object sits elsewhere. The question "
             "text is identical, so the copied states differ only through what they read from the description.", "",
             "| layer | sentence (sweep) | question tokens | final position |", "|---|---|---|---|"]
    lines += [f"| {l} | {rel.get(l, float('nan')):.2f} | {piv.loc[l, 'question']:.2f} | {piv.loc[l, 'question_last']:.2f} |"
              for l in piv.index]
    append_entry(f"P3q hand-off to the question ({a.model}) — exploratory", "\n".join(lines), [FIG / "p3_handoff.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
