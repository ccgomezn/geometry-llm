"""P3m (exploratory): does the single token where the question names the moved object carry the position the
answer uses? Copy only that token from the run where the object sits elsewhere, layer by layer. Same pairs as the
P3 sweep; only questions that name the moved object are scored."""

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
        qs = [q for q in qs if leaf in q.args]
        if not qs:
            continue
        d = run_pair_question(m, base, src, leaf, qs, layers, sites=("mention",), batch_size=a.batch_size)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[mention] pair {p}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(OUT / f"{a.model}_mention.parquet")
    c = consistency(df, n_boot=1000)
    c.to_csv(OUT / f"{a.model}_mention_consistency.csv", index=False)
    c = c[c.site == "mention"].sort_values("layer")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(c.layer, c.joint_iia, marker="o", ms=3, color="#2b57c9", label="moved object's name in the question")
    ax.fill_between(c.layer, c.jiia_lo, c.jiia_hi, color="#2b57c9", alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer patched")
    ax.set_ylabel("joint agreement")
    ax.set_title(f"Copying one token: the object's name in the question ({a.model})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p3_mention.png", dpi=170)
    plt.close(fig)

    n_pairs = df.pair.nunique()
    lines = [f"Model **{a.model}**, {n_pairs} of the P3 sweep's pairs (questions that name the moved object). "
             "Exploratory. Joint agreement after copying only the moved object's name token in the question from the "
             "run where it sits elsewhere.", "", "| layer | joint agreement [95% CI] |", "|---|---|"]
    lines += [f"| {r.layer} | {r.joint_iia:.2f} [{r.jiia_lo:.2f}, {r.jiia_hi:.2f}] |" for r in c.itertuples()]
    append_entry(f"P3m one-token patch at the question's mention ({a.model}) — exploratory", "\n".join(lines),
                 [FIG / "p3_mention.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
