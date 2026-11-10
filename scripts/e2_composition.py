"""E2 (preregistered in docs/preregistration_e2.md): move the middle object of a chain inside the model; does the
end of the chain follow? Layer sweep of patching M's sentence, with model-relative follow/stay targets."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.composition import make_chain_pairs, run_chain_pair, summarize, verdict
from geomap.models import load
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "e2"
FIG = ROOT / "results" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = load(a.model)
    layers = sorted(set(range(0, m.n_layers + 1, a.layer_step)) | {m.n_layers})
    pairs = make_chain_pairs(m, a.n, seed=11, exclude_qtypes=tuple(a.exclude))
    dfs, t0 = [], time.time()
    for p, cp in enumerate(pairs):
        d = run_chain_pair(m, cp, layers, batch_size=a.batch_size)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[e2] pair {p}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(OUT / f"{a.model}_e2.parquet")
    s = summarize(df, n_boot=1000)
    s.to_csv(OUT / f"{a.model}_e2_summary.csv", index=False)
    v, at = verdict(s)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for col, lab, c in [("follow_M", "M's answers follow M (patch took effect)", "#7a8b91"),
                        ("follow_C", "C's answers follow M (C rebuilt from sentences)", "#2b57c9"),
                        ("stay_C", "C's answers stay put (C's position already composed)", "#b4761d")]:
        ax.plot(s.layer, s[col], marker="o", ms=3, color=c, label=lab)
        ax.fill_between(s.layer, s[col + "_lo"], s[col + "_hi"], color=c, alpha=0.15)
    ax.axhline(0.8, color="#7a8b91", lw=0.6, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer at which M's sentence is patched")
    ax.set_ylabel("share of answers")
    ax.set_title(f"E2: move M in R → M → C inside the model; does C follow? ({a.model}, {len(pairs)} pairs)", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "e2_composition.png", dpi=170)
    plt.close(fig)

    show = s[s.layer.isin([0, 4, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32])]
    lines = [f"Model **{a.model}**, {len(pairs)} chain pairs (R → M → C, M's sentence first). M's sentence is copied from a "
             "scene where M (and therefore C) sits elsewhere. *follow* = answer equals the model's answer on that scene; "
             "*stay* = answer equals the model's answer on a scene where M moved but C kept its old position.", "",
             "| layer | M follows | C follows | C stays | M-questions | C-questions |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.layer} | {r.follow_M:.2f} [{r.follow_M_lo:.2f}, {r.follow_M_hi:.2f}] | "
              f"{r.follow_C:.2f} [{r.follow_C_lo:.2f}, {r.follow_C_hi:.2f}] | {r.stay_C:.2f} [{r.stay_C_lo:.2f}, "
              f"{r.stay_C_hi:.2f}] | {r.n_M} | {r.n_C} |" for r in show.itertuples()]
    lines += ["", f"Preregistered verdict: **{v}**" + (f" (C's position composed by layer {at})" if at is not None else "")]
    append_entry(f"E2 composition timing ({a.model})", "\n".join(lines), [FIG / "e2_composition.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
