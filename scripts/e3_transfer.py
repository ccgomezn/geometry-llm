"""E3 (preregistered in docs/preregistration_e3.md): in R -> M -> C, change only M's sentence, then copy C's own
(word-for-word identical) sentence from that run into the original run, layer by layer. If the answers then treat
C as sitting at its moved position with M left in place, C's sentence carries C's composed position, and it is used."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.models import load
from geomap.report import ROOT, append_entry
from geomap.transfer import make_transfer_pairs, run_transfer_pair, summarize_transfer, verdict_transfer

OUT = ROOT / "results" / "e3"
FIG = ROOT / "results" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-layer", type=int, default=40)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = load(a.model)
    layers = list(range(0, min(a.max_layer, m.n_layers) + 1, a.layer_step))
    pairs = make_transfer_pairs(m, a.n, seed=13, exclude_qtypes=tuple(a.exclude))
    dfs, t0 = [], time.time()
    for p, tp in enumerate(pairs):
        d = run_transfer_pair(m, tp, layers, batch_size=a.batch_size)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[e3] pair {p}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(OUT / f"{a.model}_e3.parquet")
    s = summarize_transfer(df, n_boot=1000)
    s.to_csv(OUT / f"{a.model}_e3_summary.csv", index=False)
    v, at = verdict_transfer(s)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for col, lab, c in [("follow_T", "answers treat C as moved, M in place (C's composed position transferred)", "#2b57c9"),
                        ("follow_S", "answers treat M as moved too (should stay low)", "#a8432f"),
                        ("stable", "control answers unchanged", "#7a8b91")]:
        ax.plot(s.layer, s[col], marker="o", ms=3, color=c, label=lab)
        ax.fill_between(s.layer, s[col + "_lo"], s[col + "_hi"], color=c, alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer at which C's (identical) sentence is copied")
    ax.set_ylabel("share of answers")
    ax.set_title(f"E3: does C's unchanged sentence carry C's computed position? ({a.model}, {len(pairs)} pairs)", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "e3_transfer.png", dpi=170)
    plt.close(fig)

    lines = [f"Model **{a.model}**, {len(pairs)} chain pairs (R → M → C). Only M's sentence differs between the two runs; "
             "C's sentence is copied (same words) from the run where M, and with it C, sits elsewhere. *Follows target* = "
             "answer equals the model's answer on a scene with C at its moved position and M unmoved.", "",
             "| layer | follows target | follows source (M moved too) | controls stable | target Qs | control Qs |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r.layer} | {r.follow_T:.2f} [{r.follow_T_lo:.2f}, {r.follow_T_hi:.2f}] | {r.follow_S:.2f} | "
              f"{r.stable:.2f} [{r.stable_lo:.2f}, {r.stable_hi:.2f}] | {r.n_T} | {r.n_ctrl} |" for r in s.itertuples()]
    lines += ["", f"Preregistered verdict: **{v}**" + (f" (from layer {at})" if at is not None else "")]
    append_entry(f"E3 composed-position transfer ({a.model})", "\n".join(lines), [FIG / "e3_transfer.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
