"""E2 robustness (deviation, dated in docs/preregistration_e2.md): restrict C-questions to those whose answers
differ across base, source and hybrid, so that "follow" cannot happen by accident when the patch does nothing."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.composition import summarize, verdict
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "e2"
FIG = ROOT / "results" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(OUT / f"{a.model}_e2.parquet")
    s = summarize(df, n_boot=1000, strict=True)
    s.to_csv(OUT / f"{a.model}_e2_strict_summary.csv", index=False)
    v, at = verdict(s)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for col, lab, c in [("follow_M", "M's answers follow M (patch took effect)", "#7a8b91"),
                        ("follow_C", "C's answers follow M (C rebuilt from M's sentence)", "#2b57c9"),
                        ("stay_C", "C's answers stay behind (C's position already composed)", "#b4761d")]:
        ax.plot(s.layer, s[col], marker="o", ms=3, color=c, label=lab)
        ax.fill_between(s.layer, s[col + "_lo"], s[col + "_hi"], color=c, alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer at which M's sentence is patched")
    ax.set_ylabel("share of answers")
    ax.set_title(f"E2, strict question set: when does C stop following M? ({a.model})", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "e2_strict.png", dpi=170)
    plt.close(fig)

    show = s[s.layer.isin([0, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])]
    lines = [f"Model **{a.model}**. Deviation from the E2 preregistration (dated there): C-questions restricted to those "
             "whose model answers differ across base, source and hybrid. Under the preregistered definition, a C-question "
             "whose answer the move leaves unchanged counts as \"follow\" even when the patch does nothing.", "",
             "| layer | M follows | C follows | C stays | C-questions |", "|---|---|---|---|---|"]
    lines += [f"| {r.layer} | {r.follow_M:.2f} | {r.follow_C:.2f} [{r.follow_C_lo:.2f}, {r.follow_C_hi:.2f}] | "
              f"{r.stay_C:.2f} [{r.stay_C_lo:.2f}, {r.stay_C_hi:.2f}] | {r.n_C} |" for r in show.itertuples()]
    lines += ["", f"Decision rule applied to the strict set: **{v}**" + (f" (layer {at})" if at is not None else "")]
    append_entry(f"E2 strict question set ({a.model}) — deviation", "\n".join(lines), [FIG / "e2_strict.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
