"""Paper figure 1: behaviour across models (accuracy without written reasoning, and error signatures)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geomap.analysis import bootstrap_ci, error_signature
from geomap.report import ROOT

ORDER = ["qwen3-1.7b", "llama3.2-3b", "qwen3-4b", "gemma3-4b", "qwen3-8b", "gemma3-12b", "qwen3-14b", "gemma3-27b"]
LABEL = {"qwen3-1.7b": "Qwen3 1.7B", "llama3.2-3b": "Llama3.2 3B", "qwen3-4b": "Qwen3 4B", "gemma3-4b": "Gemma3 4B",
         "qwen3-8b": "Qwen3 8B", "gemma3-12b": "Gemma3 12B", "qwen3-14b": "Qwen3 14B", "gemma3-27b": "Gemma3 27B"}
COLOR = {"qwen3": "#4C72B0", "llama3": "#55A868", "gemma3": "#C44E52"}


def main():
    P1 = ROOT / "results" / "p1"
    res = {k: pd.read_parquet(P1 / f"{k}.parquet") for k in ORDER if (P1 / f"{k}.parquet").exists()}
    rows = []
    for k, d in res.items():
        m = d[d["set"] == "main"]
        a3 = bootstrap_ci(m[m.n_objects == 3].correct.astype(float), groups=m[m.n_objects == 3].scene_id, n_boot=300)
        a4 = bootstrap_ci(m[m.n_objects == 4].correct.astype(float), groups=m[m.n_objects == 4].scene_id, n_boot=300)
        sig = error_signature(m, n_boot=100)
        rows.append(dict(model=k, acc3=a3, acc4=a4, hop=sig["hop"], margin=sig["log_margin"],
                         mass=float(np.exp(m["mass"].median())) if "mass" in m else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "p1_models_summary.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(df))
    for off, col, lab in [(-0.18, "acc3", "3 objects"), (0.18, "acc4", "4 objects")]:
        v = np.array([c[0] for c in df[col]])
        err = np.array([[c[0] - c[1], c[2] - c[0]] for c in df[col]]).T
        ax1.bar(x + off, v, 0.36, yerr=err, capsize=2, label=lab, alpha=0.9 if col == "acc3" else 0.55,
                color=[COLOR[k.split("-")[0].replace("3.2", "3")] for k in df.model])
    ax1.axhline(0.75, color="k", ls="--", lw=1)
    ax1.text(-0.45, 0.765, "gate 0.75", ha="left", fontsize=8)
    ax1.set_xticks(x, [LABEL[k] for k in df.model], rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("accuracy, no written reasoning")
    ax1.set_ylim(0, 1)
    ax1.set_title("Solving text-only layouts silently (light = 4 objects)", fontsize=10)
    for _, r in df.iterrows():
        fam = r.model.split("-")[0].replace("3.2", "3")
        ax2.errorbar(-r.hop[0], r.margin[0], xerr=[[r.hop[0] - r.hop[1]], [r.hop[2] - r.hop[0]]],
                     yerr=[[r.margin[0] - r.margin[1]], [r.margin[2] - r.margin[0]]], fmt="o", color=COLOR[fam], ms=5)
        ax2.annotate(LABEL[r.model], (-r.hop[0], r.margin[0]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = max(0.4, float(max(-df.hop.apply(lambda c: c[0]).min(), df.margin.apply(lambda c: c[2]).max())) + 0.05)
    ax2.plot([0, lim], [0, lim], color="gray", lw=0.8, ls=":")
    ax2.set_xlabel("errors driven by chain length\n(−coefficient of hop depth)")
    ax2.set_ylabel("errors driven by near-ties\n(coefficient of log margin)")
    ax2.set_title("Error signatures", fontsize=10)
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "paper_fig1_behaviour.png"
    fig.savefig(out, dpi=200)
    print(out)
    print(df[["model", "mass"]].to_string())


if __name__ == "__main__":
    main()
