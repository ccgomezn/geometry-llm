"""P4i (exploratory): does the learned 2-D code behave like a coordinate? Set it part of the way between the
base and the moved scene (alpha), i.e. ask for a position that no sentence states, and score the answers against
the layout with the object at that interpolated position.

Key subset: questions whose answer at the interpolated position differs from the answer in BOTH real layouts.
A model that only switches between the two stated scenes scores 0 there."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from geomap.das import SubspaceIntervention, eval_interp
from geomap.models import load
from geomap.patching import make_pairs
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p4"
FIG = ROOT / "results" / "figures"


def rate(d, col, n_boot=1000, seed=0):
    if not len(d):
        return np.nan, np.nan, np.nan
    per = d.groupby("pair")[col].agg(["sum", "count"])
    s, c = per["sum"].to_numpy(float), per["count"].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(per), size=(n_boot, len(per)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), *np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--site", default="relation")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--n-objects", type=int, nargs="+", default=[3])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=6, help="test-pair seed of the DAS run whose subspace is used")
    ap.add_argument("--label", default=None, help="label of the DAS run (see p4_das.py --label)")
    a = ap.parse_args()
    stem = f"{a.site}_L{a.layer}" + (f"_{a.label}" if a.label else "")
    m = load(a.model)
    iv = SubspaceIntervention(m.d_model, 2)
    iv.load_state_dict(torch.load(OUT / f"{a.model}_L{a.layer}_{a.site}" + (f"_{a.label}" if a.label else "") + "_k2.pt"))
    # the DAS test pairs (same seed): never used for training the subspace
    pairs = make_pairs(m, a.n, seed=a.seed, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
    df = eval_interp(m, iv, pairs, a.layer, a.site, alphas=a.alphas)
    df.to_parquet(OUT / f"{a.model}_interp_{stem}.parquet")
    df["hit"] = df.pred == df.geo
    df["is_base"] = df.pred == df.geo_base
    df["is_src"] = df.pred == df.geo_src
    novel = df.margin_ok & (df.geo != df.geo_base) & (df.geo != df.geo_src)

    rows = []
    for al, g in df.groupby("alpha"):
        ok, nv = g[g.margin_ok], g[novel.loc[g.index]]
        r = dict(alpha=al, n_ok=len(ok), n_novel=len(nv))
        for name, d, col in [("agree", ok, "hit"), ("novel", nv, "hit"), ("novel_base", nv, "is_base"),
                             ("novel_src", nv, "is_src")]:
            r[name], r[name + "_lo"], r[name + "_hi"] = rate(d, col)
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / f"{a.model}_interp_{stem}.csv", index=False)
    clean_acc = (df[df.alpha == df.alpha.min()].model_base == df[df.alpha == df.alpha.min()].geo_base).mean()
    n_cands = {"direction": 8}
    nv_all = df[novel]
    chance = np.mean([1 / n_cands.get(q, 2) for q in nv_all.qtype]) if len(nv_all) else np.nan

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for col, lab, c in [("agree", "answers match the object at the interpolated position", "#2b57c9"),
                        ("novel", "…on questions where that answer differs from both stated layouts", "#b4761d"),
                        ("novel_base", "…answer of the original layout instead", "#7a8b91"),
                        ("novel_src", "…answer of the moved layout instead", "#a8432f")]:
        ax.plot(s.alpha, s[col], marker="o", ms=4, color=c, label=lab)
        ax.fill_between(s.alpha, s[col + "_lo"], s[col + "_hi"], color=c, alpha=0.12)
    ax.axhline(chance, color="gray", ls=":", lw=0.8, label=f"chance on the novel subset ({chance:.2f})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("α: 2-D code set to base + α·(moved − base)")
    ax.set_ylabel("share of questions")
    ax.set_title(f"Setting the 2-D code to unstated positions ({a.model}, {a.site} site, layer {a.layer})", fontsize=10)
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / f"p4_interp_{stem}.png", dpi=170)
    plt.close(fig)

    lines = [f"Model **{a.model}**, the 2-D subspace learned by DAS at the {a.site} site, layer {a.layer}; {len(pairs)} "
             "of its held-out test pairs. Exploratory. The subspace is set to base + α·(moved − base); answers are "
             "scored against the layout with the object at the matching interpolated position (questions with "
             "margin ≥ 1 unit / 10° there). *Novel*: questions whose answer at that position differs from the answer "
             f"in both stated layouts (chance {chance:.2f}). Clean model accuracy on the base layouts: {clean_acc:.2f}.", "",
             "| α | agrees with interpolated layout | novel: interpolated answer | novel: original-layout answer | novel: moved-layout answer | n novel |",
             "|---|---|---|---|---|---|"]
    f = lambda r, c: f"{r[c]:.2f} [{r[c + '_lo']:.2f}, {r[c + '_hi']:.2f}]" if r["n_novel" if c != "agree" else "n_ok"] else "–"
    lines += [f"| {r.alpha:g} | {f(r, 'agree')} | {f(r, 'novel')} | {f(r, 'novel_base')} | {f(r, 'novel_src')} | {int(r.n_novel)} |"
              for _, r in s.iterrows()]
    append_entry(f"P4i 2-D code set to unstated positions ({a.model}) — "
                 + (f"preregistered '{a.label}' run" if a.label else "exploratory"), "\n".join(lines),
                 [FIG / f"p4_interp_{stem}.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
