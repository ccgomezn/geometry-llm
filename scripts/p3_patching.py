"""P3: move one object inside the model (activation patching) and check that every answer follows."""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geomap.models import load
from geomap.patching import consistency, make_pairs, run_pair
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p3"
FIG = ROOT / "results" / "figures"
SITES = ["list", "relation", "eol", "all"]


def run(m, pairs, sites, layers, bs, label):
    dfs, t0 = [], time.time()
    for p, (base, src, leaf, qs) in enumerate(pairs):
        d = run_pair(m, base, src, leaf, qs, sites, layers, batch_size=bs)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[{label}] pair {p}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    return pd.concat(dfs, ignore_index=True)


def clean_pairs(df):
    """Pairs where the model answers every question correctly on both the clean base and clean source."""
    ok = []
    for p, g in df.groupby("pair"):
        nb, ns = g[g.site == "none"], g[g.site == "source"]
        if (nb.model_ans == nb.base_ans).all() and (ns.model_ans == ns.src_ans).all():
            ok.append(p)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-sweep", type=int, default=100)
    ap.add_argument("--n-confirm", type=int, default=400)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--n-objects", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--exclude", nargs="*", default=[], help="question types excluded (preregistered: at chance)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = load(a.model)
    layers = list(range(0, m.n_layers + 1, a.layer_step))

    if "A" in a.stage:
        pairs = make_pairs(m, a.n_sweep, seed=3, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
        dfA = run(m, pairs, SITES, layers, a.batch_size, "sweep")
        dfA.to_parquet(OUT / f"{a.model}_sweep.parquet")
    else:
        dfA = pd.read_parquet(OUT / f"{a.model}_sweep.parquet")
    cA = consistency(dfA, n_boot=200)
    cA.to_csv(OUT / f"{a.model}_sweep_consistency.csv", index=False)

    piv = cA.pivot(index="site", columns="layer", values="joint_consistency").reindex(SITES)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(SITES)), ["object's list position", "its relation sentence", "end of list", "all prefix (control)"])
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_xlabel("layer")
    fig.colorbar(im, label="joint consistency")
    ax.set_title(f"P3 sweep: does moving one object move every answer? ({a.model}, {a.n_sweep} pairs)")
    fig.tight_layout()
    fig.savefig(FIG / "p3_sweep.png", dpi=150)
    plt.close(fig)

    best = {s: int(cA[cA.site == s].nlargest(2, "joint_consistency").layer.iloc[0]) for s in SITES[:3]}
    best2 = {s: sorted(cA[cA.site == s].nlargest(2, "joint_consistency").layer.tolist()) for s in SITES[:3]}

    if "B" in a.stage:
        conf_layers = sorted({l for v in best2.values() for l in v})
        pairsB = make_pairs(m, a.n_confirm, seed=4, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
        dfB = run(m, pairsB, SITES[:3], conf_layers, a.batch_size, "confirm")
        dfB.to_parquet(OUT / f"{a.model}_confirm.parquet")
        cB = consistency(dfB, n_boot=1000)
        okB = clean_pairs(dfB)
        cBc = consistency(dfB[dfB.pair.isin(okB)], n_boot=1000)
        cB.to_csv(OUT / f"{a.model}_confirm_consistency.csv", index=False)
    else:
        dfB, cB, cBc, okB = dfA, cA, consistency(dfA[dfA.pair.isin(clean_pairs(dfA))], 200), clean_pairs(dfA)

    qcols = [c for c in cB.columns if c.startswith("flip_") and c not in ("flip_rate", "flip_lo", "flip_hi")]
    top = cB.loc[cB.joint_consistency.idxmax()]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([c[5:] for c in qcols], [top[c] for c in qcols])
    ax.set_ylim(0, 1)
    ax.set_ylabel("answers that follow the moved object")
    ax.set_title(f"P3: flip rate by question type at {top.site}, layer {int(top.layer)}")
    fig.tight_layout()
    fig.savefig(FIG / "p3_flip_by_qtype.png", dpi=150)
    plt.close(fig)

    def line(r):
        return (f"| {r.site} | {int(r.layer)} | {r.joint_consistency:.2f} [{r.jc_lo:.2f}, {r.jc_hi:.2f}] | "
                f"{r.flip_rate:.2f} | {r.control_stability:.2f} | {int(r.n_pairs)} |")

    allA = cA[cA.site == "all"].joint_consistency
    readout_best = cB[cB.site.isin(["list", "eol"])].joint_consistency.max()
    rel_best = cB[cB.site == "relation"].joint_consistency.max()
    lines = [f"Model **{a.model}**. Each pair differs in one sentence that moves one leaf object; the source's activations "
             f"are copied into the base run at one site and layer, then ~10 questions are asked.",
             "",
             f"- Positive control (copy the whole prefix): joint consistency peaks at {allA.max():.2f} "
             f"(layer 0 should be ≈ the clean source run's accuracy).",
             f"- Best layers in the sweep: " + ", ".join(f"{s} → {v}" for s, v in best2.items()),
             "",
             "Confirmation run (all pairs):", "",
             "| site | layer | joint consistency [95% CI] | flip rate | control stability | pairs |",
             "|---|---|---|---|---|---|"] + [line(r) for r in cB.itertuples()] + [
             "", f"Only pairs the model answers fully correctly when unpatched ({len(okB)} pairs):", "",
             "| site | layer | joint consistency [95% CI] | flip rate | control stability | pairs |",
             "|---|---|---|---|---|---|"] + [line(r) for r in cBc.itertuples() if not np.isnan(r.joint_consistency)] + [
             "", "Against the preregistered P3 criteria:",
             f"- H-map needs joint consistency ≥ 0.6 at the object's list position or the end of list: best {readout_best:.2f} → "
             f"{'met' if readout_best >= 0.6 else 'not met'}",
             f"- H-rel needs < 0.2 at the readout positions, with the effect concentrated on the relation tokens: "
             f"readout {readout_best:.2f}, relation sentence {rel_best:.2f} → "
             f"{'met' if readout_best < 0.2 and rel_best > readout_best else 'not met'}"]
    append_entry(f"P3 patching ({a.model})", "\n".join(lines), [FIG / "p3_sweep.png", FIG / "p3_flip_by_qtype.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
