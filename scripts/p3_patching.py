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
    layers = sorted(set(range(0, m.n_layers + 1, a.layer_step)) | {m.n_layers})

    if "A" in a.stage:
        pairs = make_pairs(m, a.n_sweep, seed=3, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
        dfA = run(m, pairs, SITES, layers, a.batch_size, "sweep")
        dfA.to_parquet(OUT / f"{a.model}_sweep.parquet")
    else:
        dfA = pd.read_parquet(OUT / f"{a.model}_sweep.parquet")
    cA = consistency(dfA, n_boot=200)
    cA.to_csv(OUT / f"{a.model}_sweep_consistency.csv", index=False)

    piv = cA.pivot(index="site", columns="layer", values="joint_iia").reindex(SITES)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(SITES)), ["object's list position", "its relation sentence", "end of list", "all prefix (control)"])
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_xlabel("layer")
    fig.colorbar(im, label="joint agreement with the model's\nanswers when the object is moved")
    ax.set_title(f"P3 sweep: does moving one object move every answer? ({a.model}, {a.n_sweep} pairs)")
    fig.tight_layout()
    fig.savefig(FIG / "p3_sweep.png", dpi=150)
    plt.close(fig)

    best2 = {s: sorted(cA[(cA.site == s) & (cA.layer > 0)].nlargest(2, "joint_iia").layer.tolist()) for s in SITES[:3]}

    if "B" in a.stage:
        conf_layers = sorted({l for v in best2.values() for l in v})
        pairsB = make_pairs(m, a.n_confirm, seed=4, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
        dfB = run(m, pairsB, SITES, conf_layers, a.batch_size, "confirm")
        dfB.to_parquet(OUT / f"{a.model}_confirm.parquet")
        cB = consistency(dfB, n_boot=1000)
        okB = clean_pairs(dfB)
        cBc = consistency(dfB[dfB.pair.isin(okB)], n_boot=1000)
        cB.to_csv(OUT / f"{a.model}_confirm_consistency.csv", index=False)
    else:
        dfB, cB, cBc, okB = dfA, cA, consistency(dfA[dfA.pair.isin(clean_pairs(dfA))], 200), clean_pairs(dfA)

    sites_only = cB[cB.site != "all"]
    top = sites_only.loc[sites_only.joint_iia.idxmax()]
    qcols = [c for c in cB.columns if c.startswith("iia_") and c != "iia_rate"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([c[4:] for c in qcols], [top[c] for c in qcols])
    ax.set_ylim(0, 1)
    ax.set_ylabel("answers that follow the moved object")
    ax.set_title(f"P3: per-question agreement at {top.site}, layer {int(top.layer)}")
    fig.tight_layout()
    fig.savefig(FIG / "p3_flip_by_qtype.png", dpi=150)
    plt.close(fig)

    def line(r):
        return (f"| {r.site} | {int(r.layer)} | {r.joint_iia:.2f} [{r.jiia_lo:.2f}, {r.jiia_hi:.2f}] | {r.iia_rate:.2f} | "
                f"{r.joint_consistency:.2f} | {int(r.n_pairs_iia)} |")

    # preregistered criteria (clarified before P3 data): headline = joint agreement with the model's own answers
    ro = cB[cB.site.isin(["list", "eol"]) & (cB.layer > 0)]  # layer 0: patching the relation sentence = rerunning the source
    readout_best = ro.joint_iia.max()
    ro_layer = int(ro.loc[ro.joint_iia.idxmax(), "layer"])
    rel_at = cB[(cB.site == "relation") & (cB.layer == ro_layer)].joint_iia
    rel_at = float(rel_at.iloc[0]) if len(rel_at) else float("nan")
    ceiling = cB[(cB.site == "all") & (cB.layer == min(cB.layer))].joint_iia
    lines = [f"Model **{a.model}**. Each pair differs in one sentence that moves one leaf object; the source's activations "
             f"are copied into the base run at one site and layer, then every question is asked.",
             "",
             "**Joint agreement** = share of pairs where, after patching, *every* answer equals what the model says when "
             "the object really is elsewhere (pairs where at least one of those answers differs from the base run). "
             "Geometric joint consistency (all answers match the true layout) is shown for reference.",
             "",
             f"- Positive control (copy the whole prefix, earliest layer): {float(ceiling.iloc[0]) if len(ceiling) else float('nan'):.2f} (should be 1.00)",
             f"- Best layers in the sweep: " + ", ".join(f"{s} → {v}" for s, v in best2.items()),
             "",
             "Confirmation run:", "",
             "| site | layer | joint agreement [95% CI] | per-question agreement | geometric joint | pairs |",
             "|---|---|---|---|---|---|"] + [line(r) for r in cB.itertuples()] + [
             "", f"Pairs the model answers fully correctly when unpatched ({len(okB)} pairs):", "",
             "| site | layer | joint agreement [95% CI] | per-question agreement | geometric joint | pairs |",
             "|---|---|---|---|---|---|"] + [line(r) for r in cBc.itertuples() if not np.isnan(r.joint_iia)] + [
             "", "Against the preregistered P3 criteria (as clarified):",
             f"- H-map: joint agreement ≥ 0.6 at the object's list position or the end of list → best {readout_best:.2f} "
             f"(layer {ro_layer}) → {'met' if readout_best >= 0.6 else 'not met'}",
             f"- H-rel: < 0.2 at the readout positions, and at that layer the relation sentence carries more → "
             f"readout {readout_best:.2f}, relation sentence at layer {ro_layer}: {rel_at:.2f} → "
             f"{'met' if readout_best < 0.2 and rel_at > readout_best else 'not met'}"]
    append_entry(f"P3 patching ({a.model})", "\n".join(lines), [FIG / "p3_sweep.png", FIG / "p3_flip_by_qtype.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
