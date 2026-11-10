"""P3 necessity (exploratory, not preregistered): are the readout positions needed, even if not sufficient?

Copies the source description into the base run at every position EXCEPT the objects' list positions and the
end of the list, and compares with copying everything. If the answers depend on the list positions, leaving them
at their base values should pull joint agreement down. Uses the P3 sweep's pairs (same seed and arguments).
Also writes effect sizes (mean |change in log-probability of the base answer|) per site and layer from the sweep."""

import argparse

import pandas as pd

from geomap.models import load
from geomap.patching import consistency, make_pairs, run_pair
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p3"


def effect_sizes(model):
    d = pd.read_parquet(OUT / f"{model}_sweep.parquet")
    base = d[d.site == "none"].set_index(["pair", "qid"]).lp_base_ans
    rows = []
    for (s, l), g in d[~d.site.isin(["none", "source"])].groupby(["site", "layer"]):
        g = g.set_index(["pair", "qid"])
        rows.append(dict(site=s, layer=l, mean_abs_dlp=float((g.lp_base_ans - base.reindex(g.index)).abs().mean())))
    es = pd.DataFrame(rows)
    es.to_csv(OUT / f"{model}_effect_sizes.csv", index=False)
    return es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 14, 18, 21, 24])
    ap.add_argument("--n-objects", type=int, nargs="+", default=[3])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    es = effect_sizes(a.model)
    m = load(a.model)
    pairs = make_pairs(m, a.n, seed=3, n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
    dfs = []
    for p, (base, src, leaf, qs) in enumerate(pairs):
        d = run_pair(m, base, src, leaf, qs, ["all", "all_but_readout"], a.layers, batch_size=a.batch_size)
        d["pair"] = p
        dfs.append(d)
        if p % 10 == 0:
            print(f"[necessity] pair {p}/{len(pairs)}", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(OUT / f"{a.model}_necessity.parquet")
    c = consistency(df, n_boot=1000)
    c.to_csv(OUT / f"{a.model}_necessity_consistency.csv", index=False)

    piv = es.pivot(index="layer", columns="site", values="mean_abs_dlp")
    lines = [f"Model **{a.model}**, {len(pairs)} pairs (the P3 sweep's pairs). Exploratory, not preregistered: added after "
             "P3 showed that patching the readout positions alone changes no answer (insufficiency). Here the "
             "source description is copied everywhere *except* the objects' list positions and the end of the list.",
             "", "| layer | whole description | all but readout positions | difference |", "|---|---|---|---|"]
    for l in a.layers:
        r_all = c[(c.site == "all") & (c.layer == l)].iloc[0]
        r_but = c[(c.site == "all_but_readout") & (c.layer == l)].iloc[0]
        lines.append(f"| {l} | {r_all.joint_iia:.2f} [{r_all.jiia_lo:.2f}, {r_all.jiia_hi:.2f}] | "
                     f"{r_but.joint_iia:.2f} [{r_but.jiia_lo:.2f}, {r_but.jiia_hi:.2f}] | "
                     f"{r_but.joint_iia - r_all.joint_iia:+.2f} |")
    lines += ["", "Effect sizes in the P3 sweep (mean |change in log-probability of the base answer|, nats):", "",
              "| layer | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
    lines += [f"| {l} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for l, r in piv.iterrows() if l in (6, 10, 14, 18, 20, 24, 30)]
    append_entry(f"P3 necessity ({a.model}) — exploratory", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
