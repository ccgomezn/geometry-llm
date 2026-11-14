"""Exploratory (not preregistered): excess drop by question type x class for the closed-loop erasure test.
Reads results/erase/<model>_erase_closed_loop_test.parquet, writes ..._by_qtype.csv."""

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma3-12b")
ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()
base = f"results/erase/{a.model}_erase_closed_loop"
t = pd.read_parquet(base + "_test.parquet")
clean = t[t.cond == "clean"].set_index(["scene", "qi"])
rng = np.random.default_rng(0)
rows = []
for cfg in sorted(set(t.config) - {"-"}):
    w = t[t.config == cfg].pivot_table(index=["scene", "qi"], columns="cond", values="correct")
    w["clean"] = clean.correct
    w = w.join(clean[["cls", "qtype"]])
    for (q, c), g in w.groupby(["qtype", "cls"]):
        s = g.reset_index().groupby("scene")
        num = s.apply(lambda x: (x.random - x.probe).sum()).to_numpy()
        n = s.size().to_numpy()
        bs = [num[b].sum() / n[b].sum() for b in (rng.integers(0, len(n), len(n)) for _ in range(a.n_boot))]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        rows.append(dict(config=cfg, qtype=q, cls=c, n=len(g), clean=g.clean.mean(), probe=g.probe.mean(),
                         random=g.random.mean(), excess=(g.random - g.probe).mean(), excess_lo=lo, excess_hi=hi))
out = pd.DataFrame(rows)
out.to_csv(base + "_by_qtype.csv", index=False)
print(out.round(3).to_string(index=False))
