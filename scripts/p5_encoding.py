"""P5a + P5b (docs/preregistration_p5.md): does the P4c 2-D code change linearly with the stated offset, and does
writing it from the offset (instead of copying it from a source run) move the answers?"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from geomap.das import SubspaceIntervention, eval_das, pair_rows
from geomap.followups import cv_r2, fit_token_maps, project, sentence_diffs, writein_rows
from geomap.models import load
from geomap.patching import make_pairs, prefix_acts
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p5"
LAYER, SITE = 14, "relation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layer 1, random subspace; writes to /tmp only")
    a = ap.parse_args()
    global OUT, LAYER
    if a.smoke:
        import tempfile
        OUT, LAYER = __import__("pathlib").Path(tempfile.mkdtemp()), 1
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = load(a.model)
    kw = dict(n_objects=(3,), exclude_qtypes=("closer_than",))
    iv = SubspaceIntervention(m.d_model, 2)
    if not a.smoke:
        iv.load_state_dict(torch.load(ROOT / "results" / "p4" / f"{a.model}_L{LAYER}_{SITE}_confirm_k2.pt",
                                      map_location="cpu"))
    torch.manual_seed(123)
    rnd = SubspaceIntervention(m.d_model, 2)  # the same random control as P4
    with torch.no_grad():
        Q, Qr = iv.Q.double().numpy(), rnd.Q.double().numpy()

    # P5a
    fit_pairs = make_pairs(m, a.n_fit, seed=107, **kw)
    recs = sentence_diffs(m, fit_pairs, LAYER)
    print(f"[p5a] {len(recs)} token records from {len(fit_pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    r2 = cv_r2(project(recs, Q))
    rng = np.random.default_rng(0)
    null = np.array([cv_r2(project(recs, np.linalg.qr(rng.standard_normal((m.d_model, 2)))[0]))
                     for _ in range(a.n_null)])
    p95, p99 = np.quantile(null, [0.95, 0.99])
    v_a = ("linear encoding" if r2 >= 0.5 and r2 > p99 else
           "no linear encoding" if r2 < 0.2 or r2 <= p95 else "partial")
    pd.DataFrame([dict(cv_r2=r2, null_p95=p95, null_p99=p99, null_max=null.max(), verdict=v_a)]).to_csv(
        OUT / f"{a.model}_p5a.csv", index=False)
    print(f"[p5a] R² {r2:.3f}; null p95 {p95:.3f} p99 {p99:.3f} -> {v_a}", flush=True)

    # P5b
    maps, maps_r = fit_token_maps(project(recs, Q)), fit_token_maps(project(recs, Qr))
    test = make_pairs(m, a.n_test, seed=106, **kw)  # the P4c test pairs
    rows = pair_rows(m, test, LAYER, SITE)
    base_h, doffs = {}, {}
    for p, (base, src, leaf, _) in enumerate(test):
        spec = build_prompt(base, None, m.tok, m.family)
        lo, hi = spec.relation_spans[leaf]
        base_h[p] = prefix_acts(m, spec, [LAYER])[LAYER][lo:hi].float()
        rb = next(r for r in base.relations if r.subj == leaf)
        rs = next(r for r in src.relations if r.subj == leaf)
        doffs[p] = np.array([rs.dx - rb.dx, rs.dy - rb.dy], dtype=float)
    print(f"[p5b] {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    res = {"copy (P4c edit)": eval_das(m, iv, test, LAYER, SITE, rows=rows),
           "write-in": eval_das(m, iv, test, LAYER, SITE, rows=writein_rows(rows, base_h, doffs, maps, Q)),
           "write-in, random subspace": eval_das(m, rnd, test, LAYER, SITE,
                                                 rows=writein_rows(rows, base_h, doffs, maps_r, Qr))}
    full = pd.concat([d.assign(method=k) for k, d in res.items()])
    full.to_csv(OUT / f"{a.model}_p5b.csv", index=False)
    acc = {k: float(d.set_index("qtype").loc["ALL", "acc"]) for k, d in res.items()}
    ratio = acc["write-in"] / acc["copy (P4c edit)"]
    v_b = ("genuine encoding" if ratio >= 0.5 and acc["write-in, random subspace"] <= 0.05 else
           "consistent with an illusion" if ratio < 0.2 else "partial")
    print(f"[p5b] {acc} ratio {ratio:.2f} -> {v_b}  ({time.time() - t0:.0f}s)", flush=True)

    lines = [f"Model **{a.model}**, the preregistered P4c subspace (layer {LAYER}, moved object's sentence).", "",
             f"**P5a, natural encoding.** Change in the 2 subspace coordinates predicted linearly from the change in "
             f"the stated offset (one 2×2 map per token from the sentence end), {len(fit_pairs)} new pairs, "
             f"pair-grouped 5-fold CV: **R² = {r2:.2f}**. Random 2-D subspaces ({a.n_null}): 95th percentile "
             f"{p95:.2f}, 99th {p99:.2f}, max {null.max():.2f}. Preregistered verdict: **{v_a}**.", "",
             f"**P5b, write-in.** On the {len(test)} P4c test pairs, interchange accuracy (ALL question types):", "",
             "| edit | ALL | 95% CI |", "|---|---|---|"]
    for k, d in res.items():
        r = d.set_index("qtype").loc["ALL"]
        lines.append(f"| {k} | {r.acc:.2f} | [{r.acc_lo:.2f}, {r.acc_hi:.2f}] |")
    lines += ["", f"Write-in / copy = {ratio:.2f}. Preregistered verdict: **{v_b}**.",
              "Rules: docs/preregistration_p5.md."]
    if not a.smoke:
        append_entry(f"P5a/P5b natural encoding and write-in ({a.model}) — preregistered", "\n".join(lines), [])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
