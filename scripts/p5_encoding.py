"""P5a + P5b (docs/preregistration_p5.md): does the P4c 2-D code change linearly with the stated offset, and does
writing it from the offset (instead of copying it from a source run) move the answers?

Resumable: every finished step is saved under results/p5 and skipped on a rerun (Colab sessions drop). Progress
messages go to Slack when $SLACK_WEBHOOK_URL is set."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.checkpoint import df_step, notify
from geomap.das import SubspaceIntervention, eval_das, pair_rows
from geomap.followups import cv_r2, fit_token_maps, project, sentence_diffs, writein_rows
from geomap.models import load
from geomap.patching import make_pairs, prefix_acts
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry

SITE = "relation"
METHODS = {"copy": "copy (P4c edit)", "writein": "write-in", "writein_random": "write-in, random subspace"}


def save_maps(path, maps, maps_r):
    np.savez(path, **{f"q_{k}": v for k, v in maps.items()}, **{f"r_{k}": v for k, v in maps_r.items()})


def load_maps(path):
    z = np.load(path)
    return ({int(k[2:]): z[k] for k in z.files if k.startswith("q_")},
            {int(k[2:]): z[k] for k in z.files if k.startswith("r_")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layer 1, random subspace; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/p5)")
    a = ap.parse_args()
    out, layer = ROOT / "results" / "p5", 14
    if a.smoke:
        import tempfile
        out, layer = Path(tempfile.mkdtemp()), 1
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"P5a/b {a.model}"
    done = out / f"{a.model}_p5ab.done"
    if done.exists():
        print(f"[p5] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    kw = dict(n_objects=(3,), exclude_qtypes=("closer_than",))
    iv = SubspaceIntervention(m.d_model, 2)
    if not a.smoke:
        iv.load_state_dict(torch.load(ROOT / "results" / "p4" / f"{a.model}_L{layer}_{SITE}_confirm_k2.pt",
                                      map_location="cpu"))
    torch.manual_seed(123)
    rnd = SubspaceIntervention(m.d_model, 2)  # the same random control as P4
    with torch.no_grad():
        Q, Qr = iv.Q.double().numpy(), rnd.Q.double().numpy()

    # P5a (checkpoint: summary csv + fitted maps)
    a_csv, maps_npz = out / f"{a.model}_p5a.csv", out / f"{a.model}_p5a_maps.npz"
    if a_csv.exists() and maps_npz.exists():
        s_a = pd.read_csv(a_csv).iloc[0]
        maps, maps_r = load_maps(maps_npz)
        print(f"[p5a] resumed from {a_csv}", flush=True)
    else:
        fit_pairs = make_pairs(m, a.n_fit, seed=107, **kw)
        recs = sentence_diffs(m, fit_pairs, layer)
        print(f"[p5a] {len(recs)} token records from {len(fit_pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
        r2 = cv_r2(project(recs, Q))
        rng = np.random.default_rng(0)
        null = np.array([cv_r2(project(recs, np.linalg.qr(rng.standard_normal((m.d_model, 2)))[0]))
                         for _ in range(a.n_null)])
        p95, p99 = np.quantile(null, [0.95, 0.99])
        v_a = ("linear encoding" if r2 >= 0.5 and r2 > p99 else
               "no linear encoding" if r2 < 0.2 or r2 <= p95 else "partial")
        maps, maps_r = fit_token_maps(project(recs, Q)), fit_token_maps(project(recs, Qr))
        save_maps(maps_npz, maps, maps_r)
        s_a = pd.Series(dict(cv_r2=r2, null_p95=p95, null_p99=p99, null_max=null.max(), n_pairs=len(fit_pairs),
                             verdict=v_a))
        pd.DataFrame([s_a]).to_csv(a_csv, index=False)
        del recs
        notify(f"[{tag}] P5a done: R² {r2:.2f} (random p99 {p99:.2f}) -> {v_a}")
    print(f"[p5a] R² {s_a.cv_r2:.3f}; null p95 {s_a.null_p95:.3f} p99 {s_a.null_p99:.3f} -> {s_a.verdict}", flush=True)

    # P5b (checkpoint: one csv per edit)
    test, ctx = None, {}

    def prepared():
        nonlocal test
        if test is None:
            test = make_pairs(m, a.n_test, seed=106, **kw)  # the P4c test pairs
            ctx["rows"] = pair_rows(m, test, layer, SITE)
            ctx["base_h"], ctx["doffs"] = {}, {}
            for p, (base, src, leaf, _) in enumerate(test):
                spec = build_prompt(base, None, m.tok, m.family)
                lo, hi = spec.relation_spans[leaf]
                ctx["base_h"][p] = prefix_acts(m, spec, [layer])[layer][lo:hi].float()
                rb = next(r for r in base.relations if r.subj == leaf)
                rs = next(r for r in src.relations if r.subj == leaf)
                ctx["doffs"][p] = np.array([rs.dx - rb.dx, rs.dy - rb.dy], dtype=float)
            print(f"[p5b] {len(ctx['rows'])} rows, {time.time() - t0:.0f}s", flush=True)
        return test, ctx

    def run(key):
        t, c = prepared()
        if key == "copy":
            return eval_das(m, iv, t, layer, SITE, rows=c["rows"])
        if key == "writein":
            return eval_das(m, iv, t, layer, SITE, rows=writein_rows(c["rows"], c["base_h"], c["doffs"], maps, Q))
        return eval_das(m, rnd, t, layer, SITE, rows=writein_rows(c["rows"], c["base_h"], c["doffs"], maps_r, Qr))

    res = {}
    for key, name in METHODS.items():
        path = out / f"{a.model}_p5b_{key}.csv"
        fresh = not path.exists()
        res[name] = df_step(path, lambda k=key: run(k))
        acc = float(res[name].set_index("qtype").loc["ALL", "acc"])
        print(f"[p5b] {name}: ALL {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if fresh:
            notify(f"[{tag}] P5b {name}: ALL {acc:.2f}")
    acc = {k: float(d.set_index("qtype").loc["ALL", "acc"]) for k, d in res.items()}
    ratio = acc["write-in"] / acc["copy (P4c edit)"]
    v_b = ("genuine encoding" if ratio >= 0.5 and acc["write-in, random subspace"] <= 0.05 else
           "consistent with an illusion" if ratio < 0.2 else "partial")
    pd.concat([d.assign(method=k) for k, d in res.items()]).to_csv(out / f"{a.model}_p5b.csv", index=False)

    lines = [f"Model **{a.model}**, the preregistered P4c subspace (layer {layer}, moved object's sentence).", "",
             f"**P5a, natural encoding.** Change in the 2 subspace coordinates predicted linearly from the change in "
             f"the stated offset (one 2×2 map per token from the sentence end), {int(s_a.n_pairs)} new pairs, "
             f"pair-grouped 5-fold CV: **R² = {s_a.cv_r2:.2f}**. Random 2-D subspaces ({a.n_null}): 95th percentile "
             f"{s_a.null_p95:.2f}, 99th {s_a.null_p99:.2f}, max {s_a.null_max:.2f}. Preregistered verdict: "
             f"**{s_a.verdict}**.", "",
             f"**P5b, write-in.** On the {a.n_test} P4c test pairs, interchange accuracy (ALL question types):", "",
             "| edit | ALL | 95% CI |", "|---|---|---|"]
    for k, d in res.items():
        r = d.set_index("qtype").loc["ALL"]
        lines.append(f"| {k} | {r.acc:.2f} | [{r.acc_lo:.2f}, {r.acc_hi:.2f}] |")
    lines += ["", f"Write-in / copy = {ratio:.2f}. Preregistered verdict: **{v_b}**.",
              "Rules: docs/preregistration_p5.md."]
    if not a.smoke:
        append_entry(f"P5a/P5b natural encoding and write-in ({a.model}) — preregistered", "\n".join(lines), [])
    done.write_text(json.dumps(dict(p5a=s_a.verdict, p5b=v_b, ratio=ratio)))
    notify(f"[{tag}] COMPLETE. P5a: {s_a.verdict} (R² {s_a.cv_r2:.2f}). P5b: {v_b} "
           f"(write-in {acc['write-in']:.2f} vs copy {acc['copy (P4c edit)']:.2f}, random {acc['write-in, random subspace']:.2f})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
