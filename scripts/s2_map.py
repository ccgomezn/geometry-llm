"""Study 2 (docs/preregistration_s2_map.md): could the model's own map tell whether a point is inside the triangle?
Fit linear probes for scene-centred (x, y) at the object-list positions on new triangle scenes, decode every object's
position in the pilot's test scenes, answer every ladder question geometrically from the decoded map, and compare with
the model's own answers (from the pilot) and with the size-shortcut baseline.

Resumable: the probes are saved under results/s2/map and reused on a rerun. Progress goes to Slack if
$SLACK_WEBHOOK_URL is set."""

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.followups import decoded_coords
from geomap.models import load
from geomap.probes import GroupRidge, centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.report import ROOT, append_entry
from geomap.scoring import predict, score
from geomap.study2 import fit_magnitude_rule, magnitude_rule, map_summary, verdict_s2_map
from geomap.triangles import generate_tri_scene, ladder, ladder_answer

TEST_SEED, MAG_SEED, PROBE_SEED = 702, 701, 711


def scenes(seed: int, n: int, pools) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ts = generate_tri_scene(rng, pools[i % 2])
        out.append((ts, ladder(ts, rng)))
    return out


def fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/s2/map)")
    a = ap.parse_args()
    out, layers = ROOT / "results" / "s2" / "map", [14, 18, 21, 24, 28]
    if a.smoke:
        import tempfile
        a.model, a.n_fit, a.n, out, layers = "tiny", 80, 12, Path(tempfile.mkdtemp()), [1, 2]
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_s2_map_n{a.n}_fit{a.n_fit}"
    tag = f"S2 map {a.model}"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[s2 map] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]

    # 1. probes on new triangle scenes (checkpoint); the primary layer is fixed here, before any test data
    ppath = out / f"{run}_probes.pt"
    if ppath.exists():
        probes = torch.load(ppath)
    else:
        fit = [ts.scene for ts, _ in scenes(PROBE_SEED, a.n_fit, pools)]
        acts = cache_prefix(m, [build_prompt(s, None, m.tok, m.family) for s in fit], layers,
                            batch_size=a.batch_size)["list"]
        probes = {}
        for li, l in enumerate(layers):
            X, Y, meta = centered_xy(acts[:, li].float().numpy(), fit, np.arange(len(fit)))
            pr = GroupRidge().fit(X, Y, meta.scene.values)
            probes[l] = dict(coef=pr.coef_, cv_r2=pr.cv_r2_)
            print(f"[s2 map] probe layer {l}: CV R² {pr.cv_r2_:.3f}", flush=True)
        torch.save(probes, ppath)
        del acts
        notify(f"[{tag}] probes: " + ", ".join(f"L{l} R² {p['cv_r2']:.2f}" for l, p in probes.items()))
    primary = max(probes, key=lambda l: probes[l]["cv_r2"])

    # 2. the pilot's test scenes and the model's answers (from the pilot run when available)
    test = scenes(TEST_SEED, a.n, pools)
    flat = [(si, ts, it) for si, (ts, items) in enumerate(test) for it in items]
    texts = [it.question.text for _, _, it in flat]
    pilot = ROOT / "results" / "s2" / "pilot" / f"{a.model}_s2_pilot_n{a.n}_seed{TEST_SEED}_answers.parquet"
    if pilot.exists():
        mod = pd.read_parquet(pilot)
        print(f"[s2 map] model answers from {pilot}", flush=True)
    else:
        mod = df_step(out / f"{run}_model_answers.parquet", lambda: pd.DataFrame([
            dict(text=it.question.text, pred=predict(x)) for (_, _, it), x in zip(flat, score(
                m, [(build_prompt(ts.scene, it.question, m.tok, m.family), it.question) for _, ts, it in flat],
                batch_size=a.batch_size))]))
    if list(mod.text) != texts:
        raise ValueError("model answers do not match the current questions")
    model_pred = list(mod.pred)

    # 3. decode the map in the test scenes and answer every question from it
    tscenes = [ts.scene for ts, _ in test]
    tacts = cache_prefix(m, [build_prompt(s, None, m.tok, m.family) for s in tscenes], layers,
                         batch_size=a.batch_size)["list"]
    mag = magnitude_rule(fit_magnitude_rule([ts for ts, _ in scenes(MAG_SEED, 2000 if not a.smoke else 60, pools)]),
                         [ts for ts, _ in test])
    summaries, recs, r2_test = [], [], {}
    for li, l in enumerate(layers):
        coef = probes[l]["coef"]
        probe = SimpleNamespace(predict=lambda X, c=coef: np.asarray(X) @ c.T)
        A = tacts[:, li].float().numpy()
        X, Y, _ = centered_xy(A, tscenes, np.arange(len(tscenes)))
        P = probe.predict(X)
        r2_test[l] = float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
        dec = decoded_coords(probe, A, tscenes, np.arange(len(tscenes)))
        rows = [dict(layer=l, scene=si, rung=it.rung, stratum=ts.stratum, truth=it.truth, positive=it.positive,
                     model_pred=mp, map_pred=ladder_answer(it.question, dec[si]),
                     mag_pred=mag[si] if it.rung == 4 else None)
                for (si, ts, it), mp in zip(flat, model_pred)]
        recs += rows
        summaries.append(map_summary(pd.DataFrame(rows)).assign(layer=l, probe_cv_r2=probes[l]["cv_r2"],
                                                                 probe_test_r2=r2_test[l]))
    df, s = pd.DataFrame(recs), pd.concat(summaries, ignore_index=True)
    df.to_parquet(out / f"{run}.parquet")
    s.to_csv(out / f"{run}_summary.csv", index=False)
    v = verdict_s2_map(s[s.layer == primary])

    names = {1: "1 direct", 2: "2 chain", 3: "3 extremes", 4: "4 inside"}
    lines = [f"Model **{a.model}**. Linear probes for scene-centred (x, y) at the object-list positions, fitted on "
             f"{a.n_fit} new triangle scenes (seed {PROBE_SEED}); primary layer = best training CV R² (**L{primary}**). "
             f"Test: the pilot's {len(test)} scenes (seed {TEST_SEED}), {len(flat)} questions. The *map* answers every "
             "question geometrically from the decoded positions; the *model* answers are the pilot's. Balanced accuracy "
             "with 95% scene-bootstrap CIs; differences are paired.", "",
             "| layer (CV R² / test R²) | rung | subset | n | model | map | map − model | agreement | map − size rule |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        mm = (f"{fmt(r.map_minus_magnitude)} [{fmt(r.map_minus_magnitude_lo)}, {fmt(r.map_minus_magnitude_hi)}]"
              if r.rung == 4 else "")
        lines.append(f"| {r.layer} ({r.probe_cv_r2:.2f} / {r.probe_test_r2:.2f}) | {names[r.rung]} | {r.subset} | {r.n} | "
                     f"{fmt(r.model_bal_acc)} | {fmt(r.map_bal_acc)} [{fmt(r.map_bal_acc_lo)}, {fmt(r.map_bal_acc_hi)}] | "
                     f"{fmt(r.map_minus_model)} [{fmt(r.map_minus_model_lo)}, {fmt(r.map_minus_model_hi)}] | "
                     f"{r.agreement:.2f} | {mm} |")
    lines += ["", f"Preregistered verdict (layer {primary}, rung 4 hard): **{v}**. Rules: docs/preregistration_s2_map.md."]
    if not a.smoke:
        append_entry(f"Study 2: could the model's own map tell inside the triangle? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v)
    h = s[(s.layer == primary) & (s.rung == 4) & (s.subset == "hard")].iloc[0]
    notify(f"[{tag}] COMPLETE. {v}. Rung 4 hard: map {fmt(h.map_bal_acc)}, model {fmt(h.model_bal_acc)}")
    print("\n".join(lines))
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
