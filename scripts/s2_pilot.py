"""Study 2 pilot (docs/preregistration_s2_pilot.md): answers only. Can the model judge "inside the triangle" beyond
map-free shortcuts? Scores every question on the four-rung ladder, compares rung 4 with cheap baselines and applies
the preregistered go rule.

Resumable: the model's answers are saved under results/s2/pilot and reused on a rerun. Progress goes to Slack if
$SLACK_WEBHOOK_URL is set."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geomap.checkpoint import df_step, notify
from geomap.models import load
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.report import ROOT, append_entry
from geomap.scoring import candidate_mass, predict, score
from geomap.study2 import (box_rule, centroid_rule, feature_rule, fit_centroid_rule, fit_feature_rule,
                           pilot_summary, verdict_pilot)
from geomap.triangles import generate_tri_scene, ladder

TEST_SEED, TRAIN_SEED = 702, 701


def scenes(seed: int, n: int, pools) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ts = generate_tri_scene(rng, pools[i % 2])
        out.append((ts, ladder(ts, rng)))
    return out


def fmt(x: float) -> str:
    return "–" if isinstance(x, float) and math.isnan(x) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 8 scenes; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/s2/pilot)")
    a = ap.parse_args()
    out = ROOT / "results" / "s2" / "pilot"
    if a.smoke:
        import tempfile
        a.model, a.n, a.n_train, out = "tiny", 8, 60, Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"S2 pilot {a.model}"
    done = out / f"{a.model}_s2_pilot.done"
    if done.exists():
        print(f"[s2] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]

    test = scenes(TEST_SEED, a.n, pools)
    flat = [(si, ts, it) for si, (ts, items) in enumerate(test) for it in items]
    print(f"[s2] {len(flat)} questions in {len(test)} scenes", flush=True)
    notify(f"[{tag}] started: {len(flat)} questions")

    def model_answers():
        sc = score(m, [(build_prompt(ts.scene, it.question, m.tok, m.family), it.question) for _, ts, it in flat],
                   batch_size=a.batch_size)
        return pd.DataFrame([dict(scene=si, rung=it.rung, stratum=ts.stratum, truth=it.truth, pred=predict(x),
                                  positive=it.positive, mass=math.exp(candidate_mass(x)), text=it.question.text)
                             for (si, ts, it), x in zip(flat, sc)])

    df = df_step(out / f"{a.model}_s2_pilot_n{a.n}_seed{TEST_SEED}_answers.parquet", model_answers)
    print(f"[s2] model answers ready ({time.time() - t0:.0f}s)", flush=True)

    train = [ts for ts, _ in scenes(TRAIN_SEED, a.n_train, pools)]
    tss = [ts for ts, _ in test]
    base = {"box": box_rule(tss), "feature": feature_rule(fit_feature_rule(train), tss),
            "centroid": centroid_rule(tss, fit_centroid_rule(train))}
    s = pilot_summary(df, base)
    s.to_csv(out / f"{a.model}_s2_pilot_summary.csv", index=False)
    v = verdict_pilot(s)

    names = {1: "1 direct", 2: "2 chain", 3: "3 extremes", 4: "4 inside"}
    lines = [f"Model **{a.model}**. {len(test)} new triangle scenes (seed {TEST_SEED}), {len(flat)} questions. Balanced "
             "accuracy with a 95% scene-bootstrap CI; *hard* = P inside the triangle vs inside its bounding box but "
             f"outside the triangle. Baselines fitted on {a.n_train} other scenes (seed {TRAIN_SEED}); box and feature "
             "rules are map-free, the centroid rule is a reference only.", "",
             "| who | rung | subset | n | balanced acc [95% CI] | acc | prob. on yes/no or names |",
             "|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.who} | {names[r.rung]} | {r.subset} | {r.n} | {fmt(r.bal_acc)} [{fmt(r.lo)}, {fmt(r.hi)}] | "
                     f"{fmt(r.acc)} | {fmt(r.mean_prob_mass)} |")
    lines += ["", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_s2_pilot.md."]
    if not a.smoke:
        append_entry(f"Study 2 pilot: can the model tell inside the triangle? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v)
    h = s[(s.who == "model") & (s.rung == 4) & (s.subset == "hard")].iloc[0]
    notify(f"[{tag}] COMPLETE. verdict: {v}. Rung 4 hard: {fmt(h.bal_acc)} [{fmt(h.lo)}, {fmt(h.hi)}]")
    print("\n".join(lines))
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
