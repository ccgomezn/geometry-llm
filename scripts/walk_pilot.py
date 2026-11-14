"""Walk pilot (docs/preregistration_walk_pilot.md): answers only. Can the model tell which landmark a walk has come
back to, beyond map-free shortcuts, as the loop and the walk get longer? Scores every walk in two prompt formats,
compares with the map-free baselines and applies the preregistered go rule.

Resumable: the model's answers are saved under results/walks and reused on a rerun. Progress goes to Slack if
$SLACK_WEBHOOK_URL is set."""

import argparse
import math
import time
from pathlib import Path

import pandas as pd

from geomap.checkpoint import df_step, notify
from geomap.models import load
from geomap.report import ROOT, append_entry
from geomap.scoring import candidate_mass, predict, score
from geomap.walks import (BASELINES, FORMATS, baselines, build_walk_prompt, landmark_pool, verdict_walk_pilot,
                          walk_items, walk_summary)

SEED, NS, LOOPS, PER_CELL = 1401, (14, 20, 26), (2, 6, 8, 12), 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 4 walks per cell; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/walks)")
    a = ap.parse_args()
    out = ROOT / "results" / "walks"
    if a.smoke:
        import tempfile
        a.model, a.per_cell, out = "tiny", 4, Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"Walk pilot {a.model}"
    run = f"{a.model}_walk_pilot_n{a.per_cell}_seed{SEED}"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[walk] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[walk] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    pool = landmark_pool(m.tok)
    walks = walk_items(pool, NS, LOOPS, a.per_cell, SEED)
    log(f"{len(walks)} walks, {len(pool)} landmark words; {len(walks) * len(FORMATS)} questions")
    notify(f"[{tag}] started: {len(walks) * len(FORMATS)} questions")

    def answers(fmt):
        prompts = [build_walk_prompt(w, m.tok, m.family, fmt) for w in walks]
        sc = score(m, [(p, p) for p in prompts], batch_size=a.batch_size)
        rows = []
        for i, (w, p, x) in enumerate(zip(walks, prompts, sc)):
            pred = predict(x)
            rows.append(dict(walk=i, n=w.n, loop=w.loop, fmt=fmt, pred=pred.strip(), answer=p.answer.strip(),
                             correct=float(pred == p.answer), mass=math.exp(candidate_mass(x)),
                             **{f"b_{k}": v for k, v in baselines(w).items()}))
        return pd.DataFrame(rows)

    dfs = []
    for fmt in FORMATS:
        path = out / f"{run}_{fmt}.parquet"
        fresh = not path.exists()
        d = df_step(path, lambda f=fmt: answers(f))
        dfs.append(d)
        msg = f"{fmt}: accuracy {d.correct.mean():.3f}; by loop " + ", ".join(
            f"{l}: {d[d.loop == l].correct.mean():.2f}" for l in LOOPS)
        log(msg)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    s = walk_summary(df)
    s.to_csv(out / f"{run}_summary.csv", index=False)
    v, cfg = verdict_walk_pilot(s)

    lines = [f"Model **{a.model}**. {len(walks)} new walks (seed {SEED}): walk length N ∈ {list(NS)} moves × loop "
             f"length ℓ ∈ {list(LOOPS)} (moves since the target landmark was seen), {a.per_cell} walks per cell, 4 "
             "landmarks per walk (chance 0.25), each scored in two formats (*user*: walk in the user message; "
             "*prefill*: walk in the model's own turn). ℓ = 2 is the shortcut-sufficient control (one block out and "
             "back). Accuracy with a 95% walk-bootstrap CI; *gap* = model − best map-free baseline in that cell, "
             "paired over walks. A cell passes at accuracy ≥ 0.60 with gap CI lower bound > 0.", ""]
    for fmt in FORMATS:
        lines += [f"**{fmt}**", "", "| N | ℓ | accuracy [95% CI] | best map-free | gap [95% CI] | prob. on answers | passes |",
                  "|---|---|---|---|---|---|---|"]
        for r in s[s.fmt == fmt].itertuples():
            lines.append(f"| {r.n} | {r.loop} | {r.acc:.2f} [{r.lo:.2f}, {r.hi:.2f}] | {r.best_baseline} "
                         f"{r.base_acc:.2f} | {r.gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}] | {r.mass:.2f} | "
                         f"{'yes' if r.passes else 'no'} |")
        lines.append("")
    lines += ["Baseline accuracy by ℓ (all N, user format walks; the same walks in both formats):", "",
              "| ℓ | " + " | ".join(BASELINES) + " |", "|---|" + "---|" * len(BASELINES)]
    for l in LOOPS:
        d = df[(df.fmt == "user") & (df.loop == l)]
        lines.append(f"| {l} | " + " | ".join(f"{d[f'b_{k}'].mean():.2f}" for k in BASELINES) + " |")
    chosen = f" Main study: {cfg['fmt']} format, N = {cfg['n']}, ℓ = {cfg['loop']} (with ℓ = 2 at the same N)." if cfg else ""
    lines += ["", f"Preregistered verdict: **{v}**.{chosen} Rules: docs/preregistration_walk_pilot.md."]
    if not a.smoke and a.per_cell == PER_CELL:
        append_entry(f"Walk pilot: does the model know which landmark a long walk came back to? ({a.model}) — "
                     "preregistered", "\n".join(lines), [])
    done.write_text(v + chosen)
    notify(f"[{tag}] COMPLETE. verdict: {v}.{chosen}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
