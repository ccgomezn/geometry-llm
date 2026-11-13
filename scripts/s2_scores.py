"""Study 2, exploratory (not preregistered): does the model's yes-score separate "inside" from "just outside" even
though it almost always answers "no"? Re-scores the pilot's rung-4 questions (same scenes, seed 702), keeps the
log-probabilities of "yes" and "no", and measures how well yes − no ranks inside points above box_hard points (ROC AUC).
Compared with the size-shortcut baseline's AUC on the same scenes.

Resumable: the scores are saved under results/s2/scores and reused on a rerun."""

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
from geomap.scoring import score
from geomap.study2 import auc_ci, balanced_accuracy, centroid_score, fit_magnitude_rule, magnitude_matrix
from geomap.triangles import generate_tri_scene, ladder

TEST_SEED, TRAIN_SEED = 702, 701  # the pilot's scenes


def scenes(seed: int, n: int, pools) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ts = generate_tri_scene(rng, pools[i % 2])
        out.append((ts, ladder(ts, rng)))
    return out


def cv_threshold_ba(s: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Balanced accuracy if the yes/no cut-off were chosen on half the scenes and applied to the other half
    (both directions, averaged): what the score could do without the model's 'no' bias."""
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(s)) < len(s) // 2
    out = []
    for fit, test in ((half, ~half), (~half, half)):
        cands = np.unique(s[fit])
        thr = max(cands, key=lambda t: balanced_accuracy(y[fit], (s[fit] >= t).astype(int), 1))
        out.append(balanced_accuracy(y[test], (s[test] >= thr).astype(int), 1))
    return float(np.mean(out))


def fmt(x: float) -> str:
    return "–" if isinstance(x, float) and math.isnan(x) else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 12 scenes; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/s2/scores)")
    a = ap.parse_args()
    out = ROOT / "results" / "s2" / "scores"
    if a.smoke:
        import tempfile
        a.model, a.n, a.n_train, out = "tiny", 12, 60, Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_s2_scores_n{a.n}_seed{TEST_SEED}"
    t0 = time.time()
    m = load(a.model)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    test = scenes(TEST_SEED, a.n, pools)
    items = [(si, ts, next(it for it in lad if it.rung == 4)) for si, (ts, lad) in enumerate(test)]
    notify(f"[S2 scores {a.model}] started: {len(items)} questions")

    def scores():
        sc = score(m, [(build_prompt(ts.scene, it.question, m.tok, m.family), it.question) for _, ts, it in items],
                   batch_size=a.batch_size)
        return pd.DataFrame([dict(scene=si, stratum=ts.stratum, truth=it.truth, lp_yes=x["yes"], lp_no=x["no"],
                                  text=it.question.text) for (si, ts, it), x in zip(items, sc)])

    df = df_step(out / f"{run}.parquet", scores)
    if list(df.text) != [it.question.text for _, _, it in items]:
        raise ValueError("saved scores do not match the current questions; delete the file to rerun")
    df["s"] = df.lp_yes - df.lp_no
    print(f"[s2 scores] ready ({time.time() - t0:.0f}s)", flush=True)

    pilot = ROOT / "results" / "s2" / "pilot" / f"{a.model}_s2_pilot_n{a.n}_seed{TEST_SEED}_answers.parquet"
    same = None
    if pilot.exists():  # sanity check: the argmax here must reproduce the pilot's rung-4 answers
        p4 = pd.read_parquet(pilot).query("rung == 4").sort_values("scene")
        same = float(np.mean(np.where(df.s.values > 0, "yes", "no") == p4.pred.values))

    train = [ts for ts, _ in scenes(TRAIN_SEED, a.n_train, pools)]
    mag = fit_magnitude_rule(train)
    tss = [ts for ts, _ in test]
    df["mag"] = mag.predict_proba(magnitude_matrix(tss))[:, 1]
    df["centroid"] = [-centroid_score(t) for t in tss]
    df["pool"] = np.where(df.scene % 2 == 0, "people", "objects")

    rows = []
    hard = df[df.stratum.isin(("inside", "box_hard"))]
    for subset, d in [("hard", hard), ("hard, people names", hard[hard.pool == "people"]),
                      ("hard, object names", hard[hard.pool == "objects"])]:
        y = (d.stratum == "inside").astype(int).values
        for who, col in (("model yes − no", "s"), ("magnitude rule (map-free)", "mag"), ("centroid (reference)", "centroid")):
            est, lo, hi = auc_ci(d[col].values, y, d.scene.values)
            ba = cv_threshold_ba(d[col].values, y) if who.startswith("model") else float("nan")
            rows.append(dict(subset=subset, who=who, n=len(d), auc=est, lo=lo, hi=hi, ba_refit=ba,
                             mean_s_inside=float(d.s[y == 1].mean()), mean_s_out=float(d.s[y == 0].mean())))
    s = pd.DataFrame(rows)
    s.to_csv(out / f"{run}_summary.csv", index=False)

    lines = [f"Model **{a.model}**, exploratory (not preregistered). The pilot's {len(items)} rung-4 questions (same "
             "scenes, seed 702), re-scored keeping log P(yes) and log P(no). AUC: how often the score ranks an inside "
             "point above a box_hard point (0.5 = no separation), 95% scene-bootstrap CI. *Refit BA*: balanced "
             "accuracy if the yes/no cut-off on the model's score were chosen on half the scenes and tested on the "
             "other half (removes the 'no' bias).", "",
             "| subset | score | n | AUC [95% CI] | refit BA |", "|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.subset} | {r.who} | {r.n} | {fmt(r.auc)} [{fmt(r.lo)}, {fmt(r.hi)}] | {fmt(r.ba_refit)} |")
    lines += ["", f"Mean yes − no (log-odds): inside {hard.s[hard.stratum == 'inside'].mean():+.2f}, box_hard "
                  f"{hard.s[hard.stratum == 'box_hard'].mean():+.2f}, outside_box "
                  f"{df.s[df.stratum == 'outside_box'].mean():+.2f}."]
    if same is not None:
        lines.append(f"Check: the argmax reproduces {same:.1%} of the pilot's rung-4 answers.")
    if not a.smoke:
        append_entry(f"Study 2: does the yes-score separate inside from outside? ({a.model}) — exploratory",
                     "\n".join(lines), [])
    (out / f"{run}.done").write_text("ok")
    top = s.iloc[0]
    notify(f"[S2 scores {a.model}] COMPLETE. model AUC {fmt(top.auc)} [{fmt(top.lo)}, {fmt(top.hi)}]")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
