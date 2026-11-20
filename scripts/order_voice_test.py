"""Order and voice (docs/preregistration_order_voice.md): does the one-sentence rule follow C's own step, the
direction word of C's relation sentence, or the last sentence? The same scenes in four renderings, behaviour only.

Resumable: the scored answers are saved under results/order and reused on a rerun."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geomap.axis import CELLS, axis_bulk_items
from geomap.checkpoint import notify
from geomap.components import betas
from geomap.flow import axis_of, steps
from geomap.models import load
from geomap.ordervoice import FEATURES, RENDERINGS, design, features, render, verdict_order_voice
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry
from geomap.scoring import score

SEED = 2001
SIZES = {"own step only": 300, "agree": 300, "first step only": 300, "conflict, C larger": 300,
         "conflict, M larger": 600}
DEC = "conflict, M larger"
COL = {"own": 1, "word": 2, "last": 3}


def auc(y, x):
    """Probability that a random positive outranks a random negative (ties count half)."""
    y, x = np.asarray(y, bool), np.asarray(x, float)
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def coefs_with_ci(T: pd.DataFrame, n_boot: int, seed: int = 0):
    """Rule coefficients and pairwise differences with scene-bootstrap 95% CIs (all renderings of a scene together)."""
    F, y = design(T), T.ld.to_numpy(float)
    b = betas(y[:, None], F)[0]
    scenes = T.scene.to_numpy()
    uniq = np.unique(scenes)
    members = {g: np.flatnonzero(scenes == g) for g in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        boots.append(betas(y[idx, None], F[idx])[0])
    B = np.array(boots)
    ci = lambda v, vb: (float(v), float(np.percentile(vb, 2.5)), float(np.percentile(vb, 97.5)))
    coef = {f: ci(b[COL[f]], B[:, COL[f]]) for f in FEATURES}
    diff = {(f, g): ci(b[COL[f]] - b[COL[g]], B[:, COL[f]] - B[:, COL[g]])
            for i, f in enumerate(FEATURES) for g in FEATURES[i + 1:]}
    return coef, diff, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few questions; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/order)")
    ap.add_argument("--device", default=None, help='"auto" spreads the model over GPU and CPU memory (same bfloat16 '
                                                   'weights, slower), for GPUs too small to hold it')
    a = ap.parse_args()
    out, sizes, n_boot = ROOT / "results" / "order", dict(SIZES), 1000
    if a.smoke:
        import tempfile
        a.model, out, sizes, n_boot = "tiny", Path(tempfile.mkdtemp()), {c: 6 for c in CELLS}, 50
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_order_voice"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[order] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[order] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model, device=a.device)

    path = out / f"{run}_items.csv"
    if path.exists():
        T = pd.read_csv(path)
        log(f"reusing {len(T)} scored rows")
    else:
        items = axis_bulk_items(m, sizes, seed=SEED)
        rows, specs, qs, dropped = [], [], [], 0
        for si, (s, mid, child, qq) in enumerate(items):
            for qi, (q, cell) in enumerate(qq):
                d1, d2, truth = steps(s, mid, child, q)
                try:
                    built = [build_prompt(render(s, mid, child, r), q, m.tok, m.family) for r in RENDERINGS]
                except ValueError:
                    dropped += 1
                    continue
                for r, sp in zip(RENDERINGS, built):
                    own, word, last = features(d1, d2, r)
                    rows.append(dict(scene=si, qi=qi, rendering=r, cell=cell, axis=axis_of(q),
                                     c_first=q.args[0] == child, d1=d1, d2=d2, truth=truth, child=child,
                                     root=s.root, own_yes=own, word_yes=word, last_yes=last))
                    specs.append(sp)
                    qs.append(q)
        T = pd.DataFrame(rows)
        log(f"{len(T) // len(RENDERINGS)} questions x {len(RENDERINGS)} renderings in {len(items)} scenes "
            f"({dropped} dropped)")
        notify(f"[Order/voice {a.model}] started: {len(T)} prompts")
        sc = score(m, list(zip(specs, qs)), batch_size=a.batch_size)
        T["ld"] = [x[r.child] - x[r.root] for x, r in zip(sc, T.itertuples())]
        T["says_c"] = T.ld > 0
        T["correct"] = np.sign(T.ld) == T.truth
        T.to_csv(path, index=False)
        log("scored and saved")

    # ---- preregistered rule ----
    coef, diff, b = coefs_with_ci(T, n_boot)
    verdict = verdict_order_voice(coef, diff)

    # ---- reported ----
    acc = T.pivot_table(index="cell", columns="rendering", values="correct", aggfunc="mean").reindex(
        index=CELLS, columns=list(RENDERINGS))
    split = T.groupby(["rendering", "own_yes", "word_yes", "last_yes"]).agg(n=("ld", "size"),
                                                                             says_c=("says_c", "mean")).reset_index()
    dec = T[T.cell == DEC]
    d1w = {}
    for r in RENDERINGS:
        Tr = T[T.rendering == r]
        Fr = np.c_[np.ones(len(Tr)), Tr.word_yes, Tr.own_yes, Tr.d1, Tr.d2, np.where(Tr.c_first, 1.0, -1.0)]
        d1w[r] = float(np.linalg.lstsq(Fr, Tr.ld.to_numpy(float), rcond=None)[0][3])

    pct = lambda v: "–" if pd.isna(v) else f"{v:.2f}"
    lines = [f"Model **{a.model}**. {len(T) // len(RENDERINGS)} new C–R axis questions (seed {SEED}), each in four "
             "renderings of the same scene: sentence order MC (C's sentence last, as before) or CM (C's sentence "
             "first), and C's relation in subject voice (\"C is d east of M\") or object voice (\"M is d west of C\"). "
             "Answer = sign of logit(C) − logit(R). Scene-bootstrap 95% CIs.", "",
             "**Rule coefficients** (pooled regression of the logit difference on own_yes, word_yes, last_yes, d1, d2, "
             "name order, order and voice):", "",
             "| feature | coefficient [95% CI] |", "|---|---|"]
    for f in FEATURES:
        e, lo, hi = coef[f]
        lines.append(f"| {f}_yes | {e:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "| difference | estimate [95% CI] |", "|---|---|"]
    for (f, g), (e, lo, hi) in diff.items():
        lines.append(f"| {f} − {g} | {e:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", f"Other coefficients: d1 {b[4]:+.2f}/unit, d2 {b[5]:+.2f}/unit, C named first {b[6]:+.2f}, "
              f"CM order {b[7]:+.2f}, object voice {b[8]:+.2f}.", "",
              "**Accuracy** by kind of question and rendering:", "",
              "| kind | " + " | ".join(RENDERINGS) + " |", "|---" * (len(RENDERINGS) + 1) + "|"]
    for c in CELLS:
        lines.append(f"| {c} | " + " | ".join(pct(acc.loc[c, r]) for r in RENDERINGS) + " |")
    lines += ["", "**Share answering C**, by the three features (own, word, last):", "",
              "| rendering | own | word | last | n | answers C |", "|---|---|---|---|---|---|"]
    for r in split.itertuples():
        lines.append(f"| {r.rendering} | {r.own_yes} | {r.word_yes} | {r.last_yes} | {r.n} | {r.says_c:.2f} |")
    lines += ["", "**Decisive questions** (M's step larger and opposite) by rendering:", "",
              "| rendering | accuracy | right when C correct | right when R correct | AUC |", "|---|---|---|---|---|"]
    for r in RENDERINGS:
        d = dec[dec.rendering == r]
        y = d.truth > 0
        lines.append(f"| {r} | {d.correct.mean():.2f} | {d.correct[y].mean():.2f} | {d.correct[~y].mean():.2f} | "
                     f"{auc(y, d.ld):.2f} |")
    lines += ["", "Weight of M's step (d1 coefficient, logits/unit) by rendering: "
              + ", ".join(f"{r} {v:+.2f}" for r, v in d1w.items()) + ".", "",
              f"Preregistered verdict: **{verdict}**. Rules: docs/preregistration_order_voice.md."]
    report = "\n".join(lines)
    print(report, flush=True)
    title = f"Order and voice: what does the one-sentence rule follow? ({a.model}) — preregistered"
    if not a.smoke:
        append_entry(title, report)
    (out / f"{run}_report.md").write_text(f"## {title}\n\n{report}\n")
    done.write_text(verdict + "\n")
    notify(f"[Order/voice {a.model}] done: {verdict}")
    log("done")


if __name__ == "__main__":
    main()
