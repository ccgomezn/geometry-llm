"""Blind reply (docs/preregistration_blind_reply.md): can step-by-step reasoning get M's step from the processed state
when, while it writes, it cannot attend to M's sentence? Decisive questions in four conditions: normal, blind reply
(prompt read normally, reply blind to M's sentence), floor (M's sentence hidden from the prompt positions and the
reply) and list control (reply blind to the object list).

Resumable: replies are saved after every batch under results/blind and reused on a rerun; --rescore re-reads them."""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geomap.axis import CELLS, axis_bulk_items
from geomap.blind import blind_spans, greedy_masked, recovery, verdict_blind
from geomap.checkpoint import notify
from geomap.cot_axis import named_coords, parse_final_name, prediction_label, written_side
from geomap.flow import axis_of, steps
from geomap.models import load
from geomap.report import ROOT, append_entry
from geomap.study2 import ba_ci, balanced_accuracy

SEED = 1601
FLOW_SIZES = {"own step only": 400, "agree": 400, "first step only": 600, "conflict, C larger": 400,
              "conflict, M larger": 600}
DEC = "conflict, M larger"
CONDS = ("normal", "blind reply", "floor", "list control")
UNKNOWN = re.compile(r"not (?:given|specified|provided|stated|mentioned)|unknown|doesn't (?:say|tell)|"
                     r"does not (?:say|tell)|isn't (?:given|specified)|no information", re.I)


def blocks(cond: str, m_span, l_span):
    """(prompt_block, reply_block) for one question."""
    return {"normal": ([], []), "blind reply": ([], [m_span]), "floor": ([m_span], [m_span]),
            "list control": ([], [l_span])}[cond]


def build_items(m, n: int, flow_sizes: dict) -> pd.DataFrame:
    rows = []
    for si, (s, mid, child, qq) in enumerate(axis_bulk_items(m, flow_sizes, seed=SEED)):
        for q, cell in qq:
            if cell != DEC or len(rows) >= n:
                continue
            ids, m_span, l_span = blind_spans(s, q, mid, child, m.tok, m.family)
            d1, d2, t = steps(s, mid, child, q)
            rows.append(dict(scene=si, child=child, root=s.root, mid=mid, axis=axis_of(q), d1=d1, d2=d2,
                             truth="C" if t > 0 else "R", ids=ids, m_a=m_span[0], m_b=m_span[1], l_a=l_span[0],
                             l_b=l_span[1], question=q.text))
    return pd.DataFrame(rows)


def generate(m, items: pd.DataFrame, path: Path, max_new_tokens: int, batch_size: int, tag: str) -> pd.DataFrame:
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["i", "cond", "reply", "n_tokens",
                                                                             "truncated"])
    rows, have = done.to_dict("records"), set(zip(done.i, done.cond))
    t0, total = time.time(), len(items) * len(CONDS)
    for cond in CONDS:
        todo = [i for i in range(len(items)) if (i, cond) not in have]
        for k in range(0, len(todo), batch_size):
            idx = todo[k:k + batch_size]
            r = items.iloc[idx]
            bl = [blocks(cond, (x.m_a, x.m_b), (x.l_a, x.l_b)) for x in r.itertuples()]
            reps = greedy_masked(m, [list(x) for x in r.ids], [b[0] for b in bl], [b[1] for b in bl], max_new_tokens)
            for i, rep in zip(idx, reps):
                rows.append(dict(i=i, cond=cond, reply=m.tok.decode(rep, skip_special_tokens=True), n_tokens=len(rep),
                                 truncated=len(rep) >= max_new_tokens))
            pd.DataFrame(rows).to_parquet(path)
            print(f"[blind] {len(rows)}/{total} replies ({time.time() - t0:.0f}s)", flush=True)
            if len(rows) % (batch_size * 10) < batch_size:
                notify(f"[{tag}] {len(rows)}/{total} replies")
    return pd.DataFrame(rows)


def analyse(items, R, model, out, run, n_boot, smoke, tag, rescored=False):
    meta = items.drop(columns=["ids"]).reset_index().rename(columns={"index": "i"})
    D = R.merge(meta, on="i")
    D["final"] = [parse_final_name(r.reply, (r.child, r.root)) for r in D.itertuples()]
    D["pred"] = [prediction_label(f, r.child) for f, r in zip(D.final, D.itertuples())]
    D.to_parquet(out / f"{run}_scored.parquet")
    ba = {c: ba_ci(D[D.cond == c].truth, D[D.cond == c].pred, "C", D[D.cond == c].scene, n_boot=n_boot) for c in CONDS}

    def rec_of(d):
        b = {c: balanced_accuracy(d[d.cond == c].truth.to_numpy(), d[d.cond == c].pred.to_numpy(), "C") for c in CONDS}
        return recovery(b["blind reply"], b["floor"], b["normal"])

    est = rec_of(D)
    scenes = D.scene.unique()
    groups = {s: D[D.scene == s] for s in scenes}
    rng = np.random.default_rng(0)
    boots = [rec_of(pd.concat([groups[s] for s in rng.choice(scenes, len(scenes))])) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    v = verdict_blind((est, lo, hi), ba["normal"], ba["floor"], ba["blind reply"], ba["list control"][0])

    f2 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    ci = lambda t: f"{f2(t[0])} [{f2(t[1])}, {f2(t[2])}]"
    lines = [f"Model **{model}**. {len(items)} decisive questions (flow set's generator, seed {SEED}), greedy "
             "step-by-step replies in four conditions: *normal*; *blind reply* (prompt read normally, the reply cannot "
             "attend to M's sentence); *floor* (neither the prompt positions after M's sentence nor the reply can "
             "attend to it); *list control* (the reply cannot attend to the object list). Positions numbered as in the "
             "normal prompt. Balanced accuracy, 95% scene-bootstrap CIs; no answer = wrong.", "",
             "| condition | balanced accuracy | accuracy | no answer |", "|---|---|---|---|"]
    for c in CONDS:
        d = D[D.cond == c]
        lines.append(f"| {c} | {ci(ba[c])} | {f2(float(np.mean(d.pred == d.truth)))} | {int((d.pred == 'none').sum())} |")
    lines += ["", f"**Recovery** (blind reply − floor) / (normal − floor): {f2(est)} [{f2(lo)}, {f2(hi)}]."]
    b = D[D.cond == "blind reply"]
    ms = [written_side(named_coords(r.reply, (r.child, r.root, r.mid)), r.mid, r.root, int(r.axis)) for r in b.itertuples()]
    ok = [s == int(np.sign(r.d1)) for s, r in zip(ms, b.itertuples()) if s is not None]
    lines += ["", f"Exploratory, blind-reply replies: {len(ok)} give M a position relative to R, on the correct side in "
              f"{f2(float(np.mean(ok)) if ok else float('nan'))}; {int(b.reply.str.contains(UNKNOWN).sum())} of {len(b)} "
              "say something is not given or unknown."]
    lines += ["", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_blind_reply.md."]
    report = "\n".join(lines)
    print(report, flush=True)
    title = f"Blind reply: can written reasoning use the processed state when M's sentence is hidden? ({model}) — preregistered"
    if rescored:
        title += ", rescored"
    if not smoke:
        append_entry(title, report)
    (out / f"{run}_report.md").write_text(f"## {title}\n\n{report}\n")
    (out / f"{run}.done").write_text(v + "\n")
    notify(f"[{tag}] COMPLETE: {v}. Recovery {f2(est)} [{f2(lo)}, {f2(hi)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 4 questions, 12 tokens; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/blind)")
    ap.add_argument("--rescore", action="store_true", help="re-read saved replies (no model)")
    a = ap.parse_args()
    out, flow_sizes, n_boot = ROOT / "results" / "blind", dict(FLOW_SIZES), 1000
    if a.smoke:
        import tempfile
        a.model, a.n, a.max_new_tokens, n_boot = "tiny", 4, 12, 50
        out, flow_sizes = Path(a.out) if a.out else Path(tempfile.mkdtemp()), {c: 12 for c in CELLS}
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run, tag = f"{a.model}_blind_reply_seed{SEED}", f"Blind reply {a.model}"
    ipath, rpath = out / f"{run}_items.parquet", out / f"{run}_replies.parquet"
    if a.rescore:
        analyse(pd.read_parquet(ipath), pd.read_parquet(rpath), a.model, out, run, n_boot, a.smoke, tag, True)
        return
    if (out / f"{run}.done").exists():
        print(f"[blind] already complete; delete {run}.done to rerun", flush=True)
        return
    m = load(a.model)
    items = build_items(m, a.n, flow_sizes)
    items.to_parquet(ipath)
    print(f"[blind] {len(items)} decisive questions", flush=True)
    notify(f"[{tag}] started: {len(items)} questions x 4 conditions")
    R = generate(m, items, rpath, a.max_new_tokens, a.batch_size, tag)
    analyse(items, R, a.model, out, run, n_boot, a.smoke, tag)


if __name__ == "__main__":
    main()
