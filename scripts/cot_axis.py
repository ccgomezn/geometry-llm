"""Step-by-step reasoning on the C–R axis questions (docs/preregistration_cot_axis.md): does written reasoning fix the
one-sentence rule on the decisive questions? Same loaded weights, same questions: one-word answers (logit difference)
and greedy step-by-step replies ending with 'Answer: <name>'.

Resumable: one-word answers and replies are saved under results/cot (replies after every batch) and reused on a
rerun. Progress goes to Slack if $SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.axis import CELLS, axis_bulk_items
from geomap.checkpoint import notify
from geomap.cot_axis import (cot_axis_prompt, named_coords, parse_final_name, prediction_label, select_items,
                             verdict_cot_axis, written_side)
from geomap.flow import axis_of, steps
from geomap.models import load
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry
from geomap.scoring import score
from geomap.study2 import ba_ci, ba_diff_ci

SEED = 1601                                   # the flow set's generator
FLOW_SIZES = {"own step only": 400, "agree": 400, "first step only": 600, "conflict, C larger": 400,
              "conflict, M larger": 600}
SIZES = {"own step only": 75, "agree": 75, "first step only": 75, "conflict, C larger": 75, "conflict, M larger": 300}
DEC = "conflict, M larger"


def generate_replies(m, prompts: list[str], path: Path, max_new_tokens: int, batch_size: int, tag: str) -> pd.DataFrame:
    """Greedy replies, saved after each batch; finished prompts are skipped on a rerun."""
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["i", "reply", "n_tokens", "truncated"])
    rows = done.to_dict("records")
    tok = m.tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    todo = [i for i in range(len(prompts)) if i not in set(done.i)]
    t0 = time.time()
    for b in range(0, len(todo), batch_size):
        idx = todo[b:b + batch_size]
        enc = [tok(prompts[i], add_special_tokens=False).input_ids for i in idx]
        T = max(map(len, enc))
        ids = torch.full((len(enc), T), pad, dtype=torch.long)
        am = torch.zeros_like(ids)
        for j, x in enumerate(enc):                     # left padding for generation
            ids[j, T - len(x):] = torch.tensor(x)
            am[j, T - len(x):] = 1
        with torch.no_grad():
            gen = m.hf.generate(input_ids=ids.to(m.hf.device), attention_mask=am.to(m.hf.device),
                                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
        for j, i in enumerate(idx):
            out = gen[j, T:]
            n = int((out != pad).sum())
            rows.append(dict(i=i, reply=tok.decode(out, skip_special_tokens=True), n_tokens=n,
                             truncated=n >= max_new_tokens))
        pd.DataFrame(rows).to_parquet(path)
        print(f"[cot] {len(rows)}/{len(prompts)} replies ({time.time() - t0:.0f}s)", flush=True)
        if len(rows) % (batch_size * 10) < batch_size:
            notify(f"[{tag}] {len(rows)}/{len(prompts)} replies")
    return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 3 questions per kind, 16 tokens; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/cot)")
    ap.add_argument("--rescore", action="store_true", help="re-read the saved replies (no model) and rewrite "
                    "the report, e.g. after a parser fix")
    a = ap.parse_args()
    out, sizes, flow_sizes, n_boot = ROOT / "results" / "cot", dict(SIZES), dict(FLOW_SIZES), 1000
    if a.smoke:
        import tempfile
        a.model, a.max_new_tokens, out, n_boot = "tiny", 16, Path(tempfile.mkdtemp()), 50
        sizes, flow_sizes = {c: 3 for c in CELLS}, {c: 6 for c in CELLS}
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_cot_axis_seed{SEED}"
    tag = f"Step by step {a.model}"
    done = out / f"{run}.done"
    if a.rescore:
        T = pd.read_parquet(out / f"{run}.parquet")
        analyse(T, a.model, out, run, n_boot, a.max_new_tokens, a.smoke, tag,
                lambda msg: print(f"[cot] {msg}", flush=True), rescored=True)
        return
    if done.exists():
        print(f"[cot] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[cot] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)

    sel = select_items(axis_bulk_items(m, flow_sizes, seed=SEED), sizes)
    rows, specs, qs, prompts = [], [], [], []
    for si, q, s, mid, child, cell in sel:
        d1, d2, truth = steps(s, mid, child, q)
        rows.append(dict(scene=si, cell=cell, axis=axis_of(q), d1=d1, d2=d2, truth="C" if truth > 0 else "R",
                         child=child, root=s.root, mid=mid, question=q.text))
        specs.append(build_prompt(s, q, m.tok, m.family))
        qs.append(q)
        prompts.append(cot_axis_prompt(s, q, m.tok, m.family))
    T = pd.DataFrame(rows)
    log(f"{len(T)} questions: " + ", ".join(f"{c} {int((T.cell == c).sum())}" for c in CELLS))
    notify(f"[{tag}] started: {len(T)} questions, up to {a.max_new_tokens} tokens each")

    # ---- one word (same weights) ----
    opath = out / f"{run}_one_word.csv"
    if opath.exists():
        T["one_ld"] = pd.read_csv(opath).one_ld.values
    else:
        sc = score(m, list(zip(specs, qs)), batch_size=a.batch_size)
        T["one_ld"] = [x[r.child] - x[r.root] for x, r in zip(sc, T.itertuples())]
        T[["one_ld"]].to_csv(opath, index=False)
    T["one_pred"] = np.where(T.one_ld > 0, "C", "R")
    log("one-word answers scored")

    # ---- step by step ----
    rep = generate_replies(m, prompts, out / f"{run}_replies.parquet", a.max_new_tokens, a.batch_size, tag)
    T["reply"], T["n_tokens"], T["truncated"] = rep.reply.values, rep.n_tokens.values, rep.truncated.values
    T.to_parquet(out / f"{run}.parquet")
    analyse(T, a.model, out, run, n_boot, a.max_new_tokens, a.smoke, tag, log)


def analyse(T: pd.DataFrame, model: str, out: Path, run: str, n_boot: int, max_new_tokens: int, smoke: bool,
            tag: str, log, rescored: bool = False):
    """Reads the replies, applies the preregistered rule and writes the report (also used to rescore saved
    replies without the model)."""
    T = T.copy()
    T["one_pred"] = np.where(T.one_ld > 0, "C", "R")
    T["final"] = [parse_final_name(r.reply, (r.child, r.root)) for r in T.itertuples()]
    T["cot_pred"] = [prediction_label(f, r.child) for f, r in zip(T.final, T.itertuples())]
    T["coords"] = [len(named_coords(r.reply, (r.child, r.root, r.mid))) > 0 for r in T.itertuples()]
    side = [written_side(named_coords(r.reply, (r.child, r.root, r.mid)), r.child, r.root, int(r.axis))
            for r in T.itertuples()]
    T["written_side"] = [np.nan if x is None else x for x in side]

    # ---- preregistered rule ----
    dec, oth = T[T.cell == DEC], T[T.cell != DEC]
    ba = lambda d, col: ba_ci(d.truth, d[col], "C", d.scene, n_boot=n_boot)
    dec_cot, dec_one = ba(dec, "cot_pred"), ba(dec, "one_pred")
    diff = ba_diff_ci(dec.truth, dec.cot_pred, dec.one_pred, "C", dec.scene, n_boot=n_boot)
    oth_cot, oth_one = ba(oth, "cot_pred"), ba(oth, "one_pred")
    v = verdict_cot_axis(dec_cot, diff, oth_cot[0])

    # ---- report ----
    f2 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    ci = lambda t: f"{f2(t[0])} [{f2(t[1])}, {f2(t[2])}]"
    acc = lambda d, col: float(np.mean(d[col] == d.truth)) if len(d) else float("nan")
    lines = [f"Model **{model}**. {len(T)} C–R axis questions from the flow set's generator (seed {SEED}; the first "
             f"{SIZES[DEC]} decisive and {SIZES['agree']} of each other kind). Same weights, two conditions: *one word* "
             "(sign of logit(C) − logit(R)) and *step by step* (greedy, up to "
             f"{max_new_tokens} new tokens, last 'Answer: <name>' line; none = wrong). Balanced accuracy (mean of "
             "the accuracies when the answer is C and when it is R), 95% scene-bootstrap CIs, differences paired.", "",
             "| questions | step by step | one word | step by step − one word |", "|---|---|---|---|",
             f"| decisive (*{DEC}*) | {ci(dec_cot)} | {ci(dec_one)} | {ci(diff)} |",
             f"| the other four kinds | {ci(oth_cot)} | {ci(oth_one)} | |", "",
             "**Accuracy by kind:**", "", "| kind | n | step by step | one word |", "|---|---|---|---|"]
    for c in CELLS:
        d = T[T.cell == c]
        lines.append(f"| {c} | {len(d)} | {f2(acc(d, 'cot_pred'))} | {f2(acc(d, 'one_pred'))} |")
    hd = dec
    wc, nc = hd[hd.coords], hd[~hd.coords]
    ws = hd.dropna(subset=["written_side"])
    truth_sign = np.where(ws.truth == "C", 1, -1)
    lines += ["", f"No final answer: {int((T.cot_pred == 'none').sum())}; hit the token limit: {int(T.truncated.sum())};"
              f" median reply length {int(T.n_tokens.median())} tokens.",
              f"Exploratory, decisive questions: {len(wc)} of {len(hd)} replies write (x, y) coordinates (accuracy "
              f"{f2(acc(wc, 'cot_pred'))}), {len(nc)} do not (accuracy {f2(acc(nc, 'cot_pred'))}). Replies giving "
              f"positions to both C and R: {len(ws)}; the written positions put C on the correct side in "
              f"{f2(float(np.mean(ws.written_side.values == truth_sign)) if len(ws) else float('nan'))}, and the final "
              f"answer agrees with them in "
              f"{f2(float(np.mean(np.where(ws.cot_pred == 'C', 1, -1) == ws.written_side.values)) if len(ws) else float('nan'))}."]
    base = model.replace("-8bit", "")
    fl = ROOT / "results" / "flow" / f"{base}_flow_seed{SEED}_items.csv"
    if fl.exists():
        F = pd.read_csv(fl)
        lines.append(f"Check: one-word accuracy on the decisive questions here {f2(acc(dec, 'one_pred'))}; in the "
                     f"bfloat16 flow run of {base} (all {int((F.cell == DEC).sum())}) "
                     f"{f2(float(F.correct[F.cell == DEC].mean()))}.")
    lines += ["", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_cot_axis.md."]
    report = "\n".join(lines)
    print(report, flush=True)
    title = f"Step by step: does written reasoning fix the one-sentence rule? ({model}) — preregistered"
    if rescored:
        title += ", rescored with the corrected answer parser"
    if not smoke:
        append_entry(title, report)
    (out / f"{run}_report.md").write_text(f"## {title}\n\n{report}\n")
    (out / f"{run}.done").write_text(v + "\n")
    notify(f"[{tag}] COMPLETE: {v}. Decisive: step by step {f2(dec_cot[0])}, one word {f2(dec_one[0])}")
    T.to_parquet(out / f"{run}.parquet")
    log("done")


if __name__ == "__main__":
    main()
