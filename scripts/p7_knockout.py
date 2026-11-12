"""P7 (docs/preregistration_p7.md): what does the question read? Attention knockout from the question tokens to the
object list (where the map is decodable), to C's own sentence, and to its parent M's sentence, on R -> M -> C chains.

Resumable: every run is saved under results/p7 and skipped on a rerun. Progress goes to Slack if $SLACK_WEBHOOK_URL
is set. Before any run, a self-check on the real model confirms that a sealed span cannot influence the answer."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geomap.checkpoint import df_step, notify
from geomap.followups import attention_knockout, chain_items, knockout_summary, p7_class, p7_spans, verdict_p7
from geomap.models import load
from geomap.prompts import build_prompt
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scoring import predict, score

TARGETS = ("list", "own", "parent")


def self_check(m, flat, specs, layers, batch_size):
    """Block every later position from C's sentence and scramble its words: the scores must not change."""
    items, blocks, scrambled = [], [], []
    for (si, qi, s, q, cls, mid, child), (spec, _) in list(zip(flat, specs))[:3]:
        lo, hi = spec.relation_spans[child]
        ids = list(spec.input_ids)
        ids[lo:hi] = list(reversed(ids[lo:hi]))
        items.append((spec, q))
        scrambled.append((type(spec)(**{**spec.__dict__, "input_ids": ids}), q))
        blocks.append([(hi, lo, hi)])
    ctx = attention_knockout(m, blocks, layers)
    a = score(m, items, batch_size=batch_size, batch_ctx=ctx)
    b = score(m, scrambled, batch_size=batch_size, batch_ctx=ctx)
    worst = max(abs(x[k] - y[k]) for x, y in zip(a, b) for k in x)
    assert worst < 0.05, f"knockout self-check failed: sealed span changed the scores by {worst:.3f}"
    print(f"[p7] self-check passed (max score change {worst:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/p7)")
    a = ap.parse_args()
    out = ROOT / "results" / "p7"
    if a.smoke:
        import tempfile
        out = Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"P7 {a.model}"
    done = out / f"{a.model}_p7.done"
    if done.exists():
        print(f"[p7] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    L = m.n_layers
    windows = {"all": list(range(L)), f"0-{L // 3 - 1}": list(range(L // 3)),
               f"{L // 3}-{2 * L // 3 - 1}": list(range(L // 3, 2 * L // 3)), f"{2 * L // 3}-{L - 1}": list(range(2 * L // 3, L))}

    items = chain_items(m, a.n_test, seed=402, exclude_qtypes=("closer_than",))
    flat = [(si, qi, s, q, p7_class(q, s.root, mid, child), mid, child)
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, _) in enumerate(qs)]
    flat = [f for f in flat if f[4] is not None]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _, _, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _, _, _ in flat]
    counts = pd.Series([f[4] for f in flat]).value_counts().to_dict()
    print(f"[p7] {len(flat)} questions in {len(items)} chains, classes {counts}", flush=True)
    self_check(m, flat, specs, windows["all"], a.batch_size)

    def run(target, window):
        if target == "-":
            ctx = None
        else:
            blocks = [[(spec.prefix_len, *p7_spans(spec, f[6], f[5])[target])] for f, (spec, _) in zip(flat, specs)]
            ctx = attention_knockout(m, blocks, windows[window])
        preds = [predict(sc) for sc in score(m, specs, batch_size=a.batch_size, batch_ctx=ctx)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, cls=cls, target=target, window=window,
                                  pred=p, truth=t, correct=float(p == t))
                             for (si, qi, _, q, cls, _, _), p, t in zip(flat, preds, truth)])

    runs = [("-", "-")] + [(t, w) for w in windows for t in TARGETS]
    dfs = []
    for target, window in runs:
        path = out / f"{a.model}_p7_run_{target}_{window}.parquet"
        fresh = not path.exists()
        d = df_step(path, lambda t=target, w=window: run(t, w))
        dfs.append(d)
        by = d.groupby("cls").correct.mean().round(3).to_dict()
        msg = f"block {target} at layers {window}: acc {by}"
        print(f"[p7] {msg} ({time.time() - t0:.0f}s)", flush=True)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{a.model}_p7.parquet")
    s = knockout_summary(df)
    s.to_csv(out / f"{a.model}_p7_summary.csv", index=False)
    main_v, list_v = verdict_p7(s)

    clean = df[df.target == "-"].groupby("cls").correct.agg(["mean", "size"])
    lines = [f"Model **{a.model}**, {len(items)} new chains R → M → C, {len(flat)} questions "
             f"({', '.join(f'{c}: {int(r['size'])}' for c, r in clean.iterrows())}). The question tokens (everything "
             f"after the object list) may not attend to one span, in the given layers. Clean accuracy: "
             f"{', '.join(f'{c} {r['mean']:.2f}' for c, r in clean.iterrows())}.", "",
             "Accuracy drop (clean − knocked out), with scene-bootstrap 95% CIs:", "",
             "| blocked span | layers | CR (needs C's composed position) | CM (C vs its parent) | MR (no C) |",
             "|---|---|---|---|---|"]
    for t in TARGETS:
        for w in windows:
            cells = []
            for c in ("CR", "CM", "MR"):
                r = s[(s.target == t) & (s.window == w) & (s.cls == c)]
                cells.append(f"{r.acc_drop.iloc[0]:.3f} [{r.acc_drop_lo.iloc[0]:.3f}, {r.acc_drop_hi.iloc[0]:.3f}]"
                             if len(r) else "–")
            lines.append(f"| {t} | {w} | " + " | ".join(cells) + " |")
    lines += ["", f"Preregistered verdicts: **{main_v}**; **{list_v}**. Rules: docs/preregistration_p7.md."]
    if not a.smoke:
        append_entry(f"P7 attention knockout: what does the question read? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(f"{main_v}; {list_v}")
    notify(f"[{tag}] COMPLETE. {main_v}; {list_v}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
