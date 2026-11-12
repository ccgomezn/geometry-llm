"""P10 (docs/preregistration_p10.md): push the question to attend to the map (the object-name tokens in the list) in
layers 16-31, against a matched control (the list token before each name). Do chain answers improve?

Resumable: every run is saved under results/p10 and skipped on a rerun. Progress goes to Slack if $SLACK_WEBHOOK_URL
is set."""

import argparse
import time
from pathlib import Path

import pandas as pd

from geomap.checkpoint import df_step, notify
from geomap.followups import attention_boost, boost_summary, chain_items, p10_keys, p7_class, verdict_p10
from geomap.models import load
from geomap.prompts import build_prompt
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scoring import predict, score

BIASES = (1.0, 2.0, 4.0, 8.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 0-1; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/p10)")
    a = ap.parse_args()
    out, layers = ROOT / "results" / "p10", list(range(16, 32))
    if a.smoke:
        import tempfile
        out, layers = Path(tempfile.mkdtemp()), [0, 1]
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"P10 {a.model}"
    done = out / f"{a.model}_p10.done"
    if done.exists():
        print(f"[p10] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    items = chain_items(m, a.n_test, seed=702, exclude_qtypes=("closer_than",))
    flat = [(si, qi, s, q, p7_class(q, s.root, mid, child))
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, _) in enumerate(qs)]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _ in flat]
    q_from = [spec.prefix_len for spec, _ in specs]
    keys = {k: [p10_keys(spec)[k] for spec, _ in specs] for k in ("names", "control")}
    print(f"[p10] {len(flat)} questions in {len(items)} chains, layers {layers[0]}-{layers[-1]}", flush=True)

    def run(cond, bias):
        ctx = None if cond == "clean" else attention_boost(m, q_from, keys[cond], layers, bias)
        preds = [predict(x) for x in score(m, specs, batch_size=a.batch_size, batch_ctx=ctx)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, cls=cls, cond=cond, bias=bias, pred=p,
                                  truth=t, correct=float(p == t))
                             for (si, qi, _, q, cls), p, t in zip(flat, preds, truth)])

    runs = [("clean", 0.0)] + [(c, b) for b in BIASES for c in ("names", "control")]
    dfs = []
    for cond, bias in runs:
        path = out / f"{a.model}_p10_run_{cond}_{bias:g}.parquet"
        fresh = not path.exists()
        d = df_step(path, lambda c=cond, b=bias: run(c, b))
        dfs.append(d)
        msg = f"{cond} +{bias:g}: acc " + str(d.groupby("cls").correct.mean().round(3).to_dict())
        print(f"[p10] {msg} ({time.time() - t0:.0f}s)", flush=True)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{a.model}_p10.parquet")
    s = boost_summary(df)
    s.to_csv(out / f"{a.model}_p10_summary.csv", index=False)
    v = verdict_p10(s)

    lines = [f"Model **{a.model}**, {len(items)} new chains R → M → C, {len(flat)} questions. In attention layers "
             f"{layers[0]}–{layers[-1]}, +b is added to the attention logits from every query after the object list to "
             "the three object-name tokens in the list (*names*, where the map is decoded), or to the list token just "
             "before each name (*control*). Accuracy change vs clean; contrast = Δ(names) − Δ(control) with a "
             "scene-bootstrap 95% CI.", "",
             "| boost b | questions | clean | Δ names | Δ control | contrast [95% CI] | n |", "|---|---|---|---|---|---|---|"]
    for r in s.sort_values(["bias", "cls"]).itertuples():
        lines.append(f"| +{r.bias:g} | {r.cls} | {r.clean:.2f} | {r.d_names:+.3f} | {r.d_control:+.3f} | "
                     f"{r.contrast:+.3f} [{r.contrast_lo:+.3f}, {r.contrast_hi:+.3f}] | {r.n} |")
    lines += ["", f"Preregistered verdict (CR): **{v}**. Rules: docs/preregistration_p10.md."]
    if not a.smoke:
        append_entry(f"P10 attention boost toward the map ({a.model}) — preregistered", "\n".join(lines), [])
    done.write_text(v)
    notify(f"[{tag}] COMPLETE. {v}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
