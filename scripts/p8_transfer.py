"""P8 (docs/preregistration_p8.md): copy C's word-identical sentence from the run where M (and with it C) moved, at
every layer from L to the last. Does the base run then answer as if C sat at its moved position, M unmoved?

Resumable: results are saved in chunks of pairs under results/p8 and skipped on a rerun. Progress goes to Slack if
$SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path

import pandas as pd

from geomap.checkpoint import df_step, notify
from geomap.models import load
from geomap.report import ROOT, append_entry
from geomap.transfer import make_transfer_pairs, run_transfer_multi, summarize_transfer, verdict_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/p8)")
    a = ap.parse_args()
    out = ROOT / "results" / "p8"
    if a.smoke:
        import tempfile
        out = Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"P8 {a.model}"
    done = out / f"{a.model}_p8.done"
    if done.exists():
        print(f"[p8] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    starts = sorted(set(range(0, m.n_layers + 1, a.step)) | {m.n_layers})
    pairs = make_transfer_pairs(m, a.n, seed=501, exclude_qtypes=("closer_than",))
    print(f"[p8] {len(pairs)} pairs, starts {starts}", flush=True)

    def chunk(c0):
        ds = []
        for p in range(c0, min(c0 + a.chunk, len(pairs))):
            d = run_transfer_multi(m, pairs[p], starts, batch_size=a.batch_size)
            d["pair"] = p
            ds.append(d)
        return pd.concat(ds, ignore_index=True)

    dfs = []
    for c0 in range(0, len(pairs), a.chunk):
        path = out / f"{a.model}_p8_pairs_{c0:03d}.parquet"
        fresh = not path.exists()
        dfs.append(df_step(path, lambda c=c0: chunk(c)))
        print(f"[p8] pairs {c0}-{min(c0 + a.chunk, len(pairs)) - 1} done ({time.time() - t0:.0f}s)", flush=True)
        if fresh and (c0 // a.chunk) % 5 == 4:
            notify(f"[{tag}] {min(c0 + a.chunk, len(pairs))}/{len(pairs)} pairs done")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{a.model}_p8.parquet")
    s = summarize_transfer(df, n_boot=1000)
    s.to_csv(out / f"{a.model}_p8_summary.csv", index=False)
    v, at = verdict_multi(s)

    lines = [f"Model **{a.model}**, {len(pairs)} new chain pairs (R → M → C). Only M's sentence differs between base and "
             "source; C's word-identical sentence is copied from the source run at **every layer from L to the last** "
             "(0 = embeddings). *Follows target* = answer equals the model's answer on a scene with C at its moved "
             "position and M unmoved. L = 48 copies only final outputs, which nothing downstream reads (a null).", "",
             "| copied from layer L onward | follows target | follows source (M moved too) | controls stable | target Qs | control Qs |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r.layer} | {r.follow_T:.2f} [{r.follow_T_lo:.2f}, {r.follow_T_hi:.2f}] | {r.follow_S:.2f} | "
              f"{r.stable:.2f} [{r.stable_lo:.2f}, {r.stable_hi:.2f}] | {r.n_T} | {r.n_ctrl} |" for r in s.itertuples()]
    lines += ["", f"Preregistered verdict: **{v}**" + (f" (latest L meeting the criterion: {at})" if at is not None else "")
              + ". Rules: docs/preregistration_p8.md."]
    if not a.smoke:
        append_entry(f"P8 multi-layer transfer of the dependent's sentence ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(f"{v}; latest L {at}")
    best = s.loc[s.follow_T.idxmax()]
    notify(f"[{tag}] COMPLETE. {v}" + (f" (up to L={at})" if at is not None else "") +
           f". Best follow_T {best.follow_T:.2f} at L={int(best.layer)} (stable {best.stable:.2f}).")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
