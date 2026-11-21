"""Where does written reasoning get M's step (docs/preregistration_readback.md)? Decisive questions in four conditions:
original, flipped (M's step negated on the asked axis), and two mixes where M's sentence comes from one version and the
processed state after it (every layer, every downstream position) from the other. Greedy step-by-step replies.

Resumable: replies are saved after every batch under results/readback and reused on a rerun; --rescore re-reads them
without the model."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geomap.axis import CELLS, axis_bulk_items
from geomap.checkpoint import notify
from geomap.cot_axis import named_coords, parse_final_name, prediction_label, written_side
from geomap.flow import axis_of, flip_step, steps
from geomap.models import load
from geomap.readback import _gen, mix_spec, pad_left, readback_index, record_layers, patched, verdict_readback
from geomap.report import ROOT, append_entry

SEED = 1601
FLOW_SIZES = {"own step only": 400, "agree": 400, "first step only": 600, "conflict, C larger": 400,
              "conflict, M larger": 600}
DEC = "conflict, M larger"
CONDS = ("original", "flipped", "mixed A", "mixed B")     # mixed A: M's sentence original, downstream flipped


def build_items(m, n: int, flow_sizes: dict) -> pd.DataFrame:
    rows = []
    for si, (s, mid, child, qq) in enumerate(axis_bulk_items(m, flow_sizes, seed=SEED)):
        for q, cell in qq:
            if cell != DEC or len(rows) >= n:
                continue
            try:
                sf = flip_step(s, mid, axis_of(q))
            except ValueError:
                continue
            spec = mix_spec(s, sf, q, mid, m.tok, m.family)
            if spec is None:
                continue
            ids_o, ids_f, a, b = spec
            d1, d2, t = steps(s, mid, child, q)
            assert steps(sf, mid, child, q)[2] == -t
            rows.append(dict(scene=si, child=child, root=s.root, mid=mid, axis=axis_of(q), d1=d1, d2=d2,
                             truth_o="C" if t > 0 else "R", truth_f="R" if t > 0 else "C", start=b,
                             ids_o=ids_o, ids_f=ids_f, question=q.text))
    return pd.DataFrame(rows)


def generate(m, items: pd.DataFrame, path: Path, max_new_tokens: int, batch_size: int, tag: str) -> pd.DataFrame:
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["i", "cond", "reply", "n_tokens",
                                                                             "truncated"])
    rows = done.to_dict("records")
    have = set(zip(done.i, done.cond))
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    t0, total = time.time(), len(items) * len(CONDS)
    for cond in CONDS:
        todo = [i for i in range(len(items)) if (i, cond) not in have]
        for k in range(0, len(todo), batch_size):
            idx = todo[k:k + batch_size]
            r = items.iloc[idx]
            X = list(r.ids_o) if cond in ("original", "mixed A") else list(r.ids_f)
            ids, am, st = pad_left(X, m.tok, list(r.start))
            if cond.startswith("mixed"):
                Y = list(r.ids_f) if cond == "mixed A" else list(r.ids_o)
                rec = record_layers(m, pad_left(Y, m.tok, list(r.start))[0], am)
                with patched(m, rec, st, ids.shape[1]):
                    gen = _gen(m, ids, am, max_new_tokens=max_new_tokens)
                del rec
            else:
                gen = _gen(m, ids, am, max_new_tokens=max_new_tokens)
            T = ids.shape[1]
            for j, i in enumerate(idx):
                out = gen[j, T:]
                n = int((out != pad).sum())
                rows.append(dict(i=i, cond=cond, reply=m.tok.decode(out, skip_special_tokens=True), n_tokens=n,
                                 truncated=n >= max_new_tokens))
            pd.DataFrame(rows).to_parquet(path)
            print(f"[readback] {len(rows)}/{total} replies ({time.time() - t0:.0f}s)", flush=True)
            if len(rows) % (batch_size * 10) < batch_size:
                notify(f"[{tag}] {len(rows)}/{total} replies")
    return pd.DataFrame(rows)


def analyse(items: pd.DataFrame, R: pd.DataFrame, model: str, out: Path, run: str, n_boot: int, smoke: bool,
            tag: str, rescored: bool = False):
    meta = items.drop(columns=["ids_o", "ids_f"]).reset_index().rename(columns={"index": "i"})
    D = R.merge(meta, on="i")
    D["final"] = [parse_final_name(r.reply, (r.child, r.root)) for r in D.itertuples()]
    D["label"] = [prediction_label(f, r.child) for f, r in zip(D.final, D.itertuples())]
    text_from_original = D.cond.isin(("original", "mixed A"))
    D["truth_text"] = np.where(text_from_original, D.truth_o, D.truth_f)      # the version of M's sentence
    D["truth_down"] = np.where(D.cond.isin(("original", "mixed B")), D.truth_o, D.truth_f)
    D["answered"] = D.label != "none"
    D["follows_text"] = D.label == D.truth_text
    D.to_parquet(out / f"{run}_scored.parquet")

    def stats(d):
        a = d[d.answered]
        acc_o = a[a.cond == "original"].follows_text.mean()
        acc_f = a[a.cond == "flipped"].follows_text.mean()
        acc = (acc_o + acc_f) / 2
        f_text = a[a.cond.str.startswith("mixed")].follows_text.mean()
        return acc_o, acc_f, acc, f_text, readback_index(f_text, acc)

    acc_o, acc_f, acc, f_text, idx = stats(D)
    scenes = D.scene.unique()
    groups = {s: D[D.scene == s] for s in scenes}
    rng = np.random.default_rng(0)
    boots = [stats(pd.concat([groups[s] for s in rng.choice(scenes, len(scenes))]))[4] for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    v = verdict_readback((idx, lo, hi), acc, acc_o, acc_f)

    f2 = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    lines = [f"Model **{model}**. {len(items)} decisive questions (flow set's generator, seed {SEED}) with a valid "
             "flipped version (M's step negated on the asked axis, which flips the answer). Greedy step-by-step "
             "replies in four conditions: original, flipped, and two mixes in which M's sentence comes from one "
             "version and every layer's state at every later position from the other (mixed A: sentence original, "
             "state flipped; mixed B: the reverse). Shares over replies with an answer.", "",
             "| condition | replies with an answer | follows M's sentence | follows the downstream state |",
             "|---|---|---|---|"]
    for c in CONDS:
        d = D[(D.cond == c) & D.answered]
        down = float(np.mean(d.label == d.truth_down)) if len(d) else float("nan")
        lines.append(f"| {c} | {len(d)} of {int((D.cond == c).sum())} | {f2(d.follows_text.mean())} | "
                     f"{f2(down) if c.startswith('mixed') else '(same version)'} |")
    lines += ["", f"Accuracy original {f2(acc_o)}, flipped {f2(acc_f)}; mixed replies following M's sentence "
              f"{f2(f_text)}. **Index** (1 = follows M's sentence, 0 = follows the downstream state): "
              f"{f2(idx)} [{f2(lo)}, {f2(hi)}]."]

    # exploratory: written positions of M and C in the mixed replies
    mx = D[D.cond.str.startswith("mixed")].copy()
    m_side, c_side = [], []
    for r in mx.itertuples():
        co = named_coords(r.reply, (r.child, r.root, r.mid))
        m_side.append(written_side(co, r.mid, r.root, int(r.axis)))
        c_side.append(written_side(co, r.child, r.root, int(r.axis)))
    mx["m_side"], mx["c_side"] = m_side, c_side
    d1_text = np.where(mx.cond == "mixed A", np.sign(mx.d1), -np.sign(mx.d1))
    c_text = np.where(mx.truth_text == "C", 1, -1)
    wm = mx.m_side.notna().to_numpy()
    wc = mx.c_side.notna().to_numpy()
    both = wm & wc
    m_txt = (mx.m_side.to_numpy() == d1_text)
    c_txt = (mx.c_side.to_numpy() == c_text)
    lines += ["", f"Exploratory, mixed replies: {int(wm.sum())} give M a position relative to R, which follows M's "
              f"sentence in {f2(float(m_txt[wm].mean()) if wm.any() else float('nan'))}; {int(wc.sum())} give C one, "
              f"which follows M's sentence in {f2(float(c_txt[wc].mean()) if wc.any() else float('nan'))}; of "
              f"{int(both.sum())} giving both, {int((both & (m_txt != c_txt)).sum())} mix the versions (M from one, C "
              f"from the other). No answer: " + ", ".join(f"{c} {int(((D.cond == c) & ~D.answered).sum())}"
                                                          for c in CONDS) + "."]
    lines += ["", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_readback.md."]
    report = "\n".join(lines)
    print(report, flush=True)
    title = f"Read-back: does written reasoning take M's step from its sentence or the processed state? ({model}) — preregistered"
    if rescored:
        title += ", rescored"
    if not smoke:
        append_entry(title, report)
    (out / f"{run}_report.md").write_text(f"## {title}\n\n{report}\n")
    (out / f"{run}.done").write_text(v + "\n")
    notify(f"[{tag}] COMPLETE: {v}. Index {f2(idx)} [{f2(lo)}, {f2(hi)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 4 questions, 12 tokens; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/readback)")
    ap.add_argument("--rescore", action="store_true", help="re-read saved replies (no model)")
    a = ap.parse_args()
    out, flow_sizes, n_boot = ROOT / "results" / "readback", dict(FLOW_SIZES), 1000
    if a.smoke:
        import tempfile
        a.model, a.n, a.max_new_tokens, out, n_boot = "tiny", 4, 12, Path(tempfile.mkdtemp()), 50
        flow_sizes = {c: 12 for c in CELLS}
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run, tag = f"{a.model}_readback_seed{SEED}", f"Read-back {a.model}"
    ipath, rpath = out / f"{run}_items.parquet", out / f"{run}_replies.parquet"
    if a.rescore:
        analyse(pd.read_parquet(ipath), pd.read_parquet(rpath), a.model, out, run, n_boot, a.smoke, tag, True)
        return
    if (out / f"{run}.done").exists():
        print(f"[readback] already complete; delete {run}.done to rerun", flush=True)
        return
    m = load(a.model)
    items = build_items(m, a.n, flow_sizes)
    items.to_parquet(ipath)
    print(f"[readback] {len(items)} decisive questions with a valid flipped version", flush=True)
    notify(f"[{tag}] started: {len(items)} questions x 4 conditions")
    R = generate(m, items, rpath, a.max_new_tokens, a.batch_size, tag)
    analyse(items, R, a.model, out, run, n_boot, a.smoke, tag)


if __name__ == "__main__":
    main()
