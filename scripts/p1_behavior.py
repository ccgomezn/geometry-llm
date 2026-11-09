"""P1: behaviour screen. Which model solves the puzzles without written reasoning?"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci, chance, error_signature, gate
from geomap.models import load
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import QType, all_questions, answer_for, battery, margin
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import candidate_mass, parse_cot_answer, predict, score

OUT = ROOT / "results" / "p1"
FIG = ROOT / "results" / "figures"


def max_hop(scene, q):
    objs = set(q.args) | ({answer_for(q, scene.coords)} & set(scene.names))
    return max(scene.hop[o] for o in objs)


def make_sets(m, n_scenes, seed, near_n, abl_n, cot_n):
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    rows = []  # (set, scene, question, pool, include_list)
    sid = 0
    for n in (3, 4, 5):
        for i in range(n_scenes):
            pool = i % 2
            s = generate_scene(rng, pools[pool], n)
            rows += [("main", sid, s, q, pool, True) for q in battery(s, rng, per_type=2)]
            sid += 1
    for i in range(near_n):
        s = generate_scene(rng, pools[i % 2], 4)
        near = [q for q in all_questions(s, min_margin=0.25, min_angle=2.5)
                if margin(q, s.coords) < (10 if q.qtype == QType.DIRECTION else 1.0)]
        for j in rng.permutation(len(near))[:6]:
            rows.append(("neartie", sid, s, near[j], i % 2, True))
        sid += 1
    for i in range(abl_n):
        s = generate_scene(rng, pools[i % 2], 4)
        for q in battery(s, rng, per_type=1):
            rows.append(("list", sid, s, q, i % 2, True))
            rows.append(("nolist", sid, s, q, i % 2, False))
        sid += 1
    cot = []
    for i in range(cot_n):
        s = generate_scene(rng, pools[i % 2], 4)
        qs = battery(s, rng, per_type=1)
        q = qs[int(rng.integers(len(qs)))]
        rows.append(("nocot", sid, s, q, i % 2, True))
        cot.append((sid, s, q, i % 2))
        sid += 1
    return rows, cot


def run_cot(m, cot, max_new_tokens, batch_size=8):
    out = []
    tok = m.tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for b in range(0, len(cot), batch_size):
        batch = cot[b:b + batch_size]
        specs = [build_prompt(s, q, tok, m.family, thinking=True).input_ids for _, s, q, _ in batch]
        T = max(map(len, specs))
        ids = torch.full((len(specs), T), pad, dtype=torch.long)
        am = torch.zeros_like(ids)
        for j, x in enumerate(specs):  # left padding for generation
            ids[j, T - len(x):] = torch.tensor(x)
            am[j, T - len(x):] = 1
        with torch.no_grad():
            gen = m.hf.generate(input_ids=ids.to(m.hf.device), attention_mask=am.to(m.hf.device),
                                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
        for j, (sid, s, q, pool) in enumerate(batch):
            text = tok.decode(gen[j, T:], skip_special_tokens=False)
            ans = parse_cot_answer(text, q.candidates)
            out.append(dict(set="cot", scene_id=sid, n_objects=len(s.names), qtype=q.qtype.value,
                            max_hop=max_hop(s, q), margin=margin(q, s.coords), pool=pool,
                            correct=ans == answer_for(q, s.coords), chance=chance(q), answer=ans,
                            gen_tokens=int((gen[j, T:] != pad).sum())))
    return out


def screen(key, a):
    m = load(key)
    rows, cot = make_sets(m, a.n_scenes, a.seed, a.near_n, a.abl_n, a.cot_n)
    items = [(build_prompt(s, q, m.tok, m.family, include_list=inc), q) for _, _, s, q, _, inc in rows]
    scores = score(m, items, batch_size=a.batch_size)
    recs = []
    for (st, sid, s, q, pool, _), sc in zip(rows, scores):
        recs.append(dict(set=st, scene_id=sid, n_objects=len(s.names), qtype=q.qtype.value,
                         max_hop=max_hop(s, q), margin=margin(q, s.coords), pool=pool,
                         correct=predict(sc) == answer_for(q, s.coords), chance=chance(q),
                         answer=predict(sc), scores=json.dumps(sc), mass=candidate_mass(sc)))
    if m.family == "qwen3" and a.cot_n:
        recs += run_cot(m, cot, a.cot_tokens, batch_size=a.cot_batch)
    df = pd.DataFrame(recs)
    df.to_parquet(OUT / f"{key}.parquet")
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return df


def fmt(ci):
    return f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]"


def summarize(key, df):
    lines = [f"### {key}"]
    g = gate(df)
    lines.append(f"- 4-object accuracy: {fmt(g['acc4'])}; accuracy above chance on 2+ hop questions: "
                 f"{fmt(g['hop2_excess'])} → gate **{'PASS' if g['pass'] else 'fail'}**")
    main = df[df.set == "main"]
    tab = []
    for qt, d in main.groupby("qtype"):
        accs = [fmt(bootstrap_ci(d[d.n_objects == n].correct.astype(float), groups=d[d.n_objects == n].scene_id))
                if (d.n_objects == n).any() else "–" for n in (3, 4, 5)]
        tab.append(f"| {qt} | {d.chance.mean():.2f} | " + " | ".join(accs) + " |")
    lines += ["", "| question | chance | 3 objects | 4 objects | 5 objects |", "|---|---|---|---|---|"] + tab
    hop = [f"hop {h}: {fmt(bootstrap_ci(d.correct.astype(float), groups=d.scene_id))}"
           for h, d in main.groupby("max_hop")]
    lines.append("\n- By deepest object involved: " + "; ".join(hop))
    try:
        sig = error_signature(main)
        lines.append(f"- Error signature (standardised logistic coefficients): hop {fmt(sig['hop'])}, "
                     f"log margin {fmt(sig['log_margin'])}")
    except Exception as e:  # noqa: BLE001
        sig = None
        lines.append(f"- Error signature failed: {e}")
    if "mass" in main:
        lines.append(f"- Validity: median probability on the allowed answers {float(np.exp(main['mass'].median())):.2f} "
                     f"(5th percentile {float(np.exp(main['mass'].quantile(0.05))):.2f})")
    nt = df[df.set == "neartie"]
    if len(nt):
        lines.append(f"- Near-tie set (margin 0.25–1 unit / 2.5–10°): {fmt(bootstrap_ci(nt.correct.astype(float), groups=nt.scene_id))}")
    wl, nl = df[df.set == "list"], df[df.set == "nolist"]
    if len(wl):
        diff = wl.correct.astype(float).values - nl.correct.astype(float).values
        lines.append(f"- Object-list line: with {wl.correct.mean():.2f}, without {nl.correct.mean():.2f}, "
                     f"difference {fmt(bootstrap_ci(diff, groups=wl.scene_id.values))}")
    if (df.set == "cot").any():
        c, nc = df[df.set == "cot"], df[df.set == "nocot"]
        lines.append(f"- Chain of thought on: {fmt(bootstrap_ci(c.correct.astype(float)))} vs off "
                     f"{fmt(bootstrap_ci(nc.correct.astype(float)))} (same {len(c)} questions; "
                     f"unparsed {c.answer.isna().mean():.0%}, median {int(c.gen_tokens.median())} tokens)")
    return "\n".join(lines), g, sig


def figures(results):
    keys = list(results)
    qts = sorted(results[keys[0]].qtype.unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(keys)
    for k, key in enumerate(keys):
        d = results[key]
        d = d[(d.set == "main") & (d.n_objects == 4)]
        cis = [bootstrap_ci(d[d.qtype == q].correct.astype(float), n_boot=300) for q in qts]
        x = np.arange(len(qts)) + k * w
        ax.bar(x, [c[0] for c in cis], w, yerr=[[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]],
               label=key, capsize=2)
    ch = results[keys[0]].groupby("qtype").chance.mean()
    for i, q in enumerate(qts):
        ax.hlines(ch[q], i - 0.1, i + 0.8, colors="k", linestyles="dashed", lw=1)
    ax.set_xticks(np.arange(len(qts)) + 0.4 - w / 2, qts, rotation=20)
    ax.set_ylabel("accuracy (4 objects)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("P1: accuracy without written reasoning (dashed = chance)")
    fig.tight_layout()
    fig.savefig(FIG / "p1_accuracy.png", dpi=150)
    plt.close(fig)
    return [FIG / "p1_accuracy.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-1.7b", "qwen3-4b", "qwen3-8b"])
    ap.add_argument("--n-scenes", type=int, default=500)
    ap.add_argument("--near-n", type=int, default=200)
    ap.add_argument("--abl-n", type=int, default=200)
    ap.add_argument("--cot-n", type=int, default=100)
    ap.add_argument("--cot-tokens", type=int, default=1024)
    ap.add_argument("--cot-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reuse", action="store_true", help="reuse saved parquet files")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results, parts, chosen = {}, [], None
    for key in a.models:  # ascending size
        path = OUT / f"{key}.parquet"
        results[key] = pd.read_parquet(path) if (a.reuse and path.exists()) else screen(key, a)
        text, g, _ = summarize(key, results[key])
        parts.append(text)
        if g["pass"] and chosen is None:
            chosen = key
        print(text, flush=True)
    figs = figures(results)
    verdict = (f"**Chosen model: {chosen}**, the smallest that passes the gate." if chosen else
               "**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.")
    append_entry("P1 behaviour screen", verdict + "\n\n" + "\n\n".join(parts), figs)
    (OUT / "chosen.json").write_text(json.dumps({"chosen": chosen}))


if __name__ == "__main__":
    main()
