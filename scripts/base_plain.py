"""Base models in plain text (docs/preregistration_base_plain.md): on the flow set's "which is farther east / north,
C or R?" questions, does a pretrained-only model (or an instruct model without its chat template) read the composed
position, or is its map outvoted as in the instruct models' chat answers?

With `--shots neutral|informative` (docs/preregistration_base_fewshot.md) each test item is preceded by six worked
examples. One run = one model in one format. Resumable: answers and final-position residuals are cached under results/base_plain.
`--summary` (with the same `--shots`) applies the overall rule to the finished base-model runs and appends one entry to results/RESULTS.md."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.axis import CELLS, axis_bulk_items
from geomap.baseplain import (overall_base_plain, overall_fewshot, pick_shots, rule_answer, shot_bank, shots_text,
                              verdict_base_plain)
from geomap.cache import cache_prefix
from geomap.checkpoint import notify
from geomap.crossfamily import wilson
from geomap.flow import axis_of, axis_predict, axis_r2, layer_grid, steps
from geomap.models import load
from geomap.probes import scene_split
from geomap.prompts import build_prompt
from geomap.scenes import relation_text
from geomap.report import ROOT, append_entry
from geomap.scoring import score
from geomap.study2 import ba_ci

SEED = 1601
SIZES = {"own step only": 400, "agree": 400, "first step only": 600, "conflict, C larger": 400,
         "conflict, M larger": 600}
DEC, FIRST, OWN = "conflict, M larger", "first step only", "own step only"
CHAT = {"qwen3-8b-base": "qwen3-8b", "llama3.1-8b-base": "llama3.1-8b", "gemma3-12b-pt": "gemma3-12b",
        "gemma3-12b-pt-8bit": "gemma3-12b", "qwen3-8b": "qwen3-8b", "llama3.1-8b": "llama3.1-8b",
        "gemma3-12b": "gemma3-12b", "gemma3-12b-8bit": "gemma3-12b"}
BASE = ("qwen3-8b-base", "llama3.1-8b-base", "gemma3-12b-pt", "gemma3-12b-pt-8bit")
INSTRUCT = ("qwen3-8b", "llama3.1-8b", "gemma3-12b", "gemma3-12b-8bit")
NAMES = {"qwen3-8b-base": "Qwen3-8B-Base", "llama3.1-8b-base": "Llama 3.1 8B (base)", "gemma3-12b-pt": "Gemma 3 12B pt",
         "gemma3-12b-pt-8bit": "Gemma 3 12B pt (8-bit)", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B Instruct",
         "gemma3-12b": "Gemma 3 12B it", "gemma3-12b-8bit": "Gemma 3 12B it (8-bit)"}


def tag_of(fmt: str, shots: str) -> str:
    return fmt if shots == "none" else f"{fmt}-fewshot-{shots}"


def collect(m, sizes: dict, fmt: str, shots: str = "none"):
    items = axis_bulk_items(m, sizes, seed=SEED)
    bank = shot_bank(m) if shots != "none" else []
    if bank:
        test = {tuple(relation_text(r) for r in s.relations) for s, *_ in items}
        assert not test & {tuple(relation_text(r) for r in e["scene"].relations) for e in bank}, "bank overlaps test"
    rows, specs, qs = [], [], []
    for si, (s, mid, child, qq) in enumerate(items):
        prefix = shots_text(pick_shots(bank, set(s.names), shots, np.random.default_rng(si))) if bank else ""
        for q, cell in qq:
            d1, d2, truth = steps(s, mid, child, q)
            rows.append(dict(scene=si, cell=cell, axis=axis_of(q), c_first=q.args[0] == child, d1=d1, d2=d2,
                             truth=truth, child=child, root=s.root, mid=mid))
            specs.append(build_prompt(s, q, m.tok, m.family, fmt=fmt, shots=prefix))
            qs.append(q)
    return pd.DataFrame(rows), specs, qs, len(items)


def final_residuals(m, specs, layers, batch_size) -> torch.Tensor:
    """[N, L, d] residual stream at the last prompt position."""
    sp = [type(s)(**{**s.__dict__, "list_pos": {"final": len(s.input_ids) - 1}}) for s in specs]
    return cache_prefix(m, sp, layers, batch_size)["list"][:, :, 0].clone()


def analyse(meta: pd.DataFrame, acts: torch.Tensor, layers: list[int], n_layers: int, n_scenes: int, model: str,
            fmt: str, n_boot: int, out: Path, shots: str = "none") -> tuple[str, str]:
    tag = tag_of(fmt, shots)
    _, _, win = layer_grid(n_layers)
    tr = np.isin(meta.scene.to_numpy(), scene_split(n_scenes)[0])
    g, ax = meta.scene.to_numpy(), meta.axis.to_numpy()
    d1, d2, truth = (meta[k].to_numpy(float) for k in ("d1", "d2", "truth"))
    held = ~tr & (meta.cell == DEC).to_numpy()
    held_first = ~tr & (meta.cell == FIRST).to_numpy()
    pr = []
    for li, layer in enumerate(layers):
        X = acts[:, li].float().numpy()
        ok = np.sign(axis_predict(X, truth, ax, g, tr)) == truth
        pr.append(dict(layer=layer, r2_d1=axis_r2(X, d1, ax, g, tr), r2_d2=axis_r2(X, d2, ax, g, tr),
                       truth_dec=ok[held].mean(), truth_first=ok[held_first].mean()))
    P = pd.DataFrame(pr)
    P.to_csv(out / f"{model}_{tag}_seed{SEED}_probes.csv", index=False)
    fq = P[P.layer.between(*win["final"])].sort_values("layer")
    best = fq.loc[fq.truth_dec.idxmax()]
    n = int(held.sum())
    probe = wilson(int(round(best.truth_dec * n)), n)
    model_ci = wilson(int(meta.correct[held].sum()), n)
    ba = {c: ba_ci(meta.truth[meta.cell == c], meta.pred[meta.cell == c], 1, meta.scene[meta.cell == c],
                   n_boot=n_boot) for c in CELLS}
    r2_d1, r2_d2 = float(fq.r2_d1.max()), float(fq.r2_d2.max())
    v = verdict_base_plain(ba[DEC], ba[FIRST], ba[OWN], r2_d1, r2_d2, probe, model_ci,
                           conflict_c=None if shots == "none" else ba["conflict, C larger"])
    rules = "docs/preregistration_base_plain.md" if shots == "none" else "docs/preregistration_base_fewshot.md"

    f2 = lambda x: f"{x:.2f}"
    ci = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    rule = rule_answer(meta.d2)
    shot_desc = "" if shots == "none" else (f", preceded by six worked **{shots}** examples (seed-7 bank, no names "
                                            "shared with the test scene)")
    lines = [f"Model **{NAMES.get(model, model)}** (`{model}`), **{fmt}** prompt (begin-of-text token, scene, "
             f"`Question: …`, `Answer:`; candidates read with a leading space){shot_desc}. The flow set's questions (seed "
             f"{SEED}, {len(meta)} questions in {n_scenes} scenes); answer = sign of log p(C) − log p(R). Balanced "
             "accuracy with 95% scene-bootstrap CIs.", "",
             "| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |", "|---|---|---|---|"]
    for c in CELLS:
        k = (meta.cell == c).to_numpy()
        lines.append(f"| {c} | {ci(ba[c])} | {f2(meta.correct[k].mean())} | {f2(np.mean(meta.pred[k] == rule[k]))} |")
    lines += ["", f"Final position, layers {win['final'][0]}–{win['final'][1]} of {n_layers}: best held-out R² of M's "
              f"step {f2(r2_d1)}, of C's own step {f2(r2_d2)}. Truth probe on the {n} held-out decisive questions "
              f"(best layer {int(best.layer)}): {ci(probe)}; the model on the same questions {ci(model_ci)}. "
              f"Exploratory: probe at the window's first layer {f2(fq.truth_dec.iloc[0])}, mean over the window "
              f"{f2(fq.truth_dec.mean())}; on held-out first-step-only {f2(fq.truth_first.max())}.", ""]
    first_named = np.mean((meta.pred == 1) == meta.c_first)
    expl = f"Exploratory: the answer names the first-named object in {f2(first_named)} of all questions."
    if shots != "none":
        w = np.linalg.lstsq(np.c_[d1, d2, np.ones(len(meta))], meta.ld.to_numpy(float), rcond=None)[0]
        m_rule = np.where(d1 > 0, 1, -1)
        expl += (f" Weight of M's step relative to C's own step in the margin {f2(w[0] / w[1])} (1 = composes). "
                 "Answers following M's sentence (C iff M's step points the asked way): "
                 + ", ".join(f"{c} {f2(np.mean(meta.pred[k] == m_rule[k]))}" for c in CELLS
                             for k in [(meta.cell == c).to_numpy()]) + ".")
        zs = out / f"{model}_{fmt}_seed{SEED}_items.csv"
        if zs.exists():
            Z = pd.read_csv(zs)
            if len(Z) == len(meta) and (Z.d1.to_numpy() == meta.d1.to_numpy()).all() and (Z.child == meta.child).all():
                dk = (meta.cell == DEC).to_numpy()
                expl += (f" Agreement with the zero-shot plain run: {f2(np.mean(Z.pred == meta.pred))} of all "
                         f"questions, {f2(np.mean(Z.pred[dk] == meta.pred[dk]))} of the decisive ones.")
    chat = ROOT / "results" / "flow" / f"{CHAT.get(model, model)}_flow_seed{SEED}_items.csv"
    if chat.exists() and shots == "none":
        C = pd.read_csv(chat)
        same = len(C) == len(meta) and all((C[k].to_numpy() == meta[k].to_numpy()).all()
                                           for k in ("scene", "cell", "d1", "d2", "child", "root"))
        if same:
            cp = np.where(C.ld > 0, 1, -1)
            dk = (meta.cell == DEC).to_numpy()
            expl += (f" Same questions as the chat-format run of `{CHAT[model]}`: the answers agree on "
                     f"{f2(np.mean(cp == meta.pred))} of all questions and {f2(np.mean(cp[dk] == meta.pred[dk]))} of "
                     f"the decisive ones (chat-format balanced accuracy on the decisive questions "
                     f"{ci(ba_ci(C.truth[dk], cp[dk], 1, C.scene[dk], n_boot=n_boot))}).")
        else:
            expl += f" (The chat-format run of `{CHAT[model]}` has different questions; no comparison.)"
    lines += [expl, "", f"Preregistered verdict: **{v}**. Rules: {rules}."]
    return v, "\n".join(lines)


def run(a):
    out, sizes, n_boot = ROOT / "results" / "base_plain", dict(SIZES), 1000
    if a.smoke:
        import tempfile
        a.model, sizes, n_boot = "tiny", {c: 10 for c in CELLS}, 50
        out = Path(a.out) if a.out else Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.model}_{tag_of(a.fmt, a.shots)}_seed{SEED}"
    done = out / f"{name}.done"
    if done.exists():
        print(f"[base-plain] {name} already complete; delete {done.name} to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[base-plain] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    layers = layer_grid(m.n_layers)[0]
    if a.smoke:
        layers = [0, 1, 2]
    meta, specs, qs, n_scenes = collect(m, sizes, a.fmt, a.shots)
    log(f"{len(meta)} questions in {n_scenes} scenes; mean prompt length {np.mean([len(s.input_ids) for s in specs]):.0f}")
    notify(f"[Base/plain {name}] started: {len(meta)} questions")
    ipath = out / f"{name}_items.csv"
    if ipath.exists():
        old = pd.read_csv(ipath)
        assert (old.d1.to_numpy() == meta.d1.to_numpy()).all() and (old.child == meta.child).all()
        meta["ld"] = old.ld
    else:
        sc = score(m, list(zip(specs, qs)), batch_size=a.batch_size)
        meta["ld"] = [x[r.child] - x[r.root] for x, r in zip(sc, meta.itertuples())]
    meta["pred"] = np.where(meta.ld > 0, 1, -1)
    meta["correct"] = meta.pred == meta.truth
    meta.to_csv(ipath, index=False)
    log("answers scored")
    apath = Path(a.acts_dir or out) / f"{name}_acts.pt"
    if apath.exists():
        acts = torch.load(apath)
    else:
        acts = final_residuals(m, specs, layers, a.batch_size)
        torch.save(acts, apath)
    log(f"final-position residuals {tuple(acts.shape)}")
    v, report = analyse(meta, acts, layers, m.n_layers, n_scenes, a.model, a.fmt, n_boot, out, a.shots)
    print(report, flush=True)
    title = (f"Base models in plain text: {NAMES.get(a.model, a.model)}, {a.fmt} prompt — preregistered"
             if a.shots == "none" else
             f"Base models with worked examples ({a.shots}): {NAMES.get(a.model, a.model)} — preregistered")
    if not a.smoke:
        append_entry(title, report)
    (out / f"{name}_report.md").write_text(f"## {title}\n\n{report}\n")
    done.write_text(v + "\n")
    notify(f"[Base/plain {name}] COMPLETE: {v}")


def summary_fewshot(out: Path, shots: str, smoke: bool = False) -> str:
    rows = [(k, (out / f"{k}_{tag_of('plain', shots)}_seed{SEED}.done").read_text().strip()) for k in BASE
            if (out / f"{k}_{tag_of('plain', shots)}_seed{SEED}.done").exists()]
    overall = overall_fewshot({NAMES[k]: v for k, v in rows}, shots) if rows else "no base model finished"
    have = {k.replace("-8bit", "") for k, _ in rows}
    missing = [NAMES[k] for k in ("qwen3-8b-base", "llama3.1-8b-base", "gemma3-12b-pt") if k not in have]
    lines = [f"Preregistered verdicts per base model, six worked **{shots}** examples before each plain item:", "",
             "| model | verdict |", "|---|---|"] + [f"| {NAMES[k]} | {v} |" for k, v in rows]
    lines += ["", f"**Overall ({shots} examples): {overall}**" + (f". Not run: {', '.join(missing)}" if missing else "")
              + ". Rules: docs/preregistration_base_fewshot.md."]
    text = "\n".join(lines)
    print(text, flush=True)
    if not smoke:
        append_entry(f"Base models with worked examples ({shots}): overall — preregistered", text)
    return overall


def summary(out: Path, smoke: bool = False) -> str:
    rows = []
    for k in BASE + INSTRUCT:
        f = out / f"{k}_plain_seed{SEED}.done"
        if f.exists():
            rows.append((k, f.read_text().strip()))
    base = {NAMES[k]: v for k, v in rows if k in BASE}
    overall = overall_base_plain(base) if base else "no base model finished"
    missing = [NAMES[k] for k in ("qwen3-8b-base", "llama3.1-8b-base") if k not in dict(rows)]
    if not any(k.startswith("gemma3-12b-pt") for k, _ in rows):
        missing.append("Gemma 3 12B pt")
    lines = ["Preregistered verdicts per run (plain prompt; the instruct models in chat format fail in all three "
             "families: decisive balanced accuracy 0.37, 0.37, 0.19 for Gemma 3 12B, Qwen3-8B, Llama 3.1 8B):", "",
             "| model | kind | verdict |", "|---|---|---|"]
    lines += [f"| {NAMES[k]} | {'base' if k in BASE else 'instruct'} | {v} |" for k, v in rows]
    lines += ["", f"**Overall (base models): {overall}**" + (f". Not run: {', '.join(missing)}" if missing else "")
              + ". Rules: docs/preregistration_base_plain.md."]
    text = "\n".join(lines)
    print(text, flush=True)
    if not smoke:
        append_entry("Base models in plain text: overall — preregistered", text)
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b-base")
    ap.add_argument("--fmt", default="plain", choices=["plain", "chat"])
    ap.add_argument("--shots", default="none", choices=["none", "neutral", "informative"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 10 questions per kind, layers 0-2; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/base_plain)")
    ap.add_argument("--acts-dir", default=None, help="where to cache the residuals (default: the output folder)")
    ap.add_argument("--summary", action="store_true", help="overall verdict from the finished runs (no model)")
    a = ap.parse_args()
    if a.summary:
        out = Path(a.out) if a.out else ROOT / "results" / "base_plain"
        summary(out, a.smoke) if a.shots == "none" else summary_fewshot(out, a.shots, a.smoke)
    else:
        run(a)


if __name__ == "__main__":
    main()
