"""Connect the map to the answers, development rounds (docs/design_map_connect.md). Not preregistered: every round is
logged in results/connect_dev/LOG.md, failures included. Only development sets are used here.

  python scripts/connect_dev.py train --tag r1 [--from-layer 28 --rank 16 --lr 1e-4 ...]   -> adapter on disk
  python scripts/connect_dev.py eval  --tag r1                                              -> report vs base

Eval (base model cached once, then the adapter merged into the weights):
  behaviour: dev test (new scenes: C–R per kind, one-step C–M / M–R), new names, dev transfer (farthest / closest),
             general one-word questions;
  M1  map still readable: held-out R² of M's step and C's own step at the final position (first layer of the final
      window, and 4 later);
  M2  the map drives the answer: reflect (α = −1) or amplify (α = 4) the map's directions at the final position, against
      3 variance-matched random subspaces;
  M4  erasing the map hurts: α = 0 (projection out), same controls;
  M3  no re-reading: from that layer on, the question and answer positions may not attend to M's sentence (and, as the
      floor, in every layer)."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import map_boost as mb
from geomap.axis import CELLS
from geomap.boost import DEC, EASY, FIRST, boost_edit, feasible_k, map_directions, matched_random_subspaces, orthonormal
from geomap.checkpoint import notify
from geomap.connect import (SEED_DEV, SEED_DEV_NAMES, SEED_TRAIN, add_lora, attach_lora, first_token_accuracy,
                            general_prompts,
                            last_token_logits, lora_state, make_set, merge_lora, training_examples)
from geomap.flow import axis_r2, layer_grid
from geomap.followups import attention_knockout
from geomap.models import load, quant_config
from geomap.probes import scene_split
from geomap.prompts import build_prompt
from geomap.scoring import score
from geomap.study2 import balanced_accuracy

ROOT = Path(__file__).resolve().parents[1]
SEED_LOCATE = 3401
TRAIN_PER_CELL, DEV_PER_CELL, NAMES_PER_CELL, LOCATE_PER_CELL = 1200, 200, 60, 240
ALPHAS_M2 = (-1.0, 0.0, 4.0)
N_CTRL = 3


def window_start(m) -> int:
    """First layer of the final window (28 in Gemma 3 12B); layer 1 for the tiny test model."""
    return 1 if m.key == "tiny" else layer_grid(m.n_layers)[2]["final"][0]


def sizes(n):
    return {c: n for c in CELLS}


# ---- train ----

def train(a):
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "adapter.pt").exists():
        print(f"[connect] {a.tag}: adapter exists; delete it to retrain", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[connect] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    la = window_start(m)
    from_layer = la if a.from_layer is None else a.from_layer
    rows = make_set(m.tok, SEED_TRAIN, sizes(a.train_per_cell), one_step_per_scene=a.one_step_per_scene)
    ex = training_examples(m.tok, m.family, rows)
    kinds = pd.Series([r["kind"] for r in rows]).value_counts().to_dict()
    log(f"{len(ex)} training questions {kinds}; LoRA r={a.rank} on blocks >= {from_layer} ({a.targets})")
    targets = {"all": None, "attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("gate_proj", "up_proj", "down_proj")}
    kw = {} if targets[a.targets] is None else {"targets": targets[a.targets]}
    loras = add_lora(m.hf, m.layers_path, from_layer, r=a.rank, alpha=2.0 * a.rank, seed=a.seed, **kw)
    params = [p for v in loras.values() for p in (v.A, v.B)]
    log(f"{len(loras)} adapted linears, {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    rng = np.random.default_rng(a.seed)
    steps_per_epoch = len(ex) // a.batch_size
    total = steps_per_epoch * a.epochs if a.max_steps is None else a.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / max(total, 1)))
    hist, step, run_loss, run_acc = [], 0, [], []
    micro = a.batch_size // a.grad_accum
    for ep in range(a.epochs):
        order = rng.permutation(len(ex))
        for b in range(steps_per_epoch):
            if step >= total:
                break
            batch = [ex[i] for i in order[b * a.batch_size:(b + 1) * a.batch_size]]
            opt.zero_grad()
            for k in range(0, len(batch), micro):
                mb_ = batch[k:k + micro]
                lg = last_token_logits(m.hf, [x for x, _ in mb_], pad).float()
                tgt = torch.tensor([t for _, t in mb_], device=lg.device)
                loss = torch.nn.functional.cross_entropy(lg, tgt) * len(mb_) / len(batch)
                loss.backward()
                run_loss.append(loss.item() * len(batch) / len(mb_))
                run_acc.append(float((lg.argmax(-1) == tgt).float().mean()))
                del lg
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == total:
                hist.append(dict(step=step, loss=float(np.mean(run_loss)), acc=float(np.mean(run_acc))))
                log(f"step {step}/{total}: loss {hist[-1]['loss']:.3f}, train accuracy {hist[-1]['acc']:.3f}")
                run_loss, run_acc = [], []
    cfg = dict(tag=a.tag, model=a.model, from_layer=from_layer, rank=a.rank, alpha=2.0 * a.rank, targets=a.targets,
               lr=a.lr, epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum, steps=step, n_train=len(ex), kinds=kinds,
               train_per_cell=a.train_per_cell, one_step_per_scene=a.one_step_per_scene, seed=a.seed,
               minutes=round((time.time() - t0) / 60, 1))
    torch.save(lora_state(loras), out / "adapter.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "train_log.json").write_text(json.dumps(hist, indent=1))
    log(f"adapter saved to {out}")


# ---- eval ----

def eval_sets(m, smoke):
    k = 6 if smoke else 1
    dev = make_set(m.tok, SEED_DEV, sizes(DEV_PER_CELL // k), one_step_per_scene=1, transfer_per_scene=1)
    names = make_set(m.tok, SEED_DEV_NAMES, sizes(NAMES_PER_CELL // k), one_step_per_scene=1, held_out_names=True)
    loc = make_set(m.tok, SEED_LOCATE, sizes(LOCATE_PER_CELL // (4 if smoke else 1)), one_step_per_scene=0)
    return dev, names, loc


def spatial_scores(m, rows, bs, edit=None, batch_ctx=None):
    specs = [build_prompt(r["s"], r["q"], m.tok, m.family) for r in rows]
    sc = score(m, [(sp, r["q"]) for sp, r in zip(specs, rows)], edit=edit, batch_size=bs, batch_ctx=batch_ctx)
    pred = [max(x, key=x.get) for x in sc]
    ld = [x[r["child"]] - x[r["root"]] if r["kind"] == "C-R" else np.nan for x, r in zip(sc, rows)]
    return pred, ld, specs


def behaviour(rows, pred):
    """Balanced accuracy per C–R kind and overall; accuracy per other kind."""
    df = pd.DataFrame([dict(kind=r["kind"], cell=r["cell"], truth=r.get("truth", 0), child=r["child"],
                            answer=r["answer"]) for r in rows])
    df["pred"] = pred
    out = {}
    cr = df[df.kind == "C-R"]
    sg = np.where(cr.pred == cr.child, 1, -1)
    for c in CELLS:
        k = (cr.cell == c).to_numpy()
        out[c] = balanced_accuracy(cr.truth[k].to_numpy(), sg[k], 1) if k.any() else np.nan
    out["C-R overall"] = balanced_accuracy(cr.truth.to_numpy(), sg, 1)
    for kind in sorted(set(df.kind) - {"C-R"}):
        k = df.kind == kind
        out[kind] = float((df.pred[k] == df.answer[k]).mean())
    return out


def weight_ratio(rows, ld):
    cr = [(r["d1"], r["d2"], x) for r, x in zip(rows, ld) if r["kind"] == "C-R"]
    d1, d2, y = map(np.asarray, zip(*cr))
    w = np.linalg.lstsq(np.c_[d1, d2, np.ones(len(y))], y.astype(float), rcond=None)[0]
    return float(w[0] / w[1])


def evaluate(m, la, bs, sets, log):
    dev, names, loc = sets
    res = {}
    pred, ld, specs = spatial_scores(m, dev, bs)
    res["dev"] = behaviour(dev, pred)
    res["weight_ratio"] = weight_ratio(dev, ld)
    res["names"] = behaviour(names, spatial_scores(m, names, bs)[0])
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else m.tok.eos_token_id
    res["general"] = first_token_accuracy(m.hf, general_prompts(m.tok, m.family), pad, bs)
    log(f"behaviour: decisive {res['dev'][DEC]:.2f}, general {res['general']:.2f}")

    # M1: locate the map on the model's own residuals
    lspecs = [build_prompt(r["s"], r["q"], m.tok, m.family) for r in loc]
    lays = [la, min(la + 4, m.n_layers)]
    X_all = mb.final_residuals(m, lspecs, lays, bs)
    g = np.array([r["scene"] for r in loc])
    ax, d1, d2 = (np.array([r[k] for r in loc]) for k in ("axis", "d1", "d2"))
    tr = np.isin(g, scene_split(int(g.max()) + 1)[0])
    res["M1"] = {}
    for li, L in enumerate(lays):
        X = X_all[:, li].float().numpy()
        res["M1"][L] = dict(r2_d1=axis_r2(X, d1, ax, g, tr), r2_d2=axis_r2(X, d2, ax, g, tr))
    X = X_all[:, 0].float().numpy()
    Q = orthonormal(map_directions(X, d1, d2, ax, g)).float()
    mu = torch.as_tensor(X.mean(0), dtype=torch.float32)
    ctrl = [c.float() for c in matched_random_subspaces(X, Q, k=feasible_k(X, Q, 64), n=N_CTRL, tol=0.1, seed=7)]
    log(f"M1: layer {la} R² d1 {res['M1'][la]['r2_d1']:.2f}, d2 {res['M1'][la]['r2_d2']:.2f}")

    # M2 / M4 on the dev C–R questions
    cr = [r for r in dev if r["kind"] == "C-R"]
    cspecs = [build_prompt(r["s"], r["q"], m.tok, m.family) for r in cr]
    pos = [len(sp.input_ids) - 1 for sp in cspecs]
    toward = np.array([1 if r["answer"] == r["child"] else -1 for r in cr])
    dec = np.array([r["cell"] == DEC for r in cr])
    clean_pred, clean_ld, _ = spatial_scores(m, cr, bs)
    res["edits"] = {}
    for al in ALPHAS_M2:
        for name, B in [("map", Q)] + [(f"ctrl{j}", c) for j, c in enumerate(ctrl)]:
            p, l, _ = spatial_scores(m, cr, bs, edit=boost_edit(la, B, mu, al, pos))
            b = behaviour(cr, p)
            res["edits"][f"a{al:g}|{name}"] = dict(decisive=b[DEC], overall=b["C-R overall"],
                                                   shift=float(np.mean(((np.array(l) - np.array(clean_ld)) * toward)[dec])))
        log(f"M2/M4 α {al:g}: map decisive {res['edits'][f'a{al:g}|map']['decisive']:.2f}")

    # M3: block attention from the question / answer positions to M's sentence
    blocks = [[(sp.eol_pos + 1, *sp.relation_spans[r["mid"]])] for sp, r in zip(cspecs, cr)]
    res["M3"] = {}
    for name, layers in [("from_window", range(la, m.n_layers)), ("all_layers", range(m.n_layers))]:
        p, _, _ = spatial_scores(m, cr, bs, batch_ctx=attention_knockout(m, blocks, list(layers)))
        b = behaviour(cr, p)
        res["M3"][name] = dict(decisive=b[DEC], overall=b["C-R overall"], first=b[FIRST])
    log(f"M3: decisive with M's sentence blocked from layer {la}: {res['M3']['from_window']['decisive']:.2f}")
    return res


def report(tag, cfg, base, conn, la):
    f = lambda x: f"{x:.2f}"
    rows = [("decisive (conflict, M larger)", DEC), ("first step only", FIRST), *[(e, e) for e in EASY],
            ("C–R overall", "C-R overall"), ("one step: C vs M", "C-M"), ("one step: M vs R", "M-R"),
            ("dev transfer: farthest", "farthest"), ("dev transfer: closest", "closest")]
    L = [f"## Development round {tag}", "", "Config: " + ", ".join(f"{k} {v}" for k, v in cfg.items() if k != "kinds"), "",
         "| measure | base | connected |", "|---|---|---|"]
    L += [f"| {lab} | {f(base['dev'].get(k, np.nan))} | {f(conn['dev'].get(k, np.nan))} |" for lab, k in rows]
    L += [f"| new names: decisive | {f(base['names'][DEC])} | {f(conn['names'][DEC])} |",
          f"| new names: C–R overall | {f(base['names']['C-R overall'])} | {f(conn['names']['C-R overall'])} |",
          f"| general one-word questions | {f(base['general'])} | {f(conn['general'])} |",
          f"| weight of M's step / C's step in the margin | {f(base['weight_ratio'])} | {f(conn['weight_ratio'])} |", ""]
    L += ["Mechanism checks (decisive balanced accuracy unless stated):", "",
          "| check | base | connected |", "|---|---|---|"]
    for Lr in conn["M1"]:
        L.append(f"| M1 layer {Lr}: R² M's step / C's step | {f(base['M1'][Lr]['r2_d1'])} / {f(base['M1'][Lr]['r2_d2'])} "
                 f"| {f(conn['M1'][Lr]['r2_d1'])} / {f(conn['M1'][Lr]['r2_d2'])} |")
    for al, lab in [(-1, "M2 reflect the map (α −1)"), (0, "M4 erase the map (α 0)"), (4, "M2 amplify the map (α 4)")]:
        for res, who in ((base, "base"), (conn, "conn")):
            e = res["edits"]
            res[f"_{al}"] = (e[f"a{al:g}|map"]["decisive"], np.mean([e[f"a{al:g}|ctrl{j}"]["decisive"] for j in range(N_CTRL)]),
                             e[f"a{al:g}|map"]["shift"], np.mean([e[f"a{al:g}|ctrl{j}"]["shift"] for j in range(N_CTRL)]))
        b, c = base[f"_{al}"], conn[f"_{al}"]
        L.append(f"| {lab}: map vs random (margin shift toward truth) | {f(b[0])} vs {f(b[1])} ({b[2]:+.1f} vs {b[3]:+.1f}) "
                 f"| {f(c[0])} vs {f(c[1])} ({c[2]:+.1f} vs {c[3]:+.1f}) |")
    L += [f"| M3 M's sentence blocked from layer {la} (question + answer positions) | {f(base['M3']['from_window']['decisive'])} "
          f"| {f(conn['M3']['from_window']['decisive'])} |",
          f"| M3 floor: blocked in every layer | {f(base['M3']['all_layers']['decisive'])} | {f(conn['M3']['all_layers']['decisive'])} |",
          "", "Reading guide: a connected model that uses the map keeps a high decisive score when M's sentence is blocked "
          f"from layer {la} (the map has arrived), falls toward the rule when the map is erased or reflected (more than "
          "with random directions), and keeps the map readable (M1)."]
    return "\n".join(L)


def evaluate_cmd(a):
    out = Path(a.out)
    rdir = out / a.tag
    t0 = time.time()
    log = lambda msg: print(f"[connect] {msg} ({time.time() - t0:.0f}s)", flush=True)
    cfg = json.loads((rdir / "config.json").read_text())
    m = load(a.model)
    assert cfg["model"] == a.model, "evaluate with the model the adapter was trained on"
    la = window_start(m)
    sets = eval_sets(m, a.model == "tiny")
    bpath = out / f"base_{a.model}.json"
    if bpath.exists():
        base = json.loads(bpath.read_text(), object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
    else:
        log("evaluating the base model (once)")
        base = evaluate(m, la, a.batch_size, sets, log)
        bpath.write_text(json.dumps(base, indent=1))
    state = torch.load(rdir / "adapter.pt")
    if quant_config(a.model) is None:
        merge_lora(m.hf, m.layers_path, state)                # bf16: fold into the weights
    else:
        attach_lora(m.hf, m.layers_path, state)               # 8-bit: keep the adapter as a separate path
    log(f"adapter {a.tag} loaded; evaluating")
    conn = evaluate(m, la, a.batch_size, sets, log)
    (rdir / "eval.json").write_text(json.dumps(conn, indent=1))
    rep = report(a.tag, cfg, base, conn, la)
    (rdir / "report.md").write_text(rep + "\n")
    with open(out / "LOG.md", "a") as fh:
        fh.write(f"\n{rep}\n")
    print(rep, flush=True)
    notify(f"[Connect dev {a.tag}] done: decisive {base['dev'][DEC]:.2f} -> {conn['dev'][DEC]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--out", default=str(ROOT / "results" / "connect_dev"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--from-layer", type=int, default=None, help="first adapted block (0-based); default: final window")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--targets", choices=["all", "attn", "mlp"], default="all")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--train-per-cell", type=int, default=TRAIN_PER_CELL)
    ap.add_argument("--one-step-per-scene", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1, help="split each batch into this many pieces (memory)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate_cmd(a)


if __name__ == "__main__":
    main()
