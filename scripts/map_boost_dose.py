"""Map boost dose check (docs/preregistration_map_boost_dose.md).

Was the map boost too small? Same model, sets, located subspaces and controls as the boost run (reused from its
cache). Two additions:
- a positive control: at the boost layer and position, add D(α)·u, with u the residual direction that most directly
  raises the true answer over the other name, and D(α) = (α − 1)·s (s = mean |P(h − μ)| of the map arm), α 2–64;
- the two arms and their controls at α ∈ {16, 32, 64}.
The boost run's α 2–8 answers are reused (same questions, same weights); clean answers are scored again.

Resumable: every condition's answers are saved to results/map_boost_dose/{model}_items.csv as they finish."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import map_boost as mb
from geomap.axis import CELLS
from geomap.boost import (DEC, EASY, FIRST, answer_directions, ba_boot, boost_edit, dose_verdict, push_edit,
                          push_moves, verdict_boost)
from geomap.checkpoint import notify
from geomap.flow import layer_grid
from geomap.models import load
from geomap.report import ROOT, append_entry
from geomap.scoring import score

OLD_ALPHAS = mb.ALPHAS                       # (2, 4, 8): run by the boost test
NEW_ALPHAS = (16.0, 32.0, 64.0)
ALL_ALPHAS = OLD_ALPHAS + NEW_ALPHAS
DOSE_ALPHAS = ALL_ALPHAS                     # positive control at D(α) for each
CRIT_ALPHA = 8.0                             # criterion 1 is judged at D(8)
M_ARM = "M's step"
f2 = mb.f2


def conditions(layers):
    """(name, kind, arm, layer, α, control index); kind 'clean' | 'arm' | 'push'."""
    out = [("clean", "clean", None, None, None, None)]
    for arm in mb.ARMS:
        for layer in layers:
            for al in ALL_ALPHAS:
                out.append((f"{arm}|L{layer}|a{al:g}|real", "arm", arm, layer, al, None))
                out += [(f"{arm}|L{layer}|a{al:g}|ctrl{j}", "arm", arm, layer, al, j) for j in range(mb.N_CTRL)]
    for layer in layers:
        for al in DOSE_ALPHAS:
            out.append((f"push|L{layer}|a{al:g}", "push", None, layer, al, None))
    return out


def dose(res, layer, al, arm="map", ctrl=False):
    v = res["layers"][layer]["arms"][arm]
    return (al - 1) * (v["ctrl_norm"] if ctrl else v["norm"])


def push_stats(cr, layer, al, n_boot):
    """Positive control vs clean, per question kind: gain, 99% interval of the gain."""
    name, st = f"push|L{layer}|a{al:g}", {}
    for cell in CELLS:
        k = (cr.cell == cell).to_numpy()
        b = ba_boot(cr.truth[k], {n: mb.sign_of(cr, n)[k] for n in ("clean", name)}, cr.scene[k], n_boot=n_boot)
        s = b[name][1] - b["clean"][1]
        st[cell] = dict(real=b[name][0], gain=b[name][0] - b["clean"][0], lo99=float(np.percentile(s, 0.5)),
                        hi99=float(np.percentile(s, 99.5)))
    return st


def analyse(T: pd.DataFrame, res, layers, n_boot: int, clean_agree: float | None):
    cr = T[T.kind == "C-R"]
    rows = mb.judge(cr, layers, ALL_ALPHAS, n_boot)
    new = [c for c in rows if c["alpha"] in NEW_ALPHAS]
    arm_v = {arm: verdict_boost([c for c in new if c["arm"] == arm]) for arm in mb.ARMS}
    push = {(L, al): push_stats(cr, L, al, n_boot) for L in layers for al in DOSE_ALPHAS}
    moves = {L: push_moves(push[L, CRIT_ALPHA][DEC]["gain"], push[L, CRIT_ALPHA][DEC]["lo99"]) for L in layers}
    works = {L: any(c["works"] for c in rows if c["layer"] == L) for L in layers}
    new_works = any(c["works"] for c in new)
    v = dose_verdict(moves, works, new_works)

    lines = ["Model **Gemma 3 12B** (chat format), same located subspaces and controls as the map boost run. Positive "
             "control: at the boost layer, final position, add D(α)·u, u ∝ (final-norm scale) ⊙ (W_U[true answer] − "
             "W_U[other name]) (unit length), D(α) = (α − 1)·s, s = mean |P(h − μ)| of the map arm. Arms and controls "
             f"at α ∈ {{16, 32, 64}} added; α 2–8 answers reused from the boost run. Paired scene-bootstrap 99% "
             f"intervals ({n_boot} resamples).", ""]
    for L in layers:
        lines.append(f"- Layer {L}: s = {res['layers'][L]['arms']['map']['norm']:.3g}, mean residual norm "
                     f"{res['layers'][L]['h_norm']:.3g}; D(8) = {dose(res, L, 8):.3g}, D(64) = {dose(res, L, 64):.3g}.")
    if clean_agree is not None:
        lines.append(f"- Clean answers rescored here agree with the boost run's on {clean_agree:.3f} of questions.")
    lines += ["", f"**1. Positive control (answer direction).** Criterion at D(8): decisive gain ≥ +0.20 with its 99% "
              "interval above +0.10.", "",
              "| layer | α | dose D(α) | decisive | gain [99% CI] | first step only gain | easy gains "
              "(own / agree / C larger) |", "|---|---|---|---|---|---|---|"]
    for L in layers:
        for al in DOSE_ALPHAS:
            st = push[L, al]
            d = st[DEC]
            lines.append(f"| {L} | {al:g} | {dose(res, L, al):.3g} | {d['real']:.2f} | {f2(d['gain'])} "
                         f"[{f2(d['lo99'])}, {f2(d['hi99'])}] | {f2(st[FIRST]['gain'])} | "
                         + " / ".join(f2(st[e]["gain"]) for e in EASY) + " |")
    lines += ["", "Moves the answer at D(8): " + "; ".join(f"layer {L} **{'yes' if moves[L] else 'no'}**"
                                                          for L in layers) + ".", ""]
    lines += ["**2. Larger α (arms against their variance-matched controls).** Verdicts over the six new conditions "
              "(α 16–64); α 2–8 rows are the boost run's, shown for reference.", ""]
    for arm in mb.ARMS:
        lines += [f"Arm **{arm}**, at higher dose: **{arm_v[arm]}**", "",
                  *mb.arm_table([c for c in rows if c["arm"] == arm]), ""]

    lines.append(f"**Overall (preregistered): {v}.**")
    if v == "still too small to tell":
        first = {L: next((al for al in DOSE_ALPHAS if push_moves(push[L, al][DEC]["gain"], push[L, al][DEC]["lo99"])),
                         None) for L in layers}
        for L in layers:
            if first[L] is None:
                lines.append(f"- Layer {L}: the positive control does not reach the threshold at any dose up to D(64); "
                             "the question stays open.")
            else:
                lines.append(f"- Layer {L}: the positive control first reaches the threshold at D({first[L]:g}) = "
                             f"{dose(res, L, first[L]):.3g}. The map arm was tested at that dose (α {first[L]:g}) and "
                             "above, and no condition works: **not read at a dose that moves the answer**.")
    lines.append("")

    if new_works:
        pick = max([c for c in new if c["works"]], key=lambda c: c["dec_gain"])
        base = f"{pick['arm']}|L{pick['layer']}|a{pick['alpha']:g}"
        overall, gains, tl = mb.transfer(T, base, n_boot)
        from geomap.boost import transfer_verdict
        lines += [f"Transfer (secondary, preregistered in the boost prereg), for {pick['arm']}, layer {pick['layer']}, "
                  f"α {pick['alpha']:g}:", "", *tl, "", f"Overall transfer accuracy change {f2(overall)}. Transfer "
                  f"verdict: **{transfer_verdict(overall, gains)}**.", ""]

    # exploratory
    toward = np.where(cr.answer == cr.child, 1, -1)
    dk = (cr.cell == DEC).to_numpy()
    shift = lambda c: float(np.mean(((cr[f"ld:{c}"].to_numpy(float) - cr["ld:clean"].to_numpy(float)) * toward)[dk]))
    ctrl_shift = lambda base: float(np.mean([shift(f"{base}|ctrl{j}") for j in range(mb.N_CTRL)]))
    lines += ["Exploratory: mean shift of the decisive margin toward the true answer (logits; clean margins are "
              "large), with the size of each edit.", "",
              "| layer | α | push (D) | map real | map control | M's step real | M's step control |",
              "|---|---|---|---|---|---|---|"]
    for L in layers:
        for al in ALL_ALPHAS:
            mp, ms = f"map|L{L}|a{al:g}", f"M's step|L{L}|a{al:g}"
            lines.append(f"| {L} | {al:g} | {f2(shift(f'push|L{L}|a{al:g}'))} ({dose(res, L, al):.3g}) | "
                         f"{f2(shift(mp + '|real'))} | {f2(ctrl_shift(mp))} | {f2(shift(ms + '|real'))} "
                         f"({dose(res, L, al, M_ARM):.3g}) | {f2(ctrl_shift(ms))} |")
    onset = []
    for arm in mb.ARMS:
        for L in layers:
            a0 = next((c["alpha"] for c in rows if c["arm"] == arm and c["layer"] == L
                       and min(c["st"][e]["ctrl_gain"] for e in EASY) < -0.05), None)
            onset.append(f"{arm} layer {L}: " + (f"α {a0:g}" if a0 else "none up to α 64"))
    lines += ["", "Exploratory: first α at which the controls cost an easy kind more than 0.05: " + "; ".join(onset) + ".",
              "", "Exploratory: on the decisive questions, share of answers following C's sentence / M's sentence at the "
              "new α (clean " + "{:.2f} / {:.2f}".format(*mb.follow_shares(cr, "clean")) + "): "
              + "; ".join(f"{arm} L{L} α{al:g} " + "{:.2f} / {:.2f}".format(*mb.follow_shares(cr, f"{arm}|L{L}|a{al:g}|real"))
                          for arm in mb.ARMS for L in layers for al in NEW_ALPHAS) + ".", ""]
    lines.append(f"Preregistered verdict: **{v}**. Rules: docs/preregistration_map_boost_dose.md.")
    return v, "\n".join(lines)


def run(a):
    out, boost_dir = ROOT / "results" / "map_boost_dose", ROOT / "results" / "map_boost"
    fit_sizes, test_sizes, per_kind, n_boot = mb.SIZES_FIT, mb.SIZES_TEST, mb.TRANSFER_PER_KIND, 2000
    if a.smoke:
        import tempfile
        a.model = "tiny"
        fit_sizes, test_sizes, per_kind, n_boot = ({c: 20 for c in CELLS}, {c: 6 for c in CELLS}, 4, 100)
        out = Path(a.out) if a.out else Path(tempfile.mkdtemp())
        boost_dir = out / "boost"
    out = Path(a.out) if a.out else out
    boost_dir = Path(a.boost_dir) if a.boost_dir else boost_dir
    out.mkdir(parents=True, exist_ok=True)
    boost_dir.mkdir(parents=True, exist_ok=True)
    done = out / f"{a.model}.done"
    if done.exists():
        print(f"[dose] {a.model} already complete; delete {done.name} to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[dose] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    la = layer_grid(m.n_layers)[2]["final"][0]
    layers = [1, 2] if a.smoke else [la, la + 4]
    notify(f"[Map boost dose {a.model}] started (layers {layers})")
    if not (boost_dir / f"{a.model}_subspaces.pt").exists():
        log(f"no cached subspaces in {boost_dir}: locating again (same seeds)")
    res = mb.locate(m, a, boost_dir, layers, fit_sizes, log)
    T, specs, qs, n_scenes = mb.test_set(m, test_sizes, res["items"], per_kind)
    same = lambda df: len(df) == len(T) and (df.question == T.question).all() and (df.scene == T.scene).all()
    ipath, opath = out / f"{a.model}_items.csv", boost_dir / f"{a.model}_items.csv"
    old = pd.read_csv(opath, keep_default_na=False) if opath.exists() else None
    if old is not None:
        assert same(old), "the boost run's test set differs from this one"
    if ipath.exists():
        prev = pd.read_csv(ipath, keep_default_na=False)
        assert same(prev)
        T = prev
    elif old is not None:
        reuse = [c for c in old.columns if c.startswith(("pred:", "ld:")) and not c.endswith(":clean")]
        T = pd.concat([T, old[reuse]], axis=1)
        log(f"reused {len(reuse) // 2} conditions from the boost run")
    cr_idx = np.flatnonzero((T.kind == "C-R").to_numpy())
    pos = [len(s.input_ids) - 1 for s in specs]
    todo = [c for c in conditions(layers) if f"pred:{c[0]}" not in T.columns]
    U = None
    for i, (name, kind, arm, layer, al, j) in enumerate(todo):
        if kind == "push":
            if U is None:
                trip = [(specs[k], T.answer[k], T.root[k] if T.answer[k] == T.child[k] else T.child[k]) for k in cr_idx]
                U = answer_directions(m, trip)
            sub = [(specs[k], qs[k]) for k in cr_idx]
            sc = score(m, sub, edit=push_edit(layer, U, dose(res, layer, al), [pos[k] for k in cr_idx]),
                       batch_size=a.batch_size)
            pred, ld = [""] * len(T), [np.nan] * len(T)
            for k, x in zip(cr_idx, sc):
                pred[k], ld[k] = max(x, key=x.get), x[T.child[k]] - x[T.root[k]]
        else:
            edit = None
            if kind == "arm":
                L = res["layers"][layer]
                Q = L["arms"][arm]["Q"] if j is None else L["arms"][arm]["ctrl"][j]
                edit = boost_edit(layer, Q, L["mu"], al, pos)
            sc = score(m, list(zip(specs, qs)), edit=edit, batch_size=a.batch_size)
            pred = [max(x, key=x.get) for x in sc]
            ld = [x[r.child] - x[r.root] if r.kind == "C-R" else np.nan for x, r in zip(sc, T.itertuples())]
        T[f"pred:{name}"], T[f"ld:{name}"] = pred, ld
        T.to_csv(ipath, index=False)
        log(f"[{i + 1}/{len(todo)}] {name}")
    T = pd.read_csv(ipath, keep_default_na=False)
    agree = float((old["pred:clean"] == T["pred:clean"]).mean()) if old is not None else None
    v, report = analyse(T, res, layers, n_boot, agree)
    print(report, flush=True)
    title = f"Map boost dose check ({a.model}) — preregistered"
    if not a.smoke:
        append_entry(title, report)
    (out / f"{a.model}_report.md").write_text(f"## {title}\n\n{report}\n")
    done.write_text(v + "\n")
    notify(f"[Map boost dose {a.model}] COMPLETE: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b-8bit", help="the boost run's weights (its cache is reused)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few questions, layers 1-2; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/map_boost_dose)")
    ap.add_argument("--boost-dir", default=None, help="the boost run's folder (default results/map_boost)")
    ap.add_argument("--acts-dir", default=None, help="where the fitting residuals are cached (default: boost folder)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
