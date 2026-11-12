"""P9 (docs/preregistration_p9.md): would the model's own map have answered better? Decode every object's position
from its object-list activation with a linear probe, answer each question geometrically from the decoded map, and
compare with the model's own answers.

Resumable: the probes and the model's answers are saved under results/p9 and skipped on a rerun. Progress goes to
Slack if $SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.followups import chain_items, compare_answers, decoded_coords, p7_class, verdict_p9
from geomap.models import load
from geomap.probes import GroupRidge, centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import predict, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/p9)")
    a = ap.parse_args()
    out, layers, primary = ROOT / "results" / "p9", [18, 21, 24, 28], 21
    if a.smoke:
        import tempfile
        out, layers, primary = Path(tempfile.mkdtemp()), [1, 2], 1
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"P9 {a.model}"
    done = out / f"{a.model}_p9.done"
    if done.exists():
        print(f"[p9] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]

    # probes at the object-list positions (checkpoint: coefficients per layer)
    ppath = out / f"{a.model}_p9_probes.pt"
    if ppath.exists():
        probes = torch.load(ppath)
    else:
        rng = np.random.default_rng(601)
        scenes = [generate_scene(rng, pools[i % 2], 3) for i in range(a.n_fit)]
        acts = cache_prefix(m, [build_prompt(s, None, m.tok, m.family) for s in scenes], layers,
                            batch_size=a.batch_size)["list"]
        probes = {}
        for li, l in enumerate(layers):
            X, Y, meta = centered_xy(acts[:, li].float().numpy(), scenes, np.arange(len(scenes)))
            pr = GroupRidge().fit(X, Y, meta.scene.values)
            probes[l] = dict(coef=pr.coef_, cv_r2=pr.cv_r2_)
            print(f"[p9] probe layer {l}: CV R² {pr.cv_r2_:.3f}", flush=True)
        torch.save(probes, ppath)
        del acts
        notify(f"[{tag}] probes fitted: " + ", ".join(f"L{l} R² {p['cv_r2']:.2f}" for l, p in probes.items()))

    # test chains: the model's answers (checkpoint) and the map's answers
    items = chain_items(m, a.n_test, seed=602, exclude_qtypes=("closer_than",))
    scenes = [s for s, _, _, _ in items]
    flat = [(si, qi, q, p7_class(q, s.root, mid, child))
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, _) in enumerate(qs)]
    print(f"[p9] {len(flat)} questions in {len(items)} chains", flush=True)

    def model_answers():
        sc = score(m, [(build_prompt(scenes[si], q, m.tok, m.family), q) for si, _, q, _ in flat],
                   batch_size=a.batch_size)
        return pd.DataFrame([dict(scene=si, qi=qi, model_ans=predict(x)) for (si, qi, _, _), x in zip(flat, sc)])

    mod = df_step(out / f"{a.model}_p9_model_answers.parquet", model_answers).set_index(["scene", "qi"]).model_ans
    print(f"[p9] model answers ready ({time.time() - t0:.0f}s)", flush=True)
    tacts = cache_prefix(m, [build_prompt(s, None, m.tok, m.family) for s in scenes], layers,
                         batch_size=a.batch_size)["list"]
    recs, summaries = [], []
    for li, l in enumerate(layers):
        coef = probes[l]["coef"]
        dec = decoded_coords(SimpleNamespace(predict=lambda X, c=coef: np.asarray(X) @ c.T),
                             tacts[:, li].float().numpy(), scenes, np.arange(len(scenes)))
        for si, qi, q, cls in flat:
            truth = answer_for(q, scenes[si].coords)
            ma, pa = mod[(si, qi)], answer_for(q, dec[si])
            recs.append(dict(layer=l, scene=si, qi=qi, qtype=q.qtype.value, cls=cls, truth=truth, model_ans=ma,
                             map_ans=pa, model_correct=float(ma == truth), map_correct=float(pa == truth)))
        d = pd.DataFrame([r for r in recs if r["layer"] == l])
        summaries.append(compare_answers(d).assign(layer=l, probe_cv_r2=probes[l]["cv_r2"]))
    df, s = pd.DataFrame(recs), pd.concat(summaries, ignore_index=True)
    df.to_parquet(out / f"{a.model}_p9.parquet")
    s.to_csv(out / f"{a.model}_p9_summary.csv", index=False)
    v = verdict_p9(s[s.layer == primary])

    lines = [f"Model **{a.model}**. Linear probes for scene-centred (x, y) at the object-list positions, fitted on "
             f"{a.n_fit} new 3-object scenes. Test: {len(items)} new chains R → M → C, {len(flat)} questions. The *map* "
             "answers each question geometrically from the decoded positions; the *model* answers as usual. Accuracy "
             "against the true layout (95% scene-bootstrap CI for the difference):", "",
             "| layer (probe CV R²) | questions | n | model | map | map − model | agreement |", "|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.layer} ({r.probe_cv_r2:.2f}) | {r.cls} | {r.n} | {r.acc_model:.2f} | {r.acc_map:.2f} | "
                     f"{r.map_minus_model:+.3f} [{r.map_minus_model_lo:+.3f}, {r.map_minus_model_hi:+.3f}] | "
                     f"{r.agreement:.2f} |")
    lines += ["", f"Preregistered verdict (layer {primary}, CR questions): **{v}**. Rules: docs/preregistration_p9.md."]
    if not a.smoke:
        append_entry(f"P9 would the model's own map have answered better? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v)
    cr = s[(s.layer == primary) & (s.cls == "CR")].iloc[0]
    notify(f"[{tag}] COMPLETE. {v}. CR: model {cr.acc_model:.2f}, map {cr.acc_map:.2f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
