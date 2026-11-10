"""Precompute the interactive demo: for a few held-out scenes, move one object by editing only the learned 2-D code
in its sentence (DAS, layer 14) to many target positions, partly (alpha) or fully, and record
  - the model's answers to six fixed questions (with candidate probabilities),
  - the correct answers for the layout with the object at that (possibly unstated) position,
  - the map read back from the activations by the P2 probe (layer 21, object-list mentions),
plus references: the model reading the actually-moved text. Output: results/demo/demo_data.json."""

import argparse
import json
import time

import numpy as np
import torch

from geomap.cache import load_cache
from geomap.das import (SubspaceIntervention, _groups, interp_rows, interpolated_coords, pair_rows,
                        question_scores, row_logprobs)
from geomap.demo import anchored_positions, demo_questions, leaf_targets, readout
from geomap.models import load
from geomap.patching import prefix_acts
from geomap.probes import probe_by_layer, scene_split
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for, passes_margin
from geomap.report import ROOT
from geomap.scenes import describe, generate_scene, leaves, relation_text
from geomap.scoring import predict, score

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--edit-layer", type=int, default=14)
    ap.add_argument("--read-layer", type=int, default=21)
    ap.add_argument("--n-scenes", type=int, default=5)
    ap.add_argument("--n-targets", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-correct", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    cache, cscenes, _ = load_cache(ROOT / "cache" / a.model / "main.pt")
    _, probes = probe_by_layer(cache, cscenes, scene_split(len(cscenes)), n_boot=10, return_probes=True,
                               layers=[a.read_layer])
    probe = probes[a.read_layer]
    del cache
    print(f"[demo] probe fitted {time.time() - t0:.0f}s", flush=True)

    m = load(a.model)
    iv = SubspaceIntervention(m.d_model, 2)
    iv.load_state_dict(torch.load(ROOT / "results" / "p4" / f"{a.model}_L{a.edit_layer}_relation_k2.pt"))
    rng = np.random.default_rng(a.seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]

    def decode(H, names, root, root_true):
        Hc = H.numpy().astype(np.float64)
        P = probe.predict(Hc - Hc.mean(0))
        return anchored_positions(dict(zip(names, P)), root, root_true)

    out = dict(model=a.model, edit_layer=a.edit_layer, read_layer=a.read_layer, alphas=ALPHAS, scenes=[])
    tried = 0
    while len(out["scenes"]) < a.n_scenes:
        tried += 1
        s = generate_scene(rng, pools[tried % 2], 3, p_shuffle=0.0)
        lv = sorted(leaves(s), key=lambda n: -s.hop[n])[0]  # prefer a two-step object
        qs = demo_questions(s, lv)
        if not all(passes_margin(q, s.coords) for q in qs):
            continue
        clean = score(m, [(build_prompt(s, q, m.tok, m.family), q) for q in qs])
        if sum(predict(c) == answer_for(q, s.coords) for c, q in zip(clean, qs)) < a.min_correct:
            continue  # demo scenes: the model answers the original description correctly (>= 5 of 6)
        targets = leaf_targets(m, s, lv, a.n_targets, rng)
        if len(targets) < min(8, a.n_targets):
            continue
        spec = build_prompt(s, None, m.tok, m.family)
        list_pos = [spec.list_pos[n] for n in s.names]
        lo, hi = spec.relation_spans[lv]
        epos = list(range(lo, hi))
        base_h = prefix_acts(m, spec, [a.edit_layer])[a.edit_layer][epos].float()
        root_true = s.coords[s.root]
        sc = dict(names=list(s.names), root=s.root, leaf=lv, parent=s.parent[lv], hop={n: s.hop[n] for n in s.names},
                  coords={n: list(s.coords[n]) for n in s.names}, text=describe(s),
                  leaf_sentence=relation_text(next(r for r in s.relations if r.subj == lv)),
                  questions=[dict(text=q.text, qtype=q.qtype.value, candidates=list(q.candidates)) for q in qs],
                  base_model=[predict(c) for c in clean], base_truth=[answer_for(q, s.coords) for q in qs],
                  base_decoded=decode(readout(m, spec, None, 0, [], None, a.read_layer, list_pos), s.names, s.root,
                                      root_true),
                  targets=[])
        for t in targets:
            pair = (s, t, lv, qs)
            rows = pair_rows(m, [pair], a.edit_layer, "relation")
            groups = _groups(rows)
            src_h = rows[0]["src_h"]
            tspec = build_prompt(t, None, m.tok, m.family)
            tr = dict(pos=list(t.coords[lv]), leaf_sentence=relation_text(next(r for r in t.relations if r.subj == lv)),
                      text_model=[rows[g[0]]["model_src"] for g in groups],
                      text_decoded=decode(readout(m, tspec, None, 0, [], None, a.read_layer, list_pos), s.names,
                                          s.root, root_true),
                      steps=[])
            for al in ALPHAS:
                ra = interp_rows(m, rows, [pair], a.edit_layer, "relation", al)
                lp = row_logprobs(m, ra, iv, a.edit_layer).cpu()
                ans, probs = [], []
                for g in groups:
                    cands, cl = question_scores(ra, lp[g], g)
                    p = torch.softmax(cl.float(), 0).tolist()
                    ans.append(cands[int(cl.argmax())])
                    probs.append(dict(zip(cands, [round(v, 3) for v in p])))
                ic = interpolated_coords(s, t, lv, al)
                dec = decode(readout(m, spec, iv, a.edit_layer, epos, base_h + al * (src_h - base_h), a.read_layer,
                                     list_pos), s.names, s.root, root_true)
                tr["steps"].append(dict(alpha=al, pos=list(ic[lv]), model=ans, probs=probs,
                                        truth=[answer_for(q, ic) for q in qs],
                                        clear=[bool(passes_margin(q, ic)) for q in qs], decoded=dec))
            sc["targets"].append(tr)
        out["scenes"].append(sc)
        print(f"[demo] scene {len(out['scenes'])}/{a.n_scenes} ({len(targets)} targets, tried {tried}) "
              f"{time.time() - t0:.0f}s", flush=True)

    # headline numbers over the demo itself (full edit, alpha = 1; clear questions only)
    hit = tot = 0
    for sc in out["scenes"]:
        for tr in sc["targets"]:
            st = tr["steps"][-1]
            for mo, tru, c in zip(st["model"], st["truth"], st["clear"]):
                if c:
                    hit += mo == tru
                    tot += 1
    out["summary"] = dict(full_edit_matches_truth=hit / tot, n_questions=tot, scenes_tried=tried)
    d = ROOT / "results" / "demo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "demo_data.json").write_text(json.dumps(out))
    print(f"[demo] done {time.time() - t0:.0f}s; full edit matches truth on {hit}/{tot} clear questions", flush=True)


if __name__ == "__main__":
    main()
