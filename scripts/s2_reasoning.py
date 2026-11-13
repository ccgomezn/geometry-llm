"""Study 2 (docs/preregistration_s2_reasoning.md): can the model tell whether a point is inside the triangle when it may
think step by step? The pilot's rung-4 questions (same 400 scenes, seed 702) with the one-word instruction replaced
by "think it through step by step, then end with 'Answer: yes' or 'Answer: no'". Greedy decoding. A reply without a
final answer line counts as wrong. Compared with the same model's one-word answers (the pilot) and the size rule.

Resumable: replies are saved after every batch under results/s2/reasoning and skipped on a rerun. Progress goes to
Slack if $SLACK_WEBHOOK_URL is set."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.checkpoint import notify
from geomap.models import load
from geomap.prompts import NAME_POOLS, single_token_names
from geomap.reasoning import cot_prompt_text, mentions_coordinates, parse_final_answer, verdict_s2_reasoning
from geomap.report import ROOT, append_entry
from geomap.study2 import ba_ci, ba_diff_ci, fit_magnitude_rule, magnitude_rule
from geomap.triangles import generate_tri_scene, ladder

TEST_SEED, MAG_SEED = 702, 701


def scenes(seed: int, n: int, pools) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ts = generate_tri_scene(rng, pools[i % 2])
        out.append((ts, ladder(ts, rng)))
    return out


def fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def generate_all(m, prompts: list[str], path: Path, max_new_tokens: int, batch_size: int, tag: str) -> pd.DataFrame:
    """Greedy replies for every prompt, saved after each batch; finished batches are skipped on a rerun."""
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["i", "reply", "n_tokens", "truncated"])
    rows = done.to_dict("records")
    tok = m.tok
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    todo = [i for i in range(len(prompts)) if i not in set(done.i)]
    t0 = time.time()
    for b in range(0, len(todo), batch_size):
        idx = todo[b:b + batch_size]
        enc = [tok(prompts[i], add_special_tokens=False).input_ids for i in idx]
        T = max(map(len, enc))
        ids = torch.full((len(enc), T), pad, dtype=torch.long)
        am = torch.zeros_like(ids)
        for j, x in enumerate(enc):  # left padding for generation
            ids[j, T - len(x):] = torch.tensor(x)
            am[j, T - len(x):] = 1
        with torch.no_grad():
            gen = m.hf.generate(input_ids=ids.to(m.hf.device), attention_mask=am.to(m.hf.device),
                                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
        for j, i in enumerate(idx):
            out = gen[j, T:]
            n = int((out != pad).sum())
            rows.append(dict(i=i, reply=tok.decode(out, skip_special_tokens=True), n_tokens=n,
                             truncated=n >= max_new_tokens))
        pd.DataFrame(rows).to_parquet(path)
        k = len(rows)
        print(f"[s2 reasoning] {k}/{len(prompts)} replies ({time.time() - t0:.0f}s)", flush=True)
        if k % (batch_size * 5) < batch_size:
            notify(f"[{tag}] {k}/{len(prompts)} replies")
    return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-27b")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 12 scenes, 16 tokens; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/s2/reasoning)")
    a = ap.parse_args()
    out = ROOT / "results" / "s2" / "reasoning"
    n_mag = 2000
    if a.smoke:
        import tempfile
        a.model, a.n, a.max_new_tokens, n_mag, out = "tiny", 12, 16, 60, Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_s2_reasoning_n{a.n}_seed{TEST_SEED}_tok{a.max_new_tokens}"
    tag = f"S2 reasoning {a.model}"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[s2 reasoning] already complete ({done}); delete it to rerun", flush=True)
        return
    m = load(a.model)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    test = scenes(TEST_SEED, a.n, pools)
    items = [(si, ts, next(it for it in lad if it.rung == 4)) for si, (ts, lad) in enumerate(test)]
    notify(f"[{tag}] started: {len(items)} questions, up to {a.max_new_tokens} tokens each")

    # the same model's one-word answers from the pilot (required for the paired comparison, except in smoke runs)
    pilot = ROOT / "results" / "s2" / "pilot" / f"{a.model}_s2_pilot_n{a.n}_seed{TEST_SEED}_answers.parquet"
    if pilot.exists():
        p4 = pd.read_parquet(pilot).query("rung == 4").sort_values("scene")
        if list(p4.text) != [it.question.text for _, _, it in items]:
            raise ValueError("the pilot's rung-4 questions do not match the current ones")
        plain = p4.pred.values
    elif a.smoke:
        plain = np.array(["no"] * len(items))
    else:
        raise FileNotFoundError(f"{pilot} is needed for the comparison with one-word answers")

    prompts = [cot_prompt_text(ts.scene, it.question, m.tok, m.family) for _, ts, it in items]
    rep = generate_all(m, prompts, out / f"{run}_replies.parquet", a.max_new_tokens, a.batch_size, tag)
    mag = magnitude_rule(fit_magnitude_rule([ts for ts, _ in scenes(MAG_SEED, n_mag, pools)]), [ts for _, ts, _ in items])

    df = pd.DataFrame([dict(scene=si, stratum=ts.stratum, truth=it.truth, text=it.question.text) for si, ts, it in items])
    df["reply"], df["n_tokens"], df["truncated"] = rep.reply.values, rep.n_tokens.values, rep.truncated.values
    df["answer"] = [parse_final_answer(r) for r in df.reply]
    df["pred"] = df.answer.fillna("none")  # no final answer counts as wrong
    df["plain_pred"], df["mag_pred"] = plain, mag
    df["coords"] = [mentions_coordinates(r) for r in df.reply]
    df.to_parquet(out / f"{run}.parquet")

    rows = []
    for subset, d in (("hard", df[df.stratum.isin(("inside", "box_hard"))]), ("all", df)):
        r = dict(subset=subset, n=len(d))
        for who, col in (("reasoning", "pred"), ("plain", "plain_pred"), ("magnitude", "mag_pred")):
            r[f"{who}_bal_acc"], r[f"{who}_bal_acc_lo"], r[f"{who}_bal_acc_hi"] = ba_ci(
                d.truth, d[col], "yes", d.scene)
        for other, col in (("plain", "plain_pred"), ("magnitude", "mag_pred")):
            r[f"reasoning_minus_{other}"], r[f"reasoning_minus_{other}_lo"], r[f"reasoning_minus_{other}_hi"] = \
                ba_diff_ci(d.truth, d.pred, d[col], "yes", d.scene)
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(out / f"{run}_summary.csv", index=False)
    v = verdict_s2_reasoning(s)

    yes = df.groupby("stratum").pred.apply(lambda x: float(np.mean(x == "yes")))
    h = s[s.subset == "hard"].iloc[0]
    hd = df[df.stratum.isin(("inside", "box_hard"))]
    lines = [f"Model **{a.model}**. The pilot's {len(items)} rung-4 questions (same scenes, seed {TEST_SEED}), with the "
             "one-word instruction replaced by *think it through step by step, then end with 'Answer: yes' or "
             f"'Answer: no'*. Greedy decoding, up to {a.max_new_tokens} new tokens; a reply with no final answer line "
             "counts as wrong. *One word* = the same model's pilot answers. Balanced accuracy with 95% scene-bootstrap "
             "CIs; differences are paired.", "",
             "| subset | n | step by step | one word | size rule | step by step − one word | step by step − size rule |",
             "|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.subset} | {r.n} | {fmt(r.reasoning_bal_acc)} [{fmt(r.reasoning_bal_acc_lo)}, "
                     f"{fmt(r.reasoning_bal_acc_hi)}] | {fmt(r.plain_bal_acc)} | {fmt(r.magnitude_bal_acc)} | "
                     f"{fmt(r.reasoning_minus_plain)} [{fmt(r.reasoning_minus_plain_lo)}, {fmt(r.reasoning_minus_plain_hi)}] | "
                     f"{fmt(r.reasoning_minus_magnitude)} [{fmt(r.reasoning_minus_magnitude_lo)}, "
                     f"{fmt(r.reasoning_minus_magnitude_hi)}] |")
    cz = hd[hd.coords]
    nz = hd[~hd.coords]
    acc = lambda d: fmt(float(np.mean(d.pred == d.truth))) if len(d) else "–"
    lines += ["", f"Says yes: " + ", ".join(f"{k} {yes.get(k, float('nan')):.2f}" for k in ("inside", "box_hard", "outside_box"))
              + f". No final answer: {int(df.answer.isna().sum())}; hit the token limit: {int(df.truncated.sum())}; "
              f"median reply length {int(df.n_tokens.median())} tokens.",
              f"Exploratory: {cz.shape[0]} of {len(hd)} hard replies write (x, y) coordinates (accuracy {acc(cz)}), "
              f"{nz.shape[0]} do not (accuracy {acc(nz)}).",
              "", f"Preregistered verdict (hard subset): **{v}**. Rules: docs/preregistration_s2_reasoning.md."]
    if not a.smoke:
        append_entry(f"Study 2: can the model tell inside the triangle when it thinks step by step? ({a.model}) — "
                     "preregistered", "\n".join(lines), [])
    done.write_text(v)
    notify(f"[{tag}] COMPLETE. {v}. Hard: step by step {fmt(h.reasoning_bal_acc)}, one word {fmt(h.plain_bal_acc)}")
    print("\n".join(lines))
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
