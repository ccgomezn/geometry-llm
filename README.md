# The Model Knows Better Than It Answers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24178043.svg)](https://doi.org/10.5281/zenodo.24178043)

Code, preregistrations and results for a study of how a language model handles spatial layouts described only in text.

We give Gemma 3 (12B and 27B) small scenes such as *"Ana is 3 units north of Bo. Cy is 2 units east of Ana."* and ask
questions the text never answers directly ("Is Cy further north than Bo?"). Inside the model, a linear probe can read
every object's position, including positions that only follow from combining sentences. But the answers never use
that map:

- **The map is built but never read.** Copying it, steering along it or erasing it at every token and layer changes no
  more answers than doing the same to random directions. This holds on 3-object and 4–5-object scenes.
- **The answers come from a compact code over each stated sentence.** A 2-D subspace of the sentence that places an
  object controls the answers (72% on 12B, 83% on 27B, 0% for random subspaces). It behaves like a switch between stated
  values, not like a coordinate.
- **The answers take a shortcut.** The question reads each object's own sentence (layers 16–31) and under-uses the
  chain: answers about an object change about half as often when only its parent's sentence moves it.
- **The unused map would have answered better.** Answering geometrically from the decoded map beats the model on
  questions that need the chain (0.80 vs 0.73) and ties it elsewhere. Pushing the model's attention toward the map
  does not help.

Every confirmatory test was preregistered: its decision rule was committed to this repository before its data existed.
The git history shows the order.

- Paper (preprint, Zenodo): https://doi.org/10.5281/zenodo.24178043 · source: [`paper/draft.md`](paper/draft.md) (PDF build: `python3 paper/arxiv/build.py`)
- Plain-language write-ups: [Part 1](https://medium.com/@gfcristhian98/the-ai-worked-out-where-things-are-even-when-nobody-told-it-d4f9faed80ea),
  [Part 2](https://medium.com/@gfcristhian98/i-found-the-map-inside-the-ai-then-i-deleted-it-and-nothing-happened-70f599a6fc83)
- Interactive demo: [`demo/move-the-map.html`](demo/move-the-map.html)

## Layout

| Path | Contents |
|---|---|
| `docs/preregistration*.md` | Preregistered designs and decision rules (E1, E2, E3, P4c, P5–P10) |
| `src/geomap/` | Library: scenes and prompts, scoring, probes, patching, DAS, erasure (INLP), attention knockout/boost, checkpointing |
| `scripts/` | One script per experiment (`p2_*` probes, `p3_*` patching, `p4_*` DAS, `e3_*`, `p5_*`–`p10_*` follow-ups) |
| `tests/` | Unit tests (a tiny randomly initialised model; no GPU needed) |
| `results/RESULTS.md` | Log of every run, appended by the scripts, with hand-written notes on deviations |
| `results/` | Per-experiment outputs (CSV / parquet) and figures |
| `colab/` | Colab notebooks used for P5–P10 (resumable; results saved to Google Drive) |
| `paper/` | Paper draft, LaTeX build, blog posts and figures |
| `docs/related_work_memo.md` | Related-work memo with verified citations |

## Running

```bash
uv sync
uv run pytest                     # ~2 minutes on a laptop CPU
uv run python scripts/p5_erase.py --model tiny --smoke   # any script, end to end, on the tiny test model
```

Real runs need a GPU and access to the gated Gemma 3 weights on Hugging Face (`HF_TOKEN`). Gemma 3 12B fits a single
40 GB A100 in bfloat16. The `colab/` notebooks set everything up, including a fix for Colab's TensorFlow breaking the
transformers Gemma 3 import. Large checkpoints (`*.pt`) are stored with Git LFS.

## Citation

Cristhian Gómez Neira (2026). *The Model Knows Better Than It Answers: Gemma 3 Composes a Spatial Map Its Answers
Never Read.* Preprint, Zenodo. https://doi.org/10.5281/zenodo.24178043
