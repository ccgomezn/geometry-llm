# TMLR submission checklist

Submit at https://openreview.net/group?id=TMLR (Submit button; needs an OpenReview profile).

## Files

| What | File | How to make it |
|---|---|---|
| Paper (anonymous) | `paper/tmlr/main.pdf` | `cd paper/tmlr && latexmk -pdf main.tex` |
| Supplementary material | `paper/tmlr/build/supplementary_material.zip` (20.7 MB) | `python3 paper/tmlr/make_supplement.py` |

Both were checked on 2026-11-20. The PDF has no name, repository link, commit hash or DOI in its text or metadata.
The zip holds the code, tests, preregistrations, Colab notebooks and result files, with names, accounts, paths, the
repository URL, the earlier DOI and every commit hash removed. The script refuses to write it if any remain. All 26
test files pass from the unpacked zip.

## Form

- **Title:** The Outvoted Map: LLMs Carry Spatial Geometry to the Answer, and a One-Sentence Rule Outvotes It
- **Abstract:** copy from `main.tex` (between `\begin{abstract}` and `\end{abstract}`). Replace `\%` with `%`.
- **Submission length:** regular submission (about 8 pages of main text, 12 pages or fewer).
- **Keywords:** mechanistic interpretability; language models; spatial reasoning; world models; probing; activation
  patching; direct logit attribution; preregistration.
- **Supplementary material:** the zip above.

## Rules that matter

- **Preprints are allowed.** The non-anonymous Zenodo preprint and a later arXiv version do not conflict with TMLR. Do
  not mention them in the submission.
- **No parallel archival submission.** While the paper is under review at TMLR, do not submit it to ICML, ACL/ARR or
  another archival venue. A workshop that is declared non-archival is fine.
- **After acceptance**, switch to `\usepackage[accepted]{tmlr}` and set `\month`, `\year` and `\openreview`. Then add
  the repository link back into the Reproducibility statement.
