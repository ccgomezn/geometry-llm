# Zenodo upload: what to put in each field

Go to https://zenodo.org, log in (the "Log in with GitHub" button works), then **New upload**.

| Field | Value |
|---|---|
| Files | `Gomez-Neira-2026-The-Model-Knows-Better-Than-It-Answers.pdf` (this folder) |
| Resource type | Publication → **Preprint** |
| Title | The Model Knows Better Than It Answers: Gemma 3 Composes a Spatial Map Its Answers Never Read |
| Creators | Gómez Neira, Cristhian (Personal) — add your ORCID if you have one; affiliation optional |
| Publication date | today's date |
| Description | the abstract below |
| License | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| Keywords | interpretability; mechanistic interpretability; language models; world models; spatial reasoning; probing; causal intervention; distributed alignment search; concept erasure; preregistration |
| Related works | *Is supplemented by*: https://github.com/ccgomezn/geometry-llm (Software). *Is described by* (optional): your Medium posts and the LessWrong post |
| Language | English |

Then **Publish**. Zenodo gives the DOI immediately (format 10.5281/zenodo.NNNNNNN). Send it to Claude to add to the
repo README, the citation file and the blog posts.

Note: a published Zenodo record cannot be deleted, but you can add new versions later (e.g. after arXiv or a
workshop), and they all share one "concept DOI".

## Description (copy-paste)

When people read "A is 3 units north of B; C is 2 units east of A", they can answer questions the text never answers, as if looking at a picture. We ask whether a language model builds such a picture, and whether its answers use it. On text-only layouts, Gemma 3 12B builds a map: linear probes read each object's position from its activations (R² = 0.74), including positions that require combining sentences. But the answers do not use this map. Erasing it everywhere costs no more accuracy than erasing random directions. Instead, the answers depend on a small 2-D code over the sentence that states each position. Swapping it moves 72% of the affected answers (83% in Gemma 3 27B; random: 0%), and it acts as a switch, not a coordinate. The answers also use the chain of relations only partly, so answering from the model's own unused map beats the model on chain questions (0.80 vs 0.73). The main tests were preregistered.

Code, preregistrations (committed before each experiment's data) and all results: https://github.com/ccgomezn/geometry-llm
