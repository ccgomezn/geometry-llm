# Short paper (4 pages + references/appendix): outline

*Outline for the author (2026-11-19). Facts, numbers, structure and page budget only; the prose is yours. Every number
is from `results/RESULTS.md`; the entry is named next to it. Keep the AI-assistance disclosure.*

## The one claim

Gemma 3 builds the composed position and carries it to where the answer is written, then answers from one
sentence instead. We locate the components that do this, show that removing them leaves no correct read-out (12B),
and show that the gap between what is decodable and what is answered widens from 12B to 27B.

Everything in the paper serves that claim. Anything that does not is cut (list at the end).

## Title options

- Keep v1's: *The Model Knows Better Than It Answers* + a new subtitle, e.g. "…: Gemma 3 Carries a Composed Spatial
  Map to the Answer and Answers From One Sentence".
- Shorter: *Carried but Not Read*; *Decodable at the Answer, Ignored by It*.

Your call. "Never read" (v1) now needs the qualifier "the composed part"; the subtitle above avoids it.

## Page budget (4 pages of body)

| section | pages | carries |
|---|---|---|
| Abstract | ~150 words | the one claim + 3 numbers |
| 1 Introduction | 0.5 | question, the task, three findings |
| 2 Setup | 0.4 | task, cells, model, methods in one paragraph each |
| 3.1 Behaviour: a one-sentence shortcut | 0.4 + Fig 1 | accuracy by cell |
| 3.2 The answer position holds the right answer | 0.6 + Fig 2 | flow map, truth probe vs model |
| 3.3 Who writes the answer | 0.8 + Fig 3 | writers, brakes, removal, swap |
| 3.4 27B | 0.4 + Table 1 | replication, partial |
| 4 Related work | 0.3 | 3 threads |
| 5 Limits and what it means | 0.4 | |
| Appendix (not counted) | – | preregistration table, controls, AI disclosure, code/data |

## 1. Introduction (0.5 p)

- Probes find spatial and world-state structure in LLMs (Gurnee & Tegmark); the open question is whether answers use it.
- The task: three objects in a chain, R → M → C, each placed by one sentence relative to the previous one.
  Question: "Which is farther east, C or R?" The answer needs both steps (d1 = M's step, d2 = C's step,
  truth = sign(d1 + d2)).
- Three findings, one line each:
  1. The model answers from C's sentence alone: 0.34 correct when M's step is larger and opposite (below chance).
  2. Yet at the position where the answer is written, a linear probe reads the right answer on those same questions
     (0.86 vs the model's 0.32 on 12B; 0.98 vs 0.25 on 27B).
  3. A small set of late heads writes the shortcut; others damp it. Removing the writers while keeping everything else
     leaves no read-out that gets these questions right (0.46, linear prediction 0.47).
- All tests preregistered (one line; details in appendix).

## 2. Setup (0.4 p)

- **Scenes.** Generated text, objects placed by integer offsets, one sentence each. The five cells on the asked
  axis: *own step only* (d1 = 0), *agree*, *first step only* (d2 = 0), *conflict, C larger*, *conflict, M larger*.
  The last is decisive: shortcut and truth disagree.
- **Model.** Gemma 3 12B instruct (48 layers); replication on 27B (62 layers). bfloat16, eager attention, one GPU.
- **Measure.** logit(C) − logit(R) at the final position.
- **Methods**, one sentence each: ridge probes on held-out scenes; activation copying between runs that differ in one
  step (flow map); logit lens; exact direct logit attribution through Gemma's sandwich norms, per head and per MLP;
  name-preserving removal (*twin mean*: average of a component's value on the question and on its twin, the same
  question with C's step negated).
- **Shortcut** = coefficient of own_yes (C's step points the asked way) in a regression of the logit difference on
  own_yes, truth, d1, d2, order. Truth is in the regression so a correct read-out is not counted as shortcut.

## 3.1 Behaviour: a one-sentence shortcut (0.4 p, Fig 1)

Source: *Bulk C–R axis test* (2026-11-14 15:52), 5,250 questions, 12B.

| cell | n | accuracy |
|---|---|---|
| own step only | 750 | 0.98 |
| agree | 750 | 0.97 |
| first step only | 750 | 0.53 |
| conflict, C larger | 1500 | 0.88 |
| conflict, M larger | 1500 | **0.34 [0.30, 0.37]** |

- Follows C's step on 77% of conflict answers.
- Chance when the answer is only in the first sentence: it does not use M's sentence on its own.
- 27B is more extreme: 0.23 on *conflict, M larger*, 0.55 on *first step only* (flow entry, 2026-11-19 20:55).

**Number consistency trap:** 12B's *conflict, M larger* accuracy is 0.34 (bulk set), 0.40 (flow set, seed 1601), 0.32
(Stage 2b set, seed 1901). They are different question sets. Use 0.34 for behaviour, and name the set wherever another
number appears (e.g. "0.32 on the Stage 2b set").

## 3.2 The answer position holds the right answer (0.6 p, Fig 2)

Sources: *Sum-of-parts test* (2026-11-15 18:43); *Flow map* 12B (2026-11-15 20:33) and 27B (2026-11-19 20:55).

- **The model composes.** At the end of C's own sentence (layer 24), C's distance to R reads at R² 0.61 vs 0.15 for
  the best store of the two sentences side by side (gap +0.45 [+0.38, +0.53]). It holds on offset pairs never seen in
  training (0.51 vs −0.03). One sentence; this is the "it is a map, not two stored sentences" point.
- **The first step reaches the answer.** M's step at the final position: R² 0.84 (12B, layers 28–48), 0.89 (27B,
  layers 36–62). Same route in both: sentences → question tokens → final position.
- **The right answer is linearly there.** Truth probe at the final position, held-out *conflict, M larger*:
  12B 0.86 (layer 28) vs model 0.32; 27B 0.98 (layer 38) vs model 0.25. Held-out *first step only*: 1.00 both.
- **It is written late.** Logit lens at chance until about 60% of depth, then the answer appears within a few layers:
  12B at chance through layer 28, set by layer 36 (75% of depth); 27B at chance through layer 40, set by layer 44
  (71%).
- Preregistered verdict, both models: *reaches the answer position (the read-out ignores it)*.

**Fig 2 (the key figure):** per layer, truth-probe accuracy at the final position vs logit-lens accuracy, on
*conflict, M larger*, 12B and 27B side by side, x-axis as share of depth. The gap between the two curves is "knows
vs says". Data: `results/flow/gemma3-{12b,27b}_flow_seed1601_probes.csv` (site `final`, column `truth_conflict_m`)
and `_lens.csv` (column `acc`, cell *conflict, M larger*).

## 3.3 Who writes the answer (0.8 p, Fig 3)

Sources: *Stage 2* 12B (2026-11-15 21:55); *Stage 2b* 12B (2026-11-19 11:10).

- **Writers.** Exact attribution (sums to the model's logit difference, r = 1.000). The shortcut is written by late
  attention heads, layers 41–47 (L44H10, L47H2, L45H1, L41H0, …). The top 10 write +36; the model's net is +22; the
  rest of the model −14 (brakes). The writers read the question tokens, not C's sentence.
- **Why ordinary ablation fails (one or two sentences).** Mean ablation of these name-copying heads adds a bias that
  depends on the names (81–86% of its effect), and the shortcut came back in full: preregistered *ineffective*.
  This motivates the twin mean.
- **Removal keeping the names (preregistered):** twin mean of the top 20 at the final position.
  - Shortcut 20.3 → 6.2 [4.5, 8.0] (matched random control 28.0).
  - *Conflict, M larger* 0.32 → 0.46 [0.41, 0.51]; linear prediction (shortcut subtracted, nothing else changed)
    0.47. Verdict: *no correct path underneath the shortcut*.
- **Sufficiency.** Swapping in the twin's values for the 20 components flips the answer on 98% of *own step only*
  questions (93–99% across cells, 72% on *agree*).
- **Self-repair is brakes letting go.** About 2/3 of the removed shortcut returns (−46.3 removed, +31.2 back), and
  78% of the return is from components that pushed against the shortcut in the clean run and move to zero: MLP47
  −6.6 → −0.3, L44H11 −4.9 → −0.3, MLP46 −3.4 → +1.0, L42H1 −2.6 → −0.1. Few backup writers.

**Fig 3:** per-component shortcut coefficient, clean vs removed, for the top writers and top brakes (bar or
dumbbell). Data: `results/components/gemma3-12b_components2_repair_top20.csv`. Or a 3-bar waterfall: writers removed
−46.3 / brakes relaxing / net.

## 3.4 Replication on 27B (0.4 p, Table 1)

Sources: three 27B entries (2026-11-19 20:55, 21:03, 21:15); criteria in `docs/preregistration_replication_27b.md`,
committed before the data.

| preregistered criterion | 12B | 27B | holds |
|---|---|---|---|
| shortcut (*conflict, M larger* < 0.5) | 0.40 | 0.23 | yes |
| flow: reaches the answer position, read-out ignores it | yes | yes | yes |
| Stage 2b: top 20 removal shows no correct path | yes | *ineffective* (ratio 0.55 > 0.50) | no |
| swap of top 20 sets the answer (≥ 0.80, own step only) | 0.98 | 0.37 | no |

- Report as **partial**. What fails is the count: the top 20 do 56% of the shortcut writing in 27B vs 75% in 12B
  (80% needs 55 components vs 34). Writers in the same place (layers 47–60, 76–97% of depth), repair by the same
  mechanism (78% brakes letting go, mostly MLP57–61).
- Exploratory only: with the shortcut cancelled (swap, coefficient +1.0), accuracy is ~0.6; linear prediction on
  *conflict, M larger* 0.51. No sign of a correct path, but not a preregistered result. Say so in one sentence.

## 4. Related work (0.3 p): three threads, 2–3 citations each

All verified in `reports/Decodable but unused representations.md` / `reports/Where LLMs use internal maps.md`
(check the venue flags marked UNVERIFIED there before citing a venue).

1. **Decodable ≠ used.** Ravichander, Belinkov & Hovy 2021 (arXiv:2005.00719); Orgad et al., *LLMs Know More Than
   They Show*, ICLR 2025 (arXiv:2410.02707). Contrast: maps used in models trained from scratch (Li et al. 2023,
   Othello, arXiv:2210.13382). Vafa et al. 2024 (arXiv:2406.03689).
2. **Spatial representations in pretrained LLMs.** Gurnee & Tegmark 2024 (arXiv:2310.02207); Lepori et al. 2026 on
   Gemma 3 grids (arXiv:2602.04212). **Closest prior work:** Ma et al. 2026, EMNLP (arXiv:2606.01914): in multimodal
   LLMs the correct spatial relation stays internally available and a lexical bias overrides it. Say what is new
   here: text-only, composed position (not a stated relation), the components located, removal showing nothing
   underneath, scale trend.
3. **Shortcuts, heuristics, self-repair.** Geirhos et al. 2020 (arXiv:2004.07780); Nikankin et al. ICLR 2025
   (arXiv:2410.21272); IOI backup and negative name movers (Wang et al., arXiv:2211.00593); Hydra effect (McGrath et
   al., arXiv:2307.15771); Rushing & Nanda ICML 2024 (arXiv:2402.15390). Our brakes look like IOI's negative name
   movers; say "similar to", not "the same".

## 5. Limits and what it means (0.4 p)

- One family (Gemma 3), two sizes; one task family (2-D chains of three objects); instruct models.
- Direct effects at the final position only; the hop onto the question tokens is not intervened on.
- Probes show presence, not that a read-out is possible for the model; the truth probe is a linear upper bound on
  what a read-out could get.
- Components were selected by regression; 27B needs more than 20.
- What it means, in one or two sentences: the failure is in the read-out, not in building the map, and scale makes
  the map better without making the read-out use it.

## Appendix (not counted)

- Preregistration table: compose, bulk, flow ×2, Stage 2 ×2, Stage 2b ×2, replication criteria; prereg commit before
  data for each.
- Stage 2 mean-ablation details and the name-bias diagnosis; random-control caveat for Stage 2b (the draws include
  brakes, so the control raises the shortcut to 28; against the clean run the result is the same).
- AI assistance disclosure; code and data (repo URL, commit).

## Cut from the short paper

- The erasure story (P5c, Check B, closed-loop, fair control). It is a separate methods note at most. If kept, one
  sentence in limits: "earlier erasure claims in our v1 preprint are corrected in its v2".
- DAS, the 2-D switch at each sentence, P7–P10, the triangle study (Study 2), the walk ladder.
- Earlier 27B DAS runs.

## Open decisions (yours)

1. **New paper or Zenodo v2?** This is a different paper from v1 (different claim, new method). Option: publish it as
   a new record and add a short correction note to v1 for the erasure claims.
2. **Venue.** 4 pages fits a workshop short-paper track (the ICLR 2027 workshops, calls expected early December 2026;
   check each call's page limit and anonymity rules). TMLR has no page limit, so the short version can go there too.
3. **Optional 27B rerun with the top 55** to settle criterion 3 (~1 h Colab). Not needed for the paper; Table 1 is
   honest as is.
4. **Figures.** I can make Fig 1–3 with `paper/make_paper_figs.py` once you approve the list.
