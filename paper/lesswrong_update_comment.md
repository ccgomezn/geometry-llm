**Update: four more preregistered follow-ups (Gemma 3 12B), and a sharper picture.**

The obvious objection to the erasure null was "the task is too easy to need a map", followed by "then what *does* the model use?". Each test below had its decision rule committed before the data.

1. **Harder scenes (P6).** We reran INLP erasure on 4–5-object scenes, with questions up to 4 hops from the root.
   - Probe R² after erasure is ≤ 0.095, down from up to 0.60.
   - Deep questions have headroom: 0.65 clean accuracy against ~0.36 chance.
   - The excess drop on deep questions, relative to random erasure of equal rank, is −0.005, +0.023, +0.001 and −0.031 at layers 14, 21, 28 and 12–30.
   - Verdict: **still not used.** The +0.023 at L21 is a statistically detectable trace, a fifth of the 0.10 threshold.
2. **What the question reads (P7, attention knockout).** We blocked the question tokens from attending to one span, in all 48 layers or in one third of them.
   - **Object list** (where the map decodes): every class changes by ≤ 0.018.
   - **C's own sentence:** CM questions drop 0.351.
   - **M's sentence:** MR questions drop 0.410.
   - Both effects sit almost entirely in layers 16–31.
   - But questions needing C's composed position (C vs R) drop only 0.027 when M's sentence is blocked. So "the question re-reads the chain" is **rejected**.
3. **Multi-layer transfer (P8).** We copied C's word-identical sentence from the run where M (and with it C) moved, at every layer from L onward.
   - Even L = 0 moves only 0.13 [0.09, 0.17] of the target answers. Verdict: partial.
   - **Exploratory follow-up:** C ends up at the same new position whether only C's sentence changes or only M's sentence changes. On C-vs-R questions the model updates correctly in 0.39 of the first case but only 0.21 of the second. **The answers under-use the parent's offset.**
4. **Would the map have answered better? (P9).** We decoded positions with the list-position probe at L21 (R² 0.71) and answered geometrically.
   - **C-vs-R questions:** map 0.80 vs model 0.73, +0.069 [+0.042, +0.094]. Verdict: **the map would have answered better.**
   - **Questions needing no composition:** ties (0.91 vs 0.89; 0.90 vs 0.90).
   - L24 shows the same pattern. The advantage fades at L28 and reverses at L18, where the probe is weaker.

Updated summary: the model composes an accurate map at the object list. Its answer pathway reads each queried object's own sentence in layers 16–31, never the list, and only partially integrates the chain. A linear read-out of the unused map beats the model's own answers on exactly the questions that need composition. This looks like the "faithful world state, unfaithful decoding" pattern of Feng, Russell & Steinhardt (2024), here in a spatial setting with erasure controls.

A bug and a deviation, disclosed: in P7, a classification bug dropped direction questions about C and M alone before scoring. The CM class therefore contains only further-north/east questions. CR and MR, on which the verdicts rest, are unaffected.

Part 3 (plain-language write-up): https://medium.com/@gfcristhian98/the-ai-knew-better-than-it-answered-665e03cf3fb6. Summary page with an interactive demo: https://ccgomezn.github.io/geometry-llm/. Code, preregistrations and results: https://github.com/ccgomezn/geometry-llm. Paper: coming soon.
