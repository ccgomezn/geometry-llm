# Your three paragraphs for the front page

Write each paragraph in your own words: 3–5 plain sentences, as if explaining it to a smart friend. Use the facts
below; you don't need all of them. Send the three paragraphs to Claude, who will check the numbers and put them in the
page (no rewriting).

## Paragraph 1: the setup and the map

- You gave Gemma 3 (an open AI model from Google, 12B and 27B) small scenes in words, e.g. "Ana is 3 units north of
  Bo. Cy is 2 units east of Ana.", and asked questions the text never answers directly ("Is Cy further north than Bo?").
- Inside the model you can read every object's position with a simple decoder, R² 0.74 (0 = nothing, 1 = perfect),
  including objects like Cy whose position nobody stated (you have to chain two sentences).
- It even holds distances, which you can't get by just adding up the sentences (0.62 vs −0.05 for the best adding-up
  baseline). So it really built a map.

## Paragraph 2: the answers don't use it, and what they use instead

- Deleting that map at every word and many layers hurt the answers no more than deleting random directions. Same on
  harder 4–5-object scenes.
- Blocking the question from looking at where the map lives changed nothing (≤ 2 points).
- What the answers use instead: each object's own sentence. Two numbers inside that sentence control the answers:
  editing them moves 72% of answers (83% in the bigger 27B model; random numbers 0%). It acts like a switch between
  stated facts, not a dial on a map.
- And the model takes a shortcut: when Cy moves because its own sentence changed, the model updates its answers 39% of
  the time; when Cy moves to the same spot because Ana's sentence changed, only 21%.

## Paragraph 3: the map would have answered better

- Reading the answer off the model's own unused map is right 80% of the time on questions that need the chain, vs 73%
  for the model's actual answers. On questions stated directly, they tie (about 90%).
- Pushing the model to look at its map doesn't fix it; it just gets distracted.
- So the model holds a better answer than the one it gives, and doesn't use it.
- Optional honesty line: one model family, simple puzzles; every test was written down before its data existed.
