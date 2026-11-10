# The AI Drew a Map. Then It Didn't Look at It.

*What happens inside a language model when you describe a scene in words, and why "we can read it out" is not the same as "the model uses it".*

---

Read this: *"Ana is 3 units north of Bo. Cy is 2 units east of Ana."* Now answer: who is farthest from Bo?

You probably didn't reread the sentences. You placed three dots in your head and looked. Psychologists have argued for decades about whether that inner picture does real work or is just a side effect of thinking. I wanted to ask the same question about AI: **when a language model solves a geometry problem given only in text, does it build a picture inside?**

## The setup

I generated thousands of small layouts that no model could have seen before. Each was described only in words, and each came with questions the text never answers directly: farthest, closest, which direction, who is further north.

First, a filter. Eight open models (Qwen, Llama and Gemma, from 1.7 to 27 billion parameters) had to answer instantly, with no "thinking out loud". Most struggled. The smallest model that handled three-object layouts reliably was **Gemma 3 12B**, so that's the one I opened up.

Before running anything, I wrote down what would count as "the model builds a map" and what would count as "it just chains the sentences" (a preregistration), so I couldn't move the goalposts afterwards.

## Finding 1: the map is there

The model's internal activations can be read with a simple linear decoder. From them, I could recover every object's x and y coordinates with high accuracy (R² = 0.74). This held even for objects whose position the text never states directly: Cy's position is only implied by chaining two sentences. A decoder that only sees what each sentence says gets 0.22 on those objects; the model's internals give 0.74. So the model really does *combine* the sentences into positions. By the usual standard in AI interpretability, that's a map.

## Finding 2: the model doesn't use it

Here's the twist. Readable is not the same as used. So I did surgery. I took the internal representation where the coordinates live and swapped it for one from a layout where the object sits somewhere else. If the model answers from its map, its answers should change.

**They didn't. Not once in 390 tries.** I also pushed the representation smoothly along the very directions that encode "x" and "y". The answers should flip at predictable points; they flipped in 2% of cases.

What *does* change the answers? Swapping the original sentence about that object. Do that, and every answer moves consistently. The model builds a tidy coordinate code, and then its answers come from the sentences.

## Why this matters

1. **"We can decode it" ≠ "the model uses it."** Many claims that AI models have "world models" rest on decoders like mine. Mine passed every check I could think of and still sat entirely off the path to the answer. Interventions, not readouts, are what tell you what a model actually relies on.
2. **Imagining is about use, not existence.** The human debate was never "do mental pictures exist?" It was "do they do the work?" Here, a picture-like code exists and, where we found it, does none.
3. **Writing helps a lot.** With step-by-step reasoning switched on, one model went from 57% to 98% on the same questions. Text may be serving as the scratchpad these models lack internally.

## What's next

This result locates the information but not its *form*. The next experiment is the one that can tell a map from a chain of sentences. Take A → B → C, where C is placed relative to B, and move B inside the model. If the model has already computed C's position on a map, C should stay put while B moves. If it is just chaining sentences at question time, C should follow B. Watching *when*, layer by layer, C stops following tells us whether and when the model "draws the picture".

After that: more models, bigger scenes, and the step I care about most, from reading to *seeing*: what changes when the same model gets an actual image.

*Code, preregistration and all results are kept in one repository and will be released with the paper.*
