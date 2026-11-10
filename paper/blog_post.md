# We Found a Map Inside an AI. Then We Tried to Move It.

*Notes from an ongoing experiment on whether language models "picture" a scene they only read about.*

---

Read this: *"Ana is 3 units north of Bo. Cy is 2 units east of Ana."* Now answer: who is farthest from Bo?

You probably didn't reread the sentences. You placed three dots in your head and looked. Psychologists have argued for decades about whether that inner picture does real work or is a side effect of thinking. I want to ask the same of AI: **when a language model solves a geometry problem given only in words, does it build a picture inside, and does it use it?**

## The setup

I generated thousands of small layouts no model could have seen before, described only in words. Each came with questions the text never answers directly: farthest, closest, which direction, who is further north.

Eight open models (Qwen, Llama and Gemma, 1.7 to 27 billion parameters) had to answer instantly, without "thinking out loud". The smallest that handled three-object layouts reliably was **Gemma 3 12B**, so that's the one I opened up. Before looking inside, I wrote down in advance what would count as evidence for a map, so I couldn't move the goalposts later.

## Finding 1: there is a map

From the model's internal activations, a simple decoder recovers every object's x and y coordinates (R² = 0.74). That includes positions the text never states. Cy's position only follows from chaining two sentences, and a decoder that only sees what each sentence says gets 0.22 on objects like Cy. So the model really *combines* the sentences into positions.

## Finding 2: moving the map where we found it does nothing

Readable doesn't mean used, so I tried to move an object inside the model. I replaced the internal representation where the coordinates live with one from a layout where that object sits elsewhere. If the answers came from there, they should change.

They didn't. In 390 tries, the full set of answers never followed. Swapping the *sentence* about that object, however, moved every answer consistently.

## What that does and doesn't mean

It's tempting to conclude "the model builds a map and ignores it". I don't think we can say that yet. There are two other explanations, and both are about where we looked:

- **The model may keep more than one copy.** Overwrite one and the others outvote it.
- **The map it uses may sit somewhere else**, for example right at the sentence about the object, which is exactly where our edit worked.

What we *can* say is already useful: a decoder that passes every standard check can point at a representation that has no effect on the answer. Finding *where* information is readable isn't the same as finding *what the model relies on*.

## What we're running now

1. **Look for the map where the edit works.** Can the coordinates be read off the sentence itself, including the positions that require combining sentences?
2. **Remove the other copies.** Copy everything from the other layout *except* the place where we found the map. If the answers still follow, that copy really isn't needed.
3. **The decisive test: move the middle of a chain.** Take A → B → C, where C is placed relative to B, and move B inside the model. If the model has already worked out where C is, C stays put while B moves. If it rebuilds positions from the sentences each time, C follows B. Doing this layer by layer shows whether, and when, the model "draws the picture".

The long-term goal is the step from *reading* to *seeing*: what changes when the same model is shown an actual image.

*Code, preregistration and results are in one repository and will be released with the paper.*
