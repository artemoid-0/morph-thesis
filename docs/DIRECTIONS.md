# Directions — the research programme

> **Kind:** claims — the current plan and queue; rewritten as they change.
> **Authoritative for:** what is investigated next, why, and in what order.

The thesis stands on what is established ([`FINDINGS.md`](FINDINGS.md)).
The programme began on 2 October from its one positive regularity — *a feature helps until the model has learned what it encodes* — and asked when, where and in what role morphology helps.
Each open direction names the result that would redirect it.

**Scope** (7 October, §33.59).
Every direction serves the thesis — context-aware morphological embeddings for language models: when, why and how far morphology in the embedding helps, and how that compares with morphology in the tokenizer.
Questions about small transformers in general are parked, however interesting.

## Where we are

Each direction's answer is stated once, in FINDINGS; the table gives its status and the claim to read.

| | direction | status | the claim |
|---|---|---|---|
| D1 | the learning-progress law | **answered within one model size**; across sizes, half explained (§33.35–§33.52, §33.60) | [one curve per feature](FINDINGS.md#one-curve-per-feature-within-one-model-size); [across model sizes](FINDINGS.md#across-model-sizes) |
| D2 | the mechanism, by probing | **closed** — the pivot not met (§33.49–§33.52, §33.60, §33.114) | [across model sizes](FINDINGS.md#across-model-sizes) |
| D3 | removing, weaning, adding late | **answered**: the deficit halves by 50k and closes by 100k (§33.34–§33.52, §33.102, §33.111, §33.126) | [removing a feature](FINDINGS.md#removing-a-feature-and-adding-it-late) |
| D4 | morphology as a target | **closed** (§33.50, §33.52) | [morphology as a target](FINDINGS.md#morphology-as-a-target) |
| D5 | the rare and novel tail | **closed**, screened at full training; the unseen-word version parked (§33.47) | [the rare tail](FINDINGS.md#no-benefit-in-the-rare-tail) |
| D6 | the token-independent input | **parked — outside the thesis** (§33.59); G-71 closed (§33.58) | [what the slot did instead](FINDINGS.md#what-the-slot-did-instead-a-token-independent-input) |
| D7 | tokenization | **answered** at four seeds (§33.61–§33.81); the exposure comparison against both references (§33.110–§33.118, §33.126; Unigram at width 512) | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe); [in English](FINDINGS.md#q1-in-english-morphology-inside-bpe); [the tokenizer and the embedding](FINDINGS.md#morphology-in-the-tokenizer-and-in-the-embedding-ukrainian) |
| D8 | scale and architecture: wider, deeper, a modern block | width 512 **answered** (§33.94, §33.102, §33.118); width 1024 **answered** (§33.120–§33.124, §33.132): smaller again and lasting; the split by seed was the 100-step warm-up's; depth and the RMSNorm and SwiGLU block (built, §33.54) not run | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe) ("It peaks there"); [the null at 30M](FINDINGS.md#the-same-shape-null-is-not-neutral) |
| D9 | a second corpus per language | Ukrainian **answered** with UberText fiction (§33.85, §33.88, §33.108); no English corpus chosen | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe) ("And on a second corpus") |

**D3 — is the reliance deficit permanent?**
At full training it halves and lasts (§33.102); whether it ever closes is queue item 3.
Not queued (new code, a short block): switching off to a constant (every code to one class) instead of to zeros — removing the information but keeping a shared vector — would tell the lost information from the input's sudden change.

## Parked

**D6, the shared input vector — outside the thesis** (§33.59).
For the thesis it is the confound it explains, and the one test that could have changed that reading (G-71) is closed.
What remains is about the vector itself: the fixed against the learned vector (2 × 24 min), G-71's seeds 1–3 (~2.4 h), the vector added under rotary positions at seed 3, the 256-wide pair under rotary positions, the 256-wide spread at seeds 5–7 (P91, P92, not tested), the vector's norm against the token embeddings', and the literature on shared input vectors (BERT's segment embedding, attention sinks, massive activations, vision registers).

Also parked: morphology-factorized embeddings for parameter efficiency (Q1 partly answers it; prior work to audit first) · convergence pairs at 100k and 200k steps (G-44, G-30) · English little data (WikiText needs sub-sampling) · the literature audit's unverified cells (G-61) · lexicon file naming (G-66).

## The queue

The queue as the thesis text ranked it on 25 October (§33.140), each item tied to a limit the text states, with its cost from the recorded timings (`scripts/ops/run_times.py`).
The text is complete without them; each lifts a limit it states.
Results committed by about 31 October reach the text; the work is admitted on 4 November.

**Answered** (§33.141–§33.142): the chain of 25 October ran items 1 and 2; items 3 and 4 did not run, their conditions unmet (case kept +0.0015 over its null, not more than 0.002; the tokenizer's gain was +0.0072, not below 0.006). Nothing runs; nothing is queued.

1. **Full training at 30M parameters** (the Introduction's first and third novelties, general conclusions 4 and 7), answered (§33.142): the tokenizer gains +0.0072 (0.97%) and lasts; case stays within 0.002 of its null (+0.0015, at every seed). The runs: the nulls at width 512 under BPE 17k and endings apart, and case under BPE 17k, at 151M tokens and 50k, four seeds: 12 runs, ~16 h; then MultiBLiMP on the nulls. The tokenizer's gain and the features' end-of-training answer, each now measured at full training in the 5M model only.
2. **English endings apart at 30M parameters** (width 512, 10k, four seeds): 8 runs, ~2.2 h; answered (§33.142): it gains nothing there (+0.0002 ± 0.0004, 2/4), so the English gain is the 5M model's.
3. **What case's lead over the word form is made of at 30M**, only if case keeps more than 0.002 over its null in item 1: case from the word form and case without the leak, 8 runs, ~10.7 h. If case's gain is gone, the text does without it.
4. **The empty slots at 30M**, only if the tokenizer's gain in item 1 falls below 0.006: 8 runs, ~10.7 h.
5. **Case under endings apart at full training at 30M**, last: it would need its own leak control; not in the chain.

The text's request of 24 October (§33.136): the empty slot at width 1024 is answered (§33.139), the work's size given (§33.138), and item 1 runs:

- **Width 1024 without repeating the text** (§3.3.2, §3.12): answered (§33.141): +0.0082 (1.09%) at 20k, 0.0022 below the repeated text's, still rising. The nulls at width 1024, 20k, with the 1,000-step warm-up, both tokenizers, four seeds, on the 151M-token corpus (about 1.1 passes) and on ~38M (the repeated text under the same warm-up); then MultiBLiMP: 16 runs, ~25 h.

The text's request of 22 October is answered (§33.130–§33.135): width 1024 holds at every seed with a longer warm-up; at full training what the sentence adds over the word form is not context, for Ukrainian gender and number and for English part of speech.
The text's list of 19 October is answered (§33.117–§33.129); each item's outcome is in the journey.

**Dropped** by the sentence test (every outcome gives the same sentence, or none):
- more seeds for English morpheme segmentation (the 12–13% gap is about nineteen times the widest seed spread);
- the shape effect by language;
- why fiction's share of bits per byte is half Wikipedia's (§33.88): the thesis states the size on each corpus, and no outcome changes a claim;
- case and its controls on fiction: morphology in the embedding is answered (§33.83).

**Not requested:** a third language (a defence item); the law across model sizes, until there is a hypothesis for the half the previous word's tag leaves open.

**Maintenance**, when there is time: G-79 (three of the thesis machine's models to re-make, ~36 min on it).

The lists before this one, with each item's outcome, are in the journey (§33.83, §33.92–§33.118); the list of 15 October with its outcome sentences is its [Appendix C](record/journey/appendices.md#appendix-c--the-thesiss-list-of-15-october-and-its-outcome-sentences).

## Where runs go

The second machine is equivalent to the thesis machine up to dropout's draw (§33.109), 2.8 times faster, and free at night and at weekends.
New experiments run there, each comparison within one machine (FINDINGS, method point 11); where a comparison needs the thesis machine's runs, their partners are repeated there as twins.
It holds its own null, case and empty slot under six tokenizers at 10k and under two at width 512, and the English nulls at both widths.
How a condition is made: [`reference/conditions.md`](reference/conditions.md).
