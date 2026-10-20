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
| D3 | removing, weaning, adding late | **answered** to full training (§33.34–§33.52, §33.102, §33.111); whether the deficit closes: queue item 3 | [removing a feature](FINDINGS.md#removing-a-feature-and-adding-it-late) |
| D4 | morphology as a target | **closed** (§33.50, §33.52) | [morphology as a target](FINDINGS.md#morphology-as-a-target) |
| D5 | the rare and novel tail | **closed**, screened at full training; the unseen-word version parked (§33.47) | [the rare tail](FINDINGS.md#no-benefit-in-the-rare-tail) |
| D6 | the token-independent input | **parked — outside the thesis** (§33.59); G-71 closed (§33.58) | [what the slot did instead](FINDINGS.md#what-the-slot-did-instead-a-token-independent-input) |
| D7 | tokenization | **answered** at four seeds (§33.61–§33.81); the exposure comparison against both references (§33.110–§33.118) | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe); [in English](FINDINGS.md#q1-in-english-morphology-inside-bpe); [the tokenizer and the embedding](FINDINGS.md#morphology-in-the-tokenizer-and-in-the-embedding-ukrainian) |
| D8 | scale and architecture: wider, deeper, a modern block | width 512 **answered** (§33.94, §33.102, §33.118); width 1024 at 10k: half width 512's, split by seed (§33.120), its 20k half queue item 1; depth and the RMSNorm and SwiGLU block (built, §33.54) not run | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe) ("It peaks there"); [the null at 30M](FINDINGS.md#the-same-shape-null-is-not-neutral) |
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

The thesis text's list of 19 October (§33.117), in its order, which the text confirmed on 20 October: each item with the sentence it states for every outcome (journey §33.117) and its cost from the recorded timings (`scripts/ops/run_times.py`).
Results committed by about 31 October reach the text; the work is admitted on 4 November.
Each item's preparation runs on the CPU while the item before it trains.
In the items, §3.x and "conclusion n" are the thesis text's sections, not the journey's.

**Running:** nothing. The chain of 20 October ended at 05:32 EDT on the second machine with item 1's 10k half and the 20k pair of seed 0 (§33.120). The rest of item 1 and items 2–3 wait for hours.

1. **The tokenizer's gain beyond 30M parameters** (§3.12, §3.3.2, conclusion 2): width 1024, ~38M tokens, both nulls (BPE 17k, endings apart), seeds 0–3, at 10k and at 20k.
   The 10k half and seed 0's 20k pair ran on 20 October (§33.120): at 10k the gain is +0.0067 ± 0.0030, half width 512's, because two seeds of four fall behind early (G-81).
   - **Left:** the 20k pairs of seeds 1–3, 6 runs of 88 min (measured), **~8.8 h**: the same chain started again (`scripts/chains/second_2026-10-20.sh`), which skips the finished runs. Then MultiBLiMP on the 8 models at 20k (minutes).
   - **It decides** (P271–P273): level at 20k and within 0.003 of width 512's → *by 20k it holds at 30M parameters' level; at 10k two seeds of four had not yet regained it*; level and below → *smaller again but lasting*; still falling → *it fades*.
     Whether the 10k fall recurs at seeds 1 and 2 (P272) tells the seed's from chance (G-81).
   - **Fallback** if the hours do not come before the date: the text states the 10k half and seed 0's 20k pair as they are.
2. **Number and gender as context-aware features** (§3.10, conclusion 7): under BPE 17k and endings apart, ~38M, 10k, seeds 0–3.
   For each feature, the tagged one, its word form and its leak control, each over the null and the empty slot (both already on the second machine): 2 × 3 × 2 × 4 = **48 runs, ~3.4 h**.
   The exposure comparison for both features comes with the same runs.
   - **Preparation**, about half a day of the thesis machine's CPU:
     1. the leak audit for `ugender` and `unumber` on a sample (`morph_causality_check.py`): which words take their number and gender from the right context (agreement in the noun phrase: adjectives, determiners and numerals before their noun);
     2. the word-form variants and the leak controls, derived with `derive_lexical_codes.py` and registered beside `ucaselex` and `ucaseprep`;
     3. their codes for both streams, slimmed and sent to the second machine (~1 GB).
3. **Whether English part of speech's deficit closes** (§3.8, conclusion 5): switched off at 5,000 against never fed (`OFF0`), seeds 0–3, at a 100k budget: 8 runs of ~42 min, **~5.6 h**, on data already on the second machine.
   - WikiText-103 then repeats 6.8 times (3.4 at 50k).
     If the null's validation curve stops falling, the fallback is seeds 4–7 at 50k: 8 runs of ~21 min, **~2.8 h**.
4. **The exposure redundancy at width 512**: answered (§33.118).
5. **BLiMP's agreement at eight seeds**: answered (§33.117).
6. **Unigram at width 512**, which the text asked for on 20 October, after item 3: its null, case and empty slot, 12 runs, **~3.2 h**, data on the second machine.
   Unigram is the one exposing tokenizer whose same-point shortfall is measured at one size only.
   *It stays:* under Unigram, exposed endings make case partly redundant at both sizes (conclusion 6 gains its Unigram clause).
   *It goes:* at 30M parameters the redundancy is gone under both exposing tokenizers, and Unigram's shortfall is the 5M model's (§33.118).

In all, items 1–3 and 6 take ~21 h of the second machine (~18 h with item 3's fallback), plus half a day of the thesis machine's CPU for item 2.

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
