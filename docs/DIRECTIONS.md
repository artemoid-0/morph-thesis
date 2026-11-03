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
| D8 | scale and architecture: wider, deeper, a modern block | width 512 **answered** (§33.94, §33.102, §33.118); width 1024 **answered** (§33.120–§33.124, §33.132, §33.149): smaller again and lasting, at full training the 30M model's; the split by seed was the 100-step warm-up's; the modern block and depth **answered** at 10k (§33.149): the block keeps the gain, 8 layers halve it | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe) ("It peaks there"); [the null at 30M](FINDINGS.md#the-same-shape-null-is-not-neutral) |
| D9 | a second corpus per language | Ukrainian **answered** with UberText fiction (§33.85, §33.88, §33.108); no English corpus chosen | [Q1 in Ukrainian](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe) ("And on a second corpus") |

**D3 — is the reliance deficit permanent?**
At full training it halves and lasts (§33.102); whether it ever closes is queue item 3.
Queued as item 11 of the text's list of 30 October (new code, a short block): switching off to a constant (every code to one class) instead of to zeros — removing the information but keeping a shared vector — would tell the lost information from the input's sudden change.

## Parked

**D6, the shared input vector — outside the thesis** (§33.59).
For the thesis it is the confound it explains, and the one test that could have changed that reading (G-71) is closed.
What remains is about the vector itself: the fixed against the learned vector (2 × 24 min), G-71's seeds 1–3 (~2.4 h), the vector added under rotary positions at seed 3, the 256-wide pair under rotary positions, the 256-wide spread at seeds 5–7 (P91, P92, not tested), the vector's norm against the token embeddings', and the literature on shared input vectors (BERT's segment embedding, attention sinks, massive activations, vision registers).

Also parked: morphology-factorized embeddings for parameter efficiency (Q1 partly answers it; prior work to audit first) · convergence pairs at 100k and 200k steps (G-44, G-30) · English little data (WikiText needs sub-sampling) · the literature audit's unverified cells (G-61) · lexicon file naming (G-66).

## The queue

**Reviewed 3 November, after the text's list of 30–31 October** (journey §33.159).
That list is answered: items 1–14 in §33.143–§33.158, item 15 dropped (below). Each answer's claim is in FINDINGS. The queue of
25 October is answered too (§33.140–§33.142).
- **Costs** come from the recorded timings (`scripts/ops/run_times.py`), on the second machine unless noted.
- **Deadlines:** the work is admitted on 4 November; anything later serves the defence (7–11 December) and the article.
- **Nothing runs before its hours are set.** Each piece is prepared and rehearsed, and its predictions recorded, first.

Ranked by the claim each could move: a headline first, then the question the title asks, then the axes an examiner will ask about, then a secondary claim's firmness; within a rank, the cheaper first.

| # | candidate | the sentences it decides | cost | ready | its result moves |
|---|---|---|---|---|---|
| 1 | **A large held-out sample, and table 3.1 on it.** About 6,000 of the articles outside the 0.2 fraction: no model and no tokenizer saw them, ten times either held-out sample. Every row of the tokenizer's table is scored on it. | The headline is 1.75% on validation and 1.3% on the test articles, 2.5 standard errors apart (§33.152, §33.156). On this sample: *the gain is X%*, the figure the text gives. For each row: *it keeps its sign on unseen text*, or it is restated. | Scoring only, about 100 models: ~2 h of the second machine, ~1 h of the thesis machine. A tagger pass of minutes; the null derived on the second machine. | New code: the sample as a split (~1 h). | Fixes the headline's value. Moves no other candidate. |
| 2a | **Shuffled case at 30M** (width 512, full training). | Case's +0.0015 at 30M is positive at every seed, but its interval includes zero (p = 0.073, §33.151). Level with case: *at 30M case's content does not act*. Near zero: *it does*. | 4 runs, ~5.4 h | The codes to derive on the second machine (CPU, beside); a chain piece. | Level: 2b drops. Near zero: 2b runs. |
| 2b | **What case's +0.0015 at 30M is made of:** the word form and the leak's control at 30M. | *At 30M too, what the sentence adds is the leak*; or *context adds about 0.001*; or *nothing*. | 8 runs, ~10.8 h | Its codes exist on the second machine; a chain piece. | — |
| 3a | **The minimal pairs at 85M** (width 1024, full training). | MultiBLiMP's gain fell to +3.1 there. *Fewer errors at every size trained*, or *not at 85M*. | Scoring, ~10 min | Ready. | — |
| 3b | **Endings apart at 32k.** BPE 32k beats endings apart at 17k by 0.028 with 1.9M more parameters (§33.152). | A gain: *the rule adds to a larger vocabulary*. None: *at 32k the vocabulary holds what the rule gives*. | The tokenizer and its stream, ~2 h of CPU; 4 runs at 10k, ~25 min; BPE 32k's nulls exist on the second machine. | A tokenizer to train. | None: «Практична цінність» is restated "at 17k". |
| 3c | **Ukrainian case as a target** (§3.8 has English only). | *It teaches early and stops paying, as in English*; or *it keeps paying*. | 4 runs, ~35 min | The target code exists; a chain piece. | — |
| 4 | **100k steps at 5M.** The gain is flat from step 5,000 to 50,000. | *It lasts to 100k*, or *it fades after 50k*. | 8 runs, ~6.1 h | A chain piece. | — |

**What hours of the second machine buy:**
- ~2.5 h: items 1 and 3a;
- ~3.5 h: with 3b's runs and 3c, the stream built beside on the CPU;
- ~9 h: with 2a;
- ~20 h: with 2b, if 2a does not drop it;
- ~26 h: with 4.

**Dropped** by the sentence test (every outcome gives the same sentence, or none):
- item 15 of the list of 30 October, the word-form control with an unseen-word class: it makes the split of case's lead at 10k exact, a number rather than the answer (G-76);
- Morfessor in place of the dictionary, this week: an install and hours of training before any run, where item 9 and Unigram bracket the question;
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
