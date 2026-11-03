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
Queued as item 11 of the text's list of 30 October (new code, a short block): switching off to a constant (every code to one class) instead of to zeros — removing the information but keeping a shared vector — would tell the lost information from the input's sudden change.

## Parked

**D6, the shared input vector — outside the thesis** (§33.59).
For the thesis it is the confound it explains, and the one test that could have changed that reading (G-71) is closed.
What remains is about the vector itself: the fixed against the learned vector (2 × 24 min), G-71's seeds 1–3 (~2.4 h), the vector added under rotary positions at seed 3, the 256-wide pair under rotary positions, the 256-wide spread at seeds 5–7 (P91, P92, not tested), the vector's norm against the token embeddings', and the literature on shared input vectors (BERT's segment embedding, attention sinks, massive activations, vision registers).

Also parked: morphology-factorized embeddings for parameter efficiency (Q1 partly answers it; prior work to audit first) · convergence pairs at 100k and 200k steps (G-44, G-30) · English little data (WikiText needs sub-sampling) · the literature audit's unverified cells (G-61) · lexicon file naming (G-66).

## The queue

The thesis text's list of 30–31 October (§33.143): fifteen limits a reviewer could fairly call gaps, ranked by the text in three tiers.
- **Costs:** from the recorded timings (`scripts/ops/run_times.py`), on the second machine, with the thesis machine's in brackets.
- **Deadlines:** results committed by the morning of 2 November reach the text; a single number by noon on 3 November can still go in as a sentence; the work is admitted on 4 November.

**Answered** (§33.150): item 6. On 3,999 pairs for agreement inside the noun phrase and government, 98 of 100 valid by the author's check, endings apart makes fewer errors at every seed: 47% fewer at 5M parameters (agreement +0.6, government +1.5 points) and 63% fewer at 30M (+0.3, +0.6), both tokenizers near the ceiling; the same without the check's two kinds of error.

**Answered** (§33.148, §33.149): items 2, 7, 13, 9, 10 and 11. At 85M parameters the gain at full training is +0.0071 (1.02%), the 30M model's, and MultiBLiMP +3.1; under the LLaMA-style block it holds (+0.0152 against +0.0175); at 8 layers it halves (+0.0079 against +0.0147); the cut one letter off the boundary loses 0.0207 to plain BPE; shuffled case costs 0.0011; the switch to a constant leaves the switch to zeros' deficit (+0.0038 against +0.0039).

**Answered** (§33.145): items 1, 12 and 14. At the best of three rates (3e-3, the reference 5.2% better) the 5M model's gain at full training is +0.0136 (1.52%), 0.0037 below the default's on the same machine; between plain baselines +0.0219 (2.31%), 0.0046 above the nulls'; the 1,000-step warm-up changes nothing at width 128.

**Answered** (§33.151): the software's size, and item 5's intervals. Every row of the tokenizer's table holds at 95%
(the headline +0.0141 to +0.0189); the features' one-sided 95% bound is 0.003, not 0.002, and four seeds see a feature's
effect only from about 0.002–0.006.

**Answered** (§33.152): the test split and every vocabulary size. On the held-out test articles the gain is +0.0126
(1.3%) at 5M parameters and +0.0062 (0.83%) at 30M, positive at every seed; at nearly the same token length plain BPE 8k
loses by 0.083, and BPE 32k, with 1.9M more parameters, wins by 0.028.

**Answered** (§33.153): item 8. On UD Ukrainian-IU's gold labels the tagger's case is right for 94.2% of the words with
one, 81.1% where context decides, and 87.0% where it departs from the form's usual case (the form 8.3%).

**Answered** (§33.154): the multilingual tokenizers. XLM-R's, mBERT's and GPT-4o's keep a Ukrainian ending apart for
17–25% of the words the rule cuts; Qwen2.5's for 51%, by cutting every word into 3.5 tokens.

**Answered** (§33.155): item 4. Near 5M parameters plain BPE 17k needs 1.15 times the parameters, or 1.17 times the
training steps, to match endings apart.

**Without the GPU**, in order:
1. item 3, where the gain sits, per word, with a bootstrap over articles: why the test articles give less;
2. the probes at layer 0;
3. the figures.

**Optional**, if hours remain:
- what case's +0.0015 at 30M is made of: 8 runs, ~10.8 h;
- 100k steps at 5M: 8 runs, ~6.1 h;
- shuffled case at 30M: 4 runs, ~5.4 h;
- Ukrainian case as a target: 4 runs, ~35 min.

The queue of 25 October is answered (§33.140–§33.142): full training at 30M parameters and English at 30M. Its conditional pieces did not run: case kept +0.0015 over its null, and the tokenizer's gain was +0.0072.

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
