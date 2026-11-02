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

**Answered** (§33.148, §33.149): items 2, 7, 13, 9, 10 and 11. At 85M parameters the gain at full training is +0.0071 (1.02%), the 30M model's, and MultiBLiMP +3.1; under the LLaMA-style block it holds (+0.0152 against +0.0175); at 8 layers it halves (+0.0079 against +0.0147); the cut one letter off the boundary loses 0.0207 to plain BPE; shuffled case costs 0.0011; the switch to a constant leaves the switch to zeros' deficit (+0.0038 against +0.0039). Item 6's pairs wait for the author's check (§33.147).

**Answered** (§33.145): items 1, 12 and 14. At the best of three rates (3e-3, the reference 5.2% better) the 5M model's gain at full training is +0.0136 (1.52%), 0.0037 below the default's on the same machine; between plain baselines +0.0219 (2.31%), 0.0046 above the nulls'; the 1,000-step warm-up changes nothing at width 128.

**Running** from 31 October (`second_2026-10-31.sh`, then `second_wide_2026-10-31.sh`; §33.143). The 5M model at 151M tokens and 50k unless noted:
1. **Item 1, the learning rate:** §2.1 does not justify 3·10⁻⁴, and English at 10k put 3e-3 first (results §5.21).
   - The nulls at 1e-3 and 3e-3 at seed 0, then the better rate's seeds 1–3, decided by BPE 17k's null.
   - 11 runs, ~3.9 h [~10 h]. P321–P323.
2. **Item 12, the tokenizer without a slot:** plain baselines under both tokenizers, and BPE 17k's null at the default rate as twins.
   - 11 runs, ~3.9 h [4 runs, ~3.9 h; BPE 17k's plain baselines are on the thesis machine]. P324–P325.
3. **Item 14, the 1,000-step warm-up at width 128:** the nulls at 10k on ~38M tokens.
   - 8 runs, ~35 min [~1.6 h]. P326.
4. **Item 2, width 1024 at full training:** the nulls, with the 1,000-step warm-up; then MultiBLiMP.
   - 8 runs of ~3.7 h, ~30 h [too slow there]. P327–P329.
   - The reference does not change its cost; the nulls keep the series like for like (§33.143).

**Running on the thesis machine** from 31 October (`tulpar_2026-10-31.sh`; §33.144), where every run each is compared with already is:
- **item 10, shuffled case:** the word form's case values shuffled among word types of similar frequency, as `mstructshuf`, under BPE 17k at 10k.
  - 4 runs, ~47 min. P330–P331.
- **item 11, switching off to a constant:** English part of speech set to one class at step 5,000 of 20k, on the seeds of the switch to zeros.
  - 5 runs, ~1.9 h. P332–P333.
- **item 9, where the cut falls:** BPE 17k with the rule's words cut as often, but one letter off the boundary; the null at 10k on ~38M tokens.
  - 4 runs, ~47 min, once its data is built (~1.5 h of CPU beside the chain). P334–P336.

**Next**, on the second machine once item 2 ends (`second_block_2026-10-31.sh`, queued after it; §33.146):
- **item 7, the modern block** (rotary positions, RMSNorm, SwiGLU; built, §33.54): both nulls at 10k.
  - 8 runs, ~40 min [~1.6 h]; at full training ~2.8 h more.
- **item 13, depth:** 8 layers at width 512, both nulls at 10k.
  - 8 runs, ~3–4 h by estimate (32 min a run with its margin).

**Without the GPU**, in order:
1. the software's size;
2. 95% intervals, one-sided upper bounds and the minimum detectable effect;
3. the gain against every vocabulary size;
4. the probes at layer 0;
5. item 4, the gain as compute or model size;
6. item 3, where the gain sits, per word;
7. the test split (it needs test caches);
8. item 6, minimal pairs for agreement and government (built, §33.147), checked by the author on a random hundred, then scored;
9. item 8, the tagger on gold labels, which needs the treebank downloaded;
10. the figures.

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
