# Directions — the research programme

> **Kind:** claims — the current plan and queue; rewritten as they change.
> **Authoritative for:** what is investigated next, why, and in what order.
> Updated 14 October 2026: the queue reviewed before a six-hour window, and after it (journey §33.93–§33.94).

The thesis stands on what is established ([`FINDINGS.md`](FINDINGS.md)). The
programme began on 2 October from its one positive regularity — *a feature helps
until the model has learned what it encodes* — and asked when, where and in what
role morphology helps. Each open direction names the result that would redirect it.

**Scope** (7 October, §33.59). Every direction serves the thesis — context-aware
morphological embeddings for language models: when, why and how far morphology
in the embedding helps, and how that compares with morphology in the tokenizer.
Questions about small transformers in general are parked, however interesting.

## Where we are

| | direction | status | what it established |
|---|---|---|---|
| D1 | the learning-progress law | **answered within one model size**; across sizes, half explained | one curve per feature across schedules and corpus sizes (five English features, three schedules, four seeds each); a wider model's curve collapses earlier — about half of that is how early it reads the previous word's tag (§33.35–§33.52, §33.60) |
| D2 | the mechanism, by probing | **closed** — the pivot not met | the wider model reads the previous word's tag — what the feature carries — far earlier (+3.9 to +10.4 points at matched null BPB); on that axis the widths' gains differ by at most 0.0035 (the current word's tag: 0.0063; BPB: 0.0098) — half the width effect, not all (§33.49–§33.52, §33.60) |
| D3 | removing, weaning, adding late | **answered** within 20k budgets; whether the deficit is permanent is left open (§33.83) | removal leaves a lasting deficit (5/5) — not lost knowledge of part of speech, nor of the previous word's tag, which the model rebuilds and ends reading better; weaning keeps 70% of it; a late feature is integrated, given time (§33.34–§33.52, §33.60) |
| D4 | morphology as a target | **answered; closed** | helps early and fades (+0.0009 ± 0.0008 at the end, 5 seeds); the model knows more part of speech, which stops helping (§33.50, §33.52) |
| D5 | the rare and novel tail | **screened; parked** | no hidden benefit in rare tokens at full training (§33.47) |
| D6 | the token-independent input | **parked — outside the thesis** | the confound behind "structure helps": one learned vector shared by every input helps by itself — the vector, not a mark of the window's start (G-71 closed) — hence the same-shape null (§33.49–§33.52, §33.58, §33.59) |
| D7 | tokenization | **answered at four seeds**; holds on a second corpus and in English, there to full training on the median seed, 3/4 (§33.87, §33.88, §33.92); the gain lasts at twice the budget, at full training on 4× the text, and is larger at twice the width; case on top still helps at full training under endings apart — half of it the word's own case, half the tagger's leak (G-77, closed) | BPE that keeps endings apart beats BPE 17k by 1.6% by itself — +0.0169 at twice the budget, +0.0254 at twice the width, where a feature's gain fades — and case adds on top, more under it the longer the training; case beats its null under all six tokenizers at every seed; the law holds across them to null BPB 1.25, after which case gains more where endings are exposed — the hypothesis reversed; Unigram's early lead is real, and fades; at 151M tokens and 50k steps the lead is +0.0165 (4/4) and case still adds +0.0030 under endings apart, half of it without the leak (the word form +0.0016, 3/4; +0.0036 over its gain under BPE 17k, 4/4); of the sentence's lead over the word form the tagger's leak is 29–43% at ~38M and all of it at full training, where context without the leak adds nothing (§33.58, §33.61–§33.81) |
| D8 | scale and architecture | **width 512 answered at 10k** (§33.94); whether its lead settles is open | the tokenizer's gain in bits per byte peaks at width 256 — 1.6%, 2.6% and 1.4% at widths 128, 256 and 512 (10k), the widest one's falling along the law's axis where the others hold flat — while its grammatical gain grows, +4.3, +5.3 and +6.8 points; a fit check over widths and depths (§33.54, §33.90) |
| D9 | a second corpus per language | **tools ready; corpora to choose** | `corpus_info.py`, then a spec (§33.53) |

## D3 — Is the reliance deficit permanent?

**Question.** The deficit levels off near 0.004 BPB within a 20k budget, while the
learning rate is still a third of its peak. Does a longer schedule erase it?

**Dropped by the sentence test** (§33.83): off from the start and off at 5,000 at a 50k
budget, seeds 0–1 (piece 7 of `long_2026-10-10.sh`), would change only the budget the
claim is stated for. The claim is stated for the budgets tested.

**Optional short block, new code:** switching off to a constant (every code to one
class) instead of to zeros — removing the information but keeping a shared vector
— to see whether the deficit is the lost information or the input's sudden change.

## Generalization — how far the claims travel

Most claims rest on two conditions (§33.53); D7 has added six tokenizers on one
Ukrainian corpus size (§33.65). Varying the corpus, the tokenizer and the model
tests the claims themselves — the examiner's first question — and is now the main
line. How a
condition is made: [`reference/conditions.md`](reference/conditions.md).

**D7 — Tokenization, Ukrainian first.** Does the law hold under other
tokenizers, and does how much a tokenizer exposes inflectional endings change how
much case helps, and for how long? The law predicts that case helps more, and
longer, where the tokenizer hides it. *Design:* BPE at 4k, 8k and 32k and Unigram
at 17k, all trained on the 0.2 train split like `bpe_uk_17k`; conditions at
fraction 0.05, where the 17k reference exists (§33.31); case, lexical case and the
null at a 10k budget; ending exposure (`train_tokenizer.py --stats`) as the
independent variable. *Measured* (§33.55): stem|ending boundaries in 21.5%, 14.5%,
9.1% and 5.7% of inflected words for BPE at 4k, 8k, 17k and 32k, and 41.1% for
Unigram at 17k, which compresses like BPE at 8k — two axes, vocabulary within BPE
and the algorithm at a fixed size. The 17k reference's null at 0.05 is re-run
on the constant first (G-72, G-73). *Cost:* 1–1.5 h of CPU tagging per condition, beside
GPU chains (§33.63). Predictions are recorded when the chain is planned.

*Built — the far end of the axis* (§33.60, §33.61). What D7 lacked is a
tokenizer that exposes every inflectional ending: morphology in the tokenizer,
the thesis's alternative to morphology in the embedding, and the one comparison
the closest Ukrainian work could not make (its models did not converge). BPE
that keeps endings apart (`--algorithm bpe-endings`, `bpeend_uk_17k`): trained
and applied on text marked at every cut of a rule that reads the word alone
(pymorphy3; `morphlm/data/endings.py`), whose cuts fall on a dictionary morpheme
boundary in 97.9–99.7% of the citation forms it cuts. The same rule measures
every tokenizer's exposure (`cut|`): from 5.5% (BPE 32k) to 43.3% (Unigram 17k)
and 100% for it, at 15% more tokens per word than BPE 17k (§33.62). Its
condition at 0.05 is ready: 42.6M training tokens for the same text, 12% more
than BPE 17k's 38.0M, so at a fixed budget its models read 11% less text (§33.63).
*Answered at four seeds* (§33.65; P126–P129 and P131 right, P130 wrong): by
itself it ends +0.0174 ± 0.0014 below BPE 17k (1.6%, 4/4), where Unigram gains
nothing; case still adds +0.0048 ± 0.0006 under it; and case gains more under the
tokenizers that expose endings — the reverse of the hypothesis above.

*What it leaves* (ranked, costed and placed in "The queue", below):
- **Is case's full-training gain under endings apart the tagger's leak? Half of it**
  (G-77, closed; §33.81). With prepositions read from their form case keeps +0.0015 ±
  0.0010 (3/4), and the word form alone +0.0016 ± 0.0009 (3/4), +0.0036 ± 0.0010 (4/4) over
  its gain under BPE 17k. The leak is +0.0015 ± 0.0001 (4/4), the whole of the sentence's
  lead over the word form: context without the leak adds nothing at full training. Its
  extensions are queued low (queue 6, 7).
- **Does the tokenizer's gain show beyond bits per byte? Yes** (§33.79): on MultiBLiMP's
  Ukrainian subject–verb agreement pairs, +4.9 ± 0.4 points at full training, +4.3 ± 0.3 at
  ~38M (4/4 each); a smaller vocabulary or Unigram, cutting elsewhere, does not help.
- **Where the wider model's lead settles: at +0.022** (§33.86) — width 256, 20k, flat from
  step 7,500; 2.4% against width 128's 1.7%.
- **BPE 32k at four seeds** (§33.81): case +0.0029 ± 0.0002 (4/4); nothing moved (P140 right).
- **Case in the wider model** (queue 4).
- **Answered:** the tokenizer's gain lasts at full training on 4× the text (§33.76); why
  exposure helps case (§33.70); what the sentence's case adds over the form's — the leak at
  prepositions 29–43% in trained models, about half the rest the words the form never
  saw, context resolving another word's case +0.0002 to +0.0003 (G-38; §33.74, §33.76).

**D8 — Scale and architecture.** Does the end-of-training null hold, and where
does the law's zero crossing move, in larger models — wider, deeper — and in a
modern block (rotary positions, RMSNorm, SwiGLU)? *First step:* size the grid on
the GPU with `scripts/ops/fit_check.py` — its projections for 128 and 256 wide
should match the measured 22.3 and 37.5 minutes per 20k run — then choose the
sizes.

**D9 — A second corpus per language.** Is it Wikipedia? An English and a
Ukrainian corpus of another domain, chosen after `corpus_info.py`; each a spec, then a
condition. UberText 2.0 (news, fiction,
court decisions, social media; `reference/related_work.md`) is the obvious
Ukrainian candidate.

*A grammatical test in Ukrainian* — **done** (§33.79; `scripts/eval/multiblimp_eval.py`). MultiBLiMP 1.0
holds 2,744 Ukrainian minimal pairs (`reference/related_work.md`): the
counterpart of the English BLiMP check (no feature moved it), for the saved
Ukrainian null, case and lexical-case models — does case in the input change
agreement judgements? It needs a loader beside `scripts/eval/blimp_eval.py`.

## Closed

**D2**, the mechanism: the previous word's tag, what the feature carries, accounts
for about half of what separates the widths' curves; the pivot's line-up was not
met, so the law is stated within one model size (§33.60). **D4**, the teacher: the
law covers targets; the knowledge it adds stops paying (§33.50, §33.52). **D5**, the tail: no hidden benefit by token frequency; the
unseen-word version parked (§33.47).

## Parked

**D6, the shared input vector — outside the thesis** (§33.59). For the thesis it
is the confound it explains, and the one test that could have changed that
reading (G-71) is closed. What remains is about the vector itself: the fixed
against the learned vector (2 × 24 min), G-71's seeds 1–3 (~2.4 h), the vector
added under rotary positions at seed 3, the 256-wide pair under rotary positions,
the 256-wide spread at seeds 5–7 (P91, P92, not tested), the vector's norm
against the token embeddings', and the literature on shared input vectors (BERT's
segment embedding, attention sinks, massive activations, vision registers).

Morphology-factorized embeddings for parameter efficiency (Q1 partly answers
it; prior work to audit first) · convergence pairs at 100k and 200k steps
(G-44, G-30) · English little data (WikiText needs sub-sampling) · the literature audit's unverified cells (G-61) · lexicon file
naming (G-66).

## The queue

Reviewed 11 October, evening (journey §33.89), after a ten-hour run of experiments; again
12 October (§33.90), unchanged; and 14 October, before and after width 512 (§33.93, §33.94). Each
candidate is ranked by what its result could change in the thesis and costed from the recorded
timings, and must pass the sentence test: for each likely outcome, the sentence the thesis
would state; a candidate whose outcomes all give the same sentence, or none, is dropped.
Rewritten at every review. **Running:** nothing.

**Answered on 14 October:** larger models, width 512 at 10k (§33.94). In bits per byte the
tokenizer's gain peaks at width 256: at width 512 it is +0.0125 (1.4%, 4/4), and it falls along
the law's axis where the smaller widths' leads hold flat. On MultiBLiMP it grows: +6.8 points.

**Answered on 12 October:** English at full training (§33.92). The gain holds on the median seed,
+0.0053 (0.39%), at three seeds of four: in each tokenizer one run in four drifts after 20k. On
BLiMP the models are +1.0 point better at every seed.

**Answered on 11 October:**
- **Context-aware morphology inside the models** (§33.84). Keeping endings apart makes a
  word's own grammar more accurate in the models' states. It does not help reading case from
  context.
- **Width 256 at 20k** (§33.86): the lead settles at +0.022 (2.4%).
- **English with endings apart** (§33.87): +0.0046 (0.30%, 4/4) at 10k, +0.0055 (0.38%, 4/4) at
  20k. It does not fade.
- **A second Ukrainian corpus** (§33.88): fiction keeps the gain, +0.0097 (0.8%, 4/4) at 10k,
  and +5.3 points on MultiBLiMP.

| # | candidate | what it settles for the thesis | the sentence test: each outcome's sentence | GPU | kind · ready |
|---|---|---|---|---|---|
| 1 | **Width 512 at 20k**: both Ukrainian nulls, ~38M, seeds 0–3 (or seeds 0–1 first) | whether the widest model's falling lead settles: whether at 30M parameters the tokenizer's gain outlasts learning | **It settles at +0.010 or more:** at 30M the gain is smaller but lasting. **It keeps falling:** at 30M parameters the tokenizer's gain fades with learning, as a feature's does | ~11.3 h (~85 min a run); seeds 0–1 ~5.7 h | not prepared: a 20k knob in `scale_2026-10-14.sh` (minutes), its predictions |
| 2 | **Fiction at full training**: both fiction nulls at fraction 0.4 (~133M tokens), 50k, seeds 0–3 | whether the gain lasts on a second corpus, as it does on Wikipedia | **It holds near +0.010:** it lasts on both corpora. **It fades:** on fiction it is a short-training gain | ~7.6 h | not prepared: the two conditions at 0.4 (CPU, ~3–4 h each; one at a time, §33.87) |
| 3 | **English at full training, seeds 4–7** of both tokenizers — a follow-up, run only if the thesis asks | whether one drifting run in four is the English full-training spread | **7/8 or more:** the gain lasts in English firmly. **Fewer:** at full training the English gain is within seed variation | ~7.6 h | quiet · piece 4 with seeds 4–7 (minutes) |

**Dropped by the sentence test:**
- **Why fiction's share of bits per byte is half Wikipedia's** (§33.88). The thesis states the
  size on each corpus, and no outcome changes a claim.
- **Case and its controls on fiction.** Morphology in the embedding is answered (§33.83).

**Dependencies.** None between the three. 2 needs its conditions built first, one at a time on
the CPU (§33.87). 3 waits for the thesis to ask.

**What GPU hours buy:** 1 needs a 20k knob and its predictions (minutes), then ~11.3 h for four
seeds, or ~5.7 h for seeds 0–1 first. 2 needs its two conditions first, ~7 h of CPU one at a
time, then 7.6 h. 3 only if the thesis asks.