# Research journey — Part 13: How Far the Gain Travels

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.83–§33.96. [← Part 12](12_full_training_and_the_leak.md) · [index](README.md) · [Part 14 →](14_two_machines.md)

**Established:** the sentence test, by which a candidate experiment is dropped when every outcome
would give the thesis the same sentence (§33.83). Inside the models, keeping endings apart gives a
more accurate account of a word's own grammar, its number, gender, and case where the form shows it,
while case that only context decides is read weakly under both tokenizers (§33.84). The tokenizer's
gain travels: to English, a fifth as large and lasting to full training on the median seed (§33.87,
§33.92); to fiction, at half its share at 10k (§33.88); at width 256 it settles near +0.022
(§33.86); at width 512 it is smaller and levels off near +0.011, while the grammatical gain keeps
growing (§33.94–§33.96).

---

### 33.83 Is a direction already dead? The thesis reviewed, 11 October

**The question** (11 October): is any direction already dead — are experiments continuing
on a line whose answer is settled?

**Where it sits.** Every question the thesis asks has an answer that survived its strongest
control:
- **English, morphemes instead of BPE:** no, BPE wins by 13.5%.
- **Morphology in the embedding:** helps early and with scarce data, and fades with training,
  as the law says. At full training, what reading the sentence adds over the word form is the
  tagger's leak (§33.81).
- **Morphology in the Ukrainian tokenizer:** endings apart beats BPE 17k by 1.6–1.75%,
  lasting to full training, larger in a 4× wider model, and +4.3 to +5.3 points on MultiBLiMP.
  It is firm on one corpus and small models.
- **The method:** the same-shape null, and the leak measured in trained models.

Further runs on any of these change numbers, not answers.

The manuscript is now the bottleneck:
- the defence is around December, in English;
- chapters 1–3 were last updated on 9 October, before §33.79 and §33.81;
- the 10–15 page English abstract, the introduction and the conclusions are not written.

**What was dead.** Five queued candidates fail the sentence test, now part of every queue
review: every outcome would give the manuscript the same sentence.
- **The remaining re-runs.** §33.58 already showed that a re-run is its original's re-score:
  training is deterministic and the null's training codes never changed. Since then 19 of 20
  matched to 3 × 10⁻⁶. The one that did not stopped early, at another step, on ~1.5M tokens.
  The rest were running because `demo_figures.py` reads no quarantined run.
- **Case in the wider model.** Another condition for a feature whose answer is "fades".
- **D3 at a 50k budget.** It could only change the budget a secondary claim is stated for;
  the claim is stated for the budgets tested.
- **The two extensions of G-77** that §33.82 queued: more seeds of a gain that fades, and the
  control where case is already below its null.

The queue review of the night before had let in the last two: the procedure ranked
candidates but never dropped one. All five are dropped, and the review gains the rule: for
each outcome, the sentence; dropped if all outcomes give the same one; a follow-up to an
answered question only on the manuscript's request.

**A correction.** A BPE that keeps English endings apart had been turned down as outside
the title, read as Ukrainian. The title is not limited to one language — *Context-Aware
Morphological Embeddings for Language Models* — and English is half the thesis. New
directions are wanted wherever they fit the title.

**What fits the title** (DIRECTIONS, "The queue"):
1. **Context-aware morphology inside the models** (new). The thesis has measured injected
   morphology fading. It has not measured what the models build for themselves, which is the
   title's object as the model holds it. A linear probe reads each word's case, number and
   gender from the hidden states at the word's last token, after the whole word has been
   seen. The models compared are the null models of BPE 17k and of endings apart, which carry
   no morphology in their input. The test of context-awareness is on words whose case in
   context differs from their form's majority case: there the form alone is wrong, so only
   context can be right. Desk work, on the CPU.
2. **English with endings apart** (new; the correction above). It asks whether morphology in
   the tokenizer helps a weakly inflected language. That is also why morphemes lost in
   English: they replaced BPE, while the Ukrainian tokenizer keeps BPE and forbids one kind
   of merge.
3. **A second Ukrainian corpus (D9),** **4. width 256 at 20k** and **5. larger models (D8)**
   test how far the tokenizer's gain travels.

Each passes the sentence test: its outcomes give the manuscript different sentences.

**Predictions settled by the drop:** P160 is right on the 12 re-runs that ran, each as
predicted, and its other 8 are dropped. P161–P163 are not tested, dropped by the sentence
test.

### 33.84 Context-aware morphology inside the models

**The question** (queue 1, §33.83). The thesis measured morphology injected into the input,
and saw it fade. Does a language model build context-aware morphological representations for
itself, and does keeping endings apart make them better?

**The design** (`scripts/analysis/probe_words.py`, new):
- **Models:** the nulls of BPE 17k and of endings apart at full training (151M tokens, 50k
  steps), seeds 0–3. A null's input carries no morphology.
- **The probe:** a linear probe reads a word's case, number and gender in context, as the
  tagger labels them, from each layer's hidden state at the word's last token, after the whole
  word has been seen.
- **Data:** over the whole validation split (the same words under both tokenizers). The probe
  is trained on even windows and tested on odd ones.
- **The test of context-awareness:** case is also scored on the words whose case in context
  is not the most frequent case of their form in training (`ucaselex`). There the form alone
  is wrong by construction, and only context can be right.
- **Comparison:** each tokenizer's models are paired by seed. Seeds share initial weights, and
  the words are the same.

Predictions, before any probe was fitted:

- **P183** (~65%). Under both tokenizers, at the best layer, the probe reads case in context
  more accurately than the form's majority case does (about 90% of words), on average.
- **P184** (~60%). On the words whose case in context is not their form's, the endings-apart
  models' best-layer accuracy exceeds BPE 17k's by more than 2 points, paired by seed, at three
  seeds or more.
- **P185** (~55%). For number and for gender, the endings-apart models' best-layer accuracy
  exceeds BPE 17k's at three seeds or more.
- **P186** (~60%). On the words whose case in context is not their form's, the best-layer
  accuracy exceeds that subset's majority class by more than 15 points under both
  tokenizers.

**Results** (`probe_words_20261011T070939`): 8 models, the whole validation split, about 71,000
case-bearing words tested per model. Accuracy is at each run's best layer, mean over seeds,
paired by seed:

| 151M, 50k, the nulls | BPE 17k | endings apart | endings apart − BPE 17k |
|---|---|---|---|
| case, all words | 83.2% | 84.7% | +1.5 ± 0.3 (4/4) |
| case, where the form's majority case is right | 87.2% | 89.6% | **+2.3 ± 0.4** (4/4) |
| case, where only context can be right | 57.6% | 55.1% | **−2.5 ± 0.5** (0/4) |
| case, forms unseen in train | 76.7% | 75.7% | −1.0 ± 0.8 (2/4) |
| number | 91.8% | 94.3% | **+2.5 ± 0.1** (4/4) |
| gender | 75.7% | 81.4% | **+5.7 ± 0.2** (4/4) |

The form's own majority case is right for 84.1% and 83.8% of case-bearing words (not the
"about 90%" P183 assumed: that share counted every word, caseless ones too). Where only
context can be right, that subset's majority class scores 44.9% and 44.7%. Under endings
apart the best layer is the first or second, and accuracy falls after it. Under BPE 17k it is
the third.

- **P183 wrong.** The probes read case about as well as the form's majority case does, not
  better: 83.2% against 84.1%, and 84.7% against 83.8%.
- **P184 wrong, reversed:** −2.5 ± 0.5 (0/4).
- **P185 right:** number +2.5 and gender +5.7, 4/4 each.
- **P186 wrong:** 12.7 and 10.4 points over that subset's majority class, not 15.

**What it says.** Keeping endings apart gives the model's states a more accurate account of a
word's own grammar: its number, its gender, and its case where the form shows it. Those are
what subject–verb agreement needs, and the endings-apart models judge agreement better
(§33.79). It does not make them better at reading a word's case from its context; they are
slightly worse there. Under both tokenizers a word's case that only context decides is read
weakly, 10–13 points over the floor.

**A measure that is not yet fair.** The tagger read each word's case with the words after it
in view; the model's state at the word's last token has seen only what came before. The
"only context" words include those that only the right context decides. A fair reading takes
the word's case from the model's state one unit later, at the last token of the next word or
punctuation run. `probe_words.py --after 1` does this. Predictions before it ran:

- **P187** (~55%). Read one unit later, case on the words where only context can be right
  rises by more than 5 points over its reading at the word itself, under both tokenizers.
- **P188** (~50%). Read one unit later, the endings-apart models' accuracy on those words is
  at least BPE 17k's, paired by seed, at three seeds or more.

**One unit later** (`probe_words_20261011T071514`, `--after 1`). The state at the last token of
the next unit holds less of a word's grammar, not more:

| read one unit later | BPE 17k | endings apart | endings apart − BPE 17k |
|---|---|---|---|
| case, all words | 65.9% | 64.6% | −1.2 ± 0.4 (0/4) |
| case, where only context can be right | 49.1% | 48.1% | −1.0 ± 0.8 (1/4) |
| number | 84.4% | 82.7% | −1.7 ± 0.4 (0/4) |
| gender | 62.7% | 62.6% | −0.0 ± 0.4 (2/4) |

Case on the context-decided words falls from 57.6% and 55.1% to 49.1% and 48.1%, against a
floor of 44.8% and 44.7%. **P187 wrong** (it fell by 8.5 and 7.0 points). **P188 wrong**
(1/4). A model this size does not go back to a word it has passed. Its state moves on to the
next word, and what a word's grammar is, the model states at the word. The right-context
caveat cannot be removed this way: in these models a word's case read with what follows is
not stored anywhere a linear probe finds it.

**What it changes.**
- **The thesis can say where its positive result lives.** Keeping endings apart gives a model
  a more accurate account of a word's own grammar, the morphology the form carries: number
  +2.5, gender +5.7, and case where the form decides it +2.3 points (4/4 each, at full
  training). These are the features subject–verb agreement turns on, where those models judge
  better.
- **On the title's word, context:** neither tokenizer's models read much of a word's case
  from context beyond what the form shows, 10–13 points over the floor. Keeping endings apart
  does not improve that; it is 2.5 points lower.
- **The same pattern as injected morphology:** what the form gives lasts, and what context
  adds is small (§33.81).

Predictions: P185 right; P183, P184, P186, P187 and P188 wrong.

### 33.85 How far the tokenizer's gain travels: English, and Ukrainian fiction (prepared)

Ten hours were available on 11 October, worked in batches: results checked between them,
each next launch decided from them. Running first was the only prepared candidate, width 256 at
20k (queue 4; P158–P159), from 10:28. Prepared beside it on the CPU are queue 2 and 3, the two
tests of how far the thesis's positive result travels.

**English with endings apart** (queue 2; the correction of §33.83).
- **The rule** (`morphlm/data/endings.py`, `en_endings_v1`). It is the Ukrainian rule's twin,
  reading the word alone. spaCy's English rule lemmatizer ships tables per part of speech:
  exceptions, base forms and suffix rules. Together they give every reading of a lowercased
  form with no context, for nouns, verbs and adjectives. The cut falls after the common prefix
  with the lemma, only if every reading agrees: walks → walk|s, studies → stud|ies,
  children → child|ren. `leaves` (leaf, leave), `news` and `running` (also a noun) stay whole.
- **Checked on WikiText-103's validation text** against spaCy's in-context analysis (142,129
  words):
  - it cuts 10.2% of words;
  - 89.6% of the words it cuts are inflected in context, and every one of those is cut at the
    in-context lemma's prefix;
  - it reaches 49.6% of inflected words. What it leaves is mostly by design: suppletive forms
    (was, is → be) and forms that are also adjectives or nouns (played, used).
- **Its tokenizer** (`data/bpeend_17k.json`, BPE 17k trained on the marked WikiText-103 train
  split by the same code as `bpe_17k`). On held-out text a token boundary falls at the
  stem|ending of 58.9% of inflected words (BPE 17k: 3.8%), and of 80.3% of case-bearing ones
  (3.4%). It costs 4.4% more tokens (Ukrainian: 12%). Its validation stream decodes to BPE
  17k's 1,136,862 bytes, the same SHA-256 (`stream_roundtrip_20261011T074937`).
- **The design.** Plain baselines: no feature is involved, so the slot the nulls carry is not
  needed, and plain baselines need no tagging pass. The endings-apart ones run at 10k, seeds
  0–3, paired by seed with BPE 17k's plain baselines at 10k (3 October; 1.5266, 1.5298,
  1.5289, 1.5281). One caveat: BPE 17k's English vocabulary is 17,151 entries and the new
  one's 17,000, so the pairs share seeds, not initial weights.

**Ukrainian fiction** (queue 3, D9).
- **The corpus.** UberText 2.0's fiction subcorpus (Chaplynskyi 2023; the cleansed export,
  433 MB; downloaded 11 October; no licence stated, used for research only,
  never packed). `scripts/data/prepare_ubertext_fiction.py` splits it at its separators
  (exactly three blank lines) into 25,565 texts, 2.76M paragraphs and 1.25 billion characters.
  The 9,669 hard-wrapped texts are unwrapped (`ubertext_fiction_20261011T074342`). It is
  registered as corpus `ukfiction` (`morphlm/data/corpora.py`; a local file).
- **The splits,** at holdout 500 (novels are long): validation 2.55M characters, train at
  fraction 0.1 (~120M characters, like ukwiki's 0.05), and at 0.4 (498M, like ukwiki's 0.2),
  where both tokenizers are trained: `bpe_ukf_17k` and `bpeend_ukf_17k`.
- **The design** is ukwiki's: the nulls of both at fraction 0.1, 10k, seeds 0–3, paired by seed
  from identical initial weights.

The chain is `scripts/chains/long_2026-10-11.sh`. Piece 1 is English, piece 2 fiction, and
piece 3 English at 20k, a follow-up run only when named. Its guards:
- each condition's caches must exist;
- a smoke test for fiction;
- after seed 0, the two tokenizers within 0.05 of each other, which a broken stream would not
  be;
- disk and the time box.

It was rehearsed with the stub, ten scenarios.

Predictions, before any run:

- **P189** (~70%). At 10k the English endings-apart baselines end below BPE 17k's at four seeds
  of four.
- **P190** (~45%). By more than 0.005 BPB on average.
- **P191** (~70%). The relative gain is under half the Ukrainian one at 10k: under 0.8% of BPE
  17k's BPB. English inflects little, the rule reaches half of it, and at a fixed budget the
  model reads 4.5% less text.
- **P192** (~75%). On fiction, the endings-apart nulls end below BPE 17k's by more than 0.010
  BPB at four seeds of four.
- **P193** (~55%). The relative gain lies within a factor of 1.5 of ukwiki's 1.6% at 10k
  (1.07–2.4%).
- **P194** (~70%). On MultiBLiMP the fiction endings-apart nulls beat BPE 17k's by more than 2
  points at four seeds of four.
- **P195** (~45%). On English BLiMP the endings-apart baselines beat BPE 17k's by more than 0.5
  points on average.

### 33.86 The wider model's lead settles: +0.022 at width 256 and 20k

**The run** (`long_2026-10-10.sh`, `PIECES="4"`, 10:28–15:48 on 11 October; 8 runs of 37–41
minutes). The width guard passed: each 20k null ended below its 10k null at seed 0.

**The result** (`tokenizer_table_20261011T125318`; ~38M tokens, width 256, 20k steps, paired by
seed from identical initial weights). The endings-apart null ends at 0.8823 against BPE 17k's
0.9041, **+0.0218 ± 0.0029 (2.4%, 4/4)**; per seed +0.0176, +0.0304, +0.0188 and +0.0205.
Along the schedule:

| step | 2,000 | 5,000 | 7,500 | 10,000 | 15,000 | 19,500 |
|---|---|---|---|---|---|---|
| lead | +0.0382 | +0.0276 | +0.0229 | +0.0234 | +0.0224 | +0.0220 |

From step 7,500 to the end the lead stays between +0.0216 and +0.0234 while BPE 17k's null
falls from 0.986 to 0.905. The 10k budget's +0.0254 was its schedule's end: at step 10,000 of
this longer schedule the lead is +0.0234, and it settles near +0.022. **P158 right** (above
0.010 at every seed, +0.0218 on average). **P159 right** (smaller than +0.0254).

**What it changes.** The sentence test named two outcomes (§33.83). The first came: the lead
stays above the width-128 lead at the same budget (+0.0169, 1.6%). "Larger in a wider model"
no longer rests on a lead that was still falling. The tokenizer's gain is 1.6% *(1.7%: +0.0169
of 1.0145; §33.91)* of the null at width 128 and 2.4% at width 256 at a 20k budget, and flat in both over the second half of
training.

### 33.87 English with endings apart: it helps, a fifth as much

**The runs** (`long_2026-10-11.sh`, `PIECES="1"`, 15:49–16:35 on 11 October; 4 runs of
11.0–11.1 minutes). The seed-0 sanity guard passed: 1.5246 against BPE 17k's 1.5266.

**The result** (WikiText-103, plain baselines, 10k steps; paired by seed — the seeds do not share
initial weights, the vocabularies being 17,000 and 17,151 entries):

| seed | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| BPE 17k | 1.5266 | 1.5298 | 1.5289 | 1.5281 | 1.5283 |
| endings apart | 1.5246 | 1.5253 | 1.5233 | 1.5218 | 1.5238 |

Endings apart is **+0.0046 ± 0.0010 (4/4)** below BPE 17k (per seed +0.0019, +0.0045, +0.0056,
+0.0063). Every endings-apart run is below every BPE 17k run, so the comparison does not lean
on the pairing. That is **0.30%** of the null, against 1.6% for Ukrainian at the same budget.
**P189 right; P190 wrong** (+0.0046, not above 0.005); **P191 right** (under 0.8%).

**What it changes.**
- **Morphology inside BPE helps English too,** so the thesis's positive result is not
  Ukrainian alone. In English it is a fifth as large.
- **The size tracks inflection.** In English 18% of words are inflected (in context) and the
  rule exposes half of them; in Ukrainian nearly every content word inflects, and the rule
  exposes its ending.
- **What lost in English Q1** (morphemes instead of BPE, 13.5% worse) was replacing BPE, not
  morphology. Kept inside BPE, the same idea gains in both languages, in proportion to how
  much each inflects.
- **Its lasting** (20k, piece 3, 1.6 h) did not fit this window. It is the follow-up the queue
  names.

**A fault found on the way, in the pipeline, not in any result** (`morphlm/pipelines/bpe.py`).
Building the fiction condition under BPE 17k ran five times slower than under endings apart on
the same text: 120k against 800k tokens a minute through its last 26k paragraphs.
- **The cause:** `encode` read `encoding.offsets[idx]` for every token, and a
  `tokenizers.Encoding` builds a new list on every `.offsets`. A paragraph cost the square of
  its length: 3.2 ms per token on a 56,000-token paragraph, 180 s for the paragraph.
  `EndingTokenizer`'s own `Encoding` stores the list, which is why endings apart was not slow.
- **Why it never showed:** Wikipedia's paragraphs are hundreds of tokens; the unwrapped fiction
  holds paragraphs of 100,000 and more.
- **The fix:** read the offsets once. The outputs are unchanged, and the tests pass.

A first build of the same condition had also hung for five hours with no output, having
started a minute after the endings-apart build on a corpus neither had loaded yet. Run alone, it
went through. Build one condition of a new corpus at a time.

**BLiMP** (`blimp_eval_20261011T133856`; 67 paradigms, the CPU, paired by seed). The 10k English
models are close to chance, so BLiMP is a weak instrument here: 56.7% and 57.1% over all pairs,
**+0.3 ± 0.3 (3/4) — P195 wrong**. Within agreement it divides:
- **determiner–noun agreement,** which hangs on the noun's plural ending, the ending the rule
  cuts most (`-s`): **+4.4 ± 0.9 points (4/4)**;
- **subject–verb agreement:** −1.6 ± 0.4 (4/4);
- **anaphor agreement:** −2.0 ± 1.0 (4/4).

At this budget the English gain shows on the agreement that the cut ending carries, and not
across grammar as a whole. Ukrainian's MultiBLiMP gain (§33.79) was measured on models scoring
65–76%.

**The follow-up, prepared** (piece 3 of `long_2026-10-11.sh`, after the fiction cells). Does
the English gain last, as the Ukrainian one did (+0.0174 at 10k, +0.0169 at 20k, +0.0165 at
full training; §33.76)? The endings-apart plain baselines run at 20k, seeds 0–3, paired with
BPE 17k's at 20k (3 October: 1.4325, 1.4342, 1.4351, 1.4335). Its outcomes give different
sentences: "it lasts in English too" against "in English it is a short-schedule gain".
Predictions:

- **P196** (~65%). At 20k the English endings-apart baselines end below BPE 17k's at four seeds
  of four.
- **P197** (~50%). Their lead lies within 0.002 of the 10k lead (+0.0026 to +0.0066).

**At twice the budget** (piece 3, 18:14–19:42; 4 runs of ~22 minutes). The endings-apart plain
baselines at 20k end at 1.4280, 1.4286, 1.4275 and 1.4293 (1.4284). BPE 17k's (3 October) end at
1.4325, 1.4342, 1.4351 and 1.4335 (1.4338). The lead is **+0.0055 ± 0.0008 (0.38%, 4/4)**, per
seed +0.0045, +0.0056, +0.0076, +0.0042, and again every endings-apart run is below every BPE 17k
run. It does not fade: +0.0046 at 10k, +0.0055 at 20k. **P196 right; P197 right** (within 0.002
of the 10k lead). In English, as in Ukrainian, morphology in the tokenizer is not a crutch the
model outgrows. Full training (50k) is the next step (DIRECTIONS, "The queue").

### 33.88 A second Ukrainian corpus: fiction keeps the gain, at half the size in bits per byte and in full in grammar

**The runs** (`long_2026-10-11.sh`, piece 2, as two cells with `FIC_ONLY`):
- endings apart's four seeds at 16:38–17:26, beside the BPE condition's build (§33.87);
- BPE 17k's at 17:27–18:13;
- 11.5–12 minutes each.

The seed-0 guard passed when both existed: 1.1908 against 1.2023. Both validation streams decode
to the same 4,594,258 bytes (`stream_roundtrip_20261011T142727`). On fiction's validation text
(`tokenizer_stats_ukfiction_20261011T143605`):
- BPE 17k trained on fiction puts a boundary at the stem|ending of 9.6% of inflected words;
- endings apart puts one at 73.6%, and at 100% of its rule's cuts;
- endings apart costs 11.8% more tokens.

That matches the Wikipedia pair (9%, 77–81%, 12%).

**Bits per byte** (`tokenizer_table_20261011T151443`; fraction 0.1, 33.4M BPE tokens, 10k
steps, paired by seed from identical initial weights). The endings-apart null ends at 1.1924
against BPE 17k's 1.2021: **+0.0097 ± 0.0010 (0.81%, 4/4)**; per seed +0.0116, +0.0071,
+0.0113 and +0.0090. Along the schedule:

| step | 500 | 1,000 | 2,000 | 3,000 | 5,000 | 7,500 | 9,500 |
|---|---|---|---|---|---|---|---|
| lead | +0.042 | +0.025 | +0.023 | +0.025 | +0.014 | +0.0099 | +0.0098 |

The lead is flat over the last 2,000 steps. On Wikipedia at ~38M tokens and 10k it was +0.0174
(1.6%). **P192 wrong** (above 0.010 at two seeds of four). **P193 wrong** (0.81%, below the
range 1.07–2.4%).

**MultiBLiMP** (`multiblimp_eval_20261011T151505`). The fiction models judge agreement far
better than Wikipedia's at the same budget: BPE 17k 79.6%, against 64.8% on Wikipedia at ~38M
and 10k. Endings apart scores **84.9%, +5.3 ± 0.6 points (4/4)**: number +3.3, gender +5.5,
person +6.0, 4/4 each. On Wikipedia at 10k it was +4.3. **P194 right.**

**What it changes.**
- **The "one corpus" caveat goes.** On a second Ukrainian corpus of another register — novels
  and prose against an encyclopedia — keeping endings apart beats BPE 17k at every seed, in bits
  per byte and in agreement judgements.
- **Its size in bits per byte depends on the corpus:** 0.8% on fiction against 1.6% on
  Wikipedia, both at a 10k budget, with the same exposure and the same token cost.
- **In grammar it does not shrink:** +5.3 points against +4.3. Fiction has far more verbs in
  the first and second person and in the past tense; its models learn agreement better, and
  keeping endings apart adds as much or more.
- **The thesis's claim becomes:** morphology in the tokenizer helps on two corpora and in two
  languages. Its share of bits per byte varies with the text (0.3% to 1.6%), and its effect on
  agreement holds wherever it was measured (+4.3 to +5.3 points in Ukrainian).
- **Why fiction's share is half** is not asked here. A guess to test, not a finding: fiction's
  text is easier for both, and its validation split is a few long novels.

### 33.89 Ten hours in batches, 11 October: what ran, and the queue after

The ten hours of §33.85, 10:28–19:43, ran in batches: results were checked between them, and
each next launch was chosen by the queue and the sentence test.

| GPU | what | result |
|---|---|---|
| 10:28–15:48 | width 256 at 20k (8 runs) | the lead settles at +0.0218 (2.4%), §33.86 |
| 15:49–16:35 | English with endings apart, 10k (4) | +0.0046 (0.30%), §33.87 |
| 16:38–17:26 | fiction, endings apart's nulls (4) | — |
| 17:27–18:13 | fiction, BPE 17k's nulls (4) | +0.0097 (0.8%); MultiBLiMP +5.3, §33.88 |
| 18:14–19:42 | English at 20k (4) | +0.0055 (0.38%), §33.87 |

On the CPU, beside them:
- the probing of §33.84;
- the English rule, its tokenizer and data;
- UberText fiction's conversion, corpus spec, two tokenizers and two conditions;
- MultiBLiMP and BLiMP on the new models.

The GPU stood idle twice:
- 15:48–15:49;
- 16:35–16:38, while the endings-apart fiction cell was set to run before its BPE pair, whose
  condition was still building.

**Decisions taken during the run, and why:**
- **Plain baselines for English instead of nulls.** No feature is involved, and they need no
  tagging pass over WikiText-103, a pass `prepare_condition.py` refuses (§33.85).
- **Fiction's two cells one after the other (`FIC_ONLY`)** rather than as pairs, so the GPU ran
  while one condition built; rehearsed, twelve scenarios.
- **The BPE 17k fiction build restarted twice.** It hung for five hours at its start, and then
  ran five times slower than its pair. The second cause was found and fixed in the pipeline;
  results are unchanged (§33.87).
- **English at 20k as the last launch.** It was the follow-up the queue named, and all four
  seeds fit before 20:00, so no cell was split.

**Predictions.** P158, P159, P189, P191, P194, P196 and P197 are right. P190, P192, P193 and
P195 are wrong; each missed by its size, not its sign. The tally is 108 right, 8 half right,
72 wrong, 8 never tested, 1 withdrawn, of 197.

**The queue after** (DIRECTIONS, "The queue", rewritten):
1. fiction at full training (~7.6 h, conditions to build first);
2. English at full training (~3.8 h);
3. larger models.

Dropped by the sentence test: why fiction's share is half, and case on fiction.

### 33.90 English at full training (prepared), 12 October

Four hours of GPU were available from 03:15 on 12 October. The queue (§33.89) was reviewed first.
The two requests that had come from the thesis since then concern figures and wording and ask for
no run, so the queue stands. Of its three candidates, only English at full training fits four
hours as a whole cell.

**The larger models' sizes, measured while the piece was prepared** (`scripts/ops/fit_check.py`,
the Ukrainian null at ~38M, 20k steps, batch 32 × 256; D8's first step). The projections for the
two measured widths match their runs: 21.4 against 22.3 minutes at width 128, 36.9 against 37.5
at 256.

| width × layers | parameters | of them the core | peak GPU memory | a 20k run |
|---|---|---|---|---|
| 128 × 4 | 5.04M | 0.79M | 2.70 GB | 21.4 min |
| 128 × 8 | 5.84M | 1.59M | 3.25 GB | 32.4 min |
| 256 × 4 | 11.80M | 3.16M | 3.04 GB | 36.9 min |
| 256 × 8 | 14.95M | 6.32M | 4.33 GB | 56.0 min |
| 384 × 4 | 20.12M | 7.10M | 3.68 GB | 56.1 min |
| 384 × 8 | 27.22M | 14.20M | 5.54 GB | 88.4 min |
| 512 × 4 | 30.02M | 12.61M | 4.34 GB | 80.3 min |
| 512 × 8 | 42.63M | 25.22M | 6.79 GB | 127.5 min |

Every size fits the card at the batch the runs use. Width 512, four times width 128 and 16 times
its core, would cost ~10.7 h for both nulls at four seeds and 20k; width 384, ~7.5 h.

**The piece** (piece 4 of `scripts/chains/long_2026-10-11.sh`). The endings-apart plain
baselines run at 50k on WikiText-103, seeds 0–3. They are paired by seed with BPE 17k's plain
baselines at 50k: 1.3477, 1.3482, 1.3573 and 1.3478. BPE 17k's seed 2 sits 0.009 above its other
three, which would inflate a mean lead by about 0.002, so the size is judged by the median of the
four paired leads. Guards:
- seed 0 must end within 0.05 of BPE 17k's (1.3477), or seeds 1–3 are skipped;
- disk;
- the time box.

Rehearsed with the stub, 17 scenarios: the twelve earlier ones and five for this piece (all
pass; a seed-0 failure; the deadline; a resume after seed 0; a finished chain).

**The sentence test.** If the lead holds, the thesis says the gain lasts at full training in
English too. If it fades, it says the English gain belongs to short training, and the gain
lasts only where words inflect richly.

Predictions, before any run:

- **P198** (~70%). At 50k the English endings-apart baselines end below BPE 17k's at four seeds
  of four.
- **P199** (~50%). The median of the four paired leads lies within 0.002 of the 20k lead
  (+0.0035 to +0.0075).
- **P200** (~65%). On BLiMP's determiner–noun agreement, the 50k endings-apart baselines beat
  BPE 17k's at four seeds of four (at 10k: +4.4 ± 0.9, 4/4).

### 33.91 FINDINGS checked against its record, 12 October

A check of the thesis text against FINDINGS, the results record and the journey found seven
places where FINDINGS and its own record disagreed. Each was settled from the run records, and
corrected in FINDINGS, the summaries and the results record; the journey is annotated where it
first stated them. None changes a claim's direction.

**1. The Ukrainian law's outlier.** FINDINGS still quoted §33.38: the 50k run 0.0015 below the
other conditions. That was measured against the first null, which was not constant (G-73).
§33.58 restated the comparison, but FINDINGS kept the old figure. Recomputed from the records
(`learning_law_20261012T003054`; `learning_law.py` now takes `--refs` and `--record`), case
over the constant null at five points of learning:

| condition | 1.30 | 1.25 | 1.20 | 1.15 | 1.10 |
|---|---|---|---|---|---|
| 151M, 50k | +0.0055 | +0.0055 | +0.0051 | +0.0046 | +0.0029 |
| 151M, 5k | +0.0056 | +0.0058 | +0.0056 | — | — |
| ~38M, 10k | +0.0050 | +0.0049 | +0.0050 | +0.0045 | +0.0037 |
| ~7.6M, 10k | +0.0060 | +0.0062 | +0.0056 | +0.0045 | — |
| ~1.5M, 10k | +0.0065 | — | — | — | — |
| spread | 0.0015 | 0.0013 | 0.0006 | 0.0001 | 0.0008 |

The curves draw together as the null learns, and are level at 1.15. At 1.10 the 50k run is
0.0008 below the ~38M run, which is in its last steps there (§33.58 gave +0.0030 and +0.0036).
That is within the English law's worst deviation across schedules (0.0010, §33.38). The law in
Ukrainian stays "approximate"; the limitation "why the 50k run sits below its curve" is closed.
The 151M runs at 10k and 20k have no curve on the constant: their best steps were re-scored
(§33.56), and their nulls were not re-run.

**2. "4× wider."** Width 256 is twice width 128. It has four times the core's parameters
(3.16M against 0.79M), and 2.3 times all parameters with the embeddings (11.80M against 5.04M;
the fit check, §33.90). The phrase also stands in §33.67, §33.80 and §33.83, and in P80 and
P134; read each as twice as wide. Width 64 is half as wide, as stated.

**3. Width 128 at 20k is 1.7%, not 1.6%.** +0.0169 of BPE 17k's mean null of 1.0145 is 1.67%.
The 1.6% is the 10k figure: +0.0174 of 1.0896.

**4. English part of speech at 10k is +0.0130 ± 0.0004, at four seeds** (§33.38), not §33.33's
two-seed +0.0136. Against the restated Ukrainian case on 151M at the same budget (+0.0026,
§33.56) it is five times as large, not three.

**5. The steps the early help saves.** §33.56 listed them among the curve numbers the constant
null moves, and they had not been restated. `scripts/analysis/steps_saved.py`
(`steps_saved_20261012T003033`) reproduces §33.33's English numbers to the step:
- part of speech 86, 141, 434 and 790 steps;
- inflection 53, 78, 239 and 326 steps.

Against the constant null, Ukrainian case saves 22, 64, 162 and 118 steps at steps 1k, 2k, 5k
and 10k (2.2%, 3.2%, 3.2%, 1.2%). It was first recorded as 23, 69, 181 and 163. The range over
the three features is 1–9%.

**6. What the null shares with a feature.** At one seed every cell sees the same windows and
the same dropout masks. Initial weights match only where the parameter shapes match (§33.15).
The null's slot has 8 classes, like structure, frequency band, and the Ukrainian features (case,
its controls, number, gender). Part of speech has 19 classes and inflection 72, so those pairs
share data and dropout, not initial weights. FINDINGS had claimed all three for every feature;
it now says which features share what.

**7. The slot's +0.0097 ± 0.0015.** It is the null's: the constant code, against eight starts of
the baseline (§33.16). The slot held at zero gains +0.0105 ± 0.0035 over the baseline (§33.24).
FINDINGS had attached "even held at zero" to the first number; each now stands on its own.

### 33.92 English at full training: the gain holds on the median, at three seeds of four

**The runs** (piece 4 of `long_2026-10-11.sh`, 03:26–07:22 on 12 October; 4 runs of 57.4–59.6
minutes). The seed-0 guard passed: 1.3426 against BPE 17k's 1.3477.

**The result** (WikiText-103, plain baselines, 50k steps; paired by seed, without shared initial
weights):

| seed | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| BPE 17k | 1.3477 | 1.3482 | 1.3573 | 1.3478 | 1.3503 |
| endings apart | 1.3426 | 1.3428 | 1.3400 | 1.3481 | 1.3434 |
| lead | +0.0051 | +0.0055 | +0.0173 | −0.0003 | +0.0069 ± 0.0037 |

The median lead is **+0.0053 (0.39%)**. The mean, +0.0069 ± 0.0037 (0.51%, 3/4), is inflated by
BPE 17k's seed 2. **P198 wrong** (three seeds of four). **P199 right**: the median lies within
0.002 of the 20k lead. Along training the median lead is flat from step 10,000 to the end:
+0.0051 at 10k, +0.0048 at 20k, +0.0055 at 30k, +0.0050 at 40k, +0.0053 at the end.

**One run in four drifts.** No run has a spike: no evaluation after step 10,000 rises more than
0.0007 above the one before. But in each tokenizer one run falls behind its siblings from about
step 20,000:
- BPE 17k's seed 2 ends 0.009 above its other three;
- endings apart's seed 3 ends 0.005 above its other three, which leaves that pair level: +0.0046
  at 20k, +0.0003 at 30k, −0.0003 at the end.

At 50k the spread between a tokenizer's runs (0.008–0.010) is larger than the gain. At 10k and
20k it was not: every endings-apart run was below every BPE 17k run. Here fourteen of the
sixteen cross pairs favour endings apart, and the three typical runs of each differ by 0.0061.

**BLiMP** (`blimp_eval_20261012T043938`; 67 paradigms, paired by seed). At full training the
endings-apart models judge better overall at every seed: 59.0% → 60.0%, **+1.0 ± 0.2 points
(4/4)**, against +0.3 ± 0.3 (3/4) at 10k. Morphology +2.1 ± 0.8 (4/4). The agreement
phenomena:
- determiner–noun +3.0 ± 1.3 (3/4; seed 0 −0.3) — **P200 wrong** (three of four);
- subject–verb +1.3 ± 0.7 (3/4), negative at 10k (−1.6, 4/4);
- anaphor +4.4 ± 2.9 (3/4), negative at 10k (−2.0, 4/4).

Seed 3, level in bits per byte, judges better too: +0.4 overall, +6.0 on determiner–noun
agreement. The mean margin by which a model prefers the good sentence is smaller under endings
apart at every seed, by 0.02 to 0.11 nats. The models prefer the good sentence more often, not
by more.

**What it changes.** The sentence test named two outcomes (§33.87, §33.90): the gain holds, or
it belongs to short training. The first came, less firmly than in Ukrainian. In English the gain
lasts to full training on the median seed, at three seeds of four. The grammar test favours
keeping endings apart at every seed, by more than at 10k. In Ukrainian at full training it was
four of four, with every endings-apart run below every BPE 17k run (§33.76).

**Predictions.** P199 is right; P198 and P200 are wrong, each by one seed. The tally is 109 right,
8 half right, 74 wrong, 8 never tested, 1 withdrawn, of 200.

**What remains.** Four more seeds per tokenizer (~7.6 h) would say whether one drifting run in four
is the English full-training spread. It is a follow-up, run only if the thesis asks (DIRECTIONS,
"The queue").

### 33.93 Does the tokenizer's gain keep growing with model size? Width 512 (prepared), 14 October

Six hours of GPU were available from 05:53 on 14 October. The queue was reviewed against the
thesis text's own list of limits. Its first limit is the one the main positive result rests on:
one model of about 5M parameters and its variants half and twice as wide, so how the tokenizer's
gain changes at larger scale was not tested. No request from the thesis was open.

The three candidates:
- **Fiction at full training** (queue 1) needs ~7 h of CPU to build its conditions, then 7.6 h of
  GPU. It does not fit.
- **English seeds 4–7** (queue 2) is a follow-up the thesis has not asked for.
- **Larger models** (queue 3) answer that first limit. This is the one chosen.

**The design.** Both nulls at width 512 — twice width 256, four times width 128 — on ~38M tokens,
10k steps, seeds 0–3, paired by seed from identical initial weights (the two tokenizers' models
have the same shapes). Width 512 has 30.0M parameters, 12.6M of them in the core (fit check,
§33.90): six times the base model, sixteen times its core. At 10k steps the comparison has rows
at both smaller widths, with all four seeds: +0.0174 (1.6%) at width 128, +0.0254 (2.6%) at width
256. A 20k budget, where width 256's lead settled (+0.0218, §33.86), would cost 10.7 h. Along the
law's axis the widths are compared at the same null BPB, which no schedule decides.

**The chain** (`scripts/chains/scale_2026-10-14.sh`). A seed's two runs go together, ~44 minutes
each; fit check 39.8, and runs have taken 1.07 times the projection at widths 128 and 256.
Guards:
- each condition's caches must exist, and its smoke test must pass at width 512;
- after seed 0, each 512-wide null must end below its own 256-wide 10k null, since a wider model
  that ends worse has diverged; and the pair must end within 0.05 of each other;
- disk, and the time box: END_BY 11:50.

Seven runs fit: the pairs of seeds 0–2, then, with `FILL=1`, seed 3's BPE 17k null alone. Seed
3's endings-apart null runs at the next start (~44 min). The chain was rehearsed with the stub in
twelve scenarios, all passing; they cover each guard, the fill used once, and the resume after it.

**The sentence test.** If the lead is larger than at width 256, the thesis says the tokenizer's
gain grows with model size across a sixfold range of parameters. If it is about the same, the gain
grows from 5M to 12M parameters and then holds. If it is smaller, the gain peaks at a moderate
size, where a larger model has learned more of the endings for itself.

Predictions, before any run:

- **P201** (~80%). At width 512 the endings-apart null ends below BPE 17k's at every seed that
  runs.
- **P202** (~55%). Its mean lead is larger than width 256's at the same budget (+0.0254).
- **P203** (~50%). Where the BPE 17k null passes BPB 1.18, 1.11 and 1.04, width 512's lead is
  larger than width 256's at each (+0.0383, +0.0336, +0.0296; §33.68).
- **P204** (~50%). On MultiBLiMP, width 512's endings-apart advantage is at least width 256's at
  10k (+5.3 points) on average.

### 33.94 Width 512: the gain in bits per byte peaks at width 256; the grammatical gain keeps growing

**The runs** (`scale_2026-10-14.sh`, 06:01–11:41 on 14 October; 8 runs of 40.8–44.6 minutes). All
eight fit: the runs were quicker than the 44 minutes allowed, so the fill was not needed. The
width guard passed at seed 0: 0.8803 and 0.8664 against 0.9826 and 0.9594 at width 256, and the
pair within 0.05.

**The result** (~38M tokens, 10k steps, nulls paired by seed from identical initial weights;
`tokenizer_table_20261014T094052`):

| seed | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| BPE 17k | 0.8803 | 0.8790 | 0.8770 | 0.8777 | 0.8785 |
| endings apart | 0.8664 | 0.8655 | 0.8672 | 0.8649 | 0.8660 |
| lead | +0.0139 | +0.0136 | +0.0099 | +0.0128 | **+0.0125 ± 0.0009** |

The lead is **1.4%** of BPE 17k's null, against 1.6% at width 128 (+0.0174) and 2.6% at width 256
(+0.0254) at the same budget. **P201 right** (4/4). **P202 wrong**: smaller than width 256's, and
smaller than width 128's.

**First, a defect?** The result went against the prediction, so it was checked as width 256's was
(§33.68):
- the paired cells have the same parameter shapes (30,017,760), so the same initial weights at a
  seed (§33.15);
- every run's validation BPB falls at every evaluation, and every best step is the last, so the
  30M model does not over-fit in its 2.2 passes over the text.

**Along training** the lead starts as the smaller widths' does, then keeps falling: +0.0452 at step
500, +0.0491 at 1,000, +0.0273 at 2,000, +0.0203 at 3,000, +0.0164 at 5,000, +0.0138 at 6,000,
+0.0120 at 8,000 and 9,000, +0.0125 at the end. Its fall slows: 0.004 from step 5,000 to 7,000,
0.001 from 7,000 to 9,000.

**At matched null BPB** (`tokenizer_widths_20261014T094349`, the new
`scripts/analysis/tokenizer_widths.py`), the lead at the same point of learning:

| BPE 17k's null BPB | 1.30 | 1.18 | 1.11 | 1.04 | 0.98 | 0.95 | 0.92 | 0.89 |
|---|---|---|---|---|---|---|---|---|
| width 128 (151M, 50k) | +0.0263 | +0.0178 | +0.0162 | +0.0159 | +0.0174 | +0.0168 | — | — |
| width 256 (20k) | +0.0320 | +0.0381 | +0.0333 | +0.0279 | +0.0223 | +0.0233 | +0.0228 | — |
| width 512 (10k) | +0.0475 | +0.0408 | +0.0303 | +0.0218 | +0.0180 | +0.0163 | +0.0137 | +0.0121 |

Early in learning the widest model gains most. From null BPB 1.11 on, its lead falls below width
256's, and from about 0.98 to width 128's level and under. Width 128's lead is flat along the axis
(0.016–0.018 below 1.18), and so is width 256's at 20k (0.022–0.023 from 0.98 to 0.905).
Width 512's keeps falling to the end of its budget. **P203 wrong**: larger at 1.18 (+0.0408
against +0.0381), smaller at 1.11 and 1.04.

**MultiBLiMP** (`multiblimp_eval_20261014T094155`; 2,744 pairs, paired by seed). BPE 17k 73.5%,
endings apart 80.3%: **+6.8 ± 0.2 points (4/4)**. By phenomenon: number +8.0, gender +7.9,
person +5.3, each 4/4. That is the largest advantage of any condition, against +4.3 at width 128
and +5.3 at width 256 at the same budget. **P204 right.** The grammatical gain grows with the
model while the gain in bits per byte shrinks. As on fiction (§33.88), the two are not one
quantity.

**What it changes.** The sentence test named three outcomes (§33.93); the third came. In bits
per byte the tokenizer's gain peaks at a moderate size, width 256. At width 512 it is smaller at
the end of the budget, and it falls along the law's axis as a feature's gain does, where at widths
128 and 256 it held flat. The thesis's statement that the tokenizer's gain fades on neither axis
on which a feature's gain fades — the model's learning and its size — holds to width 256, not
beyond. In grammatical judgement the gain grows with size through 30M parameters.

Two readings are possible; neither is tested:
- **The larger model learns the ending structure from BPE 17k's tokens itself.** This is the
  law's reading: help lasts until the model has learned what it carries.
- **The 30M model is the first that is limited by the text it reads.** At a fixed number of
  steps endings apart reads 11% less text (1.9 passes over the text against 2.2).

Whether the lead settles, and where, a 20k budget at width 512 would show: both nulls, four
seeds, ~10.7 h (DIRECTIONS, "The queue").

**Predictions.** P201 and P204 are right; P202 and P203 are wrong. The tally is 111 right, 8 half
right, 76 wrong, 8 never tested, 1 withdrawn, of 204.

### 33.95 Does the width-512 lead settle? Seed 0 at 20k (prepared), 14 October

About three hours of GPU were available from 19:30 on 14 October. They run one seed of the
queue's first candidate (§33.94): both width-512 nulls at 20k, seed 0. The other seeds follow when
there are hours for them. One seed cannot give a lead with its spread. It can show whether the
lead levels off between step 10,000 and 20,000, the question §33.94 left. At the end of the 10k
schedule seed 0's lead was +0.0139, after +0.0133 and +0.0138 at steps 8,000 and 9,000, while the
four seeds' mean was still falling.

**The chain** is `scale_2026-10-14.sh` with two new knobs:
- `STEPS=20000` (88 minutes a run, twice the 10k runs' 40.8–44.6);
- `SEEDS=0`.

The width guard compares each 512-wide null with its 256-wide null at the same budget. Width 256 at
20k: 0.9000 and 0.8812 at seed 0. It was rehearsed with the stub: sixteen scenarios, the twelve of
§33.93 and four for the new knobs, all passing.

**The sentence test** is §33.94's. If the lead settles at +0.010 or more, the gain at 30M parameters
is smaller but lasting. If it keeps falling, at 30M parameters the tokenizer's gain fades with
learning, as a feature's does. One seed gives the direction, and the other seeds give the size.

Predictions, before any run, for seed 0:

- **P205** (~55%). At 20k, width 512's seed-0 lead ends at +0.010 or more.
- **P206** (~50%). Over its last 5,000 steps it changes by less than 0.002: it has levelled off.
- **P207** (~60%). At null BPB 0.92 and 0.89 it lies within 0.003 of the 10k schedule's seed-0 lead
  there (+0.0151, +0.0134): the point of learning sets it, not the schedule.

### 33.96 Width 512 at 20k, seed 0: the lead levels off near +0.011

**The runs** (`scale_2026-10-14.sh`, `STEPS=20000 SEEDS=0`, 19:33–22:31 on 14 October; 87.9 and
89.2 minutes). The width guard passed: 0.8149 and 0.8035 against width 256's 0.9000 and 0.8812 at
20k, and the pair within 0.05. No defect:
- the same shapes (30,017,760 parameters each);
- validation BPB falls at every evaluation;
- both best steps are the last.

**The lead** at seed 0 is **+0.0115** at the end (1.4% of BPE 17k's 0.8149). Along training:

| step | 1,000 | 2,000 | 5,000 | 7,500 | 10,000 | 12,500 | 15,000 | 17,500 | 20,000 |
|---|---|---|---|---|---|---|---|---|---|
| lead | +0.0467 | +0.0249 | +0.0172 | +0.0117 | +0.0132 | +0.0094 | +0.0119 | +0.0107 | +0.0115 |
| BPE 17k's null | 1.233 | 1.088 | 0.948 | 0.892 | 0.862 | 0.840 | 0.828 | 0.819 | 0.815 |

The lead falls to about +0.011 by step 7,500, while the null passes 0.89. From there to the end
it moves between +0.009 and +0.013 with no trend, though the null learns another 0.08 BPB. Over
the last 5,000 steps it lies between +0.0101 and +0.0119 and ends 0.0004 below where it stood.
**P205 right** (+0.0115). **P206 right** (−0.0004).

**The point of learning sets it, not the schedule** (`tokenizer_widths_20261014T193329`). At the
same null BPB the 20k run's lead lies within 0.0006 of the 10k run's down to 0.95, and 0.0021 to
0.0025 below it at 0.92, 0.90 and 0.89, where the 10k run was in its last steps. **P207 right**
(within 0.003).

**MultiBLiMP** (`multiblimp_eval_20261014T193238`; seed 0). BPE 17k 77.3%, endings apart 82.9%:
+5.6 points. By phenomenon: number +7.9, gender +6.5, person +3.9. At 10k, seed 0 gave +6.5.

**What it changes.** The sentence test's first outcome came, at one seed. At width 512 the
tokenizer's gain in bits per byte falls further than at the smaller widths before it levels off,
and it levels off at about +0.011 (1.4%), not at zero. Smaller than at widths 128 and 256 at 20k
(1.7%, 2.4%), but lasting, where a feature's gain falls to nothing. With §33.94, the thesis can say
the gain peaks at width 256 and holds, smaller, at 30M parameters. Seeds 1–3 (~8.8 h) give its
size and spread.

**Predictions.** P205, P206 and P207 are right. The tally is 114 right, 8 half right, 76 wrong,
8 never tested, 1 withdrawn, of 207.

