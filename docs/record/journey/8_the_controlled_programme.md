# Research journey — Part 8: The Controlled Programme

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.1–§33.19. [← Part 7](7_the_causal_reckoning.md) · [index](README.md) · [Part 9 →](9_ukrainian_and_learning.md)

## 33. The Agreement Test

*§33 kept the title of its first subsections; it grew into the whole controlled programme, continued in [Part 9](9_ukrainian_and_learning.md).*

### 33.1 BLiMP at one seed

The first BLiMP pass (G-29) scored three seed-0 checkpoints on the 18
morphology paradigms, 1,000 pairs each:

| cell | all 18 | subject–verb | determiner–noun | irregular forms |
|---|---|---|---|---|
| baseline `a7175857fdf6` | 68.1% | 59.6% | 70.1% | 87.9% |
| causal `mstruct` `648675b80ba0` | 66.9% | 59.7% | 66.7% | 85.2% |
| causal `mlemma`+`mstruct` `a0abed930f58` | 68.4% | 59.8% | 70.0% | 86.5% |

**A correction first.** When this command was proposed, `a0abed930f58` was
described as a leaky cell. It is the *causal* `mlemma`+`mstruct` run — its
record says so. No leaky checkpoint was scored; the third column is not a
lookahead contrast. The leaky cell is `979a7d1f5878` / `1bcbac60f6b5` /
`4de10670e169`. The description came from memory instead of the record, which
is the first failure the project's principles name: a claim that felt closed because no check could have failed.

Read at one seed, the feature that improves perplexity by 0.89% does not
improve agreement: subject–verb is flat, determiner–noun is 3.4 points *lower*,
and one paradigm, anaphor gender, jumps from 51.5% to 63.9%. None of this can be
believed yet. BLiMP accuracy at this scale moves by several points between
seeds — a paradigm near chance, like anaphor gender, can land almost anywhere —
and every earlier single-seed reading in this project was later revised. The
paired run over every seed is scheduled (§33.4).

But one thing does not depend on seeds. **`mstruct` carries no number.**
*this* and *these* are both `simple`; *they* and *he* both `simple`.
Determiner–noun agreement turns on exactly the difference `mstruct` cannot see,
so it has no direct way to help — and a flat result would be the expected one,
not a disappointment. The feature that *could* help is inflection.

### 33.2 The inflection feature was blind to agreement

`minfl` (§32.4) turned out to miss the point of the test. It reuses the
morpheme tokenizer's classifier, which marks inflectional morphemes on nouns,
verbs and adjectives — and returns nothing for determiners and pronouns. Yet
spaCy knows *these* is `Number=Plur` and *they* is `Number=Plur|Person=3`:
precisely the cues agreement turns on, carried by words with no inflectional
morpheme at all.

Two features replace it for this question, computed from the same parse:

- **`mfeat`**, the full Universal Dependencies feature bundle. In English the
  bundle is set by one rule table from the tagger's fine tag, so the inventory
  is **closed**: 70 bundles for alphabetic words, read from the model itself —
  no hashing, no collisions, and anything outside maps to `other` rather than
  aliasing a real class. Width 8, the same budget as `mstruct`.
- **`magr`**, agreement alone: Number × Person, eight classes.

### 33.3 A tagger feature is not causal, even shifted

§30 established that shifting a feature to the previous word removes the
*within-word* leak. It does not remove a leak that the feature's own source
carries. `mstruct` is looked up from the word's letters in MorphoLEX — no
context at all, so shifted it is strictly causal. `pos`, `minfl`, `magr` and
`mfeat` come from spaCy's **tagger**, whose window reaches about four tokens to
the right: *reading* is VBG in "reading books" but NN in "reading is fun". The
previous word's tag can depend on the word being predicted.

This is the third fairness condition again, at a new layer. Two responses:

1. **Measure it.** `scripts/audits/morph_causality_check.py` tags each word twice —
   in the full sentence, and from the prefix ending at that word. For the
   parser this comparison failed (§30.6b: a parser must build a tree over any
   fragment). A tagger has no such constraint; left-context-only tagging is
   exactly what a causal tagger does, so here the comparison measures the leak
   directly. A harness check stops it if a sentence tagged as its own prefix
   ever differs.
2. **Remove it.** `mfeatlex` gives each word type its most frequent bundle in
   the training corpus — a lexicon, looked up from the word's form like
   `mstruct`. After the shift it depends only on words already written. It
   loses the tagger's disambiguation; what it keeps is leak-free.
   `scripts/data/derive_lexical_codes.py` derives it from the cached `mfeat` codes
   and verifies known words (*the*, *these*, *they*, *were*, *their*, *was*)
   map to the bundles spaCy assigns them in context. The lexicon it writes is
   what the BLiMP scorer uses on new text, and the two were checked to produce
   identical codes.

### 33.4 Predictions, recorded before the night of 26 September

Each with what would change the reading.

- **P1. `mstruct` does not move agreement.** Over three paired seeds, its
  effect on subject–verb and on determiner–noun agreement stays within ±2
  points and does not agree in sign across all three seeds; the one-seed
  −3.4 shrinks. *Wrong if* either agreement phenomenon moves the same way at
  all three seeds by more than 2 points.
- **P2. The anaphor-gender jump does not replicate.** Fewer than three of
  three seeds positive, or a mean under +5 points. *If it does replicate*, the
  names half (G-25) should carry it and the morphology half should not.
- **P3. The tagger leak is small.** `tag` and `pos` agree between
  left-only and full tagging for 95–99% of words, `mfeat` for 96–99.5%, with
  changes concentrated one to three tokens from the sentence end.
- **P4. Most words have one reading.** The majority bundle matches the
  in-context bundle for 93–98% of word occurrences.
- **P5. `pos` helps, within a factor of two of `mstruct`** (both at 8 dims). Positive at both
  seeds, +0.006 to +0.024 BPB. *If `pos` exceeds `mstruct` by more than
  twice*, "coarse class of the previous word" becomes the honest description
  of the effect, not morphology specifically.
- **P6. Lexical inflection helps less than structure.** `mfeatlex` positive at
  both seeds, +0.002 to +0.008 — smaller than `mstruct`, because English
  inflection mostly shows in the surface tokens themselves (*-s*, *-ed*,
  *-ing*), which the model already reads. What `mfeatlex` adds is number and
  person on words that do not show them.
- **P7. The sharp one: lexical inflection moves determiner–noun agreement.**
  At both seeds, `mfeatlex` beats the baseline on determiner–noun agreement by
  at least 2 points, and beats `mstruct`. The determiner's number is the cue
  the paradigm turns on, and this is the only feature that states it.
  Subject–verb gains less or nothing: the subject is usually not the previous
  word. *Wrong if* no gain at both seeds — the model already extracts number
  from the determiner's token, and stating it adds nothing.
- **P8. Inflection adds to structure, partly.** `mstruct`+`mfeatlex` beats
  `mstruct` at both seeds, by less than `mfeatlex` alone beats the baseline.
- **P9. `mfeatlex` grows, like structure.** A coarse class of word identity,
  not identity itself: it should still be growing at 50k steps (§32.2), not
  fading.
- **P10. The headline replicates at seed 3.** `mstruct` minus baseline at
  seed 3 is positive, making four of four.

### 33.5 Tools this required

- **The rebuild can run beside training.** The cache build rewrote the token
  ids file in place even when only new facts were needed — truncating a file
  that running jobs memory-map, which can kill them. It now verifies the ids
  are unchanged and leaves the file alone, writes everything atomically, and
  accumulates codes at two bytes an entry instead of eight (10.6 GB → 2.6 GB
  for eleven facts).
- **BLiMP, paired by seed** (`blimp_eval.py`, rewritten): cells grouped by
  label and paired by seed against the baseline; accuracy and the continuous
  log-probability margin; aggregation by BLiMP's own field and phenomenon;
  per-item margins saved. Sentences are encoded once per feature
  configuration, scored in batches that are checked against one-at-a-time
  scoring before anything is written, and prefixed with '.' so the first word
  is scored — the first version dropped it.
- **A gate before the rebuild** (`preflight_rebuild.py`): the real pipeline on
  the first validation lines must reproduce the cached ids and every existing
  fact exactly, and the new facts must be plausible and word-level.
- **A latent bug fixed.** The code that introduced `minfl` left one variable
  unset whenever the BPE pipeline ran without context features — a
  `NameError` waiting in any script that encodes with it. No recorded run
  reached that path.
- **Every training run tonight carries `MORPHLM_NO_BUILD=1`**, the project's
  existing guard, so a missing cache fails at once instead of starting a
  second multi-hour build beside the first.
- **One budget for every word-class feature.** `pos` took 19 dims where
  `mstruct`, `mfeat`, `mfeatlex` and `magr` take 8. Feature dims are carved
  out of the model's 128, so a `pos` model would have had ~190k fewer
  parameters and its comparison with `mstruct` would have mixed content with
  capacity. No recorded run had used `pos` as an input; it and `minfl` now
  take 8, pinned in the tests.
- **A dormant cache-poisoning trap closed.** A coarsened root feature folds its
  codes inside `code()`, and the cache build constructed its blocks from the
  requesting run's own config — so a coarsened run that ever triggered a build
  would have stored folded codes under a key that does not mention
  coarsening, and BLiMP would have folded a coarsened model's codes twice,
  turning `none` into class 0. It never happened. One function,
  `cache_blocks`, now builds codes as the cache stores them, for both paths;
  its test fails with the fix removed.
- **Rehearsed before it ran.** `scripts/rehearsal/rehearse_night.sh` ran the
  whole night on toy data in an offline sandbox, against the original
  snapshot: the new code reproduces every cached fact on every
  token, the gate passes and refuses a corrupted cache, the ids file is never
  rewritten, every cell trains from cache, BLiMP scores every checkpoint and
  the chain routes every branch — 15 of 15.


### 33.6 What the night found

The chain ran as planned in ten hours. The rebuild took 5h53m, confirmed all
120,625,719 token ids unchanged and never rewrote the file four training jobs
were reading. The derivation verified in 69 seconds. Against §33.4:

| | predicted | found | |
|---|---|---|---|
| P1 | `mstruct` does not move agreement | subject–verb +0.5 ± 0.3 (3/4 seeds), determiner–noun −0.4 ± 1.6 (2/4) | right |
| P2 | anaphor-gender jump does not replicate; if it does, the names half carries it | +14.6, +5.6, +21.2 points (margins +0.59 to +0.85 nats; 116–241 items gained against 13–60 lost); the **morphology** half carries it (+19.0, +11.8, +22.3), the names half is inconsistent | wrong twice |
| P3 | tagger leak small: `tag`/`pos` 95–99%, `mfeat` 96–99.5% | `tag` 91.8%, `pos` 92.6%, `mfeat` 95.5%, `magr` 96.7%, `minfl` 98.4%, over 55,282 words | wrong |
| P4 | majority bundle matches in context for 93–98% | 94.08% of 84M occurrences | right |
| P5 | `pos` within 2× of `mstruct` | +0.0072, +0.0109 against `mstruct`'s +0.0073, +0.0111: **tied** at both seeds | right, sharper than predicted |
| P6 | `mfeatlex` +0.002 to +0.008, below `mstruct` | +0.0002, +0.0050: below, and not distinguishable from zero | half right |
| P7 | `mfeatlex` lifts determiner–noun agreement ≥ 2 points | −1.8 ± 0.4, lower at both seeds | wrong |
| P8 | `mstruct`+`mfeatlex` beats `mstruct` at both seeds | +0.0021, −0.0015 | wrong |
| P9 | `mfeatlex` grows | **fades**: 437% of its final gain at 5k steps | wrong |
| P10 | headline positive at seed 3 | +0.0098: four of four | right |

Five of ten wrong, and the wrong ones carry the information.

**The headline, at four seeds: +0.0114 ± 0.0022 BPB (0.85%), positive at every
seed, mean/se 5.2.** It moved from 0.89% at three seeds, as a fourth seed
should move it.

**Inflection adds nothing in English.** Strictly causal inflection alone is
within noise (+0.0026 ± 0.0024), adds nothing to structure, does not help the
agreement it names, and FADES the way causal word identity does. In hindsight
that is one mechanism: English inflection is written on the word (*-s*, *-ed*,
*-ing*) and on a few frequent function words the model learns early, so a
feature restating it helps briefly and is then learned. The thesis-title
experiment has a clean English answer — the previous word's inflection is
information the model already has. Whether that holds where inflection is rich
and fused (Ukrainian) is exactly what D3 asks.

**A syntactic class ties the morphological one — by a different route.** `pos`
matches `mstruct` to 0.0002 BPB at both seeds, at the same width. Three
things argue the two are not the same information:

- *Time.* `pos` is front-loaded, 179% of its final gain at 5k steps and
  shrinking; `mstruct` is still growing at 50k (34% at 5k, over four seeds).
  They cross; they do not coincide. Trained longer, they should separate.
- *Grammar.* On BLiMP they move opposite phenomena: `pos` lowers anaphor
  agreement (−5.2, both seeds) and raises determiner–noun (+1.2); `mstruct`
  raises anaphor (+6.9 ± 2.5, four of four) and leaves determiner–noun alone.
- *Leak.* The tagger's `pos` depends on words to its right for 7.4% of words;
  `mstruct` for none. Part of `pos`'s early help may be the next word.

So the claim "morphological structure helps" now stands beside "so does a
coarse syntactic class", and which reading is right is two experiments away:
`poslex`, the strictly causal `pos`, and `pos`+`mstruct` (§33.7).

**BLiMP.** Structure's overall BLiMP gain was +1.0 ± 0.1 at three seeds and is
+0.6 ± 0.4 (three of four) at four — not robust. The one large, consistent
effect is anaphor gender, and it may not be agreement: the baseline swings from
47.6% to 67.0% across seeds on that paradigm, which is what an arbitrary
preference between *herself* and *himself* does. `scripts/eval/blimp_item_split.py`
separates the two readings (§33.7). The scorer itself is exactly deterministic:
eight checkpoints scored in both passes agreed on all 536 paradigm accuracies.

**A thesis table was wrong.** `paper/comparisons.md` measured the whole main
group against whichever feature-free cell came last — once the learning-rate
sweep existed, a 10k-step sweep run 0.08 BPB worse than the real baseline — and
compared means over different seed sets against an unpaired "noise floor". It
now pairs every cell with its own baseline by seed, and a test encodes both
failures (G-27, closed). Figures quoted from the old table must be re-read from
the new one; the headline figures never came from it.

**Scheduling.** The rebuild ran an hour past its estimate, so the GPU idled
from 19:06 to 20:58. A chain that overlaps a CPU build should hold more GPU
work in reserve than the estimate suggests.

### 33.7 Next, with predictions recorded first

- **`poslex`** (strictly causal `pos`, derived like `mfeatlex`), seeds 0 and 1.
  **P11:** it gains less than `pos` at both seeds, keeping 50–80% of `pos`'s
  gain. *Wrong if* it matches `pos` — then the tagger leak contributed nothing
  and the tie with structure is real.
- **`pos`+`mstruct`**, seeds 0 and 1. **P12:** it beats both alone at both
  seeds, gaining at least +0.013 — they carry different information, as their
  opposite time courses and opposite BLiMP profiles suggest. *Wrong if* it
  matches either alone: then they are two views of one thing.
- **The item split** on the night's BLiMP record. **P13:** `mstruct`'s
  anaphor-gender gain appears in both directions of change (*herself > himself*
  and *himself > herself*), so it is agreement. *Wrong if* one direction gains
  and its reverse loses: then it is a shifted preference, and the BLiMP result
  retracts to "no reliable grammatical effect".
- **Longer training, later**: `mstruct` and `pos` at 100k steps, where §33.6
  predicts they separate.


### 33.8 The anaphor gain was a preference, and what that changes

**P13 was wrong.** Split by what changes between the good and the bad sentence
(`blimp_item_split.py`), structure's anaphor-gender "gain" is a shifted
preference against *itself*:

| change (good > bad) | items | baseline | `mstruct` | effect |
|---|---|---|---|---|
| *herself > itself* | 342 | 53.9% | 89.0% | +35.1 (3/3) |
| *itself > herself* | 13 | 87.2% | 41.0% | −46.2 (3/3) |
| *herself > himself* | 301 | 13.2% | 19.6% | +6.4 (2/3) |
| *himself > herself* | 163 | 94.7% | 94.1% | −0.6 |

The paradigm makes *itself* the right answer in only 29 of 1,000 items, so a
model that simply disprefers *itself* gains accuracy it did not earn. Anaphor
number agreement shows the same signature. The one true gender contrast, where
the baseline is badly biased toward *himself* (13.2% on *herself > himself*),
barely moves. **By the rule recorded before the result, structure's BLiMP
result retracts to "no reliable grammatical effect".**

The mechanism is not a scoring artifact: MorphoLEX classes every reflexive
(*herself, himself, itself, themselves, myself*) as `compound`, so the features
are identical at every position of the good and the bad sentence. The
dispreference was learned in training. The morphology half carries it most
strongly (+41.1 / −53.8), which is why P2's conditional looked backwards.

**Structure and part of speech help different targets.** Split by the
frequency of the token being predicted (paired at seeds 0 and 1, every cell
2/2): structure gains more on mid-frequency targets (0.035 and 0.030 nats per
token between 13.5k and 312k occurrences, against 0.023 and 0.018 for `pos`);
`pos` gains more on the most frequent (0.030 and 0.019 against 0.022 and
0.013). A syntactic class predicts the function words that follow; word
structure, the content words. Indirect support for P12.

**The control the `pos` tie demands.** If a syntactic class helps as much as the
morphological one, perhaps any consistent coarse grouping of word types does.
`mstructshuf` keeps `mstruct`'s eight classes but reassigns them at random
among word types of similar frequency (bins close at 64 types or a 2× frequency
ratio, so a class only moves between comparable words). Class sizes and the
frequency profile survive; the linguistic content does not. In English the most
frequent words are nearly all `simple` (71 of the top 72), so the shuffle
scrambles exactly the mid- and low-frequency words where structure's gain
lives. The derivation refuses unless token-level class shares move by under 5%
and at least 15% of tokens change class.

**A new way of working.** From here, new code runs first in short day blocks whose
results are read within hours; only code that has already run on the GPU machine goes
into a night chain.

**Predictions**, recorded before the blocks run:

- **P14** (lookahead split, `pos` and `mstruct` checkpoints): `pos`'s gain on
  CONTINUATION targets is a larger share of its total than `mstruct`'s. Under
  the shift, a word's first token only ever sees the class of the word two
  back; but the tagger saw the current word when it tagged the previous one, so
  shifted `pos` partly reveals the word being continued. *Wrong if* the shares
  are similar.
- **P15** (`mstructshuf`, seeds 0 and 1): the shuffled control keeps less than a
  third of structure's gain (under +0.004 at both seeds) — the content matters.
  *Wrong if* it keeps two thirds or more: then any coarse partition of word
  types helps, and the thesis claim must be restated.
- **P16** (100k steps, seed 0): structure's gain at 100k exceeds its gain at
  50k at the same seed (+0.0073) — it keeps growing.
- **P17** (100k steps, seed 0): `pos`'s gain falls at least 0.002 below
  structure's — tied at 50k, separated at 100k.


### 33.9 The shuffled control matches structure

The three day blocks of 27 September, against §33.7–§33.8:

| | predicted | found | |
|---|---|---|---|
| P11 | `poslex` keeps 50–80% of `pos`'s gain | +0.0098, +0.0064 against `pos`'s +0.0072, +0.0109: 90%, larger at seed 0 | wrong — the tagger leak contributed nothing measurable |
| P14 | `pos` gains more on continuation fragments than `mstruct` | the reverse: `pos` HURTS continuations (−36% and −15% of its gain) and puts it all on word-initial targets (1.8× and 1.5× their share); `mstruct` is proportional, continuations 20% | wrong, reversed |
| P15 | `mstructshuf` keeps under a third of structure's gain | +0.0081, +0.0129 against +0.0073, +0.0111: **all of it, and a little more** | wrong |

The shuffle was fair and thorough: token-level class shares moved 0.09%, and
83.2% of the tokens outside `simple` changed class. Yet the shuffled feature
matches structure in size, in time course (GROWS: 8% of its final gain at 5k
steps, structure 34%), and in where it helps (the same frequency profile,
slightly larger on rare and mid-frequency targets).

**Structure's gain does not come from morphological content.** By the rule
recorded with P15, the thesis claim is restated: a coarse word-level class of
the previous word helps by about 0.85%; which word is prefixed, suffixed,
compound or a name does not matter. What the shuffle KEPT is where the gain must
live: every word still has one consistent class; the classes keep their
frequency profile; and the most frequent words, nearly all `simple`, stay
`simple` — so function words stay marked.

**Two families, not one.** `poslex` reproduces `pos` exactly as `mstructshuf`
reproduces `mstruct`: the syntactic family is front-loaded (156–179% of its final
gain at 5k steps), helps word-initial targets and the most frequent ones, and
hurts continuations; the structure family grows through training and helps
proportionally, most on mid-frequency targets. `pos` is a real, content-bearing
syntactic effect as far as it has been tested — its own shuffle control is next.

**What the structure family is made of** — two controls, same table size and
width as `mstruct`, so the architecture is identical:

- **P18** (`mconst`, every word the same class — the null): gains under +0.003
  at both seeds; the slot alone does nothing. *Wrong if* it gains +0.006 or more:
  then much of every feature's gain is the architecture — 8 of 128 embedding
  dims given to a word-level input — not information.
- **P19** (`mfreq`, the previous word's frequency band, seven bands of equal
  token mass): keeps at least two thirds of structure's gain at both seeds; how
  common the previous word is, is the ingredient. *Wrong if* under a third: then
  consistency or function-word marking carries it, not frequency.
- **P20** (`posshuf`, `pos` shuffled among words of similar frequency):
  behaves like the structure family, not like `pos` — GROWS, under 50% of its
  final gain at 5k steps. *Wrong if* front-loaded: then even `pos`'s profile
  does not need syntactic content.
- P16 stands (structure at 100k steps, seed 0). P17 is withdrawn: the night no
  longer trains `pos` at 100k.


### 33.10 The null helps as much as structure

**P18 was wrong.** `mconst` — `mstruct`'s architecture with every word given
the same class — gained **+0.0076** at seed 0 (1.3401 against the baseline's
1.3477), as much as structure (+0.0073), the shuffle (+0.0081), frequency
(`mfreq`, +0.0079) and `pos` (+0.0072). One seed, but every cell of the
structure family now sits within 0.0008 of a feature that carries nothing.

**It looks like regularization.** Averaged over the last 10k steps, every slot
cell improves validation loss about 2.5× more than training loss (`mconst`:
validation −0.024 nats, training −0.010). `pos` is different: it fits the
TRAINING data much better (−0.031) yet ends level with the null on validation
— information that does not generalize past what the slot already gives.
`mfeatlex`, with the same 8-dim slot but 72 near-identity classes, gets no
benefit at all; the identity-like features (`mhash`, `mlemma`, `mfeatlex`) all
FADE. A high-information feature seems to be used for memorization, which
cancels the effect.

**What it changes.** Every Q2 comparison in this project set a feature against
a baseline whose token embedding is 128 wide; the feature cell's is 120. The
right control for a T-dim feature is the T-dim null. Against the null, at seed
0: structure −0.0003, the shuffle +0.0005, `mfreq` +0.0003, `pos` −0.0004,
`poslex` +0.0022. If this holds at every seed, Q2's answer becomes: no
injected linguistic fact has been shown to help beyond the slot, and reserving
a few input dimensions for a low-information word-level input regularizes a
small model — a methodological finding in the spirit of §30. P19 ("frequency
is the ingredient") was right on its number and wrong on its reason: the null
keeps the same gain.

**Predictions**, recorded before the day run and the night:

- **P21** (day, `mconst32`, seed 0): the gain at 32 dims is at least +0.0106,
  0.003 above 8 dims — the lever is how much of the embedding the slot takes.
  *Wrong if* it is no larger than +0.0076.
- **P22** (night, `mconst` at seeds 1–3): within ±0.002 of structure at every
  seed — structure carries nothing beyond the slot. *Wrong if* structure beats
  the null by more than 0.002 at two or more seeds.
- **P23** (night, widths 16 / 32 / 64): the gain rises from 8 to 32 dims and
  falls by 64, where a 64-dim token embedding is too small.

Deferred while the slot is resolved: `posshuf` (P20), seed 2 of the shuffle and
`poslex`, and the 100k-step pair (P16).


### 33.11 The headline is the slot

The night of 27 September ran the null at seeds 1–3, so structure (seeds 0–3)
pairs with it at every seed:

| paired by seed | 0 | 1 | 2 | 3 | mean ± se |
|---|---|---|---|---|---|
| the null over the baseline | +0.0076 | +0.0101 | +0.0185 | +0.0082 | **+0.0111 ± 0.0025** (4/4) |
| structure over the baseline | +0.0073 | +0.0111 | +0.0176 | +0.0098 | +0.0114 ± 0.0022 (4/4) |
| **structure over the null** | −0.0002 | +0.0010 | −0.0009 | +0.0015 | **+0.0004 ± 0.0005** (2/4) |

**P22 was right: structure carries nothing beyond the slot.** The 0.85% that
headlined this project since §31 is an architecture effect. The null also
reproduces structure's time course exactly (30% of its final gain at 5k steps
and 54% at 10k, against 34% and 57%). Against the null, nothing tested so far
clearly adds information: the shuffle +0.0017 ± 0.0011, `mfreq` +0.0007 ±
0.0003, `pos` +0.0002 ± 0.0006, `poslex` −0.0007 ± 0.0030 (two seeds each);
`mfeatlex` hurts, −0.0062 ± 0.0011.

**A correction to §33.10.** "Regularization" was read off seed 0 alone. At
every other seed and width the null improves training loss too (by 0.010–0.030
nats), and at 32 and 64 dims about as much as validation. The slot makes the
model fit better, not merely generalize better.

**Width barely matters** (seeds 0 and 1, gain over the baseline): 8 dims
+0.0089, 16 dims +0.0073, 32 dims +0.0114, 64 dims +0.0100. P21 was right
(32 dims gave +0.0114 at seed 0); P23 only in part — the peak is at 32 and it
falls at 64, but the curve is nearly flat and not monotone. Almost all of the
benefit is present at 8 dims, so this is not "fewer embedding parameters":
halving the token embedding at 64 dims does no better.

**Not the pipeline.** The loader draws windows with a generator seeded by
`train.seed` alone, and codes ride along without affecting which windows are
drawn: baseline and feature cells see the same data in the same order.
Initialization differs, as the loop's own comment warns — but initialization
noise cannot produce +0.011 at four seeds out of four.

**What the thesis now says (Q2).** No injected linguistic fact has been shown to
help once measured against the right control. Every feature in this project was
compared with a baseline whose token embedding spans all 128 input dims; the
feature cell's spans 120. The null control — the same slot, no information — is
the baseline a feature must beat, and none has. It is the third measurement
artifact this project found in its own numbers, after the lookahead leak (§30)
and the mislabelled baseline of the exported tables (§33.6), and the one that
had carried the headline.

**What the slot does** — two controls with identical architecture:

- **`mflat`**: one learned vector at EVERY position, words and punctuation
  alike. `mconst` still marks whether the previous unit was a word; this
  removes even that. **P26:** within 0.003 of the null at both seeds. *Wrong
  if* under half the null's gain — then that boundary signal is the ingredient.
- **`mzero`**: the slot's 8 dims held at exactly zero by the embedding — no
  learned vector at all; the only difference from the baseline is a token
  embedding of 120 dims instead of 128. **P25:** within 0.003 of the null at
  both seeds; the lever is input dims free of token identity. *Wrong if* under
  half the null's gain — then the learned vector matters.
- **BLiMP of the null. P24:** the null reproduces structure's BLiMP profile,
  including the learned dispreference for *itself* — structure's BLiMP shifts
  were slot effects too. *Wrong if* structure beats the null on anaphor
  agreement at three or more of four seeds.


### 33.12 A narrower token embedding is the whole effect

The day of 28 September tested what the slot does, with two controls of
identical architecture (gain in BPB, seeds 0 and 1):

| cell | over the baseline | over the null |
|---|---|---|
| the null (`mconst`) | +0.0076, +0.0101 | — |
| **`mzero`** — the slot held at exactly zero | +0.0083, +0.0071 | +0.0007, −0.0030 |
| **`mflat`** — one learned vector at every position | +0.0069, +0.0114 | −0.0007, +0.0013 |

**P26 was right:** the word-versus-punctuation signal the null still carried is
not the ingredient. **P25 was right, at its boundary:** the zero slot keeps 87%
of the null's gain, and its −0.0030 at seed 1 sits on the ±0.003 line. Its
training and validation losses move like the null's (−0.013 and −0.026 nats at
seed 0). So the effect needs no information and no learned vector: **a token
embedding 8 dims narrower than the residual stream is the whole of it.** Those
8 input dims then carry only the positional embedding, which is added across
all 128.

**BLiMP of the null.** Overall, the null shifts BLiMP as much as structure
(+0.8 ± 0.3 against +0.6 ± 0.4; structure minus null −0.2, ahead at 1 of 4
seeds): the overall shift is the slot. **P24 was wrong by its own rule** —
structure beats the null on anaphor agreement at four of four seeds (+1.0,
+0.3, +0.4, +13.6). The item split says how: the null already shows the learned
dispreference for *itself* (+15.1 where it is wrong, −26.9 where it is right),
and structure roughly doubles it (+30.6, −38.5). A stronger preference, not
agreement.

**Why a narrower embedding?** Two readings, separated by one mask on the
positional embedding (a non-persistent buffer, `None` for every existing cell,
so nothing already trained changes):

- **A clean position channel.** The free dims carry position unmixed with token
  identity. **P27** (`mzeronopos`, the slot empty at the input — no token, no
  position): keeps under half of the zero slot's gain. *Wrong if* it keeps half
  or more — then the free dims help as workspace, whatever they carry.
- **P28** (`mzeroposonly`, position ONLY in the slot, token dims pure
  identity): at least the zero slot's gain. *Wrong if* under half.
- **P29** (seeds 2 and 3 of `mzero`, the shuffle and `pos`): at four seeds,
  neither feature beats the null by more than +0.002 on average, and `mzero`
  stays within ±0.003 of it.

Worth a literature check before this is written up: the result resembles
factorized embeddings (an embedding narrower than the hidden size) and the case
for separating position from content in a transformer's input. The connection
has not been verified here.


### 33.13 Where the fresh look's theories stand

§32 made claims and predictions before any null existed. Against the evidence
of §33.9–§33.12:

| §32 claim | status | why |
|---|---|---|
| §32.1 the structure gain grows through training; the headline is a floor | **reinterpreted** | the growth is the slot's: the null grows identically (30% / 54% of its final gain at 5k / 10k steps; structure 34% / 57%). Structure's own gain is +0.0004 ± 0.0005, a floor under nothing |
| §32.2 identity FADES because the model learns it; structure GROWS because it is complementary | **refuted as stated** | a shuffled structure, the null, a flat slot and a zero slot all grow exactly as structure does; "the model never learns *suffixed* as well as MorphoLEX states it" is not what the curve showed |
| §32.2 leaky features PLATEAU — a shortcut signature | stands | untouched, though leaky cells also carry the slot |
| §32.3 morpheme tokenization learns slower | stands | Q1 has no features and no slot |
| §32.5 the 0.89% is a lower bound | **withdrawn** | it is the architecture's number |
| §32.4 inflection will behave like structure, not identity | **wrong** | `mfeatlex` fades like identity (§33.6) |
| §32.4 BLiMP will give downstream evidence | **none** | the overall shift is the slot; the anaphor gain is a learned preference (§33.8, §33.12) |

**One hypothesis replaces the old story** — a synthesis, not yet a result:
**what helps is input dims free of token identity; a feature keeps that benefit
only as far as it leaves those dims free.** Low-information features — structure,
its shuffle, frequency, the null, the flat and the zero slot — leave them free
and land on the same gain. Identity-rich features FILL them and end *below* the
null, not at it: `mfeatlex` −0.0062 against the null, causal `mhash` about
−0.008 by its record. It accounts for the old fade-versus-grow contrast, for the
flat width curve (a few free dims suffice) and for training loss improving too.

**What it changes in `docs/DIRECTIONS.md`.** The literature audit (A4) gains a
fourth fairness condition — was the embedding width matched? — and becomes the
cheapest and perhaps most important item. The auxiliary morphology loss (C5)
becomes the one design immune to both the lookahead leak and the slot: morphology
as a target, not an input. The learned-class control (D2) is answered by the
shuffle and the null. Ukrainian (E1) stays open, now with a mandatory null.

**Additive injection.** The slot exists because facts TAKE dims from the token
embedding. `frozen.inject=add` gives each fact a full-width table ADDED to a
full-width token embedding instead: no carved slot, so a feature's effect over
the baseline is its information and nothing else (+1,024 parameters, 0.1%).
Every existing cell reproduces bit for bit; the switch refuses slot controls
and frozen factors. Labels gain `-ADD`.

- **P30** (`mstruct`, added, seed 0): within ±0.002 of the baseline — structure
  carries no usable information in a slot-free design either. *Wrong if* it
  gains +0.004 or more: then information exists that the slot hid.
- **P31** (`mconst`, added): within ±0.002 of the baseline — a constant added
  vector is only a bias.


### 33.14 The additive null gains too

Structure and the null, ADDED to a full-width token embedding (seed 0, gain over
the baseline's 1.3477):

| cell | BPB | gain |
|---|---|---|
| **the null, added** | 1.3383 | **+0.0094** |
| structure, added | 1.3431 | +0.0046 |
| the null in a slot (§33.10) | 1.3401 | +0.0076 |

**P31 was badly wrong**: a near-constant vector added to every input gains more
than the slotted null — in a design that carves nothing, so the "narrower token
embedding" of §33.12 cannot be the whole story. **P30 must be scored with care.**
Added structure gained +0.0046, over its "+0.004 or more" line — but that rule's
reading, "information the slot hid", assumed the additive null would be inert.
Against its own null, added structure is **0.0048 worse**. Q2's answer survives
the slot-free design; what failed is the expectation that the design has no
architecture effect of its own.

**The confound hunt.** Every run is evaluated on the same 252,160 validation
tokens. The loader's generator is seeded before the model is built and draws the
same windows for every cell. Every `Linear` and `Embedding` is re-initialized
from N(0, 0.02) after construction, with nothing keyed on facts. Parameter
counts explain nothing: the zero slot has 137k fewer than the baseline and
gains; the added null has 1,024 more and gains most. The optimizer applies weight
decay to every parameter and clips the global gradient norm — uniform across
cells, and unusual in the first respect.

**Where that leaves the mechanism.** Several unrelated tweaks each beat the plain
baseline by 0.6–0.7%: narrowing the token embedding, adding a learned
near-constant vector, or both. One common thread would be token-INDEPENDENT
content in the input — dedicated dims or a shared vector — but before building on
it, the one remaining non-architectural difference must be excluded: every
feature cell also has an extra parameter table (shifting the random
initialization sequence) and carries codes in every batch.

- **P32** (`mzero` ADDED: zeros added, nothing carved — given the baseline's
  weights it computes the baseline bit for bit): within ±0.002 of the baseline.
  *Wrong if* it gains +0.004 or more — then the effect is not architecture at
  all but something in how feature cells are built or trained, and every result
  since §33.10 must be re-read.
- **P33** (`mflat` ADDED: one learned vector added at every input): gains
  +0.006 or more — a learned input bias is what the additive null exploits.


### 33.15 Families that share an initialization

**P32 was wrong, just past its line.** `mzero` ADDED — the baseline's exact
function plus an unused table — gained **+0.0041** at seed 0 (1.3436). **P33 was
right by its letter**: `mflat` ADDED gained +0.0062.

**Why a function-identical model can differ.** At the same seed, a run sees the
same windows (the loader's generator) and the same dropout masks (the GPU
generator; the model is initialized on the CPU). What differs is the initial
weights: an extra table shifts the CPU random stream, and every random tensor
after it is drawn differently. Verified directly:

- cells with **identical parameter shapes start from identical weights** — the
  four additive cells (zero, const, flat, structure) share 54 of 54 tensors at
  seed 0, and so do the 8-class slotted cells (`mconst`, `mstruct`, `mzero`,
  and the shuffle, `mfreq`, `mflat`, the positional controls);
- a different shape draws a different start — `pos` (19 classes) shares only
  the deterministic tensors, and the **baseline sits outside every family**.

So pairing by seed matches initialization **only within a family** — the
training loop's own comment warned that a shared seed buys reproducibility,
not variance reduction; this makes it precise.

**What stands, and what must be re-read.**

- **Within families, comparisons are fully matched** — same data, dropout and
  initialization. Structure against the null in the slotted family, +0.0004 ±
  0.0005 over four seeds, is a clean, matched null result: **Q2's "no" is
  stronger than it was.** In the additive family, against the matched control
  at seed 0: structure +0.0005, one flat learned vector +0.0021, the null's two
  vectors (word / non-word) +0.0053.
- **Every comparison with the baseline mixes architecture with a different
  random start.** The null's +0.0111 over the baseline (§33.11) contains an
  initialization component of unknown size; the control shows a different start
  alone moved the baseline's own function by +0.0041. Across three seeds the
  baseline's trajectory barely varied (±0.002 nats throughout), which makes
  that +0.0041 hard to read as ordinary noise — but seed 2 was an outlier of
  +0.03 nats the other way, so the baseline's outcomes are heavy-tailed.
- **A tool follows.** Giving a baseline an unused table of a family's shape
  makes it a family member: function-identical, initialization-matched. The
  additive family has one (`mzero` ADDED). The slotted family cannot: its
  shapes include the narrower token embedding, which changes the function.

- **P34** (`mzero` ADDED at seeds 1–3): within ±0.002 of the baseline at two
  or more of three seeds — seed 0 was an initialization outlier. *Wrong if* it
  beats the baseline by +0.003 or more at all three: then the unused table
  systematically helps, and a training-side cause must be found.
- **P35** (the additive family at seed 1, matched): structure within ±0.002 of
  the matched control; the null's two vectors at least +0.003; the flat vector
  in between.


### 33.16 The architecture effect, measured against eight starts

**P34 was right.** `mzero` ADDED — the baseline's function, started elsewhere —
against the baseline at the same seed: +0.0041, +0.0015, +0.0070, −0.0015 at
seeds 0–3. Two of the three new seeds fall within ±0.002; the unused table does
not help systematically. Seed 0 was a lucky start, as seed 2 of the baseline was
an unlucky one.

That gives **eight random starts of the baseline's function**: mean 1.3489,
standard deviation 0.0039, range 0.0137. Against them:

| | runs | mean BPB | sd |
|---|---|---|---|
| the baseline's function | 8 | 1.3489 | 0.0039 |
| the slotted null | 4 | 1.3392 | 0.0009 |
| structure (slotted) | 4 | 1.3388 | 0.0015 |

**The null improves on the baseline's function by +0.0097 ± 0.0015 BPB (0.72%)**,
and the worst null run (1.3401) beats the best baseline start (1.3436). The
architecture effect is real; the seed-paired +0.0111 of §33.11 was inflated by
the baseline's unlucky seed 2. And **the baseline is four times more sensitive
to its random start** than the slotted model — the slot improves the result and
steadies it. One lesson for every comparison before this: a seed-to-seed spread
estimated on one configuration is not the spread of another.

Q2 stands as it was, and stronger: within the slotted family — matched in data,
dropout and initial weights — structure adds +0.0004 ± 0.0005.

**Tonight**, all initialization-matched: the additive family at seeds 2 and 3
against its control, and the position test (`mzeronopos`, `mzeroposonly`)
against the slotted zero control at seeds 0 and 1.

- **P36**: at every seed with a matched control, added structure stays within
  ±0.002 of it; the null's two added vectors gain +0.003 or more at three or
  more of four seeds.


### 33.17 The literature, audited — and the case for Ukrainian

`docs/reference/literature_audit.md` holds the feature-injection literature against four
conditions, each an artifact this project measured in its own numbers:
injective, coverage-matched, causal, and **form-matched** (a same-shape null).
No audited work reports a same-shape null. Sennrich & Haddow (2016) fixed their
total input width and carved slots for features — exactly the form that gained
without information here. Vania & Lopez (2017) fed a language model gold,
context-disambiguated analyses — the configuration where §33.3 found a
right-context leak. KinyaBERT changes architecture and information together.
And the positive results cluster in morphologically rich languages — Czech and
Russian, Finnish and Turkish, Kinyarwanda — and are small or absent for English,
as ours is.

That is the case for Ukrainian (`docs/reference/ukrainian_pipeline.md`), where the hypothesis
has its best chance. Checked before any code: spaCy's Ukrainian pipeline, at the
installed version, has a statistical morphologizer that resolves syncretism
from context, at English speed. And a blocker, found on paper: every word-level
tool classifies tokens with `isalpha()` on their ByteLevel symbols, which
calls 77% of Cyrillic tokens non-alphabetic. The fix — classify decoded text —
must be opt-in, because English tokens with accents would otherwise change
class and with them the codes behind every recorded English result.


### 33.18 The Ukrainian path, built

Built while the night of 28 September ran, and rehearsed end to end on a toy
Ukrainian Wikipedia through the real pipeline and spaCy's Ukrainian model:
corpus loading (`data.corpus=ukwiki`, a title-hash split), tokenizer training
that records its word rule beside the tokenizer, a language-specific free ride
(dependency, part of speech and seven UD features, no MorphoLEX), the UD blocks
padded to 8 classes so every feature shares an initialization family with the
nulls, and the nulls themselves. The Cyrillic blocker (G-60) is fixed without
touching English: the word rule travels with the tokenizer, and every English
cell type still computes bit for bit what it computed before. On Ukrainian text
the old rule found 10,900 word-continuation fragments where the decoded rule
finds 57,724.


### 33.19 Two routes to the same gain

The night of 28 September completed two matched designs.

**The additive family at four seeds**, each against its initialization-matched
control (the baseline's function plus an unused table):

| added to a full-width embedding | seeds 0–3 | mean ± se |
|---|---|---|
| the null's two vectors | +0.0054 +0.0083 +0.0100 +0.0122 | **+0.0090 ± 0.0014** (4/4) |
| one flat learned vector | +0.0022 +0.0092 +0.0112 +0.0116 | **+0.0085 ± 0.0022** (4/4) |
| structure | +0.0006 +0.0096 +0.0080 +0.0109 | +0.0073 ± 0.0023 (4/4) |

**P35 and P36 were wrong about structure**: it gains as much as the nulls,
and seed 0 — where every added vector looked nearly inert — was the odd seed
out. **Q2's answer is unchanged**: against the flat vector, the same-shape
null without information, structure adds −0.0013. The additive design carries
its own architecture effect — one learned vector added to every input — of the
same size as the slot's.

**The position test**, matched with the slotted zero control:

| slotted | against the zero slot | against the baseline |
|---|---|---|
| the zero slot (reference) | — | +0.0083, +0.0071 |
| the slot empty — no token, no position | −0.0081, −0.0051 | +0.0002, +0.0020 |
| position only in the slot | −0.0039, −0.0028 | +0.0044, +0.0043 |

**P27 was right**: take position out of the free dims and the zero slot's gain
disappears — it works as a **clean position channel**. **P28 half right**:
moving position out of the token dims as well keeps only about half the gain;
the model uses position both ways.

**Two routes, then, to the same ~0.7% over the plain baseline**: a clean
position channel (the carved slot) and a learned vector added to every input
(the additive design). Both say the plain baseline's input — token identity
and learned absolute position mixed in all 128 dims — leaves something on the
table for a model this small. And in both designs, against a null of the same
shape, no linguistic feature adds anything.


