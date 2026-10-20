# Research journey — Part 11: Morphology in the Tokenizer Helps, and Lasts

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.61–§33.70. [← Part 10](10_the_learning_law.md) · [index](README.md) · [Part 12 →](12_full_training_and_the_leak.md)

**Established:** BPE that never joins a stem to its inflectional ending, the cut read from the word
form alone by a dictionary rule checked against two morpheme dictionaries (§33.61–§33.62), beats
plain BPE 17k by +0.0174 ± 0.0014 BPB (1.6%, 4/4) at a 10k budget, every tokenizer's stream decoding
to the same text (§33.65–§33.66). The lead holds at twice the budget and is larger in a wider model,
the two axes along which a feature's gain fades (§33.68). On top of each of the six tokenizers case
beats its null (§33.65); its gain sits on the tokens that hold an ending, while the extra it gains
where endings are exposed comes mostly from words the rule does not cut (§33.69–§33.70).

---

### 33.61 Morphology in the tokenizer: the ending rule, checked, and BPE that keeps endings apart

**The closest Ukrainian work, read in full** (Borodavko 2025, a UCU master's
thesis; `docs/reference/related_work.md`). It built morpheme vocabularies from
three dictionaries — Yatsenko (1981), Polyga (2009) and a private Kyiv
university database, 35k–205k words, scanned and corrected — and tokenizers
built on them, against BPE trained on UberText. The morpheme tokenizers
compressed worse: 25.9–29.1 tokens per text against 20.0–25.3 for BPE at 15k to
3k, "no significant improvement", which it attributes to the dictionaries'
mismatch with modern text. The language-model comparison never took place: its
GPT-2 models did not converge. Two lessons for the design, nothing that changes
it: no language-model evidence on morphology in a Ukrainian tokenizer exists
there, so a controlled test in bits per byte is new; and its planned metric,
top-1 and top-5 next-token accuracy, cannot be compared across tokenizers, which
is why this project measures BPB. Its literature chapter cites Cotterell et al.
(2018) for what Arnett and Bergen (2025) found; the register cites the latter.

**The rule** (`morphlm/data/endings.py`, `uk_endings_v1`), as prototyped in
§33.60 with three changes. A cut is made only if every pymorphy3 reading of the
form gives the same one, so an ambiguous form stays whole (`його`, a pronoun or a
form of `йога`); indeclinable words, abbreviations and words in other scripts
are never cut. And a dictionary check (below) removed the soft sign and `й` from
the nouns' citation endings: `кінь`, `музей` and `-ість` nouns have no ending in
the citation form, and cutting `-ь` off put a noun's cut inside a morpheme in a
fifth (Polyga) to a quarter (Yatsenko) of the nouns' cuts. The rule's
constants, pymorphy3's version and dictionary, and the cuts of 30 probe words
make a fingerprint.

**Checked against the dictionaries** (`scripts/audits/ending_gold_check.py`, on
the two dictionaries Borodavko's repository publishes — no licence is stated, so
they are read where they are cloned and never copied here; pymorphy3
2.0.6 with its Ukrainian dictionary 2.4.1, 417,330 lexemes). On their citation
forms:

| | words | cut by the rule | cut on a morpheme boundary | cut isolates the final morpheme |
|---|---|---|---|---|
| Polyga 2009 | 34,817 | 82.7% | **99.7%** | 96.4% |
| Yatsenko 1981 | 110,783 | 73.5% | **97.9%** | 93.6% |

Adjectives agree at 99.4–99.8%. Most of the rest is convention — reflexive verbs,
where the rule cuts before `-тися` and the dictionaries before `-ся`; nouns in
`-ція`, which Yatsenko keeps as one morpheme — or dictionary noise from the scans.
Uncut words are mostly adverbs and nouns with no ending in the citation form.

**The tokenizer** (`--algorithm bpe-endings`, named `bpeend_uk_<k>k`). Training
text is marked at every cut with a noncharacter the pre-tokenizer splits on and
removes, so BPE learns no merge across a cut. Marking the training text alone
is not enough: on the toy corpus (vocabularies of 2k and 4k) BPE trained that way
still fuses a third of the endings when it encodes unmarked text — 67–69% of
cuts exposed, against 21–26% for plain BPE — because merges learned inside other
words (`книга` in `книгарня`) join a stem and its ending again. So encoded text
is marked too: every tokenizer is opened through `tokenization.load_tokenizer`,
which wraps this one so that it marks what it encodes and maps offsets back to the
unmarked text; it refuses to run under another fingerprint, and a test finds no
`Tokenizer.from_file` anywhere else. Exposure is then 100% by construction, for
about 8% more tokens per text on the toy corpus. The tokenizer statistics now
report exposure at the rule's cut as well (`cut|`), for every tokenizer; a run's
record names its tokenizer, the rule and the fingerprint; and `verify.py` runs
every test suite it finds rather than a list.

**Rehearsed on the toy corpus, real code end to end.** The CLI trained
`bpeend_uk_1.5k` (6,902 cuts in 97,888 characters of training text); the
statistics gave it 100% exposure at the cut, against 46.6% for BPE at 600 and
54.5% for Unigram at 1,500 (by the common-prefix measure 83.1%, 52.0% and
52.4%); `prepare_condition.py` built its condition — the same 17,712 validation
words as under BPE, every code constant within a word, alignment 99.6% — the
smoke test passed for the null, case and lexical case, and the three cells
trained.

Predictions, for the condition at ~38M tokens and a 10k budget (seed 0):

- **P125** (~65%). BPE 17k that keeps endings apart reads 5–15% fewer characters
  per token than BPE 17k (3.59): between 3.05 and 3.41.
- **P126** (~55%). Its null ends below BPE 17k's null (1.0891 BPB): morphology
  in the tokenizer helps by itself.
- **P127** (~60%). Read at null BPB 1.25, case's gain under it lies within 0.002
  of the five tokenizers' mean (+0.0056): the law, not the tokenizer, sets the
  gain, even where every ending is exposed.

### 33.62 The sixth tokenizer, measured — and nothing outside the project

**Trained on the GPU machine** (7 October): `bpeend_uk_17k` from 5,290,743
training rows, about six and a half minutes of marking, 17,000 tokens. Its
fingerprint, `f8bb86d44819b4b1`, is the rehearsal machine's: the rule cuts every
word there exactly as here. The dictionary check reproduced to the word (Polyga:
82.7% of words cut, 99.7% of cuts on a morpheme boundary, 96.4% isolating the
final morpheme; Yatsenko: 73.5%, 97.9%, 93.6%). Its condition at 0.05 is being
prepared.

**The six tokenizers, measured both ways** (3,000 held-out paragraphs):

| tokenizer | characters per token | tokens per word | words split | ending exposure, common prefix | at the rule's cut | ending its own token |
|---|---|---|---|---|---|---|
| BPE 4k | 2.73 | 2.37 | 64.7% | 21.5% | 20.2% | 19.8% |
| BPE 8k | 3.13 | 2.04 | 55.0% | 14.5% | 13.9% | 13.8% |
| BPE 17k | 3.59 | 1.73 | 43.3% | 9.1% | 8.8% | 8.7% |
| BPE 32k | 3.96 | 1.53 | 34.2% | 5.7% | 5.5% | 5.5% |
| Unigram 17k | 3.07 | 1.85 | 62.9% | 41.1% | 43.3% | 43.2% |
| **BPE 17k, endings apart** | **3.21** | **1.99** | 64.4% | 80.3% | **100.0%** | 99.8% |

**P125 right**: 3.21 characters per token, 10.6% fewer than BPE 17k. The two
exposure measures order the first five tokenizers alike and agree within 2.2
points, so §33.58's reading of D7 stands on either; the sixth is at 100% of its
own cuts, and 80.3% by the common prefix, which places endings differently
(§33.60). Exposing every ending costs 15% more tokens per word than BPE 17k —
between Unigram 17k and BPE 8k — so at a fixed number of steps it reads less
text; comparisons stay at equal null BPB.

**Nothing outside the project** (7 October). The dictionaries'
clone had been made in the home directory; it now lives in `external/` —
gitignored, skipped by `check_docs.py`, never packed — where the dictionary check
reads it by default. `pack.sh` wrote its snapshot beside the project; it now
writes to `packs/`. The rule since: commands write only inside the project.

**A blind spot in proving archives.** Every archive is proved by rebuilding the
GPU machine's state and comparing it with the working copy. The comparison,
`diff -r --exclude=data`, skips every directory named `data` — `morphlm/data/`
and `scripts/data/` included — so those two folders went unchecked in every
proof made that way. A tree comparison that skips generated folders
only at the top level (`scripts/rehearsal/tree_diff.py`) found no drift: every
earlier change there had shipped. It is the comparison from now on.

### 33.63 The sixth condition, ready — and what its numbers showed about the tools

**Prepared on the GPU machine** (7 October, results_73).
`prepare_condition.py` built `bpeend_uk_17k` at fraction 0.05 in an hour — the
pilot, the tagging pass (58 minutes), the null, the zero slot and lexical case —
and every gate passed: case, both nulls and lexical case present in both splits
and constant within every word. It is the other five conditions' text: the same
235,673,205 bytes of training text, the same validation words (147,694; the
others 147,693–147,733, differing only where a tokenizer splits a character), a
lexicon of 1,138,979 word types whose majority case matches the case in context
for 90.02% of occurrences (89.89–90.02% under the others), and 5.59% of
validation words unseen in train.

| tokenizer | train tokens | validation tokens | aligned, validation / train | `dep` and `pos` constant within words (validation) | tagging pass |
|---|---|---|---|---|---|
| BPE 4k | 49,433,222 | 423,931 | 99.70% / 99.69% | 99.91% | 1 h 21 min |
| BPE 8k | 43,197,136 | 371,617 | 99.78% / 99.76% | 99.93% | 1 h 16 min |
| BPE 17k | 37,972,206 | 325,867 | — | 99.92% | — |
| BPE 32k | 34,600,899 | 296,325 | 99.80% / 99.76% | 99.92% | 1 h 14 min |
| Unigram 17k | 44,485,169 | 378,426 | 91.66% / 91.62% | 99.89% | 1 h 31 min |
| **BPE 17k, endings apart** | **42,600,731** | **363,158** | 99.83% / 99.79% | 99.84% | 58 min |

Twelve percent more tokens than BPE 17k for the same text, close to BPE 8k, as
the measurement on 3,000 paragraphs had it (§33.62). At a fixed 10k steps its
models therefore read about 11% less text than BPE 17k's — which P126, its null
below BPE 17k's, has to overcome.

**Two columns that look like differences and are not.** Alignment is the
highest of the six only because the extra tokens all fall inside words, which
always align; the tokens that do not, which hold only whitespace, number about
614 in validation, within the other BPEs' 581–1,284. Within-word agreement on
dependency and part of speech is the lowest of the six for the reason recorded in
§33.22: after a double space, spaCy's whitespace token lends its labels to the
next word's first token (part of speech `none`, dependency `function`). That
happens to the same words under every BPE — 356–358 word tokens in validation
have no part of speech under BPE 4k, 8k, 32k and this one alike, 0.24% of the
words, and about 53,000 in training, 0.31% — and it shows as disagreement only
where such a word has more than one token: 60 of them under BPE 32k, 111 under
8k, 177 under 4k, 227 under the tokenizer that splits off nearly every ending.
The Universal Dependencies features — case, and everything D7 compares — skip
whitespace tokens and are 100% constant.

**The pilot's estimate missed the same way for every condition.** It put this
train split at 36.3M tokens and 2.2 hours of tagging; it was 42.6M and 58
minutes. All five conditions logged at 0.05 missed alike: tokens 14–15% low,
time 1.7–2.3 times high. The token count assumed that validation, 1 document in
2,000, holds 1 byte in 2,000 × 0.05 of the training slice's; it holds 1 in 118,
not 1 in 100 — a property of the documents it drew, the same under every
tokenizer. The time was the pilot's own rate, which includes reading through
the whole corpus to find the validation documents; tagging follows the bytes,
not the tokenizer. `prepare_condition.py` now takes the slice's bytes from
another tokenizer's cache of it (here 43.0M tokens, 1% high), each new cache
records how long its tagging took, and the time given is the latest such
build's or, failing one, the pilot's rate as an upper bound (the five passes took
37–50% of it). The planning figure for a condition at 0.05 is 1–1.5 hours of
CPU: 58 minutes on the morning of 7 October, 1 h 14 min to 1 h 31 min during the
night chain, beside its GPU runs.

**The tagger, misnamed in every Ukrainian record** (G-74). A run's record carries
the environment it ran in — library versions, the GPU, the spaCy model and its
version — because none of it can be recovered later (`morphlm/provenance.py`).
The code looked only for the English model, so every Ukrainian run and audit says
`en_core_web_sm` 3.8.0. The tagger actually used is in each run's config
(`uk_core_news_sm`); its version was never written down. Records now name the
tagger the run or audit used, with its version, and list every project tagger
installed; the exported environment table takes an old run's tagger from its
config and marks the version unrecorded. The version that tagged every Ukrainian
cache so far can still be read off the GPU machine if it has not been
reinstalled since: `scripts/ops/tagger_record.py` records each tagger's version
and install time against the earliest evidence of each corpus's caches — their
files, the cache checks and derivations recorded for them, their runs — and says
whether that installation tagged them all. G-74 stays open until it has run.

P126 and P127 wait for the condition's three cells at seed 0 (~35 minutes).

### 33.64 A long chain for 7 October: the sixth tokenizer at four seeds, and D7 at four

**In sync** (7 October, 12:31: 8–9 hours before the results come
back). The snapshot taken after the last archive equals the working copy file
for file (`tree_diff.py`); its results end with the night of 6 October's last
run, so nothing has trained since, and its audits hold results_73's.

**What fills the hours** (`scripts/chains/long_2026-10-07.sh`), in the queue's
order of priority. First the sixth tokenizer: its null, case and case's word form
at seed 0, then at seeds 1–3 — the comparison the thesis turns on, morphology in
the tokenizer against morphology in the embedding, and the two together. Then
D7's seeds 1–3 under Unigram 17k and BPE 4k, 8k and 32k, with BPE 17k's word
form: whether anything orders by ending exposure, and whether Unigram's early
lead is real. All at ~38M tokens and a 10k budget, as at seed 0. By the minutes
6 October's runs took — BPE 4k 6.8 per run, 8k 8.0, Unigram and BPE 17k
10.8–11.8, 32k 16.1 — about 9.2 hours in all.

**The promotion rule.** The sixth tokenizer's condition, smoke test and
derivations have run on the GPU machine; its training runs have not. So
they go first, each tokenizer behind its own readiness check and smoke test, and
a tokenizer that fails either has its runs skipped while the chain goes on with
the established ones.

**A time box** (`END_BY`). The chain starts a tokenizer's unfinished runs for a
seed only if all of them would end before `END_BY`, by those minutes with one to
spare: a null never runs without its case, and what does not fit is left whole
for the next chain, which skips what is done. Started at about 12:45 with
`END_BY` eight and a half hours on, everything should fit except BPE 32k at
seed 3.

**Rehearsed.** With the stub `python` (which now takes several failure patterns
and can print what a chain greps for): 51 runs in the intended order; a failed
smoke test and an unprepared condition each skip only their tokenizer; a
deadline leaves whole sets, never a null alone; half-finished sets resume with
only their missing runs; a finished chain starts nothing; an `END_BY` that
`date` cannot read stops it before anything runs. For real on the toy corpus: the
endings-apart cells trained at seeds 1–3 through the wrapper's cache;
unprepared conditions were skipped; started again after two finished runs were
removed, it re-ran exactly those two; and every run's record now names its
tagger, `uk_core_news_sm` 3.8.0 with the taggers installed, and its tokenizer's
rule and fingerprint.

Predictions (P126 and P127 stand, for seed 0):

- **P128** (~45%). Morphology in the tokenizer, by itself. Paired by seed — the
  two vocabularies are the same size, so the two nulls start from the same
  weights — the endings-apart null ends below BPE 17k's (1.0891, 1.0910, 1.0874,
  1.0911) by more than 0.001 on average over seeds 0–3. Against it: in the same
  10k steps it reads 11% less text (§33.63). For it: Unigram 17k matched BPE 17k
  while reading 15% less (§33.58).
- **P129** (~65%). Morphology in the embedding still helps where the tokenizer
  exposes every ending: case beats its own null under endings apart at all four
  seeds (under BPE 17k: +0.0018 to +0.0051, 4/4).
- **P130** (~60%). Unigram's early lead was seed 0's: over seeds 0–3 its case gain
  at null BPB 1.40 lies within 0.003 of BPE 17k's at the same level (seed 0:
  +0.0112 against +0.0046).
- **P131** (~65%). At this size the sentence's case beats the word form's: over
  seeds 0–3 case's gain exceeds its word form's by more than 0.001 under at least
  five of the six tokenizers (seed 0: by 0.0021–0.0033 under all five) — although
  at 151M tokens and 50k steps the lead was +0.0006 ± 0.0004 (§33.30). It is the
  context-aware half of the thesis's question, asked where features still help.

### 33.65 Morphology in the tokenizer helps — and case helps more where endings are exposed

**The chain** (7 October, results_74). 48 runs in 8 h 20 min, every smoke test
passed, nothing failed; `END_BY` left exactly what the estimate said, BPE 32k at
seed 3 (three runs, ~49 minutes). The endings-apart runs took 10.9–11.7 minutes
each, as BPE 17k's. The numbers below are `scripts/analysis/tokenizer_table.py`'s,
from the run records (~38M tokens, 10k steps; gains over each tokenizer's own
null, paired by seed).

**The tagger, recorded** (G-74 closed). `tagger_record.py` on the GPU
machine: `uk_core_news_sm` 3.8.0 was installed on 28 September at 22:39 UTC, eight
minutes before the first Ukrainian cache was checked, and has not been
reinstalled since — it tagged every Ukrainian cache. `en_core_web_sm` 3.8.0,
installed in April, tagged every English one.

**Morphology in the tokenizer, by itself.** BPE 17k that keeps endings apart and
plain BPE 17k have the same vocabulary size (17,004 with the special tokens), the
same 5,043,168 parameters and, at each seed, the same initial weights. Their
nulls:

| seed | BPE 17k | endings apart | difference |
|---|---|---|---|
| 0 | 1.0891 | 1.0714 | 0.0177 |
| 1 | 1.0910 | 1.0715 | 0.0195 |
| 2 | 1.0874 | 1.0739 | 0.0135 |
| 3 | 1.0911 | 1.0722 | 0.0188 |

**+0.0174 ± 0.0014 BPB, 4/4 — 1.6%**, although in the same 10k steps the
endings-apart model reads 11% less text (§33.63). Unigram 17k, which cuts words
differently and into even more tokens, is level with BPE 17k (−0.0005 ± 0.0004):
it is where the cuts fall, not how many there are. **P126 right; P128 right.**

**Is it the same text?** Bits per byte compare tokenizers only if each scores the
same text. Two reasons it does: a tokenization is a function of the text, so
distinct texts get distinct token sequences and the model's probability of each
text's sequence is a valid code length, whatever the cuts; and none of the six
tokenizers normalizes text — the endings-apart one removes only its own mark,
which never occurs in text (a text holding it is refused). On the toy corpus the
validation streams of four tokenizers, endings apart among them, decode to the
same text byte for byte (one hash). `scripts/audits/stream_roundtrip.py` checks
the real caches the same way, in seconds; it refuses a stream missing a single
token. Not yet run on the GPU machine: until it passes, the 1.6% is open
(G-75).

**Morphology in the embedding, on top** (end of the budget):

| tokenizer | null | case | word form | case over word form | case at step 2,000 |
|---|---|---|---|---|---|
| BPE 4k | 1.2233 | +0.0044 ± 0.0004 (4/4) | +0.0014 ± 0.0004 | +0.0029 ± 0.0006 | +0.0090 |
| BPE 8k | 1.1551 | +0.0040 ± 0.0009 (4/4) | +0.0011 ± 0.0008 | +0.0029 ± 0.0003 | +0.0071 |
| BPE 17k | 1.0896 | +0.0032 ± 0.0007 (4/4) | +0.0011 ± 0.0009 | +0.0021 ± 0.0003 | +0.0051 |
| BPE 32k (3 seeds) | 1.0448 | +0.0030 ± 0.0002 (3/3) | +0.0011 ± 0.0003 | +0.0019 ± 0.0003 | +0.0048 |
| Unigram 17k | 1.0902 | +0.0054 ± 0.0007 (4/4) | +0.0020 ± 0.0007 | +0.0033 ± 0.0001 | +0.0055 |
| **endings apart** | **1.0723** | **+0.0048 ± 0.0006 (4/4)** | +0.0017 ± 0.0007 | +0.0032 ± 0.0003 | +0.0045 |

Case beats its null under endings apart at every seed (**P129 right**), and the two
together end **+0.0222 ± 0.0017** below BPE 17k's null (2.0%). Morphology in the
tokenizer and in the embedding add up; neither makes the other redundant.

**Exposure works the other way round.** D7 was built on a hypothesis: case should
help more, and longer, where the tokenizer hides endings. Paired by seed — the
same initial weights — case gains more under Unigram 17k, which exposes 43% of
endings, than under BPE 17k, which exposes 9%: **+0.0021 ± 0.0006 (4/4)**, at the
same null BPB (1.0902 against 1.0896). Under endings apart, which exposes all, it
gains +0.0016 ± 0.0010 more (4/4), with a null 0.017 further along. Along the
law's axis (case over its null at matched null BPB):

| null BPB | BPE 4k | 8k | 17k | 32k | Unigram 17k | endings apart |
|---|---|---|---|---|---|---|
| 1.40 | +0.0091 | +0.0066 | +0.0045 | +0.0028 | +0.0094 | +0.0067 |
| 1.25 | +0.0054 | +0.0058 | +0.0049 | +0.0048 | +0.0051 | +0.0051 |
| 1.15 | — | — | +0.0045 | +0.0048 | +0.0077 | +0.0088 |
| 1.10 | — | — | +0.0037 | +0.0043 | +0.0059 | +0.0066 |

At 1.25 all six lie within 0.0006 of their mean, +0.0052: the law holds across
tokenizers there (at seed 0, endings apart +0.0040 against the five's +0.0056:
**P127 right**). Later the two tokenizers that expose endings gain again — over
BPE 17k at 1.15, paired by seed, Unigram +0.0031 ± 0.0006 and endings apart
+0.0043 ± 0.0013 (4/4 each); at 1.10, +0.0022 ± 0.0006 and +0.0029 ± 0.0009 — while
the BPE curves fall. So the law sets case's gain until null BPB about 1.25; after
that, case keeps more where endings are their own tokens. It is not the tagger's
right-context leak: case's word form, which reads no context, shows the same
pattern at about two thirds of the size — at 1.15, +0.0021 ± 0.0008 (Unigram) and
+0.0033 ± 0.0017 (endings apart) over BPE 17k, 4/4 each; at the end of the budget
only +0.0009 ± 0.0006 and +0.0006 ± 0.0013. A mechanism to test, not yet tested:
the causal feature carries the previous word's case, which governs and agrees
with the current word's ending; where the ending is its own token, predicting it
is a step the feature informs directly.

**Unigram's early lead is real** (**P130 wrong**). At null BPB 1.40 Unigram's case
gains +0.0094 against BPE 17k's +0.0045 — paired, +0.0049 ± 0.0009 (4/4) — and
every Unigram seed (+0.0084 to +0.0112) lies above every BPE 17k seed (+0.0023 to
+0.0057).

**The sentence's case against the word form's** (**P131 right**). Under all six
tokenizers case read from the sentence beats case read from the word form, by
+0.0019 to +0.0033 (4/4 each, 3/3 for BPE 32k) — where at 151M tokens and 50k
steps it led by +0.0006 ± 0.0004 (§33.30). The word form keeps about a third of
case's gain. Whether the lead is context or the tagger's right-context leak, this
does not separate (G-38): 5.0% of case values depend on the words to their
right, concentrated on prepositions, whose case is the next noun's (§33.21) — so
the shifted feature at a noun after a preposition can carry that noun's own case.
Recoding prepositions with their word form's case would remove that part of the
leak and keep the rest of the context.

**Tools.** `scripts/analysis/tokenizer_table.py` writes this section's tables from
the records (`results/audits/tokenizer_table_*.json`); `learning_law.py` draws
the six tokenizers as a fifth panel; `scripts/audits/stream_roundtrip.py` checks
that every tokenizer's stream is the same text.

**Left from the chain:** BPE 32k at seed 3, and the round trip on the real caches.
Predictions P126–P131: five right, one wrong.

### 33.66 The same text under every tokenizer — and the work moves to one machine

**G-75 closed.** `scripts/audits/stream_roundtrip.py` ran on the GPU machine's caches at
00:17 on 8 October, Kyiv time (`stream_roundtrip_20261007T211736`), after results_74
was packed and before it was read. On the validation split at fraction
0.05, where every number of §33.65 was measured, the six tokenizers' streams — from
296,325 tokens (BPE 32k) to 423,931 (BPE 4k), 363,158 with endings apart — each
decode to 1,989,866 bytes, the count recorded when its cache was built, and all six
to one SHA-256. The check could have failed: it refuses a stream missing a single
token, and a tokenizer that dropped or altered a character would have changed the
count or the hash. So each tokenizer's bits per byte in D7 is a code length for the
same text, and the 1.6% of §33.65 stands on that. No open gap could now change a
stated claim.

**The work moves to one machine** (8 October). From here the analysis runs on the GPU
machine itself — the repository, the GPU and the `morph` environment within reach. What
carried the work between two machines is retired:

- **The packs.** Before they went, every file in the 72 packs still on disk — 1,451,
  results_2 to results_74 — was compared with its copy in the project by SHA-256. None
  was missing: 1,400 were byte-identical; 26 were newer in the project (21 early run
  records later given provenance fields by `scripts/ops/backfill_results.py`, two runs
  re-run under the same id and packed again in results_32, three logs packed while
  their chain still ran); the 25 manifests held only what each run's record holds.
  Which runs each pack carried — what a citation such as "results_74" refers to — is
  kept in `docs/record/packs.md`. The packs, the packer's state and the packer
  (`pack_results.py`) went to the trash. `scripts/ops/chain_status.sh` now lists the
  runs finished in the last day instead of those not yet packed, and carries the
  packer's one safeguard, the duplicate-seed check (§33.48): each run that shares its
  label and seed with another run on disk is flagged. Over every run it flags exactly
  the two baselines G-26 re-ran on purpose, identical to the last digit (results
  §5.33). `pack.sh` stays, for snapshots of the project.
- **The offline sandbox.** Its setup script, the archive proofs (`tree_diff.py`) and the
  snapshot-based rehearsal (`rehearse_night.sh`) went to the trash. The toy corpora,
  the offline `datasets` stand-in and the stub `python` stay: a condition takes 1–1.5
  hours of tagging, so toy scale is how new data code is tested within minutes — run
  from a copy of the code, since a run writes into `results/` wherever it starts.
- **The rules** (8 October). The short, bold chains and the long, stable ones stay
  apart. Git was not used from here until 11 October — the working copy was the state —
  so a run record's `git_commit` from that period does not name its code. Nothing is
  deleted outright; what is removed goes to the trash.

From here a batch of results is cited by its chain (`long_2026-10-07.sh`), not by a
pack number. Left from §33.65: BPE 32k at seed 3.

### 33.67 A long chain for 8 October: does morphology in the tokenizer last?

**Where it points.** D7 leaves the thesis two kinds of morphology to weigh: case in
the embedding, which helps until the model has learned it and then fades (§33.31–§33.35),
and BPE that keeps endings apart, +1.6% at one budget, one width and one corpus size
(§33.65). Whether the tokenizer's gain fades like the feature's is what the comparison
turns on. Features fade along two axes — training (the law, §33.35) and width (a 4×
wider model's part-of-speech gain collapses early, §33.40, §33.60) — so the eight hours
of 8 October put the tokenizer on both.

**What the 10k records already show** (`scripts/analysis/tokenizer_table.py
--null-steps`, which now compares each null with the reference null along training as
well; `tokenizer_table_20261008T052848`; four seeds, paired). The endings-apart null's
lead over BPE 17k's is +0.0454 at step 500, +0.0240 at 2,000 — and then holds: +0.0178 at
3,000, +0.0163 to +0.0172 from 4,000 to 9,000, +0.0174 at the end, 4/4 at every step,
while the BPE 17k null falls from 1.21 to 1.09. Unigram 17k, which also cuts words into
more tokens, leads early as well (+0.0217 at step 500, +0.0176 at 1,000), and its lead is
gone by step 3,000 (−0.0004; −0.0026 at 4,000 and 5,000; −0.0005 at the end). Case over
its null fades along the same steps: under BPE 17k from +0.0052 at step 3,000 to +0.0032,
under endings apart from +0.0082 to +0.0048. Unigram's lead fades like a feature's; the
endings-apart lead stopped fading at step 3,000.

**The chain** (`scripts/chains/long_2026-10-08.sh`; ~38M tokens, the D7 condition;
pieces in priority order, 2 and 3 as decided on 8 October):

1. both nulls at twice the budget, 20k steps, seeds 0–3 (~3.1 h);
2. both nulls in a 4× wider model, width 256, 10k steps, seeds 0–3 (~2.6 h);
3. case under both tokenizers at 20k, seeds 0–3 (~3.1 h);
4. BPE 32k at seed 3, left from §33.65 (~50 min).

By recorded minutes the pieces take ~9.7 hours; in eight, pieces 1 and 2 and about half
of piece 3 fit, and `END_BY` leaves the rest whole for the next chain. A **CPU track**
beside them prepares BPE 17k with endings apart at fraction 0.2 — the 151M-token text,
for the third axis, more data, at the full 50k budget where case gave nothing, in the
next chain. The disk had 3.7 GB free; the condition takes ~8.8 GB (BPE 17k's at 0.2:
7.9 GB, its feature codes under their own key); space was freed (75 GB free).
Its pilot ran here in 71 s: the validation stream is the 0.05 one (363,158 tokens), the
gate passed, and the slice's bytes give ~171.3M training tokens; tagging at most 9.3 h
by the pilot's rate.

**Guards**, recorded before the chain ran; each skips only what depends on it:
after seed 0 of piece 1, each 20k null must end below its own 10k null at seed 0
(1.0891, 1.0714), or the rest of pieces 1 and 3 is skipped; after seed 0 of piece 2,
each 256-wide null must end below its tokenizer's 128-wide null, or the rest of piece 2
is skipped; no run starts with less than 1 GB free, and the CPU track only with 12 GB;
a smoke test per tokenizer and width; `END_BY`. None branches on an effect's size:
every piece informs whichever way the earlier ones fall, and choosing from seed 0 is a
known trap.

**Tools.** `run_ids.py --bpb` prints a finished run's bits per byte, for the guards. The
stub `python` takes a map of outputs (`STUB_MAP`), so a rehearsal can give two runs two
numbers, and a call it fails prints nothing, as a real one finding nothing would.
Rehearsed with it from a copy of `scripts/` under `scratch/`, eleven scenarios, each
checked by its number of runs and its messages: every piece in order (27 runs); each
guard failing; `END_BY` cutting piece 4 alone, and cutting everything while a guard
finds no run to compare; no disk; a finished chain starting nothing; conditions not
ready; a resumed piece; a width smoke test failing; an unreadable `END_BY`.

Predictions:

- **P132** (~70%). The tokenizer's gain lasts to twice the budget: at 20k steps, paired
  by seed, the endings-apart null ends below BPE 17k's by more than 0.010 on average,
  at every seed. For: the lead held at +0.016–0.017 from step 3,000 to 10,000 while the
  null fell 0.12 BPB. Against: with twice the training BPE 17k may learn the cut itself,
  as the model learns what features give it.
- **P133** (~55%). And the 20k lead lies within 0.004 of the 10k lead: +0.0134 to +0.0214.
- **P134** (~60%). In a 4× wider model (width 256, 10k) the endings-apart null ends below
  BPE 17k's by more than 0.008 on average, at every seed. Against: the wider model's
  part-of-speech gain collapsed (§33.40).
- **P135** (~60%). The lead at width 256 is smaller than at width 128 (+0.0174).
- **P136** (~65%). Under BPE 17k, case's gain over its null at the end of the 20k budget is
  below +0.002 (at 10k +0.0032 ± 0.0007; at 151M tokens and 20k steps −0.0003 ± 0.0004).
- **P137** (~65%). Under endings apart, case still beats its null at 20k, on average and at
  three seeds or more (at 10k +0.0048 ± 0.0006, 4/4).
- **P138** (~50%). At 20k, paired by seed, case gains more under endings apart than under
  BPE 17k by more than 0.001 (at 10k +0.0016 ± 0.0010).
- **P139** (~65%). Together, endings apart and case end below BPE 17k's null at 20k by more
  than 0.015 (at 10k +0.0222 ± 0.0017).
- **P140** (~85%). Under BPE 32k at seed 3, case beats its null by more than 0.0015 (seeds
  0–2: +0.0030 ± 0.0002).
- **P141** (~75%). The endings-apart condition at 0.2 holds 167.8M–171.2M training tokens:
  BPE 17k's 151.1M times the ratio measured at 0.05 (1.122), within 1% — the pilot's
  171.3M ran 1% high at 0.05.
- **P142** (~65%). Its tagging pass takes 3.5–5.5 h beside the GPU runs (the five passes at
  0.05 took 37–50% of their pilot-rate estimates).

P136–P139 are read once piece 3 has all four seeds, in this chain or the next.

### 33.68 Morphology in the tokenizer lasts with training, and is larger in a wider model

**The chain of 8 October** ran 22 runs in 7 h 53 min (08:33–16:26). Every smoke test
and both guards passed, nothing failed; `END_BY` left case at 20k for seed 3 and BPE
32k at seed 3 — five runs, ~1.7 h. A 20k null took 21.9–22.1 minutes, a 20k case run
22.9–23.1, a 256-wide null 19.0–19.7. Beside them the CPU track built the endings-apart
condition at 0.2 in 4 h 50 min.

**Twice the budget** (`tokenizer_table_20261008T211258`; four seeds, paired). At the
end of a 20k budget the endings-apart null ends **+0.0169 ± 0.0007** below BPE 17k's
(4/4), against +0.0174 ± 0.0014 at 10k. Along the 20k schedule the lead retraces the
10k one — +0.0454 at step 500, +0.0173 at 3,000 — then stays between +0.0154 and
+0.0170 to the end, while BPE 17k's null falls from 1.21 to 1.015 BPB. On the law's
axis the two schedules draw one flat line: about 0.017 wherever the null stands
between 1.21 and 1.015. **P132 right; P133 right.**

**A 4× wider model** *(twice as wide; §33.91)* (`tokenizer_table_20261008T211300`). At width 256 and 10k steps the
lead is **+0.0254 ± 0.0012** (4/4) — 2.6% of BPE 17k's 0.9849, where width 128 gives
1.6%. **P134 right; P135 wrong**: larger, not smaller. At matched null BPB the wider
model's lead is larger everywhere — +0.0383 at 1.18, +0.0336 at 1.11, +0.0296 at 1.04,
+0.0267 at 1.006 — against 0.016–0.017 at width 128. Unlike width 128's it falls along
training, from +0.0383 at step 2,000 to +0.0254 at 8,000, and holds there to the end
(+0.0254 at 9,000 and at 10,000), as the learning rate runs out; whether it has
settled, a longer budget at width 256 would show. The 256-wide endings-apart nulls
also agree with each other closely: 0.9590–0.9605, against 0.9826–0.9879 under BPE
17k. A result against the prediction was looked at for a defect first: the paired
256-wide runs have the same shapes (11,795,168 trainable parameters, so each seed
pairs initial weights), differ in nothing but the tokenizer's path, and no run's
validation BPB rises anywhere along training — no instability inflates the gap.

So on both axes where a feature's gain fades, morphology in the tokenizer does not: a
feature's gain falls as the null learns and collapses earlier in a wider model
(§33.40, §33.60); the tokenizer's lead holds as the null learns, and is larger in the
wider model. Unigram's early lead, which faded by step 3,000 (§33.67), is the
exception that shows it is not more tokens per word that lasts but where the cuts fall.

**Case on top at 20k** (seeds 0–2; seed 3 is left). Under BPE 17k case's gain has faded
to +0.0005 ± 0.0008 (2/3) — the law's course: on its axis +0.0032 at null BPB 1.10,
+0.0015 at 1.04, +0.0007 at 1.02, near where 151M tokens and 20k steps put it
(−0.0003, §33.58). Under endings apart it is **+0.0041 ± 0.0011** (3/3), and still
+0.0042 at null BPB 1.00. Paired by seed, case gains +0.0037 ± 0.0019 more under endings
apart (3/3; +0.0038 at matched null BPB 1.04 and 1.02), and together the two end
+0.0205 ± 0.0015 below BPE 17k's null (3/3). Where endings are their own tokens case
keeps helping after the model has learned enough for it to stop helping under BPE 17k:
the exposure effect of §33.65, growing with training. P136–P139 are read when seed 3
has run; on three seeds all four lie on their right side.

**More data, prepared.** The endings-apart condition at 0.2 holds 169,569,389 training
tokens for the same 938,814,912 bytes as BPE 17k's 151,133,566 — the ratio 1.1220,
against 1.1219 at 0.05 (**P141 right**). Its tagging pass took 4 h 43 min beside the GPU
runs (**P142 right**). Every gate passed; each word's majority case matches the case in
context for 89.75% of occurrences (90.02% at 0.05), and 3.05% of validation words are
unseen in train (5.59% at 0.05).

**Tools.** `tokenizer_table.py` now reports, beside each compared step, where the
reference null stands — the law's axis — and each tokenizer with its feature against
the reference null; it reproduces §33.65's +0.0222 ± 0.0017.

Predictions P132–P142: P132–P134, P141 and P142 right; P135 wrong; P136–P140 wait for the
five runs left.

### 33.69 Why exposure helps case: the loss by what each token is

**The question** (DIRECTIONS D7). Case gains more where endings are their own tokens:
over BPE 17k, +0.0021 ± 0.0006 under Unigram 17k and +0.0016 ± 0.0010 under endings
apart at 10k (§33.65), and +0.0037 ± 0.0019 under endings apart at 20k (three seeds,
§33.68). The feature at every token of a word wᵢ carries the case of the word before
it, wᵢ₋₁; each position predicts the next token. So it can inform two things: the rest
of wᵢ — its later tokens, predicted from inside wᵢ — and wᵢ₊₁'s first token. Under
BPE that keeps endings apart, wᵢ's ending is a later token, predicted from its stem's
position, where wᵢ₋₁'s case is in the input; a preposition's or an adjective's case
says much about it. Under BPE 17k the ending is mostly fused into wᵢ's first token,
predicted from the end of wᵢ₋₁, where the feature still describes wᵢ₋₂. If that is the
mechanism, case's gain sits on ending tokens, and the tokenizers' difference in it sits
in the words the rule cuts.

**The tool** (`scripts/analysis/loss_by_kind.py`, desk, minutes of GPU). Every saved
model of the null, case and case's word form is scored again on the validation stream,
token by token, as training evaluated it — its own context facts, the same windows —
and each target is classified by the word rule the causal shift reads (`word_spans`)
and by the ending rule's cuts in the stream's text (`mark_text`, the cuts the
endings-apart tokenizer itself makes): a word's first token, holding the ending's start
or not; a later token holding it (at or across the cut — under endings apart, the
ending itself); any other later token; punctuation and digits. Per kind, a feature's
gain over its null is counted in bits per byte, so the kinds add up to the whole gain,
paired by seed; and per word class — cut by the rule, not cut, not a word — the same
words under every tokenizer, so a difference between tokenizers is placed in the words
it comes from. Before anything is reported, each model's re-scored bits per byte must
equal its record's.

Predictions, for the 10k runs at ~38M tokens (four seeds) and the 20k runs (case at
seeds 0–2):

- **P143** (~70%). Under endings apart, ending tokens carry a share of case's gain at
  least 1.5 times their share of the targets. Against: the feature may help mostly with
  the next word's first token, through the chain of agreement.
- **P144** (~65%). More than half of case's extra gain under endings apart over BPE 17k at
  10k comes from words the rule cuts.
- **P145** (~60%). More than half of case's extra gain under Unigram 17k over BPE 17k at
  10k comes from words the rule cuts.
- **P146** (~55%). At 20k, more than half of the extra gain under endings apart comes from
  words the rule cuts, and within them more from ending tokens than from stem tokens.
- **P147** (~90%). Every re-scored model reproduces its recorded bits per byte within
  0.00001.

### 33.70 Case helps by predicting endings — and the exposure effect lies elsewhere

**The tool held** (`loss_by_kind_20261008T212841`, 10k; `loss_by_kind_20261008T213019`,
20k; 48 s and 21 s of GPU). All 50 models re-scored to their recorded bits per byte
within 1.9 × 10⁻⁸ (**P147 right**), and their kinds add up to every recorded gain. The
mapping from the text's cuts to tokens reproduces the tokenizers' measured exposure on
the validation text: a token starts at 9.0% of the cuts under BPE 17k, 44.2% under
Unigram 17k and 100.0% under endings apart (§33.62: 8.8%, 43.3%, 100.0%), with the same
71,623 cut words of 147,694–147,733 under all three.

**Case helps by predicting endings, under every tokenizer** (10k, four seeds, gains in
bits per byte over each tokenizer's null):

| | BPE 17k | Unigram 17k | endings apart |
|---|---|---|---|
| tokens holding an ending's start, share of targets | 22.0% | 18.9% | 19.7% |
| their part of case's gain | +0.0035 ± 0.0005 (108%) | +0.0032 ± 0.0002 (60%) | +0.0028 ± 0.0002 (58%) |
| cut words | +0.0036 ± 0.0005 | +0.0041 ± 0.0003 | +0.0040 ± 0.0004 |
| uncut words | −0.0001 ± 0.0002 | +0.0013 ± 0.0003 | +0.0009 ± 0.0002 |
| not words | −0.0003 ± 0.0002 | −0.0001 ± 0.0002 | −0.0000 ± 0.0002 |
| case's whole gain | +0.0032 ± 0.0007 | +0.0054 ± 0.0007 | +0.0048 ± 0.0006 |

Under endings apart the ending tokens carry 2.9 times their share of the targets
(**P143 right**). Under BPE 17k the whole gain is on endings that fall in a word's later
token (+0.0032, 4/4); where the ending is fused into the word's first token — predicted
from the end of the word before, where the feature still describes the word two back —
case gives almost nothing (+0.0003). The causal structure of §33.69, seen in the loss.

**But the exposing tokenizers' extra gain is not in the cut words.** Cut words gain about
the same under all three (+0.0036 to +0.0041). The extra over BPE 17k — +0.0016 ± 0.0010
under endings apart, +0.0021 ± 0.0006 under Unigram — comes from words the rule does not
cut: +0.0010 ± 0.0003 and +0.0014 ± 0.0004 (4/4 each), mostly their first tokens; cut
words hold 25% and 24% of it (**P144 wrong; P145 wrong**). At 20k (three seeds) the extra,
+0.0037 ± 0.0019, is spread: cut words +0.0014 ± 0.0012, uncut words +0.0012 ± 0.0007,
non-word targets +0.0011 ± 0.0001 — 37% in cut words (**P146 wrong**; within cut words the
endings do carry more than the stems, +0.0025 against +0.0009).

**What fades under BPE 17k is the rest of the stream.** At 20k case still helps BPE 17k's
endings (+0.0021 ± 0.0006, 3/3) but costs everything else — uncut words −0.0008 (0/3),
non-word targets −0.0007 (0/3) — so its whole gain is +0.0005. Under endings apart those
costs do not appear (uncut +0.0004, non-word +0.0004 at 20k). So the exposure effect is
not that case helps exposed endings more; it is that under BPE 17k the feature becomes a
cost away from the endings, and under the exposing tokenizers it does not.

**The word form tells the two apart.** Case read from the word form alone helps BPE 17k's
endings almost as much as case from the sentence (+0.0021 against +0.0032), but under
endings apart barely (+0.0005 against +0.0028). There the previous word's ending is itself
a token in the input, and what the form says of its case is in plain view; what the
sentence's case adds on the endings is what the form does not show — case syncretism
resolved by context, or the tagger's right-context leak (G-38): the leak sits on
prepositions, whose case is the next noun's, and the next noun's ending is exactly such a
target. Splitting the ending targets by whether the word before is a preposition would
place it.

Predictions P143–P147: P143 and P147 right; P144, P145 and P146 wrong.

