# Research journey — Part 9: Ukrainian, and Before the Model Has Learned It

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.20–§33.32. [← Part 8](8_the_controlled_programme.md) · [index](README.md) · [Part 10 →](10_the_learning_law.md)

**Established:** Ukrainian case, against its same-shape null, adds nothing at the end of training,
as in English (§33.30), and every informative feature speeds early learning in both languages and
loses the advantage by the end (§33.28). Case helps where the model has not yet learned it, with
little data (§33.31) or on a short budget with all the data: what keeps a feature useful is how far
the model has learned, and read from the word form alone case keeps part of the gain, so it is case,
not the tagger's right context (§33.32). A gate on the word rule caught words glued by split
punctuation and paragraphs glued together (§33.21–§33.22). The Ukrainian gains were later restated
smaller against a constant null (§33.56).

---

## 33. The Agreement Test — continued

### 33.20 What we know, and Ukrainian begins

`docs/FINDINGS.md` now states the thesis's claims one by one — each with its
evidence, its firmness and the sections that reached it — as the bridge from
this record to the writing.

Ukrainian starts in day blocks (`docs/reference/ukrainian_pipeline.md`). Its first
measurement came from the toy corpus while the causality check was being
extended to it: *на* is tagged accusative or locative **by the noun that
follows** — so shifted case can hand the model the case of the very word it
is predicting, the English tag leak in a stronger form. `ucaselex`, each form's
majority case, is its leak-free control, built into the first family from the
start. Predictions P37–P40 are recorded before any Ukrainian result.


### 33.21 A gate that held: words glued by split punctuation

Blocks U0 and U1 ran on 29 September (`docs/reference/ukrainian_pipeline.md`). The dump
gave a train fraction of 0.2; the tokenizer trained losslessly; the pilot
aligned 99.80% of tokens and `ucase` fired on 73% of word tokens. But every
feature was only **97–98% constant within words**, and the go/no-go rule
demanded 100%, so the block stopped before an 8-hour build.

Real treebank sentences reproduced it at 0.08%; their violating words showed
the mechanism. A ByteLevel token can hold only the continuation bytes of a
character, and the decoded rule counted every such fragment as part of a
letter — so split quotes, dashes and ellipses, frequent in Wikipedia, were
glued into their neighbours: *„Порше“*, *бігла—весь*, *день…*. The fix gives
a fragment the class of the token before it, and classes a token holding a
split character's first byte by its Unicode block. The acceptance test —
**words are exactly the text's letter runs** — failed on the old rule and
passes on the new for every treebank paragraph but one (a split ×), including
through a tokenizer that splits most letters; the model's GPU word finder
agrees with numpy's at every one of 208,944 word positions; English tables and
words are unchanged. The rule groups tokens but never touches the cache, so
nothing is rebuilt.

**P37 was wrong**: 5.0% of case values depend on the words to their right,
fewer than English tags' 7–8% — Ukrainian forms carry their case. The changes
concentrate on prepositions (*на, з, за, у, в, під*), whose case is the next
noun's: small in share, placed exactly where a shifted feature would carry the
target's form. `ucaselex` stays in the family.


### 33.22 The real cause: paragraphs glued together

U1b refuted P41: under the fixed word rule the pilot still agreed only 97–98%
within words — the same numbers to the hundredth. The pilot holds 487 fragment
tokens; §33.21 found a real mechanism and mistook it for the one at work.

The pattern pointed elsewhere: violations sat on nominal features (gender
3.2%, case 2.4%) and hardly on verbal ones (tense 0.1%). Neither the
GPU machine's exact library versions nor Wikipedia-like characters (stress marks,
IPA, no-break spaces, units) reproduced it; Wikipedia's **structure** did. The
corpus loader split articles into paragraphs and dropped the newline; the
stream builder strips and concatenates rows. A heading or list item without
final punctuation — *Історія*, *Примітки*, a list of names — therefore ran
into the next paragraph's first word: *ІсторіяМісто*, one "word" across two
paragraphs, nouns on both sides. Heading-rich test articles gave 92.6% within
words under the old loader and 100.00% under the fixed one, which gives each
Ukrainian paragraph its newline back. English never met it — WikiText's
paragraphs end in punctuation and its headings in `=` — and its streams and
keys are untouched. Unlike §33.21, this fix changes the data: Ukrainian
streams gain one newline per paragraph, and the key gains `:nl` so the glued
pilot cannot be reused.

The lesson for the method: to reproduce a failure, reproduce the data's
structure, not only its alphabet.

**P42 right.** The rebuilt pilot: every Ukrainian feature 100.00% constant
within words; the stream grew by 11,251 tokens and exactly 11,251 bytes, one
newline per paragraph; 5,004 glued words came apart. The last 0.08%, on `dep`
and `pos` only, is a double space: spaCy's whitespace token lends its `SPACE`
tag to the first token of the next word — verified, and left alone because
changing it would change English alignment.


### 33.23 Two tracks in one night

The Ukrainian train cache takes about nine hours of CPU and none of GPU, so the
night of 29 September runs two tracks (`scripts/chains/night_2026-09-29.sh`):
the cache, its derivations and a health gate on the CPU, while the GPU gives
English cells resting on two seeds their seeds 2 and 3; then, when both are
done, the first Ukrainian family at seed 0 — gated, because Ukrainian training
has not yet run on the GPU machine. The gate: both derivations pass
their verification, `ucase`, `mconst` and `ucaselex` are present and at least
99.9% constant within words on train and validation (a new exit-code mode of
`check_cache.py`), and the smoke test passes. Rehearsed twice offline:
every branch with the stub (English always runs; each failure stops its track
and skips U3; the chain always completes), and the Ukrainian half for real on
treebank text — cache, derivations, gate, smoke, four runs, tables.

Predictions for the English track, recorded before it runs:

- **P29** (recorded 28 September, not yet run): at seeds 2–3, neither the zero
  slot nor shuffled structure beats the null by more than 0.002 on average.
  Tonight tests those two; `pos` follows another night.
- **P43.** At seeds 2 and 3, as at 0 and 1, both position controls fall short
  of the zero slot they are matched with: the empty slot keeps under half of
  the zero slot's gain over the baseline, position-only between a quarter and
  three quarters of it.

The Ukrainian predictions P38–P40 stand as recorded (§33.20).


### 33.24 The first Ukrainian family: early help, late cost

The night of 29 September completed both tracks (results_50). The Ukrainian
train cache: 151.1M tokens in 5 h 22 min, 99.76% aligned. The lexicon: 2.65M
word types; a form's majority case matches its in-context case for **89.74%**
of occurrences, so **P38 was wrong** (predicted under 85%) — Ukrainian case
is less ambiguous than assumed. All nine known words verified; the gate passed
on both splits.

**The first Ukrainian family, seed 0**, the gap to the null (+ = better):

| step | 2k | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|
| contextual case (`ucase`) | **+0.0074** | +0.0039 | −0.0001 | −0.0013 | −0.0017 | −0.0022 | **−0.0025** |
| lexical case (`ucaselex`) | +0.0051 | +0.0015 | −0.0026 | −0.0025 | −0.0028 | −0.0032 | −0.0034 |
| the baseline | −0.0031 | −0.0055 | −0.0049 | −0.0054 | −0.0045 | −0.0050 | −0.0052 |

**P39 and P40 were wrong**: case does not end above the null, and lexical case
ends further below it than ±0.002. By the reading rule recorded with them,
neither gains — so at this seed Q2's "no" extends to a fusional language. The
shape is new: case **speeds early learning** — the model is handed what it has
not yet learned from word forms — the advantage is gone by step 10k, and a
deficit grows to the end, the fade English identity-like features showed.
The architecture effect replicates: the null leads the baseline by about
0.005 from step 5k on (+0.0052, 0.55%). Case's late cost sits almost entirely
on mid-frequency tokens (4k–113k training occurrences: ~99% of the gap for
contextual case), where the architecture effect is spread evenly. One working
hypothesis, not yet a finding (G-65): an informative code occupies the slot's
free dims, which the architecture effect uses as a clean position channel.
Contextual case ends +0.0009 above lexical case — no sign yet of the tagger's
right context paying off.

**English at four seeds.** **P29 right**: the zero slot sits at the null
(−0.0006 ± 0.0013); shuffled structure stays under 0.002 (+0.0015 ± 0.0005)
though positive at every seed — and real structure does no better than its
shuffled version (−0.0012 ± 0.0008). **P43 half right**: both position controls
fall short of the zero slot at every seed (−0.0082 ± 0.0012 without position
in the slot, −0.0062 ± 0.0022 without it in the token dims), but the shares of
the gain each keeps did not replicate (59% and −93%; 40% and 6%) — ratios over
a baseline outside the family. What stands: the zero slot's +0.0105 ± 0.0035
needs position in both places; the slot adds a channel free of token identity,
it does not replace the usual one.

Predictions for the Ukrainian seeds 1–3 (`scripts/chains/night_2026-09-30.sh`):

- **P44.** Over seeds 0–3 both case features end below the null on average,
  and neither ends above it at more than one seed.
- **P45.** The early advantage replicates: at step 2,000 contextual case is
  ahead of the null at every seed.
- **P46.** The null beats the Ukrainian baseline at every seed, by at least
  +0.003 on average.
- **P47.** Contextual and lexical case end within ±0.002 of each other on
  average: the tagger's right context buys no final advantage.


### 33.25 A day for G-65: does case cost the position channel?

At seed 0 Ukrainian case ends below its same-shape null (§33.24). One
hypothesis (G-65): an informative code occupies the slot's free dims, which
the architecture effect uses as a clean position channel (§33.19, confirmed at
four seeds in §33.24). Two designs test it on 30 September in day blocks, each
a matched pair — identical initial weights, checked tensor by tensor:

- **Two slots**: a zero slot (a guaranteed clean channel) beside case, against
  a zero slot beside the null: `mzero,ucase` against `mzero,mconst`.
- **Additive**: case added to the full embedding, where it occupies no dims of
  its own, against the null added the same way: `ucase-ADD` against
  `mconst-ADD`.

Both rest on one new derivation — the zero slot's codes derived from the
null's file, so a single file holds `ucase`, `mconst` and `mzero` — rehearsed
end to end on treebank Ukrainian, with smoke tests passing in both designs.
The rehearsal exposed a derivation that could read its own previous output as
its source; derivations now never do.

Predictions, with the reason I lean against my own hypothesis — English part
of speech carries as much information in the same slot and costs nothing
(+0.0002 against its null):

- **P48** (~55%). Beside a zero slot, case still ends below its null by at
  least half its single-slot deficit (≤ −0.0012). If it ends within 0.001 of
  the null instead, G-65's hypothesis gains its first support.
- **P49** (~60%). Added to the full embedding, case ends below the additive
  null by at least 0.001.
- **P50** (~80%). At step 2,000, case leads its null in both designs: the
  early help does not depend on where the feature enters.


### 33.26 D2: with a clean channel beside it, case gains

Block D2 (results_51), seed 0, case minus its matched null:

| design | 2k | 10k | 30k | end |
|---|---|---|---|---|
| one slot (§33.24) | +0.0074 | −0.0001 | −0.0017 | −0.0025 |
| **two slots: a zero slot beside** | +0.0069 | +0.0046 | +0.0016 | **+0.0017** |
| additive | +0.0095 | +0.0040 | +0.0006 | −0.0013 |

**P48 was wrong** — I leaned against G-65's hypothesis, and with a clean
position channel guaranteed, case keeps a lasting gain. **P49 right** (−0.0013,
barely past −0.001). **P50 right**: the early help appears in every design.

The frequency buckets show where. On mid-frequency tokens (4k–113k training
occurrences), the sign flips with the design: case costs −0.013 to −0.026 nats
per token in one slot, **gains +0.006 to +0.011 beside a zero slot**, and costs
in the additive design as in one slot. The refined hypothesis: case competes
with position wherever the two share dims; a protected position channel
removes the conflict. First support, at one seed.

One caution. Across architectures — different shapes, so different initial
weights — the two-slot null (0.9468) sits well below the one-slot null
(0.9424), while two-slot case (0.9451) sits where one-slot case did (0.9449).
Part of the matched gain may be a weak two-slot null rather than a stronger
case; cross-family comparisons need more than one seed.

Block D3 replicates the pair at seeds 1 and 2. Predictions:

- **P51.** Case beside a zero slot ends above its matched null at both new
  seeds.
- **P52.** At both, it gains on the mid-frequency tokens (4k–113k) as a whole.
- **P53.** Over the seeds both architectures share after tonight (0–2), the
  two-slot null ends at least 0.002 worse than the one-slot null: a second
  free slot that holds nothing costs Ukrainian something.


### 33.27 D3: seed 0 was the favourable draw

Block D3 (results_52) gave the two-slot pair seeds 1 and 2. Case minus its
matched null:

| seed | step 2k | end | mid-frequency tokens (nats/token) |
|---|---|---|---|
| 0 (D2) | +0.0069 | +0.0017 | +0.0087 |
| 1 | +0.0087 | +0.0012 | +0.0009 |
| 2 | +0.0079 | **−0.0018** | **−0.0133** |

**P51 and P52 were wrong**: seed 2 ends below its null and its mid-frequency
gain reverses. Over three seeds the pair is indistinguishable,
**+0.0003 ± 0.0011** — the "first support" of §33.26 was one favourable seed,
and is withdrawn. What replicates is the early help, +0.007 to +0.009 at step
2,000 at every seed. A lesson for the Ukrainian runs: their matched
differences spread more than English ones (sd 0.0019 here, about 0.001 in
English), so no Ukrainian result is read from one seed.

G-65 narrows to what the design can still answer: does a zero slot remove the
single-slot deficit, even without producing a gain? That is a difference of
two matched differences, and tonight's single-slot seeds 1–3 supply the other
half. An English version was considered and set aside: the only English
feature with a clear single-slot deficit (inflection bundles, 72 classes) does
not share the null's parameter shapes, and those that do show no deficit.

Block D4 completes the two-slot pair at seed 3 and gives the additive arm its
seed 1. Predictions:

- **P54.** Over seeds 0–3, case's matched gap is larger beside a zero slot
  than in one slot — the difference of differences is positive on average.
- **P55.** Over seeds 0–3, case beside a zero slot ends within ±0.002 of its
  null on average: the zero slot removes the cost, and case still adds nothing.
- **P56.** At seed 1, added to the full embedding, case again ends at least
  0.001 below its null.


### 33.28 D4, and the one effect every informative feature shares

Block D4 (results_53): beside a zero slot, case at seed 3 ends +0.0035 over
its null — over four seeds **+0.0011 ± 0.0011** (3 of 4 positive), so **P55
right**. Added to the full embedding, case at seed 1 ends +0.0001, so **P56
wrong**; over two seeds −0.0006 ± 0.0007. P54 waits for tonight's single-slot
seeds.

Reading the trajectories across both languages, one pattern holds without
exception. Each informative feature, against its matched null:

| feature | step 2,000 | end |
|---|---|---|
| English part of speech (2 seeds) | +0.0132, +0.0170 | −0.0004, +0.0008 |
| English lexical part of speech (2) | +0.0097, +0.0143 | +0.0023, −0.0037 |
| English inflection bundles (2) | +0.0085, +0.0102 | −0.0074, −0.0051 |
| English frequency band (2) | +0.0038, +0.0025 | +0.0004, +0.0009 |
| English structure (4) | +0.0002 to −0.0027 | −0.0009 to +0.0015 |
| Ukrainian case, every design (7 runs) | +0.0058 to +0.0114 | −0.0025 to +0.0035 |

**Every feature that informs the next word — part of speech, inflection,
case, frequency — speeds early learning, at every seed in both languages; the
advantage is gone by the end of training in every case.** Structure, which
says little about the next word, gives none. The information is real and the
model uses it; a 5M-parameter model on 100M+ tokens simply learns it for
itself within the first ~10k steps. This is the strongest positive statement
the project can make about linguistic features, and it points to where they
might still pay: where the model cannot learn the information itself — little
data or little training — the regime in which the audited positive results
cluster (low-resource, morphologically rich languages). That is a testable
hypothesis for the next day blocks.


### 33.29 Toward the demo: little data

The thesis demo is on 1 October at 18:00. The last GPU hours before it: the
follow-on chain (`scripts/chains/night_2026-10-01.sh`), which starts itself
when the chain of 30 September ends (~midnight) and runs ~5 h; then one
morning chain ending by ~14:00; then figures and tables.

The one experiment that could still turn Q2 positive follows from §33.28:
every informative feature helps early and loses the advantage once the model
learns the information itself — so where it cannot, with little data, the
advantage might survive to the end. The follow-on chain builds two
little-data Ukrainian caches — corpus fractions 0.002 and 0.01, ~1.5M and
~7.6M train tokens against 151M, the validation split unchanged — and trains
the null and contextual case at both sizes, seeds 0–3, for 10k steps with the
cosine over 10k (best step reported, so over-fitting is read at its best
point). Gated as before: health gate on both splits, smoke test. Beside it,
English inflection bundles get seeds 2 and 3 for the early-help record.
Rehearsed with the stub on every branch — including waiting for the previous
chain, where a substring match would have waited forever on anything showing
the file's name, so the pattern is anchored — and for real on treebank
Ukrainian at a fractional size.

Predictions:

- **P57** (~50%). At ~1.5M train tokens, contextual case ends above its null
  at three or four of four seeds, by at least +0.002 on average.
- **P58** (~65%). At ~7.6M tokens its average gain is smaller than at ~1.5M.
- **P59** (~80%). English inflection bundles at seeds 2 and 3 lead the null
  at step 2,000 and end below it, as at seeds 0 and 1.


### 33.30 Ukrainian at four seeds, and a chain that ran twice

The chain of 30 September (results_54) completed the Ukrainian family and
English part of speech at four seeds.

| comparison, seeds 0–3 | values | mean | verdict |
|---|---|---|---|
| contextual case over the null | −0.0025 −0.0021 +0.0006 +0.0005 | −0.0009 ± 0.0008 | P44 narrowly wrong |
| lexical case over the null | −0.0034 −0.0027 +0.0012 −0.0009 | −0.0015 ± 0.0010 | |
| contextual over lexical case | +0.0009 +0.0006 −0.0006 +0.0015 | +0.0006 ± 0.0004 | **P47 right** |
| case at step 2,000, over the null | +0.0074 +0.0053 +0.0050 +0.0064 | **+0.0060 ± 0.0005** | **P45 right** |
| the null over the baseline | +0.0052 +0.0014 +0.0020 +0.0006 | +0.0023 ± 0.0010 | P46: 4/4, but under +0.003 |
| case beside a zero slot minus case in one slot | +0.0042 +0.0033 −0.0025 +0.0029 | +0.0020 ± 0.0015 | P54 right in sign, within noise |
| second empty slot's cost | −0.0044 +0.0040 −0.0006 +0.0001 | −0.0002 ± 0.0017 | P53 wrong |
| English part of speech over the null | −0.0004 +0.0008 +0.0013 +0.0030 | +0.0012 ± 0.0007 | **P29 right** |
| English part of speech at step 2,000 | +0.0132 +0.0170 +0.0119 +0.0138 | **+0.0140 ± 0.0011** | |

**Q2 in Ukrainian: no.** Case — from the sentence or from the word form —
adds nothing at the end of training, as in English; contextual case beats
lexical case by only +0.0006 ± 0.0004, so the tagger's right context buys
nothing (G-62 closed). The single-slot deficit that motivated G-65 was mostly
seed 0 (−0.0009 ± 0.0008 over four seeds), and the zero slot's relief
(+0.0020 ± 0.0015) is within noise: G-65 closes as not established. What
four seeds make firm is the early help — case ahead of its null at step 2,000
at every seed, part of speech likewise in English — and the architecture
effect in Ukrainian, positive at every seed but a quarter of its English
relative size.

**A chain that ran twice.** The follow-on chain was started twice; both
copies waited, both began at 00:02, and for 45 minutes trained the same run
into the same folder at half speed, the two CPU tracks building the same
caches concurrently. A status check caught it — two chain
processes, two identical runs, 7 GB on the GPU, a doubled log — and one copy
was frozen, its children killed, then removed; the caches were re-verified
(atomic writes of identical content: all gates pass). Two tools now make the
mistake impossible and the check one command: every chain sources
`scripts/chains/guard.sh`, a lock held by the chain and everything it starts,
so a second copy refuses to start; and `scripts/ops/chain_status.sh` shows every
chain, flags a duplicate (a chain's own background track is not one), the
run in training, the GPU and the runs finished but unpacked — both tested on
tonight's exact situations.


### 33.31 With little data, case helps

The follow-on chain (results_55) ran the little-data family: the null and
contextual case at ~1.5M and ~7.6M Ukrainian train tokens, seeds 0–3, 10k
steps with the cosine over 10k, scored at each run's best step.

| train tokens | case over its matched null | gap at 2k → later | best step |
|---|---|---|---|
| **~1.5M** | **+0.0096 ± 0.0016** (4/4; 0.77%) | +0.0079 → +0.0097 at 4–5k, held | 4,500–5,000 (over-fits after) |
| **~7.6M** | **+0.0051 ± 0.0002** (4/4; 0.46%) *(the four runs give +0.00516, +0.0052 to four decimals, as §33.32's table has it; annotated 19 October 2026)* | +0.0080 → +0.0056 at 8k, shrinking | 9,999 (still improving) |
| 151M (50k steps, §33.30) | −0.0009 ± 0.0008 | +0.0060 → gone by ~10k | — |

**P57 and P58 right.** This is the project's first positive answer to Q2,
and against the same-shape null, so it is information, not architecture. The
trajectories show the mechanism of §33.28 directly: with abundant data the
model learns case and the feature's advantage disappears; with 7.6M tokens it
learns it slowly; with 1.5M it never catches up. The gain is spread over
every token-frequency band — not concentrated where the tagger's right
context could leak (prepositions). **P59** holds at seed 2 (English
inflection +0.0065 at step 2k, −0.0012 at the end).

Two alternative readings remain, and the day chain of 1 October
(`scripts/chains/day_2026-10-01.sh`, the last before the demo) tests both:
the tagger's right context, with lexical case at both little-data sizes —
its lexicon now built from the full corpus (`--lexicon-fraction`), so a small
corpus does not starve it of coverage (sizes are nested, so every
little-data training word is covered); and short training, with all the data
at the same 10k steps. It adds the middle of the curve (~38M tokens) and the
baseline at little data. Rehearsed on every branch with the stub — including
the guard refusing a second copy — and for real on treebank Ukrainian.

- **P60** (~65%). Lexical case beats the null at three or four of four seeds
  at both little-data sizes, with at least half of contextual case's gain:
  the gain is case, not right-context leakage.
- **P61** (~65%). With all the data and the same 10k steps, case ends within
  ±0.002 of the null: the gain belongs to scarce data, not to short training.
- **P62** (~70%). At ~38M tokens case's gain lies between zero and its
  ~7.6M gain (+0.0051).
- **P63** (~55%). At both little-data sizes the null beats the baseline on
  average.


### 33.32 Not scarce data — an unfinished lesson

The day chain of 1 October (results_56) ran every control, 32 runs, and
finished at 10:53. Case over its matched null, 10k steps, 4 seeds each:

| | contextual case | word form only | null over baseline |
|---|---|---|---|
| ~1.5M tokens | +0.0096 ± 0.0016 | +0.0070 ± 0.0013 | +0.0079 ± 0.0016 |
| ~7.6M | +0.0052 ± 0.0002 | +0.0034 ± 0.0004 | +0.0065 ± 0.0017 |
| ~38M | +0.0052 ± 0.0007 | | |
| 151M | **+0.0044 ± 0.0004** | | |
| 151M, 50k steps (§33.30) | −0.0009 ± 0.0008 | | |

**P61 wrong — and it reframes §33.31.** With all the data and the same 10k
steps case still gains +0.0044: what keeps the feature useful is not scarce
data but how far the model has learned. At a fixed 10k-step budget the gain is
about +0.005 for any corpus from 7.6M to 151M tokens (**P62** wrong by a hair:
+0.0052 against "below +0.0051"); the smallest corpus gives the most because
there the model stops improving after ~5k steps and never learns case well.
The 50k run shows the learning directly — case's lead +0.0060 at step 2k,
+0.0017 at 10k, −0.0002 at 20k. **P60 right**: case read from the word form
alone keeps 73% and 65% of the gain, so it is case, not the tagger's right
context (G-67 closed). **P63 right**: the architecture effect is larger where
data is scarce. **P59 right** at four seeds: English inflection leads its null
by +0.0077 ± 0.0011 at step 2k and ends −0.0046 ± 0.0013 below it.

The claim, then: *a feature helps a model that has not yet learned what it
encodes — early in training, on a short budget, or when data is too scarce to
learn it — and adds nothing once the model has.* **P64** (stated before the
runs, ~75%): all the data at 20k steps puts case's gain between +0.0044 and
−0.0009. **Right** (results_57): **+0.0012 ± 0.0002** (2 seeds, both
positive). On the same 151M tokens, the gain falls with training length —
+0.0044 at 10k steps, +0.0012 at 20k, −0.0009 at 50k.

**Demo figures** (`scripts/analysis/demo_figures.py`) draw every number from the raw
records. Inspecting them caught two things a slide would have carried: the
first figure's title claimed no feature beats its null, where part of speech
and the shuffled control sit slightly above zero — it now says what the data
shows, every feature within 0.002 of its null; and the 50k-step point, offset
on a log axis to avoid overlap, read as a larger corpus — it now sits at its
true 151M tokens, labelled as the same data trained longer.

---

