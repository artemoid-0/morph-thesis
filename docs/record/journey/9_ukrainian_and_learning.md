# Research journey — Part 9: Ukrainian, and Before the Model Has Learned It

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.20–§33.32. [← Part 8](8_the_controlled_programme.md) · [index](README.md)

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
| **~7.6M** | **+0.0051 ± 0.0002** (4/4; 0.46%) | +0.0080 → +0.0056 at 8k, shrinking | 9,999 (still improving) |
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

### 33.33 After the demo: English at a short budget, and a switch

The demo of 1 October is over; experiments resume in small blocks.

**What the early help is worth.** Interpolating each null's curve between
evaluations gives the step at which it reaches the feature's level — the
training the feature saves:

| feature | at step 1k | 2k | 5k | 10k |
|---|---|---|---|---|
| English part of speech | 86 steps (8.6%) | 141 (7.0%) | 434 (8.7%) | 790 (7.9%) |
| English inflection | 53 (5.3%) | 78 (3.9%) | 239 (4.8%) | 326 (3.3%) |
| Ukrainian case | 23 (2.3%) | 69 (3.5%) | 181 (3.6%) | 163 (1.6%) |

The early help is worth **2–9% of training steps** — consistent, and modest. *(Ukrainian case,
restated against the constant null: 22, 64, 162 and 118 steps, 1.2–3.2% — 1–9% in all; §33.91.)*
(The first measurement, read off the 500-step evaluation grid, gave identical
ratios for every feature; it could only resolve "within one interval".)

**Block A1** (results_58): English part of speech against its null at a
10k-step budget, seeds 0–1: **+0.0140, +0.0133 — +0.0136 ± 0.0004**. **P65
right.** The gap holds through the whole short schedule (+0.0159 at step 1k,
+0.0136 at the end), against +0.0096 at step 10k of the 50k run and +0.0002 at
its end. The finding of §33.32 holds in both languages: a feature helps a model
that has not yet learned what it encodes. At the same budget English part of
speech helps three times as much as Ukrainian case.

**The switch.** Does early help leave a lasting trace, or only arrive early?
`frozen.feature_off_step = K` closes a gate on every feature's output at step K
— the slot becomes a zero slot — and resets the best-so-far and the patience
counter, so the reported result describes the model *without* the feature and
the jump after the switch cannot end training. The gate exists only when set
(every other model and checkpoint is unchanged — logits bit-identical to the old
code), is saved with the checkpoint, and adds no parameters, so a switched cell
starts from exactly the weights of the cell that keeps its feature. The label
marks it (`-OFF2500`); the cache key does not change. Tested on the real
embedding (gate, gradient, checkpoint, initial weights) and in a real run.

**Block B1** compares, at the 10k budget, English part of speech switched off
from the start (K = 0) and after the first quarter (K = 2,500), seeds 0–1;
A1's runs are the third arm, the feature kept throughout. All three share
their initial weights.

- **P66** (~65%). Switched off at step 2,500, part of speech keeps less than
  half its gain: at the end, the K = 2,500 model sits closer to K = 0 than to
  the model that kept the feature.
- **P67** (~65%). The switch costs at once: at step 3,000 the K = 2,500 model
  trails the K = 0 model.

### 33.34 A crutch, not a teacher

Block B1 (results_60; the upload was named `results_59.zip`) switched English
part of speech off at the 10k budget, seeds 0–1. With A1's runs as the third
arm, all three share their initial weights:

| at the end of 10k steps | seed 0 | seed 1 |
|---|---|---|
| feature kept throughout | 1.5122 | 1.5121 |
| switched off at step 2,500 | 1.5256 | 1.5246 |
| off from the start | 1.5234 | 1.5238 |

Switched off at 2,500, part of speech keeps **−19% and −7%** of its gain: the
model ends *below* the one that never had the feature (−0.0015 on average).
**P66 right**, beyond its own terms. Step by step, against off-from-start:

| step | 1k | 2k | 2,500 | 3k | 5k | 7,500 | 10k |
|---|---|---|---|---|---|---|---|
| switched at 2,500 | +0.0160 | +0.0157 | **−0.0294** | −0.0092 | −0.0046 | −0.0021 | −0.0015 |
| kept throughout | +0.0160 | +0.0157 | +0.0151 | +0.0134 | +0.0116 | +0.0116 | +0.0114 |

**P67 right**: at the switch the model falls from 0.016 ahead to 0.029 behind
a model that never had the feature — it had routed information through the
slot, and the route is suddenly empty — and seven thousand steps of recovery
never quite close the gap. Before the switch the switched and kept runs follow
the same trajectory, as identical weights and an open gate must.

The early help is **a crutch, not a teacher**: the model does not learn better
representations from the feature, it learns to use it; taken away, the reliance
costs more than the feature gave. Matched within its own family, keeping part of
speech is worth +0.0114 at this budget (+0.0136 against the null, a different
family).

Two follow-ups, one block (C1), seeds 0–1, at the same budget:

- **P68** (~70%). Switched off later, at step 7,500 — more reliance, less time
  to recover — part of speech ends at least 0.003 below off-from-start.
- **P69** (~70%). Switched **on** at step 5,000 (`frozen.feature_on_step`, the
  reverse gate: closed until K, then open), it ends with at least half of the
  kept-throughout gain over off-from-start (≥ +0.0057): the feature is
  information the model can pick up mid-training.

### 33.35 A programme after the demo

With the thesis secure and months to the defence, the question became where
investment could give a strong result. What the project has ruled out is
narrower than "morphology in embeddings does not work": segmentation does not
beat BPE overall (Q1), and linguistic features *added to the input* of a small
model add nothing *at the end*, once it has learned what they encode — and act
as a crutch before. Other routes stay open: morphology as a training target, as
a scaffold removed gradually, in the rare and novel tail that averages hide,
and the question of *when* the model learns what a feature encodes.

**The observation that set the programme.** Plotted against the null's own
validation BPB — how far the model has got — a feature's gain from every run
traces nearly one curve (`scripts/analysis/learning_law.py`): an inverted U,
peaking early and falling as the model learns. English part of speech collapses
across schedules (+0.0134 and +0.0139 at null BPB 1.55 under 50k and 10k
schedules). Ukrainian case collapses approximately, with a residual that grows
with schedule length (+0.0035 at 50k to +0.0056 for a 38M corpus at 10k, at
1.10). The little-data gains are explained: a 1.5M-token model plateaus near
the curve's peak. Each feature has its own zero crossing — structure first
(1.484), then frequency band (1.412), inflection (1.404), part of speech from
the word form (1.361) — while contextual part of speech never reaches it at this
scale. This is exploratory, not pre-registered; it sets direction D1 and its
first registered test.

The programme ([`docs/DIRECTIONS.md`](../../DIRECTIONS.md)) has six directions — the
law (D1), its mechanism by probing (D2), scaffolding (D3), the teacher (D4), the
tail (D5), the position channel (D6) — each with a day test, night-ready work and
a pivot criterion; it runs on two queues: decision-independent work at night, anything that could change
the plan by day.

### 33.36 The first programme night

The night of 2 October runs the night-ready queue
(`scripts/chains/night_2026-10-02.sh`), in priority order: English part of
speech at a 10k budget at seeds 2–3 with its null; the switch at seeds 2–3; D1's
sampling — English part of speech at 5k and 20k budgets, seeds 0–3; Ukrainian case
at a 20k budget, seeds 2–3. The chain waits while any training run is still
going — a process whose command line *starts* `python -m morphlm.cli run`; a
bare substring would also have matched a shell that merely mentions it, which
the rehearsal caught — and the guard refuses a second copy. Rehearsed with the
stub: 28 runs, each consolidation cell extending exactly the seeds on record.

- **P70** (~90%). At seeds 2 and 3, English part of speech at a 10k budget again
  beats its null by at least +0.003 at each seed.
- **P71** (~70%). The crutch replicates: switched off at 2,500, part of speech
  ends at or below off-from-start (gain over it ≤ +0.0005) at seeds 2 and 3.
- **P72** (~70%). English part of speech at 5k and 20k budgets lands on the
  gain-against-learning curve: wherever the curves overlap, within ±0.002 of
  the 50k run's gain at the same null BPB.
- **P73** (~75%). Ukrainian case at a 20k budget ends between its 10k and 50k
  results (+0.0044 and −0.0009) at seeds 2 and 3.

### 33.37 The follow-on of 2 October

At 09:07 the night chain had started 17 of 28 runs, on time for about 13:40;
with the results read near 19:00, a follow-on fills the rest
(`scripts/chains/day_2026-10-02.sh`, ~4.8 h; it waits for the night chain). The
English feature cells at a 10k budget reuse the 10k nulls now run at seeds 0–3.
Structure is the law's negative control: no early help, so no gain predicted at
a short budget either.

- **P74** (~80%). At seeds 2–3 each of C1's cells — off at 7,500, on at 5,000 —
  has the same sign of gain over off-from-start as at seeds 0–1.
- **P75** (~75%). English inflection at a 10k budget beats its null by at least
  +0.003 on average, positive at every seed — though it ends below its null at
  50k steps.
- **P76** (~70%). Ukrainian case at a 5k budget beats its null by more than its
  10k-budget gain (+0.0044) on average.
- **P77** (~75%). English structure at a 10k budget stays within ±0.002 of its
  null at every seed.
- **P78** (~70%). English frequency band and part of speech from the word form
  each beat the null by at least +0.002 on average at a 10k budget.

### 33.38 The law holds; a critical period

Both chains of 2 October completed (results_62; C1's seeds 0–1 in results_61).

**The learning-progress law, in English, holds almost exactly.** English part
of speech traces the same gain-against-learning curve under 5k, 10k, 20k and
50k schedules: the worst deviation from the 50k curve is 0.0010 over 67
compared points, the average 0.0002–0.0005. **P72 right.** Feature by feature,
50k and 10k schedules agree within 0.0004 at null BPB 1.55 (inflection +0.0069
and +0.0073; part of speech from the word form +0.0101 and +0.0105; frequency band
+0.0052 and +0.0050; structure +0.0011 and +0.0007). The curve predicts the
short-budget gains: at a 10k budget, part of speech **+0.0130 ± 0.0004**, part
of speech from the word form **+0.0099 ± 0.0005**, inflection **+0.0067 ±
0.0014** (4/4 — though it ends 0.0046 below its null at 50k), frequency band
**+0.0046 ± 0.0008**, structure +0.0008 ± 0.0008 — the order of their curves.
**P70, P75, P78 right**; **P77 wrong** by one seed (structure +0.0028 at seed
2; its mean is at the null). In Ukrainian, case at 5k, 10k and 20k budgets and
on a 38M corpus cluster on one curve (at null BPB 1.10: +0.0051, +0.0049,
+0.0056); the 50k run sits 0.0015 below them, the one outlier. *(Against the first null.
Against the constant null it is 0.0008 below the ~38M run at 1.10 and level at 1.15; §33.91.)* **P73, P76
right** (20k: +0.0015 ± 0.0004; 5k: +0.0074 ± 0.0007).

**The switch family, four seeds**, English part of speech at a 10k budget,
against off-from-start:

| | mean | share of the kept gain |
|---|---|---|
| kept throughout | +0.0111 ± 0.0003 | 100% |
| off at 2,500 | −0.0023 ± 0.0006 | −7% to −33% |
| off at 7,500 | −0.0118 ± 0.0003 | −94% to −120% |
| on at 5,000 | +0.0010 ± 0.0004 | 0% to 17% |

**P68, P71, P74 right; P69 wrong.** Reliance grows with exposure: switched off
at 7,500 the model falls from 0.011 ahead to 0.041 behind, and ends 0.012 below
a model that never had the feature. And a feature switched *on* halfway is
barely used — +0.0010 against +0.0111 from the start — after its own jump at the
switch (−0.032). Together: **a critical period** — a feature is integrated if it
is present while the representations form; arriving late, it is largely ignored;
removed late, it costs most. One confound is open (G-68): a closed gate passes no
gradient, so a feature switched on meets an embedding table still at its random
initial values, and a cosine schedule has halved the learning rate by step 5,000.

### 33.39 Day block D: the confound, and a wider model

Two day tests. **G-68**: English part of speech switched on at step 5,000 of a
**20k** budget — the learning rate still near its peak, 15,000 steps left —
with off-from-start for comparison; the feature kept throughout at 20k is
already run. **The width pilot** (D1): null and part of speech at a 10k budget
in a model four times wider *(twice as wide, with four times the core's parameters; §33.91)*
(d_model 256; the feature keeps its 8 dims), seed 0.

Preparing the pilot found that a run's label ignored the model's size: a
256-wide run would have been labelled like a base run, and every analysis
groups by label — it would have entered the base cell as a fifth seed. The
rule the code states for training settings (non-default values mark the label)
now covers the model: `_w256`, `_L6`, `_h8`; defaults add nothing. Rebuilt from
their saved configurations, all 392 recorded runs keep their labels (two old Q1
runs differ under the old code and the new alike: they predate the surface
settings).

- **P79** (~60%). Switched on at 5,000 of a 20k budget, part of speech ends
  with less than half of the kept-throughout gain over off-from-start: the
  critical period survives time and learning rate.
- **P80** (~60%). The 4× wider model's part-of-speech gain at a 10k budget lies
  within ±0.002 of the base 50k curve's gain at the same null BPB: the law holds
  across capacity.

### 33.40 Two pivots

Day block D (results_63, results_64), seed 0. Both predictions wrong; both
redirect the programme as the pivot criteria said they would.

**G-68: slow integration, not a critical period.** Switched on at step 5,000
of a **20k** budget, English part of speech ends +0.0039 over off-from-start —
**63%** of the kept gain (+0.0061), against 9% at a 10k budget. Its share
climbs steadily: 14% by step 8k, 40% by 12k, 54% by 15k, 63% at the end. A
late feature is integrated slowly — its table starts untrained, the model
rebuilds around it — not ignored. **P79 wrong.** The removal side of §33.38
stands; "a late feature is largely ignored" was the short schedule.

**The width pilot: the law does not cross model size.** A model four times
wider (d_model 256; 19–20 minutes per 10k run, 1.6× the base) starts on the
base curve (+0.0136 against +0.0140 at null BPB 1.94) — then its gain falls
away early: +0.0031 at null BPB 1.58, where the base model gains +0.0133; it
stays near +0.002, and the base curve comes down to meet it (+0.0022 against
+0.0039 at 1.384). Worst deviation −0.0111, mean 0.0046. **P80 wrong** — the
D1 pivot. At equal overall BPB the wider model has already learned what part of
speech encodes; overall BPB measures "how much the model has learned" across
schedules and corpus sizes, not across capacity. If "learned it" is the model's
own knowledge of the feature, a probe's accuracy should be the axis on which
widths collapse — D2 moves to the centre.

The night of 2–3 October (`scripts/chains/night_2026-10-03.sh`, ~6.8 h) maps
both: G-68 at seeds 1–3; the 256-wide model at seeds 1–3; a 64-wide model at
seeds 0–3 — if capacity shifts the curve, a narrower model should shift it the
other way; and the 256-wide model at a 20k budget, seed 0.

- **P81** (~70%). At each of seeds 1–3, part of speech switched on at 5,000 of
  a 20k budget ends with more than a third of the kept gain over off-from-start.
- **P82** (~75%). The wider model's early fall replicates: at null BPB 1.55 its
  gain is below +0.006 at each of seeds 1–3 (the base curve: +0.013).
- **P83** (~60%). The 64-wide model's gain at the end of a 10k budget is at least
  the base curve's gain at the same null BPB: a narrower model keeps its gain
  longer.

### 33.41 The chain widened: four directions in one night

Before it started, the night chain of §33.40 grew to ~11 hours, so that one
night advances four directions.

**D6 without new code.** During the 50k runs the architecture effect — the
null over the plain baseline — is positive from the start and grows: +0.0056 at
step 2k, +0.0060 at 10k, +0.0095 at 20k, +0.0112 at the end (4/4 seeds at every
point). But a run *budgeted* for 10k steps, its learning rate annealed by step
10k, shows **+0.0003** (seed 0: the 128-wide baseline from the September
learning-rate sweep, whose training settings match today's field for field; a
plain baseline exercises none of the code changed since, so it serves as seed
0). If that holds, the free dims help optimization at a high learning rate rather
than representation — in line with the slotted runs' fourfold stability across
seeds. Tonight: the plain baseline at a 10k budget (seeds 1–3) and at 20k (seeds
0–3), and a 256-wide baseline at 20k (seed 0) beside the 256-wide null.

**D3, the mirror of G-68.** Part of speech switched off at 5,000 of a 20k
budget (seeds 0–3): given 15,000 steps, can the model shed its reliance?

- **P84** (~70%). At a 10k budget the architecture effect is small: the null
  beats the plain baseline by less than +0.003 on average over seeds 0–3
  (+0.0060 at step 10k of the 50k runs).
- **P85** (~60%). At a 20k budget it lies between: above the 10k-budget value
  and below the 50k value (+0.0112), on average over seeds 0–3.
- **P86** (~60%). Switched off at 5,000 of a 20k budget, part of speech ends
  within ±0.0015 of off-from-start on average over seeds 0–3: given time, the
  model sheds its reliance.

### 33.42 The night of four directions

The chain of §33.41 completed (results_65, 34 runs).

**D3 — adding is reversible in time, removing is not.** Switched on at step
5,000 of a 20k budget, part of speech recovers 63%, 54%, 82% and 104% of the
kept gain at seeds 0–3 (+0.0049 against +0.0067): slow integration, four seeds.
**P81 right; G-68 closed** — the 9% at a 10k budget was the schedule. Switched
*off* at 5,000 of a 20k budget, with 15,000 steps to recover, the model still
ends **0.0042 below** off-from-start, at every seed (−0.0044, −0.0012, −0.0047,
−0.0064) — more than the 0.0023 left by a switch at 2,500 of a 10k budget.
**P86 wrong.** A late feature can be learned; a removed one leaves a lasting
deficit. The likely mechanism: leaning on the feature while learning is fastest
displaces what the model would have learned by itself, and that is hard to make
up later.

**D1 — no single curve across widths.** At null BPB 1.55 the 256-wide model
gains +0.0023, +0.0082, +0.0009, +0.0029 (mean +0.0036; the base curve
+0.0134). **P82 wrong** by one seed; the fall replicates on average. It is
abrupt: the 256-wide curve follows the base curve to null BPB ~1.70, collapses
by 1.60, and crosses zero near 1.37 (20k budget, seed 0: −0.0022 at the end).
The 64-wide model gains *less* than the base model at every common null BPB
(+0.0112 against +0.0135 at 1.70), ending 0.0028 below the base curve at its own
null BPB. **P83 wrong.** The slot is a fixed 8 dims — 12.5% of a 64-wide
embedding, 3% of a 256-wide one — so across widths the feature's share of
capacity differs; within a width the same-shape null keeps each comparison fair.

**D6 — the architecture effect builds with training, and survives width.** The
null over the plain baseline: **+0.0044, +0.0043, +0.0050** at a 10k budget
(seeds 1–3), **+0.0078 ± 0.0009** at 20k (4/4), +0.0112 at 50k; in the 256-wide
model **+0.0057** at 20k (seed 0). The September seed 0 at a 10k budget
(+0.0003) is the outlier on which §33.41's high-learning-rate hypothesis rested;
whether that run differs from current code is G-69. **P84 wrong** (+0.0035 with
seed 0 counted); **P85 right**.

### 33.43 A bit-for-bit reproduction

**G-69 closed.** The 10k-budget plain baseline at seed 0, rerun under current
code (results_66, run 12e3d62efc84), matches the September learning-rate
sweep's run (cfbb4ce93538) to the last digit of float precision — validation
BPB 1.5265532254632217 — at all 21 evaluations, in validation BPB and training
loss alike. A run made weeks of code changes earlier reproduces exactly: the
pipeline is deterministic along this path, and the September run was never a
different code era. Its small architecture effect is seed variation; over four
seeds the effect at a 10k budget is **+0.0035 ± 0.0011** (4/4). That is below
the 50k run's value at step 10k (+0.0060 ± 0.0007) but far from zero: the
high-learning-rate hypothesis of §33.41 falls.

Two runs now held one cell's seed 0. A quarantine rule (`sweep_duplicate`)
moves the September copy aside; dry-run against 234 runs it matches that one
run and nothing else.

**The night of 3–4 October** (`scripts/chains/night_2026-10-04.sh`, ~12 h): the
256-wide model at a 20k budget, seeds 1–3 — null, part of speech, plain
baseline; then English inflection, part of speech from the word form,
frequency band and structure at a 20k budget, seeds 0–3.

- **P87** (~75%). The 256-wide model's part of speech ends a 20k budget at or
  below its null (gain ≤ +0.0005) at each of seeds 1–3.
- **P88** (~75%). The architecture effect at 256 wide and a 20k budget is
  positive at each of seeds 1–3.
- **P89** (~70%). At null BPB 1.55 each of the four features' 20k-budget gain
  lies within ±0.001 of its 50k value.

### 33.44 A rule written for one generation of runs

The dry run of the quarantine (before `--apply`) listed, beside the
September duplicate, **nine valid runs** under an old rule, `causal_partial`:
every plain baseline run since 2 October — today's G-69 rerun among them. The
rule flags runs with `causal_identity` set that lack the current causal-shift
marker. The chains pass the causal flags to every run, plain baselines
included; a baseline has no feature block, so it cannot carry the marker — nor
have the defect, having nothing to shift. Written when only feature runs set the
flag, the rule misfired on a later generation of runs. Fix: a run is in scope
only if it has features; a regression test pins it. Over all 340 runs on hand,
the complete dry run now lists exactly the September duplicate.

The rehearsal had included the G-69 rerun, so it must have flagged it too — but
it was read only around the new rule. The procedure now includes what the near miss
taught: a dry run is read in
full, every rule and every run, and a new rule is checked together with the old
ones. Methodological finding 6 — analysis code needs the scrutiny the models get
— extends to the tools that decide which runs count.

### 33.45 Tests the suite never ran

Checking why a new test's line was missing from a suite's output found that
each test file ran a hand-kept list of its tests — and four tests had never been
added to one: G-63's regression test for the fragment-aware word rule (since 1
October), the feature switch's (2 October), and today's two, the model size in
the label and the quarantine sparing plain baselines. Each passed when called
directly; "PASS" at the end of a file was true of the listed tests only, so none
of the four guarded anything. Each file now runs every test function it defines,
in order — 63 tests, all run, all pass, the four among them. The same lesson as
§33.44, one level up: the tools that certify the code need the scrutiny the code
gets.

### 33.46 The wide model's crossing, and the slot as consistency

The night chain of §33.43 ran shortened to 8.5 hours (results_67, 17 runs):
`FEAT_SEEDS="0 1"` kept the features at a 20k budget to seeds 0–1, so that it
ended within the nine hours available; its log records the seeds it ran.

**P87 right.** At a 20k budget the 256-wide model's part of speech ends at or
below its null at every seed: −0.0022, −0.0002, −0.0008, −0.0002 (mean −0.0008 ±
0.0005), where the 128-wide model still gains +0.0075. The wider model has
passed its zero crossing within 20k steps.

**P88 wrong — and the reason is the finding.** The architecture effect at 256
wide is +0.0057, +0.0200, −0.0033, +0.0028 (mean +0.0063 ± 0.0049). The variance
is all in the plain baseline: its final BPB spans 1.2740–1.2977 across seeds
(spread 0.0237), the null's 1.2751–1.2777 (0.0026), part of speech's 0.0007. No
run spiked — every curve falls smoothly; the baseline's seed 1 is worse from the
start, its seed 2 better. At this width the slot's clearest effect is
**consistency**: without it, where training ends depends on the seed; with it,
every seed lands in the same place — the "4× more stable" of 128-wide 50k runs,
now 9×.

**P89 wrong by one feature.** At null BPB 1.55 and a 20k budget, part of speech
from the word form, frequency band and structure match their 50k values within
0.0007 (+0.0001, +0.0002, −0.0007); inflection sits +0.0020 above (+0.0088
against +0.0069, two seeds — the noisiest feature, ±0.0014 at 10k). Seed noise
or a schedule effect: G-70.

The night of 4–5 October (`scripts/chains/night_2026-10-05.sh`, ~8.1 h): the four
features at a 20k budget, seeds 2–3; the 256-wide baseline and null at a 20k
budget, seeds 4–7.

- **P90** (~65%). Over four seeds, inflection's 20k-budget gain at null BPB
  1.55 lies within ±0.001 of its 50k value (+0.0069): the two-seed deviation was
  noise.
- **P91** (~70%). Over seeds 0–7, the 256-wide plain baseline's spread of final
  BPB is at least four times the null's.
- **P92** (~60%). Over seeds 0–7, the 256-wide null beats the plain baseline on
  average, at six or more of the eight seeds.

### 33.47 A screen before the long chain

The night chain of §33.46 did not run. A better rhythm replaced it:
a short screen first — one sharp test per open direction — then a long chain
shaped by what it shows. Its G-70 runs move into the screen; P91 and P92 (the
wide model at seeds 4–7) into the long chain after it.

**D5, screened at the desk.** Every result records loss by token-frequency
bucket. At the end of 50k runs no benefit hides in the rare tail: English part
of speech keeps its gain only in the two most frequent buckets (+0.0112 and
+0.0159 nats, 4/4 seeds), noise in the rarest; English inflection is *worse*
than its null in rare and mid-frequency tokens (−0.019 to −0.036 nats, 0/4
positive) — where its end deficit lives; Ukrainian case is neutral to negative
across the tail. By BPE-token frequency, not word rarity; with this prior the
unseen-word version is parked.

**Three new modes**, each inert unless set (logits of existing models
bit-identical; labels unchanged), each with a test that runs in the suite and a
real run on the toy corpus:

- `model.position = rope` (D6): rotary positions in attention, none added at the
  input; label `_rope`.
- `frozen.feature_dropout_until = K` (D3): weaning — each token keeps its
  features with a probability falling linearly to 0 at step K; evaluation scales
  by it; best-tracking resets at K; label `-DROP<K>`.
- `frozen.role = target` (D4): the fact as an auxiliary target, predicted from the
  final hidden state with weight `aux_weight` (0.1); nothing injected; its head
  made last, so every language-model weight starts exactly as a baseline's; its
  loss added only to the training step, never to evaluation; label `-TGT0.1`, the
  shuffled-label null `-SHUF-…-TGT0.1`.

The RoPE run exposed the provenance record assuming a learned position table;
it now counts positional parameters as zero when absent and the auxiliary head
apart from the transformer core.

**The screen** (`scripts/chains/screen_2026-10-05.sh`, ~2.7 h, seed 0, 20k budget):

- **P95** (~35%). Part of speech as an auxiliary target ends at least +0.002
  better than both the shuffled target and the plain baseline: a lasting teacher.
- **P93** (~65%). Under rotary positions the architecture effect is below +0.002
  (+0.0078 ± 0.0009 with learned positions): it needs position in the slot.
- **P94** (~55%). Weaned gradually to zero by step 5,000, part of speech ends less
  than 0.0021 below off-from-start — half the abrupt switch's deficit at 5,000.

### 33.48 Building D1 and D2 while the screen runs

**D1 needs no new model code.** Both part of speech and the null occupy 8 dims
(part of speech's 19 classes learned into 8), and the existing tiling lever,
`frozen.amplify`, repeats a block k times: a 256-wide model with its slot tiled
×2 gives the feature 16 of 256 dims — the base model's share. Tiling repeats
values, so it raises the feature's share and prominence, not its capacity. But
`amplify` never reached the label — a third hazard like the model size (§33.39):
a tiled run would have joined its untiled cell as more seeds. All 356 recorded
runs used `amplify=1`, so no collision ever happened; the label now carries
`-x<k>`.

**D2: checkpoints and a probe.** `MORPHLM_CHECKPOINT_EVERY=N` — an environment
switch, not a config field, so a checkpointed run keeps its label — saves
`ckpt/step_NNNNNN.pt` beside `model.pt`; a toy run with and without it is
bit-identical. `scripts/analysis/probe.py` rebuilds each checkpoint, feeds the
model its own features, and trains a linear probe from every layer's hidden
states to the probed feature's labels (alternating windows train and test,
sentinel tokens left out, the two datasets' tokens asserted identical). Layer 0
is the input — what token identity alone says. Its record goes to
`results/audits/` with a UTC timestamp in the filename, the form by which the
packer collects audits.

**A fourth hazard, found preparing the pilot.** Run ids hash the whole
configuration; every field added since a cell ran — the switches, today's three
modes — gives a re-run of that cell a *new* id under the *same* label and seed.
G-69's September duplicate arose this way. The pilot's checkpointed nulls
therefore take a new seed (4), not a re-run of seed 0; and the packer now lists
**DUPLICATE SEEDS** — any packed run sharing its label and seed with another
run on disk — tested on a planted duplicate. `scripts/ops/run_ids.py` gives the
id a command line would produce, checked against a real run.

**Day block E** (`scripts/chains/day_2026-10-05.sh`, ~80 min; it waits for the
screen chain itself, since between a chain's runs there are seconds with no
training): the 128- and 256-wide nulls at a 10k budget, seed 4, with checkpoints
every 500 steps, then the part-of-speech probe on both; and the 256-wide model
with its slot tiled ×2, null and part of speech.

- **P96** (~60%). At null BPB ~1.65 — where the 256-wide gain collapses — the
  256-wide null's deepest-layer part-of-speech accuracy is at least 2 points
  above the 128-wide null's at the same null BPB: the wider model knows part of
  speech earlier.
- **P97** (~55%). With its slot doubled to the base model's share, the 256-wide
  model's part-of-speech gain at null BPB 1.55 stays below +0.006: the width
  effect is not the slot's share.

### 33.49 The screen's verdicts, and time budgets

The screen and block E completed (results_68: 11 runs, two probe records).

| direction | result (seed 0 unless noted) | prediction |
|---|---|---|
| G-70 | inflection at 20k over four seeds, at null BPB 1.55: +0.0070 against +0.0069 at 50k | **P90 right** — G-70 closed |
| D1 | slot tiled ×2 (the base model's share): +0.0024 at null BPB 1.55 (×1: +0.0036) | **P97 right** — the collapse is not the slot's share |
| D2 | at null BPB ~1.65, deepest-layer part-of-speech accuracy 0.867 (256 wide) against 0.854 (128) | **P96 wrong** — 1.3 points, not 2 |
| D6 | RoPE: null over plain baseline +0.0073 (learned positions: +0.0078 ± 0.0009) | **P93 wrong** — the effect survives |
| D4 | the real target over the matched shuffled one +0.0018, over the plain baseline +0.0014 | **P95 wrong** — under +0.002, but positive |
| D3 | weaned to 5,000: −0.0023 against off-from-start (the abrupt switch at 5,000: −0.0044) | **P94 wrong** by 0.0002 — half the deficit |

**D6 changes its question.** Under rotary positions nothing positional enters
the input, and eight content-free dims help as much as before: the effect is not
a clean position channel. What such dims do — a bias- or sink-like channel — is
the new question. RoPE itself is far better than learned positions here (1.3908
against 1.4325 BPB for the plain baseline at 20k).

**D2 locates the width effect in context.** At equal null BPB the two widths
read part of speech from their *input* equally well (0.838 and 0.839); the wider
model's deeper layers add twice as much (+0.029 against +0.015). It learns
contextual part of speech earlier — the knowledge the feature would have supplied.
Probe accuracy narrows the two widths' gap but, on this one comparison, does not
close it: at equal accuracy (~0.867) the 128-wide model still gains +0.0134, the
256-wide +0.0091.

**Time budgets replace day and night.** The hours available are not fixed, and
several long chains may follow one another; a chain is now planned from one
number, the hours until its results are read. Short blocks take the boldest,
plan-changing tests; long chains the quiet, mutually independent work: two
queues, by time budget. "Night-ready" becomes
long-chain-ready, with the same two conditions.

**The long chain of 5 October** (`scripts/chains/long_2026-10-05.sh`, ~7.5 h, 19
runs) gives the screen's promising tests seeds 1–3, extends the probe to 20k
budgets, and adds D1's second seed.

- **P98** (~55%). Over seeds 0–3 the real auxiliary target beats the matched
  shuffled one at a 20k budget by at least +0.001 on average, at three or more
  seeds.
- **P99** (~75%). Over seeds 0–3 the architecture effect under RoPE is positive
  at every seed, with a mean of at least +0.004.
- **P100** (~65%). Over seeds 0–3 weaning to 5,000 leaves at most 70% of the
  abrupt switch's deficit.
- **P101** (~60%). Probed along 20k budgets, the 256-wide null's deepest-layer
  part-of-speech accuracy exceeds the 128-wide null's by at least 1 point at
  every matched null BPB between 1.75 and 1.45.
- **P102** (~65%). With its slot tiled ×2, the 256-wide model's gain at null BPB
  1.55 stays below +0.006 at seed 1 as well.

### 33.50 The teacher fades faster than the crutch

The long chain of §33.49 completed (results_69, 19 runs, two probe records).

| | seeds 0–3, 20k budget unless noted | prediction |
|---|---|---|
| D4 | real target over its shuffled twin: +0.0018, +0.0015, −0.0011, −0.0008 — **+0.0003 ± 0.0007** | **P98 wrong** |
| D6 | under RoPE, null over plain baseline: **+0.0057 ± 0.0006**, 4/4 (learned positions: +0.0078 ± 0.0009) | **P99 right** |
| D3 | weaned to 5,000: −0.0029 ± 0.0016 against the abrupt −0.0042 ± 0.0011 — **70%** of the deficit | **P100 right** |
| D2 | probe along 20k budgets: the 256-wide null leads the 128-wide by +0.8 to +1.3 points at every matched null BPB, 1.75–1.45; input layers equal | **P101 wrong** — below 1 point at the ends |
| D1 | tiled ×2, seed 1: +0.0037 at null BPB 1.55 (seed 0: +0.0024) | **P102 right** |

**D4 answered — the law covers targets.** Along the 20k budget the real target
beats the shuffled one by +0.0094 at step 1k, +0.0087 at 2k, +0.0041 at 5k,
+0.0020 at 10k, +0.0006 at 15k, +0.0003 at the end — early help that fades, like
the feature as an input (+0.0150 at 1k, still +0.0075 at the end), only faster.
The contrast fits §33.34: as a target the feature pushes the model to learn part
of speech itself, and once it has, nothing is left to add; as an input it is
leaned on, so its gap lasts longer. Any auxiliary head helps a little — the
shuffled target too (+0.0006 ± 0.0004 over the plain baseline) — the morphology
in it adds nothing at the end. Seed 0's +0.0018 (§33.49) was within this spread.

**D6:** the effect survives rotary positions at four seeds, a little smaller.
RoPE itself beats learned positions by +0.0443 ± 0.0011 BPB (plain baseline,
4/4). **D3:** weaning's reduction is modest and varies by seed (one seed ends
above off-from-start, one equals the abrupt switch). **D2:** the wider model's
contextual lead holds at every matched null BPB while the input layers agree —
the width effect lives in what context adds. **D1:** the slot's share is ruled
out at two seeds.

### 33.51 An empty slot under rotary positions, and resumable chains

One run as a small piece (6 October): the zero slot under rotary positions,
seed 0 (run 16a11dd9bf91) — 8 input dims held at exactly zero, and nothing
positional anywhere in the input. **1.3910**, against the plain baseline's 1.3908
and the constant slot's 1.3835. **P103 wrong** (stated before the run and recorded
after it, ~55%: at least half the constant slot's benefit). The zero and constant
slots share their initial weights (§33.15), so the matched contrast is clean:
the constant beats the empty slot by **+0.0075**.

Read with the record, the slot needs a **token-independent signal** in its dims.
With learned positions the "empty" slot was never empty — position filled it, and
zero, flat and constant slots gained alike (§33.12); one learned vector *added* to
every input dim beat its initialization-matched control by +0.0085 ± 0.0022 (4/4;
§33.14–§33.15). Under rotary positions a plain model's input holds token identity
and nothing else: an empty slot then gives nothing, a constant one +0.0075. Every
linear layer here has a bias and every LayerNorm a shift, so this is not a missing
bias in the usual sense: the constant enters *before* normalization and changes
what LayerNorm does — one hypothesis is that it softens it, letting each block see
the scale of the token's embedding. At the desk, by token frequency, the RoPE
null's gain is even across the vocabulary (+0.015 to +0.020 nats in every bucket,
4/4), where with learned positions it concentrated in rare tokens (+0.061 in the
rarest bucket, +0.0005 in the most frequent).

**Resumable chains.** The hours available vary, and a long chain may have to stop.
`scripts/ops/run_ids.py --done` tells a chain whether a run's cell and seed already
have a finished result — the identity every analysis groups by, which also finds
an equivalent run recorded under an older id; a half-written result does not
count. The long chain of 6 October (`scripts/chains/long_2026-10-06.sh`, ~9.7 h,
24 runs in 10 pieces) skips every finished run and every recorded probe, and
`PIECES` selects pieces: stopped and restarted, it repeats only the run that was
in progress. Rehearsed against the 386 recorded runs: 24 runs; the 4 with
checkpoints receive the checkpoint setting through the skip function, and no other
run does; 4 probes, each reading its checkpointed run; no duplicate seed. After a
simulated stop, 21 runs and 3 probes; with `PIECES="5 9"`, 4 runs. The 256-wide
null at 20k, seed 4, already existed — the checkpointed run of §33.49 — so that
piece runs only its plain baseline.

- **P104** (~55%, stated before its run and recorded after it). The flat slot under
  rotary positions — one learned vector in the 8 dims at every position, no word
  marker — gains at least +0.0037 over the plain baseline at seed 0, half the
  constant slot's +0.0073.
- **P105** (~70%). Under rotary positions, one learned vector ADDED to every input
  dim beats zeros added the same way — its initialization-matched control — by at
  least +0.004 at seed 0: the signal needs no dims of its own.
- **P106** (~75%). At each of seeds 1–3 under rotary positions, the constant slot
  beats the zero slot of the same family by at least +0.004.
- **P107** (~65%). At step 6,000 — 1,000 steps after the switch — the model
  switched off at 5,000 reads part of speech from its deepest layer at least 1
  point less accurately than the model off from the start (seed 4): early
  reliance displaced the model's own learning of it.
- **P108** (~55%). At step 2,000 the real-target model's layer-2 part-of-speech
  accuracy exceeds the shuffled-target model's by at least 1 point (seed 4): the
  teacher speeds the model's own learning upstream of its head.
- **P109** (~70%). Over seeds 0–3 at a 20k budget, word form, frequency band and
  structure each lie within ±0.001 of their 50k gain at null BPB 1.55.

### 33.52 What the input needs, and what the probes found

The long chain of 6 October ran all ten pieces in 9 hours 5 minutes (results_70:
24 runs, the zero-slot piece of §33.51, and four probe records), with no
duplicate seed.

**D6 — one learned vector shared by every input.** Under rotary positions, final
BPB at a 20k budget:

| seed | plain baseline | zero slot | flat slot | constant slot | zeros added | flat added |
|---|---|---|---|---|---|---|
| 0 | 1.3908 | 1.3910 | 1.3832 | 1.3835 | 1.3929 | 1.3836 |
| 1 | 1.3909 | 1.3853 | 1.3849 | 1.3852 | 1.3867 | 1.3822 |
| 2 | 1.3898 | 1.3886 | 1.3842 | 1.3843 | 1.3884 | 1.3840 |
| 3 | 1.3866 | 1.3879 | 1.3801 | 1.3822 | — | — |

Within a family — identical initial weights — the flat slot beats the zero slot by
**+0.0051 ± 0.0017** (4/4), the constant slot beats it by +0.0044 ± 0.0016 (4/4),
the flat slot equals the constant slot (+0.0007 ± 0.0005), and one vector added
to every input dim beats zeros added the same way by **+0.0060 ± 0.0016** (3/3).
Against the plain baseline: flat slot +0.0064 ± 0.0004 (4/4), flat added +0.0072
± 0.0008 (3/3), constant slot +0.0057 ± 0.0006 (4/4), zero slot +0.0013 ± 0.0015
(2/4), zeros added +0.0012 ± 0.0018 (2/3). **P104 right** (+0.0076 at seed 0);
**P105 right** (+0.0093); **P106 wrong** — at seed 1 the constant slot beats the
zero slot by only +0.0001, because that seed's zero slot is the outlier (0.0056
better than its baseline); over four seeds the margin is +0.0044 ± 0.0016, 4/4.

Under rotary positions, then, a single learned vector shared by every input
helps, whether it has dims of its own or is added to all of them; dims merely kept
free of token identity do not; whether the vector marks words is irrelevant.
Added to every input, the vector is a reparameterization of the token embedding —
a shared vector plus each token's row reaches nothing the rows alone cannot — so
the gain lies in how the model learns, not in what it can represent.

**A confound found while writing this down (G-71).** The causal shift gives the
first word of every window the sentinel class, since no earlier word is known
(`causal_shift`). The flat and constant slots, and the flat vector added,
therefore also mark the start of each window — one to three tokens of 256 —
which a rotary model cannot otherwise see; the zero slot and zeros added mark
nothing. With learned positions the window start is visible anyway, and there
one vector added to every input still beat zeros added by +0.0085 ± 0.0022 (4/4;
§33.15) — evidence for the vector itself. Under rotary positions the two are not
yet separated. Without the causal shift a flat slot is a pure constant, and it
starts from the same initial weights (the shift reads only tokenizer tables, with
no random draw): one run separates them, with no new code.

**D3 — the deficit is not lost knowledge of part of speech.** Seed 4 adds −0.0027:
over five seeds the model switched off at 5,000 of a 20k budget ends **0.0039 ±
0.0009** below off-from-start (0/5 above). The gap levels off — −0.0051 at step
10k, −0.0044 at 12.5k, −0.0039 at the end — while the learning rate is still
1.1e-4 at 12.5k (a third of its peak); weaning's gap is flat at −0.0029 from step
7,500. The probe (seed 4, deepest layer; the model is fed its own features):

| step | 1k | 4k | 5k | 6k | 8k | 12k | 20k |
|---|---|---|---|---|---|---|---|
| off from the start | 0.815 | 0.859 | 0.866 | 0.870 | 0.874 | 0.879 | 0.882 |
| off at 5,000 | 0.831 | 0.867 | 0.854 | 0.867 | 0.873 | 0.879 | 0.882 |

Before the switch the switched model reads part of speech better — its input
carried the previous word's tag. One training step after it, the deepest layer is
1.2 points behind; 1,000 steps later 0.3; from then on level, and at layer 2 a
little ahead (+0.6 at the end) — while its BPB stays behind. **P107 wrong.**
Whatever the lasting deficit is, it is not the model's knowledge of part of speech.

**D4 — the teacher teaches; the knowledge stops paying.** Seed 4: the real target
beats the shuffled one by +0.0029; over five seeds **+0.0009 ± 0.0008** (3/5). The
probe (seed 4):

| step | 1k | 2k | 5k | 10k | 20k |
|---|---|---|---|---|---|
| layer 0, real / shuffled | 0.860 / 0.804 | 0.862 / 0.829 | 0.866 / 0.843 | 0.864 / 0.850 | 0.864 / 0.850 |
| layer 2, real / shuffled | 0.874 / 0.819 | 0.885 / 0.839 | 0.893 / 0.863 | 0.896 / 0.877 | 0.900 / 0.884 |
| layer 4, real / shuffled | 0.875 / 0.819 | 0.892 / 0.838 | 0.911 / 0.867 | 0.916 / 0.879 | 0.921 / 0.884 |

The real-target model encodes part of speech better at every layer from step
1,000 — its input embeddings included (+5.5 points at layer 0) — **+4.6 at layer 2
by step 2,000 (P108 right)**, and still +1.4, +1.6 and +3.7 at layers 0, 2 and 4 at
the end (layer 4 is the one the head is trained to read). The extra knowledge
persists while the advantage in prediction fades: knowing more part of speech than
language modelling itself acquires does not help predict text.

**D1 — the law per feature, at three schedules, four seeds each.** At null BPB
1.55 the 20k-budget gains match the 50k ones within 0.0004: part of speech from the
word form +0.0100 (50k: +0.0101), frequency band +0.0055 (+0.0052), structure
+0.0010 (+0.0011), inflection +0.0070 (+0.0069). **P109 right.** End gains at a
20k budget over seeds 0–3: part of speech +0.0075 ± 0.0008, word form +0.0050 ±
0.0006, inflection +0.0022 ± 0.0011, frequency band +0.0014 ± 0.0005, structure
−0.0008 ± 0.0006 — the order of their curves.

**D2 — probe accuracy does not line up the widths (desk).** With the 20k-budget
probes of both nulls (seed 4) and the part-of-speech gain over seeds 0–3 at the
same steps: at deepest-layer accuracy 0.870 the base model gains +0.0133, the 4×
wider one +0.0076; at 0.880, +0.0096 and +0.0036. The wider model's gain falls
away even at equal knowledge of part of speech — D2's pivot: "learned it" is not
the linear decodability of the current word's part of speech. One candidate
remains: these probes read the *current* word's tag, while the feature carries the
*previous* word's — which is what the model must infer from context for the
feature to become redundant.

**256 wide, seed 4:** plain baseline 1.2757, null 1.2735 (+0.0022). Over five seeds
the null leads by +0.0055 ± 0.0039 (4/5), and the baseline's spread of final BPB
is 5.7 times the null's (0.0237 against 0.0041). P91 and P92 await seeds 5–7.

### 33.53 Tools for other conditions

Every claim so far rests on two conditions — WikiText-103 with a 17k BPE, and
Ukrainian Wikipedia with its own 17k BPE. Other corpora, tokenizers and models
were opened (6 October), and testing how far the claims travel is the
least drifting use of the remaining time: it is the examiner's first question.
Tools first, then chains (`docs/reference/conditions.md`).

**Corpora as specs** (`morphlm/data/corpora.py`). A corpus is a dataset, a text
field, a language and tagger, a word rule, a split rule (`native`, or `hash` —
the Ukrainian rule, now general) and a serving rule (`rows`, or `paragraphs`
with their newline, G-64). `stream.load_corpus` reads only the spec. WikiText
keeps its single historical call and refuses a fraction (its cache key carries
none). An unknown corpus fails when the configuration is checked; a tagger of
another language fails before the first document. On toy corpora — the toy
WikiText, and a toy Ukrainian Wikipedia built from the UD_Ukrainian-IU
treebank's real sentences (`scripts/rehearsal/make_toy_ukwiki.py`) — the new
code rebuilt all 34 cache files of the old code byte for byte, and the
tokenizer and its sidecar; a test reimplements the old Ukrainian loader verbatim
and requires the same rows at every split, size and holdout.
`scripts/data/corpus_info.py` looks at a candidate dataset before it gets a spec.

**Tokenizers** (`scripts/data/train_tokenizer.py`, `morphlm/data/tokenization.py`).
One training path, shared with `cli build bpe` (byte-identical tokenizers for
BPE in both languages). Standard names (`bpe_uk_8k`), and a tokenizer is never
overwritten: its name is in every cache key built from it. That matters because
**Unigram training is not deterministic** in the `tokenizers` library — three
trainings with the old code, even on one thread, gave the same 387 pieces with
scores differing by up to 0.02 — so a Unigram file is an artifact, not a
recipe. The sidecar records the recipe and the file's sha256. The statistics
measure what a tokenizer does to words on held-out text: tokens per word, words
split, and **ending exposure** — how often a token boundary falls exactly
between stem and ending in inflected words (школ|и, студент|ам). On the toy
corpus the gradient is the expected one: case-bearing endings exposed in 54.6%
of words at 600 tokens, 38.6% at 1,500, 26.1% at 4,000, and 51.6% for Unigram at
1,500 — a larger vocabulary hides endings inside whole-word tokens, Unigram
exposes them.

**One command per condition** (`scripts/data/prepare_condition.py`): the Ukrainian
recipe for any corpus × size × tokenizer — a timed validation pilot that
estimates the train build, its check, the tagging pass, the null, the zero slot
and the word-form variants, and a final gate on both splits. It calls the
existing commands, skips what is done (a ready condition: 10 seconds) and stops
at the first failure. Rehearsed end to end on the toy corpus with a new
tokenizer, then four runs trained on the new condition.

**G-66 closed.** Outside WikiText a lexicon is named for the corpus and size it
was built from (`lexicon_ucase_bpe_uk_8k_ukwiki0.05.npz`); readers fall back to
the historical name. All 16 derived codes files stayed byte-identical. While
refactoring, the heading check of G-37 showed again — `startswith(' =')` after
`strip()` never fires — and now says so in the code.

The model tools — a fit check for larger models, RMSNorm and SwiGLU blocks —
follow in §33.54.

### 33.54 Tools for other models

The second half of the tools (§33.53): two block options and a way to size a
model before chaining it.

**RMSNorm and SwiGLU** (`model.norm`, `model.mlp`; `docs/reference/apparatus.md`,
§2.1a). RMSNorm keeps a gain and drops the mean subtraction and the shift;
SwiGLU gates the MLP with a hidden width of 8d/3 (344 at d=128), so its weights
match the GELU MLP's within 1%. Each marks the label (`_rmsnorm`, `_swiglu`);
with rotary positions the block is LLaMA-style. RMSNorm draws no random number,
so at the same seed a model with it starts from exactly the weights of its
LayerNorm twin — a matched pair, and a sharp test for D6: with no mean
subtraction, a constant input vector no longer shifts what the norm removes.
Default models are untouched: their outputs are bit-identical to the old code's
in seven kinds of cell (baseline, causal part of speech, the null, rotary
positions with the zero slot, 256 wide, the auxiliary target, the switch).

**The fit check** (`scripts/ops/fit_check.py`) builds the model a command line
would build, times the training step as the loop runs it — AdamW, clipping, the
real batch shape — and one validation pass as the loop runs it, and projects the
run. Over a grid of widths and depths it sizes a scale study in one call. Writing
it showed that the validation pass runs without `torch.no_grad()`: it builds
graphs it never uses — about 2% of a run's time, no change to any result — and
the fit check measures it as it is.

Rehearsed end to end on the toy Ukrainian condition of §33.53: runs with rotary
positions, RMSNorm and SwiGLU train, record their parameters (core 796,992
against the standard 793,344) and reload from their checkpoints with their
settings. The two new fields change the run ids of every run made from now on, as
each field added since September has (§33.48); labels and cache keys of all 411
recorded configurations are unchanged, and chains match cells by label and seed.


### 33.55 What the tokenizers expose, and codes read across corpus sizes

**The 17k condition at 0.05 is complete**: the zero slot and lexical case
derived — 5.58% of validation words are unseen in its 38M-token train split —
and every gate passed (`prepare_condition.py`, a few minutes, no tagging).

**Four Ukrainian tokenizers** (D7), trained on the 0.2 train split like
`bpe_uk_17k` (5.29M rows, 2.74M distinct words) and lossless by round trip:
BPE at 4k, 8k and 32k, and Unigram at 17k. Measured on 3,000 held-out
paragraphs — 15,740 inflected words, 12,701 of them nouns, proper nouns or
adjectives (`train_tokenizer.py --stats`):

| tokenizer | characters per token | tokens per word | words split | stem\|ending boundary | case-bearing words |
|---|---|---|---|---|---|
| BPE 4k | 2.73 | 2.37 | 64.7% | 21.5% | 20.4% |
| BPE 8k | 3.13 | 2.04 | 55.0% | 14.5% | 13.8% |
| BPE 17k | 3.59 | 1.73 | 43.3% | 9.1% | 9.0% |
| BPE 32k | 3.96 | 1.53 | 34.2% | 5.7% | 5.6% |
| Unigram 17k | 3.07 | 1.85 | 62.9% | **41.1%** | **43.7%** |

BPE hides endings as it grows: each doubling of the vocabulary removes about a
third of the remaining stem|ending boundaries, the words absorbed whole.
**Unigram at 17k exposes four and a half times as many endings as BPE at 17k**
while compressing like BPE at 8k: it splits more words, and splits them where
the morphology is. When a boundary falls at the ending, the ending is nearly
always one token (the "own token" share trails the boundary share by at most
half a point), so the two measures say the same thing. D7 therefore has two
axes: the vocabulary within BPE (exposure 21.5% → 5.7%, compression changing
with it) and the algorithm at a fixed size (9.1% against 41.1% at 17k). The law
predicts that case helps most where endings are hidden — BPE 32k — and least
where they are exposed — Unigram 17k. Predictions are recorded when the chain
is planned; the four conditions at 0.05 are being prepared on the CPU.

**Codes read across corpus sizes (G-72).** The condition's own checks showed
it. Before its derivations, the validation check reported the null firing on
92.9% of word tokens; after them 78.7% — though the null had not been derived
again — and the zero slot 87.8%. A corpus's validation split is the same text
at every fraction, so the Ukrainian validation streams at 0.002, 0.01, 0.05 and
0.2 are the same 325,867 tokens under four cache stems. The guard that keeps
codes with their stream compared token counts, and the check, a derivation's
source lookup and a run's own lookup each took any file with the right count —
in name order, or the fullest. For built codes, spaCy's tags of the same
tokens, that is harmless. For derived codes it is not: a null or a word-form
lexicon is built from its fraction's own train split, and a word unseen there
gets the class `none` — in training, the mark of a non-word. The three numbers
are three lexicons: 0.2's (7.1% of validation word tokens marked unseen),
0.05's (12.2%), and 0.01's (21.3%), which the 0.05 zero slot's derivation had
copied into its file.

Reconstructed from the run records and the derivation records — and reproduced
exactly on the toy corpus — from the night of 1 October, when the 0.01 null was
derived (its stem sorts first), **every Ukrainian null run read the 0.01
fraction's validation null, whatever its own fraction**:

| null runs, 4 seeds each | read | unseen marks against its own lexicon |
|---|---|---|
| ~1.5M tokens (0.002), 10k steps | 0.01's | fewer |
| ~38M (0.05), 10k steps | 0.01's | more |
| 151M (0.2), 5k, 10k and 20k budgets | 0.01's | more |

The chain logs, where every run printed the file it reused, confirm the
reconstruction exactly: 24 runs reused the 0.01 validation null — the 20 above
and the four 0.01 runs — 18 reused 0.2's own files (the null at 50k and added,
the zero slot, lexical case), 8 reused the 0.01 lexical-case file, and every
English reuse in the logs is of its own stream's files.

Unaffected: the ~7.6M runs (0.01, their own), the 151M runs at 50k and every
zero-slot run (all before that night), contextual case (built codes), lexical
case (its files at 0.002, 0.01 and 0.2 all hold the 0.2 lexicon by design —
identical rows), and English (one stream per tokenizer). So the nulls behind
§33.31–§33.32 (case over its matched null at ~1.5M, ~38M, and 151M at 10k and
20k steps; P57, P60–P64), §33.38 (Ukrainian case at 5k and 20k; P73, P76) and
the Ukrainian points of the learning law (§33.35) were scored on another
fraction's lexicon. If unseen marks hurt a null, more of them than its own
inflate the gain (~38M; 151M at 5k–20k) and fewer flatter the null (~1.5M).

**The fix.** A derived file is read only by the stream it was derived for
(`stream.stream_codes_files`, now used by the loader, by a derivation's source
lookup and by the check). Built files are still shared by token count, and an
older build file still lends a fact the exact file lacks. A test builds three
fractions' files and requires every run to read its own; it fails under the old
rule. The check names the file each fact was read from. On the toy corpus the
old code reproduced the symptom exactly (the null at 44.9% both before and after
the derivations, the zero slot 39.0%); after the fix both read 39.0%. The 0.05
zero slot's file must be derived again; its own row was right, the copied null
row was 0.01's.

**The audit** (`scripts/audits/codes_audit.py`) lists, for every recorded run,
the validation file it read under the old rule — over the files that existed
when it started — against its own stream's file, and checks every copied row
against the file that derived it. With `--score` it re-scores the affected
runs' saved models — the best step's weights — on the codes they were scored
on, which must reproduce the recorded BPB or the reconstruction is wrong; on
their own fraction's codes; and on the constant itself. The chain logs give a
second witness: every run printed the file it reused (`[codes reuse] …`).
Rehearsed on the toy corpus: the as-read scores reproduce the records to four
decimals.

**A question under it (G-73).** Even on its own lexicon the null is not constant
at validation: every word unseen in train is marked `none`, a mark the null met
only on non-words in training. The contextual feature's codes carry no such
mark — spaCy tags every word — so the null may carry a handicap that grows as
the data shrinks. On the toy corpus, where 38% of validation words are unseen,
a null trained for 600 steps scores 0.0083 better on the constant itself — its
training codes exactly, every word one class. `--nulls` measures this for every
constant null; if it is not negligible, the null's validation codes become the
constant, the existing checkpoints are re-scored, and the comparisons restated.

Predictions, before the audit:

- **P110** (~85%). On its own fraction's codes the null scores better than
  recorded at every seed at ~38M tokens and at 151M with 5k, 10k and 20k
  budgets, and worse at ~1.5M.
- **P111** (~75%). On the constant itself every Ukrainian null scores better
  than on its own codes, at every seed.
- **P112** (~50%). For the 151M null at 50k that improvement is smaller than
  0.001.
- **P113** (~65%). For every English constant null it is smaller than 0.0005.

### 33.56 What the nulls read, and a null that was not constant

**The audit** (`scripts/audits/codes_audit.py`, 6 October). Of 343 recorded runs
that read context codes, 319 read their own stream's file, 4 another stream's
file with identical rows (lexical case at ~1.5M tokens, whose lexicon is the
0.2 one by design), and 20 read different codes — exactly the 20 null runs of
§33.55. None would miss its codes under the new rule. One file held a foreign
copy, the 0.05 zero slot's null row (7.1% of tokens); derived again, every
copied row equals its original. Re-scored on the codes the audit says they
read, all 105 saved models — the 20, and 85 constant nulls that read their own —
reproduce their recorded BPB to four decimals, so the reconstruction is exact.

**G-72, the size of it.** On its own fraction's codes each affected null scores
differently at every seed: worse by 0.0019 ± 0.0001 at ~1.5M tokens (its own
lexicon marks more words unseen), better by 0.0010 at ~38M and by 0.0013–0.0014
at 151M with 5k, 10k and 20k budgets. **P110 right.**

**G-73: the null was not constant.** On the constant itself — its training
codes' rule, every word one class — every Ukrainian null scores better than on
its own codes, at every seed:

| null runs | constant − own codes | validation word tokens marked unseen |
|---|---|---|
| 151M, at 5k, 10k, 20k and 50k budgets alike | −0.0005 | 7.1% |
| 151M, the null added to the token dims | −0.0007 | 7.1% |
| ~38M | −0.0010 | 12.2% |
| ~7.6M | −0.0020 ± 0.0001 | 21.3% |
| ~1.5M | −0.0039 ± 0.0002 | |
| English, 71 runs | at most −0.0002 (mean −0.0001) | |
| zero slots, both languages | 0.0000 | (codes never reach the model) |

**P111, P112 and P113 right.** The handicap grows with the share of held-out
words unseen in the null's train split, and a morphologically rich language
has many more unseen word forms: in English it is noise, in Ukrainian it is the
size of the effects measured. It contradicts the null's own definition — "every
word given the same class" (`MorphConstBlock`) — so it is a bug, not a design:
the constant was a lexicon lookup, and a word missing from the lexicon got
`none`.

**The fix.** `--transform const` now gives every word the class at every split,
seen in train or not, and the null codes new text (BLiMP) the same way, with no
lexicon. Train codes cannot change — every train word is in its lexicon — and a
derivation that replaces a file now reports how many tokens it changed: on the
toy corpus, 0 in train, the unseen words in validation. A constant's codes file
declares the rule (`rules`); a run of the null refuses codes that do not, with
the command that fixes them; and every run records the codes file and rules it
read per split (`data.<split>.codes_file`), so the next audit needs no
reconstruction. `scripts/data/rederive_constants.py` replays every constant
derivation in the records, in order, resumable. Quarantine rule `const_unseen`
takes the 34 Ukrainian runs with `mconst` among their facts; English runs stay
(at most 0.0002), and so does case beside a zero slot.

**Restated** — Ukrainian, against the constant null, from the saved best-step
models (the audit's printed scores, four decimals):

| | recorded | restated |
|---|---|---|
| case over the null at 10k steps, ~1.5M tokens | +0.0096 ± 0.0016 | **+0.0076 ± 0.0016** (4/4) |
| ~7.6M | +0.0052 ± 0.0002 | **+0.0031 ± 0.0002** (4/4) |
| ~38M | +0.0052 ± 0.0007 | **+0.0032 ± 0.0007** (4/4) |
| 151M | +0.0044 ± 0.0004 | **+0.0026 ± 0.0004** (4/4) |
| 151M at 5k steps | +0.0074 ± 0.0007 | **+0.0055 ± 0.0006** (4/4) |
| 151M at 20k steps | +0.0015 ± 0.0004 | **−0.0003 ± 0.0004** (1/4) |
| 151M at 50k steps | −0.0009 ± 0.0008 | **−0.0013 ± 0.0008** (1/4) |
| word form alone, ~1.5M | +0.0070 ± 0.0013 | +0.0050 ± 0.0013 (66% of case's gain) |
| word form alone, ~7.6M | +0.0034 ± 0.0004 | +0.0014 ± 0.0004 (43%) |
| word form alone, 151M at 50k | −0.0015 ± 0.0010 | −0.0020 ± 0.0010 |
| the null over the plain baseline, ~1.5M | +0.0079 ± 0.0016 | +0.0099 ± 0.0016 |
| ~7.6M | +0.0065 ± 0.0017 | +0.0085 ± 0.0017 |
| 151M at 50k | +0.0023 ± 0.0010 | **+0.0028 ± 0.0010** |

**The claim stands, smaller.** Before the model has learned case, case helps —
at every seed and every corpus size at a 10k budget: about +0.003 from 7.6M to
151M tokens, +0.008 at 1.5M — and on 151M it is gone by 20k steps, not 50k.
Read from the word form alone it keeps two-thirds of the gain at 1.5M and
two-fifths at 7.6M, so **P60 is half right restated** (at least half was
predicted). The other verdicts hold: P57, P58, P63, P64 (−0.0003, between
+0.0026 and −0.0013), P73 and P76 right; P61 and P62 still wrong (+0.0026 is
outside ±0.002; at ~38M +0.0032 against +0.0031 at ~7.6M — still not below).

**What re-scoring cannot restate.** Numbers read from evaluations during
training — Ukrainian case at step 2,000 (+0.0060 ± 0.0005), the law's Ukrainian
points (§33.35), the steps the feature saves (§33.33) — are curves, not saved
models. The handicap was the same 0.0005 at four training lengths on 151M, so
they move by about the measured offset: case at step 2,000 ≈ +0.0055; the law's
points at null BPB 1.10 ≈ +0.0030 (50k), +0.0023 (20k), +0.0031 (10k), +0.0036
(~38M). Re-running the Ukrainian nulls makes them exact: training is
deterministic (G-69: bit-identical at 21 evaluations) and the null's training
codes are unchanged, so the same models come back with every evaluation on the
constant — and their best-step BPB must equal the audit's re-scores, a check.

### 33.57 The constants replayed, and the night of 6 October

**The replay** (`rederive_constants.py`, 18:48–19:00): all 13 constant derivations
in the records — seven English (the null, its three widths, the three zero
slots) and six Ukrainian — derived again. Every train split changed by **0
tokens**, 120.6M English and 151.1M Ukrainian among them; only held-out words
unseen in train changed: 1.44% of English validation tokens (2.60% of test), and
in Ukrainian 5.54% at 151M, 9.47% at ~38M, 16.54% at ~7.6M and 26.89% at ~1.5M.
The audit then found every copied row equal to its original and all 20 constant
files constant on every word, the rule declared. The quarantine's dry run named
exactly the 34 runs expected.

**The night chain** (`scripts/chains/night_2026-10-06.sh`, ~7.7 h, six resumable
pieces, ordered by what each settles):

1. **G-71** — the flat slot and the flat vector added, *without* the causal
   shift, under rotary positions (seed 0): a pure constant, the same initial
   weights, no window-start marker.
2. **G-73** — the ~38M-token Ukrainian null again, seeds 0–3: the 17k reference
   of D7, and the first check that a re-run reproduces the audit's re-score.
3. **G-73** — the 151M null at 50k again, seeds 0–1: early help at step 2,000,
   and Q2's end, on the constant.
4. **D7** — each new tokenizer whose condition is ready: the null, case and
   lexical case at ~38M tokens, 10k steps, seed 0; and lexical case under BPE 17k.
5. **G-73** — the 151M null at 50k, seeds 2–3.
6. **D7** — the tokenizers not ready in piece 4, waiting while they are prepared.

Before piece 2 the chain makes sure the 34 runs are out of `results/` (else
`--done` would take them for finished), moving them only if the dry run is
exactly the one read. If no preparation is running when it starts, the chain
prepares the D7 conditions itself on the CPU. Rehearsed on the toy corpus, real
code end to end: the quarantine moved 34 stand-in runs; every piece trained its
cells; a toy Unigram condition, prepared by the chain's own CPU track, was not
ready in piece 4, waited for in piece 6, then trained; a second pass skipped every
run; an unexpected quarantine skipped pieces 2, 3 and 5. The rehearsal caught one
fault: piece 1's smoke test took the default English tokenizer path, which a toy
has not got — the tokenizer is now a knob (`ENTOK`) the smoke test shares with
the runs.

Predictions:

- **P114** (~65%). Without the causal shift the flat slot under rotary positions
  keeps at least half its gain over the plain baseline at seed 0 (≥ +0.0038; with
  the shift +0.0076): the shared vector, not the window-start marker, carries it.
- **P115** (~65%). Without the causal shift the flat vector added keeps at least
  half its gain over the plain baseline at seed 0 (≥ +0.0036; with the shift
  +0.0072).
- **P116** (~85%). Re-run on the constant, each Ukrainian null ends within
  0.0002 of the audit's re-score of its quarantined checkpoint.
- **P117** (~70%). At step 2,000 the re-run 151M null evaluates 0.0002–0.0008
  below its quarantined record at every seed re-run: the handicap is about the
  same early as late.
- **P118** (~65%). Under each of the four new tokenizers case beats its null at
  the end of a 10k budget on ~38M tokens (seed 0).
- **P119** (~55%). Case gains less under Unigram 17k (endings exposed in 41% of
  inflected words) than under BPE 17k (9%), seed 0.
- **P120** (~55%). Case gains more under BPE 32k (5.7%) than under BPE 4k
  (21.5%), seed 0.

One seed per tokenizer reads a direction, not a size: the per-seed spread of
case's gain at ~38M is about 0.0014.

### 33.58 The night of 6 October: the vector, the exact null, five tokenizers

The chain ran all six pieces in 7 hours 34 minutes (results_71: 23 runs). No
preparation was running when it started, so its CPU track prepared all four
tokenizer conditions itself, each in 1 h 15 min to 1 h 31 min of tagging; the
8k condition was ready only for piece 6, which ran it. (The preparation's pilot
estimate, 2.4–2.7 h, runs about 1.8 times long: the pilot's fixed start-up cost
is counted as tagging. About 1.3 h per condition at 0.05 is the planning figure.)

**G-71 closed: the vector, not the window-start marker.** Without the causal
shift — a pure constant, no sentinel at the window's first word, the same
initial weights — under rotary positions at seed 0 the flat slot ends 1.3840
and the flat vector added 1.3849, against the plain baseline's 1.3908:

| seed 0, rotary, 20k | with the shift (marks the window start) | without it |
|---|---|---|
| flat slot, over the plain baseline | +0.0076 | **+0.0068** (89%) |
| flat vector added, over the plain baseline | +0.0072 | **+0.0059** (82%) |

**P114 and P115 right.** The marker is worth 0.0008 and 0.0013 here — the two
cells of each pair start from the same weights — and the shared vector carries
the rest.

**G-73 closed: the re-runs are the audit's re-scores.** Run again on the
constant, every null reproduced the audit's re-score of its quarantined
checkpoint to the sixth decimal (seven of eight exactly; one 50k seed by
0.000003, its best step 49,500): training is deterministic and the null's
training codes were unchanged, so the restated best-step numbers of §33.56 are
exact. **P116 right.** Along training the handicap grew from 0.0002 at step
1,000 and 0.0004 at step 2,000 to 0.0005 from step 5,000 on, at every seed
(0.00038–0.00047 at step 2,000): **P117 right.** Against the re-run nulls,
Ukrainian case leads at step 2,000 by **+0.0056 ± 0.0005** (4/4; first recorded
+0.0060 ± 0.0005), and the 151M curve now matches the ~38M one at the same null
BPB — +0.0055 against +0.0049 at 1.25, +0.0030 against +0.0036 at 1.10.

**D7, seed 0 — case under five tokenizers** (~38M tokens, 10k steps; gains over
each tokenizer's own null):

| tokenizer | endings exposed | null BPB | case | word form | case at step 2,000 |
|---|---|---|---|---|---|
| BPE 4k | 21.5% | 1.2239 | +0.0047 | +0.0020 | +0.0082 |
| BPE 8k | 14.5% | 1.1542 | +0.0017 | −0.0004 | +0.0052 |
| BPE 17k | 9.1% | 1.0891 | +0.0028 | +0.0002 | +0.0064 |
| Unigram 17k | 41.1% | 1.0890 | +0.0034 | +0.0001 | +0.0066 |
| BPE 32k | 5.7% | 1.0449 | +0.0035 | +0.0011 | +0.0053 |

**P118 right**: case beats its null under every tokenizer, and leads early under
every one. **P119 and P120 wrong**: Unigram, which exposes the most endings,
gains a little more than BPE at the same size, and BPE 32k less than BPE 4k. At
one seed nothing here orders by ending exposure — BPE 17k's four seeds spread
0.0014 — so the exposure hypothesis has no support yet. Read at the same null
BPB (the law's axis), case's gain is about the same under every tokenizer: at
1.25, +0.0055 (4k), +0.0052 (8k), +0.0060 (17k), +0.0057 (Unigram), +0.0054
(32k). The one departure is Unigram early: +0.0112 at null BPB 1.40 against
+0.0046 for BPE 17k — the opposite of what hidden endings would predict. So far,
what sets case's gain is how much the null has learned, not how the tokenizer
cuts words.

Two side observations. At the same 10k steps Unigram 17k reaches the same BPB as
BPE 17k (1.0890 against 1.0891) while reading 15% less text per step (3.07
characters per token against 3.59). And across BPE sizes the null's BPB falls
from 1.2239 to 1.0449 — the larger vocabulary reads more text per step and holds
more parameters (1.8M at 4k, 8.8M at 32k, the same 0.8M core) — which is why the
comparison across tokenizers is made at equal null BPB.

### 33.59 What the feature carries, probed — and the thesis kept in scope

**D2: the previous unit, probed.** At each position a causal feature carries the
class of the *previous* unit — the previous word, or a run of punctuation (its
last token's class) — and the window's first unit gets the sentinel (§30). The
probes so far read the *current* word's tag. `probe.py --previous` labels every
position with what the feature would carry there, shifted by the model's own
functions (`word_runs` over `_word_starts`, then `causal_shift`), word starts from
the model's tables. A test checks the labels three ways, on Ukrainian text whose
letters and dashes the tokenizer splits into byte fragments and in a window that
opens inside a word: against units read independently from the lexicon's word
spans; with the model's tables as with the tokenizer's; and against the model
itself — a causal embedding fed the current labels puts in its slot exactly what
a plain one puts there for the probe's labels. It fails for a shift by one token,
for split characters ignored and for a wrong sentinel. Rehearsed end to end on toy
runs of two widths, with checkpoints.

**The desk half** (`scripts/analysis/probe_axis.py`): every probed null's
accuracy at each checkpoint, paired with the feature's gain at the same step over
the seeds both cells share; the widths compared within each budget at six levels
across the accuracy range both reach; and each condition's accuracy at common
null BPB. On the current-word records it reproduces §33.52 — at 0.870, +0.0133
against +0.0076; at 0.880, +0.0097 against +0.0036 (recorded +0.0096; the
interpolation) — and gives the verdict the pivot needs: along the 20k budgets the
widths' gains differ by up to **0.0063** (at 0.874), along the 10k budgets by up to
0.0052. Up to an accuracy of about 0.85 they agree within 0.0013 in both budgets;
they part where the wider model's gain collapses. At matched null BPB the wider
null leads on the current word's tag by +0.8 to +1.3 points (1.75 to 1.45).

**The block** (minutes of GPU): the previous-unit probe on six checkpointed runs,
seed 4 — the four nulls, 128 and 256 wide at 10k and 20k budgets (D2), and D3's
switched pair, off from the start and off at 5,000. A check before anything is
read: until step 5,000 the switched model's input carries exactly the probe's
label, so at its layer 0 it must read the previous unit's tag far better than the
model never fed it (by 10 points or more) — and about level after the switch.

Predictions:

- **P121** (~15%). The pivot. By `probe_axis.py --previous`, the two widths line
  up along the 20k budgets: at every common level of the deepest layer's
  previous-unit accuracy their part-of-speech gains differ by at most 0.0015 (on
  the current word's tag: up to 0.0063). Against it: the wider model's gain falls
  to the base model's final +0.0075 by step 2,000, so to line up it would have to
  know the previous word's tag at step 2,000 as well as the base model does at
  20,000.
- **P122** (~60%). At each of the seven matched null BPB levels from 1.75 to
  1.45 (20k budgets), the wider null's deepest-layer lead on the previous unit's
  tag exceeds its lead on the current word's (+0.8 to +1.3 points): the width
  effect lies in reading context.
- **P123** (~60%). At the end of the 20k budget the model switched off at 5,000
  reads the previous unit's tag at the deepest layer within 0.5 points of the
  model never given the feature: the lasting deficit (§33.52) is not missing
  knowledge of what the feature carried either.
- **P124** (~55%). At the end of its 20k budget the 128-wide null reads the
  previous unit's tag at the deepest layer below 0.85 (the current word's: 0.881)
  — at the end it still lacks much of what the feature tells it, as the
  feature's remaining +0.0075 would suggest.

**The thesis kept in scope** (7 October: every theory presented
and every test run must serve the thesis — context-aware morphological embeddings
for language models). D1–D5 and D7–D9 ask when, why, and how far morphology in
the embedding helps. D6 asks how a shared input vector helps a small transformer.
For the thesis it matters as the confound it explains: what looked like word
structure helping was the slot itself, which is why every feature is measured
against a same-shape null — and G-71, the one test that could have changed that
reading, is closed (§33.58). Its remaining tests — the fixed against the learned
vector, G-71's seeds 1–3, the vector added under rotary positions at seed 3, the
256-wide pair under rotary positions — and the 256-wide spread at seeds 5–7 (P91,
P92) are about the vector itself: parked, P91 and P92 not tested. FINDINGS now
presents the vector as that confound, and D8 drops its RMSNorm test of D6.

**Other tokenizers** (are BPE variants enough?). A
controlled comparison of 64 models (350M–2.4B parameters) found the best subword
tokenizers — BPE, Unigram, WordPiece and two BPE variants — statistically
indistinguishable on downstream tasks (Schmidt et al. 2024, arXiv 2402.18376);
and morphological alignment did not explain why morphologically complex
languages are modelled worse — data size, counted in bytes, did (Arnett and
Bergen, arXiv 2411.14198). D7 already crosses the algorithm (BPE, Unigram) with
the vocabulary size, and at the same null BPB case gains about the same under
all five (§33.58). Tokenizers that differ in kind are outside the design: superword
tokens span several words, so a word's feature has no token of its own
(SuperBPE, arXiv 2503.13423); byte-level patching needs its own architecture
(BLT, arXiv 2412.09871). What D7 lacks is the far end of its own axis: a
tokenizer that exposes every inflectional ending — morphology in the tokenizer,
the thesis's alternative to morphology in the embedding (morphology-aware BPE:
MorphBPE, arXiv 2502.00894). For Ukrainian, a BPE that never merges across a
stem|ending boundary, the ending found from the word form alone so that it reads
no context. A candidate (DIRECTIONS D7): code first, then a
condition (~1.3 h of tagging) and its runs.

### 33.60 The previous word's tag: half the width gap, and what the switch leaves

The probe ran on all six runs (results_72: six records, about six minutes). The
check passed: until step 5,000 the switched model's input carries the probe's
label, and its layer 0 reads it at 0.997–1.000, against 0.33–0.36 for the model
never fed it; after the switch, 0.362 against 0.359. The labels are what the model
reads.

**What the nulls know of the previous word's tag** (seed 4; majority class 0.181;
from the current token alone, layer 0, 0.44): far less than of the current
word's. At the end of the 20k budgets the deepest layer reads it at **0.619** (128
wide; best layer 0.655) and 0.731 (256 wide; best 0.805), against 0.881 and 0.895
for the current word's tag. **P124 right.** The wider model's lead lies where
context is read: at matched null BPB from 1.75 to 1.45 it leads by **+3.9 to +10.4
points** on the previous word's tag, against +0.8 to +1.3 on the current word's.
**P122 right.**

**The pivot: closer, not lined up.** On the previous word's tag the two widths'
gains differ by up to **0.0035** along the 20k budgets and 0.0025 along the 10k
budgets, above the 0.0015 the pivot set. **P121 wrong.** Yet each axis closer to
what the feature carries cuts the widest gap — on the null's BPB 0.0098 (20k) and
0.0099 (10k), on the current word's tag 0.0063 and 0.0052, on the previous word's
0.0035 and 0.0025 — and the mean gap halves (+0.0034 to +0.0017, 20k). At the top
of the shared range the widths meet exactly: the base model at the end of its 20k
budget and the wider one at step 2,000 both read the tag at 0.62 and both gain
+0.0075 — the coincidence P121's own argument held unlikely. The deepest layer
lines up best (the others: 0.0036–0.0042 along 20k). One probed seed per width.

D2 closes as its pivot said: the law is stated within one model size. It closes
with a mechanism in hand. The width effect is largely how early a model reads the
previous word's part of speech from context — about half of what separates the
widths' curves, not all of it. And the feature stops paying long before the null
knows the tag fully: the wider model's gain has collapsed (+0.0033 at step 3,000)
when its deepest layer reads the tag at 0.68.

**D3 — the switched model rebuilds the tag, and ends knowing more of it.** One
checkpoint after the switch its deepest layer reads the previous word's tag 9.5
points below the model never fed it (0.443 against 0.538); 1,000 steps later level
(0.556, 0.557); ahead from then on. At the end of the 20k budget: +2.0 points at
layer 4 (0.635 against 0.615) and +1.8 at layer 3, but −1.6 at layer 2 and −1.2
at layer 1. **P123 wrong by its rule** — 2 points, the other way. The lasting
deficit (−0.0039 BPB) is not a lack of what the feature carried: the switched
model reads it from context as well as the never-fed one, at its deep layers
better.

**The proposed tokenizer, first look at its ending rule** (desk; the toy
Ukrainian corpus, 74k words). The D7 exposure measure takes the stem to be the
longest common prefix of a word and its lemma. Where the lemma's own ending
begins like the word's, that puts the boundary inside the ending — `книгами`
(lemma `книга`) becomes `книга|ми`, `важливих` becomes `важливи|х` — and citation
forms are not counted at all. A rule that reads the word form alone — the lemma
from pymorphy3's best parse with no context, minus its citation ending by part of
speech (`книга` → `книг`), the ending whatever follows the stem — gives
`книг|ами`, `книг|а`, `важлив|их`, `зібра|лися`, `написа|в`. It splits 54% of word
tokens and declines where the stem changes beyond recognition (`ішов`, `дітьми`);
where both rules split a word, they disagree on 9% of boundaries. Its faults:
alternations shift the boundary (`кни|зі`), and indeclinable words are split
(`метр|о`). Whatever rule the tokenizer uses, D7's exposure must then be measured
by the same rule, for every tokenizer.

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

### 33.71 The manuscript begins

From 9 October the thesis manuscript is drafted from what is established. Its documents —
the manuscript, the article, the practice report — live in `manuscript/`, outside the
repository. `paper/`, which scripts redraw after every chain, is never edited by hand:
anything edited there would be overwritten by the next chain's export.

### 33.72 paper/ becomes exports/, kept current

`paper/` was born in September for the journal article: the revision register took the
article's tables and its reproducibility appendix from `paper/runs.md`,
`paper/parameters.md` and `paper/environment.md`, generated by `export_results.py` so
that no number was copied by hand. The article itself (`manuscript/publikatsiia/`,
published, and not to change) was written from them and from the results record — its
Fig. 1 is results §5.15 exactly, drawn outside the project. Since then the folder had
become the general export of the run records, and its old name suggested a manuscript.
It was renamed `exports/` (9 October) and is kept current: the manuscript takes its
numbers from it.

It had not been current. The four demo figures were drawn on 1 October and never again;
fig3 still plotted the Ukrainian numbers G-72 and G-73 restated (+0.0096 at ~1.5M tokens
where the restated value is +0.0076). Redrawn, fig3 kept two of its six points and fig4
lost its two little-data bars: they rest on Ukrainian nulls quarantined under G-73 and
not yet re-run. A figure that silently drops points reads as another result, so
`demo_figures.py` now names every run each figure is planned to show and does not draw a
figure with any of them missing; the stale fig3 and fig4 went to the trash, and the
Ukrainian re-runs that would restore them are in the queue. `scripts/ops/refresh_exports.sh`
regenerates every table and figure in one command; chains end with it, and it runs
after every desk result.

Two arms of the English Q1 comparison are in the record, and the article and FINDINGS
had headlined different ones — the deterministic arm (13.15%, against Unigram 12.29%,
held-out test 12.0%) and the arm with an unsupervised fallback (13.5%, 12.6%, 12.9%).
FINDINGS now names both; the manuscript must name its arm.

*Corrected 9 October 2026: the article is accepted for publication — the editorial
board's letter of 7 October — not yet published. It does not change, apart from what its
proofs bring. Since §33.71 the manuscript's folders are renamed (`manuscript/article/` was
`publikatsiia/`); the paths above stay as written.*

### 33.73 The queue reviewed: what each candidate settles for the thesis

**Why now** (9 October). The manuscript has begun from what is established, and the next
experiments are chosen by what they would change in it: each candidate ranked by its
importance for the thesis, estimated from the recorded timings, with what depends on what
and which results would move the rest aside. The review is standing practice from here, a
step before every chain, and its product lives in one place, DIRECTIONS' "The queue",
rewritten at each review. This section records how the
first one was reached and the predictions for everything it prepared.

**How a candidate ranks.** By the claim its result could change: first what could
overturn or qualify a headline the manuscript states; then the question the thesis is
named for; then what completes the evidence for a claim the manuscript already cites;
then what carries a claim along an axis an examiner will ask about (data, width, model,
corpus); then the firmness of a secondary claim; within a rank, the cheaper first. The
manuscript's story since §33.65 has four supports (the manuscript's plan): the
method, the law, the two caught false positives, and **morphology in the tokenizer lasts
where morphology in the embedding fades**. The fourth rests on one corpus size (~38M
tokens) and budgets that end with the null at 1.015 bits per byte; at 151M tokens and 50k
steps — where case in the embedding gave nothing (§33.58) — it has not been measured. So:

1. **More data at full training, the tokenizer** — the endings-apart null at 151M tokens
   and 50k steps against BPE 17k's: whether the fourth support holds as stated.
2. **More data at full training, case on top** — "they add up" where under BPE 17k case
   ends below its null.
3. **The title's question** (G-38) — what the sentence's case adds over the word form's:
   context, or the tagger's right-context leak. FINDINGS leaves it "not yet separated";
   a thesis named *context-aware* should not. At the desk first (minutes), then a
   control in which prepositions carry their form's case (~1.6 h, new code).
4. **The five runs left on 8 October** — case at twice the budget is cited at three seeds.
5. **Where the wider model's lead settles** — "larger in a wider model", measured while
   the lead was still falling.
6. **The Ukrainian re-runs** — no claim moves (their numbers are exact, §33.58); they
   restore fig3 and fig4, and the manuscript has no figures of its own yet.
7. **Case in the wider model**; 8. **a Ukrainian grammatical test** (MultiBLiMP; needs
   the data downloaded); 9. **D3 at a 50k budget** (English); then D8 and D9.

What could move the rest is written beside each in DIRECTIONS; the largest: if the
tokenizer's lead does not survive full training (P152 wrong), the fourth support becomes
"fades later than a feature", the width questions (5, 7) lose their point, and placing
the fade comes first.

**Prepared.**

- `scripts/chains/long_2026-10-09.sh` — every quiet candidate (1, 2, 4–7, 9) as resumable
  pieces in rank order, ~25.9 h in all; `END_BY` keeps a start within the hours
  available, and `PLAN=1` prints, from the records and without running anything, what would
  run by then — the plan for any number of hours. It starts with a **reproduction check**:
  one of the twenty quarantined nulls (BPE 17k at 151M tokens, 5k steps, seed 0) is made
  again and must reproduce the audit's re-score of its checkpoint, 1.196103, within
  0.00001, or nothing else starts. The runs this chain pairs with were made between 29
  September and 8 October; a change in the code since would break every pairing without a
  sign, and a run whose twin's answer is known in advance is the one test that shows it.
  Then a 151M sanity guard (the endings-apart null at seed 0 must end below its own 20k
  null at ~38M, 0.9974), a width guard (each 256-wide 20k null below its 10k one), smoke
  tests per condition and width, disk. Rehearsed with the stub from a copy under
  `scratch/`, sixteen scenarios, each checked by its number of runs and its messages:
  every piece in order (54 runs); the reproduction check failing; each guard failing;
  the time box cutting the sets of three, and cutting the check itself; no disk; a
  finished chain; a condition not ready; smoke tests failing for width and for English;
  one piece alone; the plan at eight hours and in full; an unreadable `END_BY`.
- **G-38 at the desk:** `loss_by_kind.py --previous` puts every target in one class by the
  word the feature describes at the position that predicts it — read through the model's
  own boundary rule and shift — a preposition or another word, whose case in context
  differs from its form's majority case or not, or whose form gives none (unseen in
  train); it reports what case from the sentence adds over case from the form in each.
- **G-38's control:** `ucaseprep`, case in context except at prepositions — the word types
  whose majority tag in train is ADP — which carry their form's majority case at every
  occurrence (`derive_lexical_codes.py --recode-pos ADP`, with checks that every word
  not recoded keeps its value and that known prepositions are recoded and known pronouns
  are not). Its smoke test passed; its codes are derived only if the desk split points
  at prepositions.
- `scripts/ops/run_times.py` — recorded minutes by label, the estimates' source.

**Predictions**, before any of it ran.

G-38 at the desk (10k at ~38M under BPE 17k, Unigram 17k and endings apart, four seeds;
151M at 50k under BPE 17k, four seeds):

- **P148** (~55%). Under endings apart, more than half of what case from the sentence adds
  over case from the word form on the tokens holding an ending's start (+0.0023, §33.70)
  sits where the described word is a preposition whose case in context differs from its
  form's. Against: adjectives and nouns whose case the form leaves ambiguous are far more
  numerous, and their case in context speaks to the next word's ending too.
- **P149** (~65%). Under each tokenizer, the targets where both models are given the same
  input carry less than a fifth of the sentence's whole extra (+0.0021 to +0.0033).
- **P150** (~55%). Under endings apart, words whose form gives no case — unseen in train —
  carry less than a quarter of it.
- **P151** (~60%). At 151M tokens and 50k steps, where the sentence leads by +0.0006 ±
  0.0004 (§33.30), the preposition class's part is positive at three seeds or more: a
  leak gives what the past cannot, so it fades more slowly than context.

The long chain (paired by seed, from identical initial weights):

- **P152** (~75%). At 151M tokens and 50k steps the endings-apart null ends below BPE 17k's
  (0.9442 on average) at every seed, by more than 0.010 on average. For: at ~38M tokens the
  lead held near 0.017 while the null fell from 1.21 to 1.015, and is larger in a wider
  model. Against: with four times the text BPE 17k may learn what the cut gives.
- **P153** (~50%). That lead lies within 0.004 of the 20k lead at ~38M (+0.0169): +0.0129 to
  +0.0209.
- **P154** (~65%). Along the 50k schedule the final lead is at least three quarters of the
  lead at step 10,000: no fade over the last 40,000 steps.
- **P155** (~55%). Under endings apart, case beats its null at 151M and 50k on average and at
  three seeds or more (under BPE 17k there: −0.0013 ± 0.0008).
- **P156** (~70%). Paired by seed, case gains more under endings apart than under BPE 17k
  there, by more than 0.002.
- **P157** (~75%). Endings apart with case ends below BPE 17k's null there by more than 0.012.
- **P158** (~60%). At width 256 and a 20k budget the endings-apart null ends below BPE 17k's
  by more than 0.010 at every seed, and by more than width 128's 20k lead (+0.0169) on
  average.
- **P159** (~65%). That lead is smaller than at width 256 and 10k (+0.0254): still falling
  as the null learns.
- **P160** (~90%). Every re-run null reproduces the audit's re-score of its quarantined
  checkpoint on the constant (`codes_audit_20261006T151153`) within 0.00001.
- **P161** (~60%). At width 256 and 10k, case under BPE 17k ends less than +0.002 above its
  null (at width 128: +0.0032 ± 0.0007).
- **P162** (~65%). Paired by seed, case gains more under endings apart than under BPE 17k at
  width 256, by more than 0.001.
- **P163** (~50%). Switched off at step 5,000 of a 50k budget, English part of speech ends
  more than 0.002 below the model never fed it, at both seeds (at 20k: 0.0039 ± 0.0009,
  5/5).

P136–P140 wait for piece 3. Predictions for G-38's control are recorded once the desk
split has placed what it controls.

### 33.74 What the sentence's case adds: prepositions, and words the form never saw

**The split ran** (`loss_by_kind_20261009T035142`, 10k at ~38M, three tokenizers;
`loss_by_kind_20261009T035358`, 151M at 50k, BPE 17k; 49 s and 18 s of GPU). All 48
models re-scored to their records within 1.7 × 10⁻⁸. Each target is classed by the word
the feature describes at the position that predicts it, read through the models' own
boundary rule and shift; the classes are the stream's own codes — its part of speech, its
case in context and its word form's majority case. What case from the sentence (`ucase`)
gains over case from the word form (`ucaselex`), paired by seed, four seeds:

| described word | targets | BPE 17k | Unigram 17k | endings apart | 151M, 50k, BPE 17k |
|---|---|---|---|---|---|
| a preposition whose case in context is not its form's | 1.9–2.0% | +0.0007 | +0.0013 | +0.0013 | +0.0005 |
| a preposition, the same case | 6.4–7.0% | +0.0004 | +0.0006 | +0.0006 | +0.0003 |
| another word whose case in context is not its form's | 4.8–5.3% | +0.0002 | +0.0002 | +0.0002 | +0.0001 |
| a word unseen in train (the form gives `none`) | 1.9–3.2% | +0.0008 | +0.0007 | +0.0009 | +0.0003 |
| every other target — both models given the same input | 83–84% | −0.0000 | +0.0006 | +0.0002 | −0.0005 |
| **the whole extra** | | **+0.0021 ± 0.0003** | **+0.0033 ± 0.0001** | **+0.0032 ± 0.0003** | **+0.0006 ± 0.0004** |

Every cell of the first four rows is positive at four seeds of four. **Prepositions hold
most of it**: 55%, 57% and 60% of the extra at 10k, and on the tokens holding an ending's
start 64%, 62% and 71%. A preposition's case in context is the case of the noun after it
(на + Acc for a direction, на + Loc for a place), which the tagger reads from that noun —
the very word whose ending the feature then helps predict: G-38's right-context leak,
where §33.21 measured it. That it shows where the two inputs agree as well is the same
information seen from the other side: in the sentence's model a preposition's case is
always the next noun's, so even its majority value tells that model what the word-form
model can only guess. **Words the form never saw hold a fifth to two-fifths** (38%, 21%,
28%) — and there the word-form model is not merely uninformed but misled: it ends below
the null on those targets, −0.0006, −0.0005 and −0.0007 (0/4 each). Its `none` marks, in
training, a word without case — a verb, an adverb — and at validation it marks every word
the lexicon lacks: the null's fault of G-73 met again, in the control for the tagger's
leak (G-76). **Context resolving another word's case holds 5–8%** (+0.0002 under each).

At 151M tokens and 50k steps the whole lead is +0.0006 ± 0.0004 (§33.30), but not
because the leak has gone: the prepositions still give +0.0008 (4/4 in each class) and
the unseen words +0.0003, while on the targets where both models get the same input the
sentence's model is 0.0005 worse — case there costs away from the endings, as under BPE
17k at twice the budget (§33.70).

So what the "context-aware" feature adds over the word form's is, at ~38M tokens and 10k
steps, mostly what the tagger read from the right at prepositions and what the lexicon
never saw — context in the sense the thesis's title means, the sentence resolving a word
the form leaves ambiguous, holds under a tenth of it. **P148 wrong by its rule** (the
prepositions whose case differs hold 46% of it on the ending tokens under endings apart,
not over half; with the prepositions whose case agrees, 71%); **P149 wrong** (the targets
with the same input hold 21%, 36% and 24% of the extra — under Unigram the rest of the
stream too, +0.0006 ± 0.0001); **P150 wrong** (unseen words, 28%; on the ending tokens
14%); **P151 right** (+0.0005, 4/4).

**What it changes.** FINDINGS stops saying "context or leak, not yet separated": at the
desk they are separated, and it is mostly the leak. The word form's shares of case's gain
— two-fifths to two-thirds at little data (§33.32), about a third at ~38M — are lower
bounds, since the form's model loses on the words it never saw: the claim they support,
"it is case, not only the tagger's right context", only gains. G-76 is opened for the
word-form variants' `none` (it would not change a claim; the split separates it). And
the desk split pointed where the queued control looks, so the control was prepared.

**G-38's control, prepared.** `ucaseprep`'s codes were derived for BPE 17k and endings
apart at 0.05 (`derive_lexical_codes_20261009T035502`, `…T035553`; 40 s each): 415 and 414
word types whose majority tag is ADP, 11.4% of word occurrences; 97.6% of them are tagged
ADP in context, and 99.7% of the occurrences tagged ADP are recoded; a recoded
preposition's value changes at 23.3% of its occurrences (2.65% of all words; 1.1–1.2% of
tokens); every word not recoded kept its value, known prepositions were recoded and known
pronouns and nouns were not, and both caches pass the gate for the new fact. The block
(`scripts/chains/short_2026-10-09.sh`, ~1.6 h): `ucaseprep` at 10k, seeds 0–3, under both
tokenizers, beside the null, case and word form already recorded there; after seed 0 each
recoded run must end within 0.005 of case — the inputs differ at 2.65% of words, so a
larger gap means codes misread. Rehearsed with the stub: five scenarios.

Predictions for the control (paired by seed with case, its word form and the null):

- **P164** (~70%). Under endings apart, case with prepositions read from their form keeps
  less than half of case's lead over the word form (+0.0032): `ucaseprep` gains less than
  +0.0016 over `ucaselex`.
- **P165** (~70%). And more than a fifth of it (+0.0006): the unseen words' part stays.
- **P166** (~65%). Under BPE 17k it keeps less than half (`ucaseprep` over `ucaselex` below
  +0.0011).
- **P167** (~65%). Under endings apart its gain over the null on the tokens holding an
  ending's start is below +0.0017 (case: +0.0028; the prepositions' part there +0.0016).
- **P168** (~75%). Split as above, `ucaseprep` against `ucaselex` under endings apart: the
  two preposition classes — the same input in both now — hold less than a fifth of its lead.

Predictions P148–P151: P151 right; P148, P149 and P150 wrong.

*P160 refined, 9 October, before any run.* The ~1.5M-token nulls overfit: their best
step lies mid-schedule (4,500–5,000 of 10,000; they stopped at 7,500–8,000), and the
audit re-scored the checkpoint chosen on the old codes. On the constant a re-run may
choose a neighbouring step and score below that re-score without any change in the code.
So P160 holds where a re-run's best step is its twin's, within 0.00001, and at ~1.5M
tokens the re-run ends at or below the re-score. The chain's reproduction check uses the
5k null at 151M tokens, whose best step is its last.

### 33.75 Ten hours, and two decisions

**Two decisions** (9 October). Packs leave out `manuscript/` (`pack.sh`; `--list` does not
show it). MultiBLiMP's Ukrainian file is
downloaded into `external/multiblimp/` — 2,744 minimal pairs, all subject–verb agreement:
person 1,181, gender 1,041, number 522; CC BY 4.0 — for queue 8, a loader at the desk.

**Ten hours, rounded up.** The queue's top in order: the short block (3, ~1.6 h) and the
tokenizer and case at full training (1 and 2, ~7.9 h) take ~9.5 h; the half hour left
holds no whole cell, and this time the plan rounds up rather than down. So
the next whole cell joins: case at 20k for seed 3 under both tokenizers (~48 min), which
completes P136–P139; BPE 32k's seed 3 (P140) stays queued. `PIECE3=case` selects it. By
planned minutes ~10.3 h; with no `END_BY`, nothing is cut short by a slow run. One command,
the short block first, then the long chain with `PIECES="0 1 2 3" PIECE3=case`. Rehearsed:
the knob's two scenarios added, eighteen of eighteen pass. Predictions: P136–P139,
P152–P157, P160 (its reproduction check), P164–P168.

*The first launch, 07:16, ended in two seconds:* the launching shell had no conda
environment active, so `python` was not on PATH. Every smoke test "failed", no run
started, and the long chain stopped at its reproduction check, as designed: nothing
trained and nothing was written to `results/`. `guard.sh`, sourced by every chain, now
puts the `morph` environment's Python on PATH when none is there, and so does the long
chain's `PLAN`. Checked from a bare shell; both rehearsals pass. Relaunched with the
same command.

*Old material cleared, 9 October.* The archives that carried the work between the two
machines until 8 October — 120 of them, code, documents and records, all superseded by the
working copy — went to the trash. Three cited papers moved into `external/literature/` (`docs/reference/related_work.md`). Two older
documents were read first and then trashed: the article's draft v2, whose numbers unique to
it are all in the results record as the pre-repair and intermediate figures, and the
April research journey of the "tokenizer-free" project, whose decisions and facts §1–§10
already hold, and whose verdicts on tokenizer output Q1 overturned.

### 33.76 Morphology in the tokenizer lasts at full training — and case on top of it, in part through the leak

**The chain of 9 October** (`short_2026-10-09.sh`, then `long_2026-10-09.sh` with
`PIECES="0 1 2 3" PIECE3=case`) ran 19 runs from 07:18 to 16:50: the short block in 1 h 28
min, the long chain in 8 h 4 min. Every guard passed. A 50k run at 151M tokens took 53.9–54.5
minutes, a 10k run of the control 10.8–11.1.

**The reproduction check held.** The quarantined BPE 17k null at 151M tokens and 5k steps,
seed 0, made again, ended at 1.1961031 against the audit's re-score of its checkpoint,
1.196103 (best step 4,999 both): the code that made the records this chain pairs with
still makes them. One of P160's twenty; the rest wait for the re-runs.

**The tokenizer at full training, on four times the text** (`tokenizer_table_20261009T143624`;
151M tokens, 50k steps, paired by seed from identical initial weights). The endings-apart
null ends **+0.0165 ± 0.0008** below BPE 17k's (4/4; per seed +0.0153, +0.0178, +0.0151,
+0.0179) — 1.75% of BPE 17k's 0.9442, against 1.6% at ~38M tokens and 10k steps. Along the
schedule the lead retraces the shorter ones — +0.0456 at step 500, +0.0252 at 2,000, +0.0190
at 3,000 — and from step 5,000 to the end stays between +0.0161 and +0.0176, while BPE 17k's
null falls from 1.126 to 0.944. **P152 right; P153 right; P154 right** (the final lead is
99% of the lead at step 10,000). Across two corpus sizes and three budgets at width 128 the
tokenizer's gain is one number, about 0.017 bits per byte, wherever the null stands between
1.21 and 0.944. A feature's gain falls along the same axis to nothing; this does not.

**Case on top, at full training.** Under endings apart case beats its null at every seed,
**+0.0030 ± 0.0009** (4/4), where under BPE 17k it ends at −0.0013 ± 0.0008 (1/4); paired by
seed the difference is +0.0044 ± 0.0011 (4/4), and the two together end +0.0195 ± 0.0013
below BPE 17k's null. **P155, P156 and P157 right.** On the law's axis, under BPE 17k case's
gain crosses zero between null BPB 1.05 and 1.00 and ends −0.0014 at 0.94; under endings apart
it is +0.0046 at 1.00 and still +0.0033 at 0.94.

**Twice the budget, four seeds** (`tokenizer_table_20261009T143907`). With seed 3, case at 20k
steps under BPE 17k is +0.0010 ± 0.0008 (3/4), under endings apart +0.0038 ± 0.0008 (4/4),
the paired difference +0.0029 ± 0.0016 (4/4), and the two together +0.0208 ± 0.0011 below BPE
17k's null. **P136, P137, P138 and P139 right.**

**G-38's control: the leak, measured in trained models** (`tokenizer_table` at 10k;
`loss_by_kind_20261009T143714`; 32 models re-scored within 1.7 × 10⁻⁸). With prepositions
reading their form's case, every other word its case in context:

| ~38M, 10k, gain over the null | case | prepositions read from their form | word form |
|---|---|---|---|
| BPE 17k | +0.0032 ± 0.0007 | +0.0026 ± 0.0007 (4/4) | +0.0011 ± 0.0009 |
| endings apart | +0.0048 ± 0.0006 | +0.0035 ± 0.0009 (4/4) | +0.0017 ± 0.0007 |

Reading prepositions from their form costs +0.0006 ± 0.0002 under BPE 17k and +0.0014 ±
0.0004 under endings apart (4/4 each): **the leak is 29% and 43% of the sentence's lead over
the word form**, not the 55–60% the desk split placed after prepositions (§33.74). A model
trained without the leak recovers part of what the leak gave a model trained with it; the
per-target split compared two trained models, not a model with its counterfactual, and
overstated the part. The control keeps +0.0015 ± 0.0003 and +0.0018 ± 0.0002 over the word
form (4/4 each). Placed as before: nothing after prepositions (+0.0000 in both preposition
classes — so the "agreeing" prepositions' share of the sentence's lead was the leak's too,
through how far its model could trust the value); about half after words unseen in train
(+0.0008, +0.0009), where the word form misleads its model (G-76); +0.0002 and +0.0003 where
context resolves another word's case; +0.0004 and +0.0005 spread over targets with the same
input. **P164 wrong** (the control keeps 57% under endings apart, not under half); **P165
right**; **P166 wrong** (71% under BPE 17k); **P167 right** (+0.0012 ± 0.0002 on the ending
tokens); **P168 right** (the preposition classes hold none of the control's lead).

The leak also carries part of the exposure effect. With prepositions read from their form,
case under endings apart gains +0.0008 ± 0.0012 (3/4) more than under BPE 17k at the end of
the 10k budget, against +0.0016 ± 0.0010 for case: half. Along the way it is mostly not the
leak — at null BPB 1.15, +0.0035 ± 0.0015 (4/4) against +0.0043 ± 0.0013.

**And at full training?** The same split of case's gain over its null at 151M and 50k
(`loss_by_kind_20261009T143815`; `--previous` now takes one feature, against the null): under
endings apart +0.0017 of the +0.0030 sits after prepositions (+0.0009 and +0.0008, 4/4 each),
+0.0003 after another word whose case the form leaves ambiguous, +0.0010 ± 0.0008 (3/4) on
targets where no word's case differs; under BPE 17k the prepositions give +0.0010 and the rest
of the stream costs −0.0023 (0/4). So most of the tokenizers' difference at full training is
that case costs nothing away from the endings under endings apart (+0.0033 of the +0.0044 in
that class) — §33.70's pattern at full training — while over half of what case gains there
sits where the leak lives. Under endings apart the leak can reach its target: a noun's ending
is a token of its own, predicted from the noun's stem, where the feature describes the
preposition and carries the case the tagger read from that very noun; under BPE 17k the
ending is mostly fused into the noun's first token, predicted before the preposition is
described. How much of the +0.0030 survives without the leak is the first question of the
next chain: the desk split overstated the leak at 10k, and only a trained control measures it.

Predictions: P136–P139, P152–P157, P165, P167 and P168 right; P164 and P166 wrong; P160 one
twentieth read; P140 and P158–P163 wait.

### 33.77 The queue reviewed after the chain of 9 October

The review of §33.73 named what would move the queue; two of its pivots were met. The
tokenizer's gain held at full training, so the width axis is next for it, and larger
models and a second corpus rise behind it — the gain has held on every axis tried, at one
corpus. And case on top held too, but over half of its full-training gain sits where the
tagger's leak can now reach its target: a claim the manuscript is about to state could
change, so it is the first candidate, and the register has an open gap that could change a
claim again (G-77). The order (DIRECTIONS, "The queue"): G-77 at full training; MultiBLiMP,
for a measure beyond bits per byte; width 256 at 20k; BPE 32k's last seed; larger models
and a second corpus; the re-runs; case at width 256; D3 at 50k.

**Prepared.** `scripts/chains/long_2026-10-10.sh` — a reproduction check on another of the
quarantined nulls (BPE 17k at 151M tokens and 5k steps, seed 1: the audit's re-score,
1.193635), then case's word form and the control under endings apart at 151M and 50k, seeds
0–3, then the earlier quiet pieces. The control's codes at 0.2 are derived and gated
(`derive_lexical_codes_20261009T144030`: 672 preposition types, 2.66% of word occurrences
changed, as at 0.05), the word form's were derived with the condition. Rehearsed with the
stub, sixteen scenarios. `PLAN=1` puts the word form alone at ~4.1 h, both halves of G-77 at
8 h, and the whole chain at 25.1 h. `loss_by_kind.py --previous` now also places a single
feature's gain over its null.

Predictions for G-77 (151M tokens, 50k steps, under endings apart, paired by seed with case
and the null):

- **P169** (~55%). Case's word form beats its null on average and at three seeds or more.
  For: at 10k it kept a third of case's gain under endings apart (+0.0017 of +0.0048), and
  under endings apart case costs nothing away from the endings at full training (§33.76).
  Against: at 151M under BPE 17k the word form ends below the null (−0.0020 ± 0.0010).
- **P170** (~65%). Paired by seed, the word form gains more under endings apart than under
  BPE 17k (−0.0020) by more than 0.002.
- **P171** (~65%). Case with prepositions read from their form beats its null on average and
  at three seeds or more.
- **P172** (~55%). It keeps more than half of case's gain there (more than +0.0015).
- **P173** (~70%). Case beats it by more than +0.0005 on average: the leak is still worth
  something at full training, as the prepositions' part was under BPE 17k (P151).

MultiBLiMP's predictions are recorded when its loader is written.

### 33.78 Ten hours, 10 October

Ten hours were available. By the queue (§33.77): the reproduction check, both halves of
G-77 and BPE 32k's last seed take 530 planned minutes; of the seventy left, width 256 at 20k
(328) does not fit, and the next whole cells are re-runs. Piece 5 now takes its cells by
condition, each condition's four seeds together (`RERUNS`), so a time box can never split
one. Two fit: the 5k null at 151M tokens (seeds 2–3; seeds 0 and 1 are the two
reproduction checks) and the ~1.5M-token null at 10k — 18 runs, 584 planned minutes, ~9 h
by the run times of 9 October. The ~7.6M-token cell would leave three of its seeds for
later, so it waits. `PIECES="0 1 2 3 5" RERUNS="0.2:5k 0.002:10k"`, `END_BY` ten hours on;
rehearsed, eighteen scenarios. Predictions: P140, P160, P169–P173.

### 33.79 A grammatical test in Ukrainian: MultiBLiMP

**Why** (queue 2, §33.77). The tokenizer's gain holds on every axis tried — in bits per
byte. MultiBLiMP 1.0's Ukrainian set (2,744 pairs, `external/multiblimp/ukr/data.tsv`) breaks
subject–verb agreement in one verb of a sentence — number (522), gender (1,041) or person
(1,181) — which is what a Ukrainian verb's ending carries, and what keeping endings apart
exposes. A model is right when it prefers the grammatical sentence. It runs on the CPU,
beside the chain.

**The tool** (`scripts/eval/multiblimp_eval.py`). The English evaluator's scorer and its
batch check, with a Ukrainian paragraph's start: each sentence is encoded alone by the
checkpoint's own pipeline — its tokenizer with the ending marks, the tagger reading the
sentence — after a newline, the token before every paragraph in training, which is not
scored. Features are coded as the caches code them. Before anything is scored, the first
40 paragraphs of each configuration's validation stream are re-encoded by this path and
must reproduce the cached ids and every cached code exactly: they do, for the null under
both tokenizers, case under endings apart and the word form under BPE 17k; with the null
coded through its lexicon, as before G-73, 8 of 40 paragraphs fail — the check bites.

Predictions, before any pair was scored (overall accuracy over all pairs; paired by seed):

- **P174** (~60%). At 151M tokens and 50k steps the endings-apart nulls judge agreement better
  than BPE 17k's, by more than 0.5 points on average and at three seeds or more. For: the
  agreement sits in the ending, a token of its own there. Against: BPE 17k has seen the
  common verb forms whole, and the decision may hang on the subject, far back.
- **P175** (~70%). Every null at 151M and 50k scores between 65% and 92%.
- **P176** (~65%). Under BPE 17k at 151M and 50k, case moves accuracy by less than 1 point on
  average: the feature names a case, not the number, gender or person a verb agrees in, and in
  English no feature moved BLiMP (§33.9).
- **P177** (~55%). At ~38M tokens and 10k steps, endings apart beats BPE 17k by more than 0.5
  points at three seeds or more.
- **P178** (~50%). At ~38M and 10k, BPE 4k, which exposes 21.5% of endings, beats BPE 32k,
  which exposes 5.5%, by more than 1 point on average (three paired seeds).
- **P179** (~50%). At 151M and 50k, endings apart's advantage is largest in gender, the
  past-tense ending (-в, -ла, -ло, -ли), of the three.

### 33.80 Figures for the thesis, in Ukrainian

The thesis needs its figures in Ukrainian, without titles — the template captions every
figure below it — legible at 16 cm, and a figure of the tokenizer's lead, the thesis's main
positive result having none.
`scripts/analysis/thesis_figures.py` draws them in Ukrainian and English into
`exports/figures/uk/` and `exports/figures/en/`, and `refresh_exports.sh` runs it:

- **Shared settings:** each figure is exactly 16 cm wide at 300 dpi, with axis labels and
  legends at 10 pt. Ukrainian versions use decimal commas and the thesis's terms; panels are
  lettered.
- **Colours:** the categorical palette of the data-visualisation guidance, in its fixed
  order. It is validated there for line charts; the validator itself needs Node, which this
  machine lacks, so the palette is used unchanged.
- **The tokenizer figure:** colour marks data and budget, and line style marks width (the
  256-wide run is dash-dot in its condition's colour), so no two lines need hues told apart
  that the palette does not separate.
- **Missing runs:** as in `demo_figures.py`, a figure whose planned runs are missing is not
  drawn.
- **The law figure:** five panels in one row were 3 cm each at 16 cm. They became two
  figures of two panels, Ukrainian case (by data and budget; by tokenizer) and English part
  of speech (by schedule; by width), each legend under its panel.
- **Checked by rendering:** every version was looked at as rendered, and labels that clipped
  or collided were moved until none did.

**A confound to close first** (recorded after the 151M accuracies were seen, before this check
ran). A pair's two sentences can split into different numbers of tokens, and a model can
prefer one for its length. Under endings apart the two verb forms are mostly one stem and two
endings; under BPE 17k one form may be a single token and the other two. The tool now also
reports accuracy on the pairs whose two sentences have the same number of tokens under every
tokenizer scored together (`--by-length`).

- **P180** (~75%). On those pairs, at 151M tokens and 50k steps, the endings-apart nulls still
  beat BPE 17k's by more than 2 points, at every seed.

**Results** (`multiblimp_eval_20261010T065127`, 151M; `…T065611`, ~38M; `…T070238`, the 151M nulls
by length). Every configuration's scoring path reproduced its 40 validation paragraphs, and
every batch check passed. Overall accuracy, mean over seeds, and the paired difference:

| | BPE 17k | endings apart | endings apart − BPE 17k |
|---|---|---|---|
| 151M, 50k — all 2,744 pairs | 71.4% | 76.3% | **+4.9 ± 0.4** (4/4) |
| number · gender · person | 71.7 · 75.5 · 67.6 | 75.7 · 81.4 · 72.2 | +4.0 · +5.9 · +4.5 (4/4 each) |
| the 1,287 pairs of equal token length under both | 72.5% | 77.5% | +5.0 ± 1.1 (4/4) |
| ~38M, 10k — all pairs | 64.8% | 69.0% | **+4.3 ± 0.3** (4/4) |
| number · gender · person | 62.5 · 66.7 · 64.0 | 70.8 · 74.1 · 63.8 | +8.3 · +7.3 · −0.2 |

At ~38M and 10k the six tokenizers' nulls score:

| BPE 4k | BPE 8k | BPE 17k | BPE 32k | Unigram 17k | endings apart |
|---|---|---|---|---|---|
| 63.5% | 64.1% | 64.8% | 64.8% (3 seeds) | 64.7% | **69.0%** |

Exposing more endings by a smaller vocabulary (BPE 4k, 21.5%) or another algorithm (Unigram,
43%) does not help — BPE 4k is 1.2 points below BPE 17k (4/4). Only cutting exactly at the
ending does: where the cuts fall, as in bits per byte. Case, over its own null, moves nothing:
−0.2 ± 0.4 and +0.2 ± 0.5 at 151M (BPE 17k, endings apart), the word form +0.1 ± 0.4;
+0.4 ± 0.1 and −0.3 ± 0.4 at ~38M. As English BLiMP (§33.9), no feature moves grammatical
judgements.

**P174 right** (+4.9 ± 0.4, 4/4); **P175 right** (70.7–71.8% and 75.4–77.0%); **P176 right**
(−0.2 ± 0.4); **P177 right** (+4.3 ± 0.3, 4/4); **P178 wrong** (BPE 4k is 1.0 ± 0.2 below
BPE 32k, 0/3); **P179 right** (gender +5.9, person +4.5, number +4.0); **P180 right** (+3.4,
+6.6, +2.9 and +6.9 on the equal-length pairs).

**What it changes.** The tokenizer's gain is not only a shorter code. Models trained on text
whose inflectional endings are their own tokens prefer the grammatical sentence of a
subject–verb agreement pair more often, at both sizes. At full training the gain is +4.9
points, in number, gender and person alike. It is not because their pairs differ in length.
Morphology in the embedding moves these judgements not at all. So the thesis's main positive
result now has a second measure, one that is not bits per byte. G-07 ("no downstream task")
stands, but the Ukrainian results have a grammatical test beside the English BLiMP. At ~38M
and 10k person agreement does not move (−0.2) while number and gender do (+8.3, +7.3); at
full training all three do. In Ukrainian, person is marked mostly in the present tense's
endings, gender and number in the past's; why the first comes later is not asked here.

**And along the axes the lead was measured on** (the same day, predictions first). The lead in
bits per byte holds at twice the budget and is larger at 4× width (§33.68). Does the
agreement gain follow it? The 20k and 256-wide nulls at ~38M, four seeds each:

- **P181** (~55%). At width 256 (10k) the endings-apart nulls' advantage over BPE 17k's is
  larger than at width 128 (+4.3 points).
- **P182** (~60%). At 20k (width 128) it lies within 1.5 points of its 10k value (+2.8 to
  +5.8).

It does (`multiblimp_eval_20261010T071828`, 20k; `…T072101`, width 256). At twice the budget
BPE 17k's nulls score 68.5% and endings apart's 73.0%, **+4.5 ± 0.5** (4/4); at width 256,
69.7% against 75.0%, **+5.3 ± 0.3** (4/4), in number, gender and person alike (+5.7, +7.1, +3.7).
**P181 right; P182 right.** Across the conditions measured the agreement gain moves with the
lead in bits per byte: +4.3 at ~38M and 10k, +4.5 at 20k, +4.9 at 151M and 50k, +5.3 in the
4× wider model.

### 33.81 Case at full training without the leak: the word's own case lasts, and the sentence's lead was the leak

**The chain of 10 October** (`long_2026-10-10.sh`, `PIECES="0 1 2 3 5" RERUNS="0.2:5k
0.002:10k"`) ran its 18 runs from 09:54 to 18:52, nine of the ten hours, every guard passing.
A 50k run at 151M tokens took 54.0–54.3 minutes. The first two word-form runs took 58.4 and
55.4: they ran beside the morning's CPU work, MultiBLiMP's scoring and the figures
(§33.79–§33.80). A 10k run under BPE 32k took 15.6–16.1
minutes, a 5k run at 151M 5.4. A run at ~1.5M tokens took 8.1–8.6, each stopped early, at
7,500 or 8,000 steps.

**The second reproduction check held.** The quarantined BPE 17k null at 151M tokens and 5k
steps, seed 1, made again, ended at 1.1936354; the audit's re-score of its checkpoint is
1.193635.

**G-77** (`tokenizer_table_20261010T170427`, `…T170434`): 151M tokens, 50k steps, under
endings apart, paired by seed from identical initial weights.

| over the null, 151M, 50k, endings apart | gain | seeds |
|---|---|---|
| case read from the sentence (the tagger) | +0.0030 ± 0.0009 | 4/4 |
| case with prepositions read from their form | +0.0015 ± 0.0010 | 3/4 |
| case read from the word form alone | +0.0016 ± 0.0009 | 3/4 |

The three are compared in pairs, by seed:
- **The leak** (case over the control) is worth **+0.0015 ± 0.0001** (4/4).
- **The control over the word form** is **−0.0001 ± 0.0003** (per seed +0.0001, +0.0003,
  +0.0001, −0.0009).
- **Case over the word form** is +0.0014 ± 0.0002 (4/4).

**At full training the sentence's whole lead over the word form is the tagger's leak.** At
~38M tokens and 10k steps the control kept +0.0018 ± 0.0002 (4/4) over the word form (§33.76).
Context beyond the word form, without the leak, helps the shorter training and is gone by the
end of the longer one.

The law's axis shows how (the four seeds' curves averaged, each kept to its best step):
- **The control's lead over the word form** is +0.0011 to +0.0013 from null BPB 1.25 to 1.05,
  +0.0007 at 1.00 and 0.98, and +0.0002 at 0.95 and 0.94. It falls as a feature's gain falls.
- **The leak's part** (case over the control) does not fall to zero: +0.0019 and +0.0020 at 1.25
  and 1.15, +0.0008 to +0.0015 from 1.10 to 1.00, and +0.0012 to +0.0014 from 0.98 to the end.

The leak carries the answer, the case of the noun being predicted. The model cannot read that
from the words before it, so the leak does not stop paying. The honest part of context the
model learns to read for itself, as it learns part of speech.

**The word form's gain lasts where endings are tokens.** At full training under BPE 17k the
word form ends below its null (−0.0020 ± 0.0010, 1/4). Under endings apart it ends above it
(+0.0016 ± 0.0009, 3/4): paired, **+0.0036 ± 0.0010 (4/4)** more. Along the axis it goes:

| null BPB | 1.15 | 1.00 | 0.94 |
|---|---|---|---|
| endings apart | +0.0053 | +0.0029 | +0.0017 |
| BPE 17k | +0.0026 | −0.0020 | −0.0020 |

So the exposure effect at full training (§33.76: +0.0044 for case) holds in a feature that reads
no context and so cannot leak. With the tokenizer, the word form ends +0.0181 ± 0.0013 (4/4)
below BPE 17k's null.

**P169 right** (+0.0016, 3/4). **P170 right** (+0.0036 ± 0.0010). **P171 right** (+0.0015,
3/4). **P172 right on its letter only.** The control keeps 50.05% of case's gain: +0.0015051
against +0.0015036, half of +0.0030072. In substance it is a tie, half. **P173 right**
(+0.0015 ± 0.0001, 4/4).

**Where the gains sit** (`loss_by_kind_20261010T170915`; 16 models re-scored on the CPU, the
largest difference from their records 1.1 × 10⁻⁶ bits per byte).
- **The leak's +0.0015 sits after prepositions:** +0.0009 where a preposition's case in context
  differs from its form's, +0.0005 where it agrees (4/4 each), nothing after other words.
  +0.0013 of it is on ending tokens: the noun's ending, where §33.76 placed it.
- **The gains without the leak sit mostly elsewhere.** The word form's +0.0016 is +0.0007 on
  the stems of cut words and +0.0005 on uncut words (the first token of each), and +0.0005 on
  endings (33%). Case's gain on endings is +0.0021 (68%).

What the word's own case gives a fully trained model under endings apart is mostly the start
of the next word, not its ending.

**MultiBLiMP** (`multiblimp_eval_20261010T171043`): the word form under endings apart at 151M
moves agreement judgements by +0.2 ± 0.4 points (3/4) over its null. Nothing, as case did
(§33.79).

**What it changes.** §33.77 named three outcomes. The first came, and with it half of the
third:
- **"Case keeps helping at full training where endings are tokens" is free of the leak.** It
  holds as the word's own case, at half the size the sentence's case showed, at three seeds of
  four, and the word form gains +0.0036 more there than under BPE 17k at every seed.
- **What the sentence adds over the word form at full training is all leak.** Without the leak,
  context adds +0.0015 to +0.0018 at ~38M and 10k (4/4 each) and nothing at full training.
- **For the title's question:** a context-aware embedding helps a model that has not yet
  learned to read the context itself, and a fully trained model of this size reads it itself.
  It is the learning law, in the part of the feature that is context.
- **G-77 closes;** FINDINGS restates both paragraphs.

**BPE 32k's fourth seed** (`tokenizer_table_20261010T170757`). At seed 3 case beats its null by
+0.0027: **P140 right**. At four seeds:
- case gains +0.0029 ± 0.0002 (4/4), and the word form +0.0008 ± 0.0003 (3/4);
- case leads its word form by +0.0022 ± 0.0003 (4/4);
- the null ends at 1.0447, +0.0450 ± 0.0007 below BPE 17k's.

On MultiBLiMP (`multiblimp_eval_20261010T171508`) BPE 32k's null at four seeds scores
64.6%, −0.1 ± 0.2 points against BPE 17k's (2/4). Case over it moves +0.0 ± 0.1. BPE 4k
stays below it, by 1.1 ± 0.2 (0/4), so P178's verdict stands.

D7's six tokenizers are now all at four seeds, and nothing moved.

**The re-runs** (P160), each against the codes audit's re-score of its quarantined twin
(`codes_audit_20261006T151153`):
- **151M tokens, 5k steps:** seeds 2 and 3 reproduce their re-scores to 10⁻⁷, as seeds 0 and 1
  did.
- **~1.5M tokens, 10k steps:** seeds 0–2 reproduce too, at best steps 5,000, 4,500 and 4,500.
  Seed 3 ends 0.0000864 below its re-score (1.2606262 against 1.2607126). Its twin's best step
  under the old codes was 5,000, and its own on the constant is 4,500: the mid-schedule case
  the refinement allowed for.

Eight of P160's twenty are read, all as predicted.

Predictions: P140 and P169–P173 right (P172 on its letter only); P160 eight twentieths read;
P158, P159 and P161–P163 wait.

### 33.82 One more hour, 10 October

One more hour was available. The queue after §33.81 is in
DIRECTIONS, "The queue", rewritten. Of what is left, only the re-runs come in whole cells under
an hour: a condition's four seeds at about 11 minutes each.
- **The cell:** of their conditions, ~7.6M tokens at 10k is the one both `fig3_budget` and
  `fig4_architecture` (`demo_figures.py`) wait for. fig4 lacks nothing else.
- **The plan and launch:** `PLAN=1` put it at four runs, ~48 minutes. It was launched at 20:06
  with `PIECES="5" RERUNS="0.01:10k"` and `END_BY` an hour on.
- **Predictions:** P160 covers it.
- **Beside it, on the CPU, the GPU hidden from both:** the split of G-77's models by what each
  target is, and MultiBLiMP on the word form at 151M and on BPE 32k's fourth seed.

**The hour's cell** (20:06–20:50; four runs, 10.8–11.1 minutes each, the first two beside the
CPU work). The ~7.6M-token nulls at 10k reproduce their quarantined twins' re-scores to
10⁻⁷ at every seed: 1.1037241, 1.1065840, 1.1042292 and 1.1063754, each at its twin's best
step, 9,999. Twelve of P160's twenty are read, all as predicted, and every Ukrainian number
restated at that size stands as restated. `fig4_architecture` is drawn again
(`exports/figures/`). `fig3_budget` still lacks the 151M nulls at 10k and 20k (queue 3).

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
