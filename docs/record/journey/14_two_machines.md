# Research journey — Part 14: Two Machines, and a Null That Is Not Neutral

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.97–§33.116. [← Part 13](13_how_far_the_gain_travels.md) · [index](README.md) · [Part 15 →](15_the_list_of_19_october.md)

**Established:** a run repeated on a second machine differs from its original only by dropout's
draw, so every comparison is made within one machine (§33.101, §33.106, §33.109). In the 5M
Ukrainian model the same-shape null is not neutral: its slot costs where endings are exposed and
helps under BPE (G-78; §33.106, §33.110, §33.113, §33.116). In English too, what part of speech read
from the sentence adds over the word form is mostly the tagger's leak (§33.107). Width 512's gain
settles at +0.0102 (§33.102); fiction's gain holds at full training (§33.108); English part of
speech, removed, leaves a deficit that halves by 50k, and Ukrainian case leaves none (§33.102); the
probes hold at five seeds (§33.111, §33.114).

---

### 33.97 Width 512 at 20k, seed 1 (prepared), 15 October

On 15 October the thesis's list of the experiments it needs arrived, in order. It was ranked by
what each result changes in the text, with the sentence for every outcome. Its first item is the
rest of §33.96: width 512 at 20k, seeds 1–3. About three hours were available from 18:40, enough
for seed 1's pair (`STEPS=20000 SEEDS=1`; 2 × 88 minutes). The width guard reads seed 0's pair, now
on record. At 10k seed 1 ended where seed 0 did: +0.0136 against +0.0139, with +0.0133 and +0.0129
at steps 8,000 and 9,000.

Predictions, before the runs:

- **P208** (~65%). At 20k, width 512's seed-1 lead ends at +0.010 or more.
- **P209** (~55%). Over its last 5,000 steps it changes by less than 0.002.
- **P210** (~65%). Seeds 0 and 1 end within 0.003 of each other (seed 0: +0.0115).

### 33.98 The thesis's list, and its ninth item: the law in Ukrainian is within its seed spread

The list of 15 October (§33.97) asks for nine experiments in order. Each would let the text drop a
qualifier it now carries, and each comes with the sentence it states for every outcome.
DIRECTIONS, "The queue", now follows it. About 50 hours of GPU; results committed by about 31
October reach the text.

**The ninth item needs no GPU.** In Ukrainian the law was "approximate": the conditions' curves
differ by 0.0015 at null BPB 1.30 and by 0.0007 at 1.10 (§33.91). Are those gaps larger than
the seeds make them? `learning_law.py --spread` (`learning_law_20261015T154512`) reads each seed's
own curve at each point. It then deals the per-seed values to the conditions at random, 20,000
times, and counts how often the gap between the conditions' means is as large as observed:

| null BPB | gap between conditions | as large in random deals | each condition's seeds, sd |
|---|---|---|---|
| 1.30 | 0.0015 | 24% | 0.0006–0.0014 |
| 1.25 | 0.0013 | 50% | 0.0010–0.0015 |
| 1.20 | 0.0006 | 88% | 0.0009–0.0015 |
| 1.15 | 0.0001 | 100% | 0.0004–0.0015 |
| 1.10 | 0.0007 | 57% | 0.0011–0.0014 |

At every point the conditions differ no more than their seeds do. FINDINGS now says "in Ukrainian
within the seed spread". Averaging each seed's own reading gives +0.0030 and +0.0036 at 1.10,
§33.58's figures. Interpolating the mean curve gives §33.91's +0.0029 and +0.0037. The two
methods differ by that 0.0001.

**The literature, read.** The full texts of four audited works settled the audit's open cells,
recorded in `docs/reference/literature_audit.md` and `related_work.md`:
- Alexandrescu & Kirchhoff do not fix the input width, and their Turkish factors come from text
  disambiguated over the sentence;
- Sennrich & Haddow's width is fixed at 500, with contrastive runs and no null;
- Vania & Lopez predict whole words from 200-dimensional inputs, with gold analyses annotated in
  sentence context;
- KinyaBERT's ablation changes architecture and information together.

No audited work ran a random-feature control.

### 33.99 A causal tagger for English part of speech (the list's item 2, prepared)

Item 2 asks whether, for English part of speech too, what the sentence adds over the word form is
mostly the tagger's right-context leak, as it was for Ukrainian case (§33.81). It needs a control:
part of speech with the leak removed, over the same-shape null.

**Where the leak sits.** `morph_causality_check.py --by-type`
(`morph_causality_check_20261015T154900`) tags 10,000 training sentences in full and each word from
its prefix:
- 7.27% of words change part of speech without their right context, as §33.3 found on validation
  text;
- "to" alone is 8.3% of the changes (preposition before a noun, particle before a verb);
- the rest is spread over thousands of word types, by majority class: proper nouns 27%, adjectives
  26%, nouns 17%, particles 8%, verbs 7%, prepositions 5%. The next word decides whether a word
  modifies it or heads its phrase.
- Words whose tag changes at least 20% of the time, seen five times or more, number 813 types and
  hold only 60% of the changes.

In Ukrainian the leak sat at prepositions, where a word list could remove it (`ucaseprep`). In
English no word list does.

**So the control is a causal tagger:** every word tagged from its left context alone (`poscausal`,
`scripts/data/derive_causal_pos.py`). spaCy's tagger sees four tokens on each side. Tagged from the
window of its last five tokens, a word gets the tag its whole prefix gives: 100.000% of 9,992
training words (2 tokens 99.5%, 3 tokens 99.9%). So the derivation tags every word in its own
five-token window.

**The parse, and why the control goes without it.** `pos` as the pipeline builds it passes the tags
through rules that read the dependency parse: "which" as a determiner or a pronoun, "have" as an
auxiliary or a verb. Without the parser, 0.68% of the cached codes come out otherwise. With the
parser, a prefix test is not a measure of the right context. A parser given a prefix that ends in
"the" has nothing to attach it to, and calls it a pronoun 4,498 times in 4,499; "a", "an", "this"
and "these" likewise. With it, 19.42% of words "change"
(`morph_causality_check_20261015T160946`, `--full-pipeline`, 3,000 sentences), most of them that
artifact, as §30.6b found for the parser itself. A parse needs the whole sentence, so a causal
tagger cannot make those refinements. They are information from the right, and `poscausal`
reads the tagger's tags through the tag map alone.

**Two checks that can fail**, run before anything is written:
- the paragraphs re-encoded as the stream builder encodes them must give the cached ids, token for
  token;
- the full pipeline must reproduce the cached `pos` codes exactly on them, which proves the
  alignment and the class order.

On 300 validation paragraphs (26,315 tokens) both hold, and the causal codes equal `pos` at 91.66%
of tokens. The training split needs about 4–5 hours of CPU on eight processes; the GPU part, the
causal control at 10k and 20k at four seeds, is about 2.3 hours.

### 33.100 Width 512 at 20k, seed 1: levelled off too, at +0.0097

**The runs** (`STEPS=20000 SEEDS=1`, 18:42–21:41 on 15 October). The width guard read seed 0's pair.
No defect: validation BPB falls at every evaluation, and both best steps are the last. An
unrelated GPU job was started and crashed during the second run; the run was unaffected.

**The lead** at seed 1 is **+0.0097** (BPE 17k 0.8135, endings apart 0.8038). Along training:
+0.0266 at step 2,000, +0.0163 at 5,000, +0.0100 at 7,500, +0.0110 at 10,000, +0.0084 at 12,500,
+0.0094 at 15,000, +0.0083 at 17,500. Over the last 5,000 steps it lies between +0.0080 and
+0.0103 and ends 0.0003 above where it stood. Like seed 0, it falls until about step 7,500 and
then holds.
- **P208 wrong:** +0.0097, just under +0.010.
- **P209 right:** +0.0003.
- **P210 right:** the two seeds end 0.0018 apart.

**Two seeds** (`tokenizer_widths_20261015T184449`): the lead is +0.0106 (+0.0115, +0.0097), 1.3% of
BPE 17k's null. At matched null BPB: +0.0402 at 1.18, +0.0206 at 1.04, +0.0168 at 0.95, +0.0121 at
0.92, +0.0099 at 0.88, +0.0097 at 0.82.

**MultiBLiMP** (`multiblimp_eval_20261015T184315`). Seed 1: BPE 17k 76.9%, endings apart 83.3%,
+6.4 points; seed 0, +5.6.

**What it changes.** Two seeds of four both level off, at about +0.010 (1.3%): smaller than at
widths 128 and 256, but lasting. Seeds 2–3 (~5.9 h) give the size with its spread.

**Predictions.** P209 and P210 are right; P208 is wrong. The tally is 116 right, 8 half right,
77 wrong, 8 never tested, 1 withdrawn, of 210.

### 33.101 A second machine, and the night of 16 October (prepared)

**The machine:** an RTX 5090 Laptop GPU (24 GB), 24 CPU cores, 93 GB of memory. Everything for this
work lives in one folder on it:
- the Python environment, with the thesis machine's versions (torch 2.11 with its own CUDA 13.0
  runtime);
- the code, as a `git archive` of a named commit;
- the caches it needs, sent with SHA-256 checks or built there from public sources.

**Calibration** (`device_calibration_20261015T232350`; the four runs in
`results/audits/device_calibration_20261016/`). The null at width 128, ~38M tokens, 10k steps,
seeds 0–2, was rerun there and compared with the thesis machine's runs:
- **Agreement:** the 5090 ends −0.00017, +0.00015 and −0.00053 away, with no consistent sign.
  That is well within a cell's seed spread (0.0037 here), but not zero. A seed's two tokenizers
  therefore always train on the same machine, and every run's record names its GPU
  (`environment.gpu`).
- **Repeatability:** seed 0 run twice there differs by at most 2.3e-8 at any evaluation. The
  thesis machine repeats bit for bit (G-69).
- **Speed:** a run alone takes 245 seconds, 2.8 times faster. Three at once take 787 seconds
  each, so there is no gain from running in parallel; runs go one at a time.

**The night** (`scripts/chains/night_2026-10-16.sh`, eleven hours) takes the items of the list
(§33.98) whose comparisons are all new runs, in the list's order as far as the data allows:
- **width 512 at 20k, seeds 2–3** (item 1);
- **English part of speech switched off** from the start and at step 5,000, at 50k, seeds 0–3
  (item 4a; §33.42's 20k design, at full training);
- **fiction at full training**, both nulls at fraction 0.4, 50k, seeds 0–3 (item 3);
- **Ukrainian case switched off** the same two ways under endings apart, ~38M tokens, 20k,
  seeds 0–3 (item 4b);
- **English plain baselines at 50k, seeds 4–7** (item 6), last, and likely cut short by the time
  box.

Guards:
- each cell's caches must be present (`scripts/ops/cells_ready.py`);
- a piece's first pair must end within 0.05 of each other;
- a run starts only when 6 GB of the GPU are free, waiting while anyone else holds it;
- disk, and the time box.

It was rehearsed with the stub in eight scenarios, all passing.

**On the CPU beside it:**
- **The fiction data:** `scripts/data/fiction_full.sh` downloads the fiction export, converts it,
  and builds both conditions at 0.4. Three checks against the thesis machine's hashes come first:
  the download, the converted corpus, and the validation streams, which are the same text at
  every fraction. Only when all three hold may the fiction piece run.
- **Item 2's causal tagger** (`derive_causal_pos.py`) over WikiText-103's training split.

The width 512 and fiction pairs, and both pairs of each switch, are compared within the machine.
English seeds 4–7 join seeds 0–3 from the thesis machine.

Predictions, before any run:

- **P211** (~75%). At width 512 and 20k, seeds 2 and 3 each end with the endings-apart null below
  BPE 17k's by +0.006 to +0.015.
- **P212** (~60%). Each levels off: over its last 5,000 steps the lead changes by less than 0.002.
- **P213** (~65%). The four seeds' mean lead lies between +0.008 and +0.013.
- **P214** (~55%). At 50k, English part of speech switched off at 5,000 ends below the model never
  fed it at three seeds of four or more.
- **P215** (~55%). Its deficit at 50k is smaller than at 20k (under 0.0039).
- **P216** (~65%). On fiction at full training, the endings-apart null ends below BPE 17k's at four
  seeds of four.
- **P217** (~55%). Its lead lies within 0.003 of fiction's 10k lead (+0.0067 to +0.0127).
- **P218** (~50%). Ukrainian case switched off at 5,000 of 20k under endings apart ends below the
  model never fed it at three seeds of four or more.
- **P219** (~55%). Its deficit is smaller than English part of speech's at 20k (under 0.0039).
- **P220** (~65%). At 50k, English endings apart ends below BPE 17k at three or more of seeds 4–7.
- **P221** (~55%). Over eight seeds, the median lead lies within 0.002 of the 20k lead (+0.0035 to
  +0.0075).

**Launched** at 03:00 Kyiv on 16 October, with the time box at 14:00. Before launch, a cell of each piece
ran 30 steps through the real loader on the second machine. That found three gaps the readiness check
and the smoke tests could not, each fixed before any piece reached it:
- the single-fact codes files sent for `pos` and `ucase` had been written one-dimensional; they were
  rewritten as one row each, contents unchanged;
- three data files the English pipeline reads through its configuration had not been sent;
- one package (`openpyxl`) was missing from the environment.

The causal tagger's validation split passed both its checks there: the same ids, and the cached `pos`
reproduced on all 252,255 tokens. Its causal codes equal `pos` at 92.22% of tokens; the validation file
is written, and the training split follows.

### 33.102 The night of 16 October: width 512 settles, a removed feature's deficit halves by 50k, and case leaves none

The chain of §33.101 completed on the second machine at 04:50 local time, two hours inside its time
box: 28 runs, every piece but fiction, whose data build stopped (below). Each comparison below is
within one machine, and each record names its GPU.

**Item 1 — width 512 at 20k, four seeds.** At seeds 2 and 3 the endings-apart null ends below BPE
17k's by +0.0107 and +0.0088; over the last 5,000 steps the lead moves by −0.0005 and +0.0006. With
seeds 0–1 (§33.96, §33.100) the lead is **+0.0102 ± 0.0006 (4/4)**, 1.25% of the null's bits per
byte, and every seed has levelled off. **P211, P212 and P213 right.**

**Item 4a — English part of speech switched off at 5,000 of 50k.** Against the model never fed it,
the switched model ends −0.0011, +0.0022, +0.0036 and +0.0031 behind: **+0.0019 ± 0.0011**, a deficit
at three seeds of four, half the 20k budget's 0.0039 ± 0.0009. The gap closes slowly (mean over the
four seeds):

| step | 5k | 6k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|
| switched minus never fed | +0.0413 | +0.0110 | +0.0049 | +0.0038 | +0.0032 | +0.0024 | +0.0019 |

- Seed 0 recovers by step 30,000 and ends 0.0011 ahead; seeds 1–3 keep 0.0022–0.0036.
- At step 20,000 the gap is +0.0038, as at the end of a 20k budget (0.0039), though there the learning
  rate had annealed and here it is still high. The further 30,000 steps halve it.

**P214 right** (three seeds of four), **P215 right** (+0.0019, under 0.0039). The deficit lasts well
past where a 20k budget ends, and shrinks with time.

**Item 4b — Ukrainian case switched off at 5,000 of 20k, under endings apart (~38M tokens).** The
switched model ends −0.0022, −0.0012, −0.0003 and +0.0007 against the model never fed case:
**−0.0008 ± 0.0006**, level or better at three seeds of four. The jump at the switch (+0.0239 at step
5,000) is gone by step 7,500 (+0.0002), and from step 10,000 the switched model is ahead (−0.0008 to
−0.0010). **P218 wrong**; P219 (under English's 0.0039) **right**, trivially: there is no deficit.
Removal's lasting deficit is not general. English part of speech leaves one; Ukrainian case under
endings apart does not. One reading, untested: with endings apart the case ending is a token of its
own, so what the feature carried stays in the input.

**Item 6 — English at full training, seeds 4–7.** Endings apart ends below BPE 17k at every seed:
+0.0160, +0.0016, +0.0065, +0.0009 (+0.0062 ± 0.0035). With seeds 0–3 (§33.92), over eight seeds:
**median +0.0053, mean +0.0066 ± 0.0024, 7 of 8**. Seed 4's BPE 17k run ends about 0.009 above the
typical run, as seed 2's did on the thesis machine. Endings apart's highest runs, seeds 3 and 7, end
about 0.005 above its typical one. With eight seeds the median stays the measure, as at four.
**P220 right** (4/4), **P221 right** (median +0.0053, against 20k's +0.0055). BLiMP for seeds 4–7
is tonight's (§33.103).

**Item 2 — the causal tagger's training split.** On the second machine
(`derive_causal_pos_20261016T001413`, 69 minutes):
- 120,625,719 tokens, the ids identical to the cache;
- the full pipeline reproduces the cached `pos` on the first 20,000 paragraphs (2,044,348 tokens);
- the causal tag equals `pos` at 92.09% of tokens (validation: 92.22%).

Its codes were copied to the thesis machine with SHA-256 checks at both ends. Its runs pair there
with `pos`, `poslex` and the null.

**Item 3 — fiction: the build stopped.** The download, the conversion and both validation streams
matched the thesis machine's (`ubertext_fiction_20261016T000257`, `check_cache_20261016T000349`,
`check_cache_20261016T000601`). Both builds tagged 1,046,000 of the training split's 1,121,272
paragraphs (92.5M tokens) in 2 h 04 min. Then they stopped on paragraph 1,046,116: the Bible, in
Ohienko's translation, as one line of 2,713,718 characters, past spaCy's limit of a million. It is
the only paragraph over the limit at 0.4; the next longest has 587,473.

The fix: the pipeline lifts the limit to 4 million characters, the corpus's longest document being
3.6 million, so a paragraph is still tagged whole.
- Measured on this paragraph, the Ukrainian pipeline needs ~3 GB per million characters (1.5 GB at
  500,000).
- No text under the old limit is touched, so no cache built before changes.

P216 and P217 wait.

**A question the night raised (G-78).** Case switched off from the start holds its slot at exactly
zero: an empty slot, in effect the zero slot of §33.11.
- Under endings apart at 20k its four seeds end at 0.99398 on average. That is 0.0036 better than the
  null's four seeds on the thesis machine (0.99756), and level with case there (0.99373; case 0.0002
  ahead).
- In English the two are level: at 20k, part of speech switched off from the start ends at 1.4252
  against the null's 1.4261, both on the thesis machine. FINDINGS states the same for learned
  positions.

If the Ukrainian gap holds on one machine, an empty slot matches case's gain over the same-shape null
(+0.0038 ± 0.0008 at 20k under endings apart, §33.76). The null would still measure what case carries
against the slot's constant, but it would not be the best model without case. This comparison crosses
machines. The calibration put them within 0.0005 at 10k on the Ukrainian null, but no run has been
repeated at 20k. Both questions go to one night (§33.103).

**Predictions.** P211–P215 right, P218 wrong, P219–P221 right; P216–P217 wait for fiction. The tally
is 124 right, 8 half right, 78 wrong, 8 never tested, 1 withdrawn, 2 pending, of 221.

### 33.103 Twins and the empty slot, on one machine (prepared, 17 October)

Two questions first, then what remains of the list for the second machine
(`scripts/chains/night_2026-10-17.sh`):

- **Do the machines differ at 20k and 50k?** TWINS are runs of the thesis machine repeated on the
  second with the same arguments and seeds:
  - Ukrainian, 20k, seeds 0–3: the null and case under endings apart, and the null under BPE 17k;
  - English, 50k, seeds 0–1: both plain baselines.

  Twins are compared with their originals, never pooled with them. They say whether the machines
  differ like a seed effect (then a comparison within a machine stands, and so does an average of
  such comparisons over machines) or by an offset (then nothing is compared across machines).
- **Does the empty slot beat the null on one machine (G-78)?** The null switched off from step 0:
  the null's own initial weights, its slot held at zero. Under endings apart it runs beside the twins
  of the null and case; under BPE 17k, beside the null's twin.
- **The grammar tests of items 6 and 1:** BLiMP on English seeds 4–7, MultiBLiMP on width 512 at 20k,
  seeds 2–3. Their checkpoints are on the second machine; the data is sent, not downloaded.
- **Fiction (item 3)**, with §33.102's fix, its data rebuilt on the CPU from the start.

About 8.7 hours of GPU with margins (7.8 at the last night's times):

| piece | runs | time |
|---|---|---|
| twins and the empty slot under endings apart | 12 | ~1.7 h |
| English twins | 4 | ~1.4 h |
| BLiMP and MultiBLiMP | — | ~0.25 h |
| fiction | 8 | ~3.5 h |
| twins and the empty slot under BPE 17k | 8 | ~1.2 h |

Rehearsed with the stub in ten scenarios, all passing. The Ukrainian twins and English endings
apart's twins keep the thesis machine's run ids. The September BPE 17k baselines' configuration has
gained fields since, at defaults that reproduce it (G-69), so their twins have new ids. On the thesis
machine the twins' records go to `results/audits/device_twins_20261017/`, never to `results/`.

Predictions, before any run:

- **P222** (~60%). At 20k each of the twelve Ukrainian twins ends within 0.001 of its original, and
  their mean difference lies within ±0.0005: the machines differ like a small seed effect, not by an
  offset.
- **P223** (~55%). At 50k each English twin ends within 0.002 of its original, and endings apart's
  lead at seeds 0 and 1 is reproduced within 0.002.
- **P224** (~60%). On one machine, under endings apart at 20k, the empty slot ends better than the
  null by more than 0.002 on average over seeds 0–3 (across machines: 0.0036).
- **P225** (~55%). On one machine, case ends within 0.001 of the empty slot on average over seeds 0–3.
- **P226** (~45%). Under BPE 17k at 20k the empty slot also ends better than the null by more than
  0.002 on average.
- **P227** (~70%). On BLiMP endings apart judges better overall than BPE 17k at three or more of
  seeds 4–7 (seeds 0–3: +1.0 ± 0.2, 4/4).
- **P228** (~75%). On MultiBLiMP at width 512 and 20k endings apart judges better at seeds 2 and 3,
  by 4 to 8 points each (seeds 0–1: +5.6, +6.4).

Each question has a different sentence for each outcome, so both pass the sentence test.
- **Twins.** If P222 holds, the methods state the machines' difference as measured, and averages over
  both machines stand. If the twins show an offset, every comparison is stated within one machine.
- **The empty slot.** If P224 holds, the Ukrainian gains are restated against the empty slot as well,
  and G-78 asks where else it holds. If it fails, the gap of §33.102 was the machine, and the null
  stands as the reference.

### 33.104 A day on both machines: the comparison, G-78 on each, and item 2 (prepared, 17 October)

Both machines have about eleven hours.

**The second machine** runs §33.103's chain, launched at 02:41 local time (09:41 Kyiv) with its
time box at 13:45. Fiction's data is rebuilt beside it, with the fix of §33.102. The machine has left
the office network, so data now goes to it over the direct route, compressed.

**The software is the same on both machines:** torch 2.11.0 with CUDA 13.0 and cuDNN 9.19, and every
computation in single precision (no mixed precision; TF32 off for matrix products). Only the GPU
differs (Ada, sm_89, against Blackwell, sm_120), and the driver (580 against 595). A twin's
difference can therefore come only from each GPU's own kernels rounding differently, and from
training amplifying that.

**The thesis machine** (`scripts/chains/tulpar_2026-10-17.sh`, ~4.2 hours):
- **The empty slot on this machine too.** G-78 is answered on the machine where the null and case
  ran, each seed paired from identical initial weights. Each run is also a twin of the second
  machine's empty slot (§33.103), so the comparison runs in both directions.
- **Item 2:** `poscausal` at 10k and 20k, seeds 0–3. Its codes came from the second machine,
  SHA-256-checked (§33.102). A 30-step run through the real loader passed. `poscausal` has
  `pos`'s table shape (19 classes, 8 dims), so at each seed it starts from `pos`'s initial weights.
  - At 10k, `pos` leads the null by +0.0130 and the word form (`poslex`) by +0.0099: the sentence
    adds +0.0031 over the word form.
  - At 20k: +0.0075 and +0.0050, so +0.0025.

  Item 2 asks how much of that is the tagger's right-context leak.

The rest of each machine's hours is decided once the twins and the empty slot have reported.
Rehearsed with the stub in seven scenarios, all passing.

Predictions, before any run:

- **P229** (~85%). `poscausal` beats the null at every seed, at 10k and at 20k.
- **P230** (~50%). At 10k `poscausal` keeps less than half of `pos`'s lead over the word form: it
  ends less than +0.0016 ahead of `poslex` on average.
- **P231** (~50%). At 20k likewise: less than +0.0013 ahead of `poslex`.
- **P232** (~65%). Paired by seed, `poscausal` ends behind `pos` at both budgets by more than 0.0005
  on average: the leak is measurable.
- **P233** (~60%). On the thesis machine the empty slot ends better than the null by more than 0.002
  on average over seeds 0–3, each seed paired from identical initial weights.
- **P234** (~65%). Each of those four runs ends within 0.001 of its twin on the second machine.
- **P235** (~55%). On the thesis machine, case ends within 0.001 of the empty slot on average.

**Item 2's sentence for each outcome.** If P230 and P231 hold, in both languages what the sentence
adds over the word form is mostly the tagger's leak. If not, English part of speech carries context
of its own. G-78's sentences are those of §33.103.

### 33.105 Why twins differ, item 5's probes at four seeds, and each machine's afternoon (prepared, 17 October)

**What the first twins showed.** At step 0 a twin and its original evaluate to the same bits per byte
within 2 × 10⁻⁹ (2.573648769 against 2.573648771): the weights are initialised on the CPU, alike on both
machines, and the evaluation's arithmetic agrees. Yet the training loss of the very first batch differs
(9.7766 against 9.7732), and the runs drift apart from there. The model's dropout (0.1, in every layer)
draws its masks on the GPU from torch's CUDA generator, and a kernel's threads take their random numbers
by their place in its launch grid, which depends on the GPU (24 multiprocessors against 82). A twin is
then the same run with dropout drawn differently: the same initial weights and batches, a different
dropout draw. Two tests:
- **the masks** (`scripts/audits/dropout_masks.py`, seconds on each machine): dropout's masks for the
  base model's activation shapes, seeded as training seeds them, and fixed arithmetic (a matrix
  product, a softmax, a layer norm) on inputs made on the CPU;
- **dropout off**: BPE 17k's null at ~38M and 10k with dropout 0, seeds 0–1, on both machines. What is
  left between the twins is the GPUs' rounding alone. A run without dropout is a different model, so
  its label carries the setting (`_do0`; no existing label changes: all 578 runs used 0.1).

**Item 5 on the second machine** (the thesis's list): the four probes the thesis cites at one seed,
seed 4, repeated at seeds 0–3. The runs are repeated with a checkpoint every 1,000 steps and probed as
seed 4 was (every layer; 21 checkpoints):
- after removal: part of speech switched off from the start and at 5,000, 20k — the word's tag and
  the previous word's;
- the auxiliary target, real and shuffled, 20k — the word's tag.

Each run is also a twin of the thesis machine's run of that cell and seed. `probe.py` reproduces seed 4's
record within 6 × 10⁻⁸ at every step on the CPU, so the probe does not depend on the machine.

**Width 512 at 20k, seed 1**, both nulls as twins: item 1's four seeds are two per machine.

**G-78 first, then item 5.** After two seeds on the thesis machine the empty slot leads the null there
(−0.0019, −0.0035 bits per byte at 20k under endings apart, paired from identical initial weights),
and case leads the empty slot (−0.0019, −0.0027). The null's handicap therefore looks real on the
machine the thesis's runs came from, so the next hours map it:
- on the second machine, at 10k: the empty slot under both tokenizers, beside twins of the null and
  case (`g78at10k`), before item 5;
- on the thesis machine, at full training (151M tokens, 50k steps, endings apart), where FINDINGS'
  strongest Ukrainian case claim is stated against the null (+0.0030 ± 0.0009): the empty slot at
  seeds 0–3 (`fullempty`, ~4 h). Then item 7 (`ucaseprep` under BPE 17k at full training, its codes
  derived this morning: 671 preposition types read from their form, 1.22% of tokens changed) for as
  many seeds as the time box allows.

The chains: `scripts/chains/second_followup_2026-10-17.sh` (pieces masks, nodropout, g78at10k,
w512twins, proberemoval, probetarget; ~5.5 h after the second machine's chain) and
`scripts/chains/tulpar_followup_2026-10-17.sh` (masks, nodropout, fullempty, item7). Both start when
their machine's chain ends. Rehearsed with the stub in eight scenarios, all passing.

Predictions, before any of these runs:

- **P236** (~85%). The two GPUs draw different dropout masks from the same seed for every shape; the
  CPU's masks are the same on both machines.
- **P237** (~75%). The fixed arithmetic agrees between the GPUs: each result's float64 sum within
  10⁻⁶ of the other machine's, relative.
- **P238** (~50%). With dropout off, each pair of twins at 10k ends within 0.0002.
- **P239** (~65%). With dropout off, the twins differ by less than 10⁻⁵ at the first evaluation
  (step 500).
- **P240** (~60%). At 10k under endings apart, on the second machine, the empty slot ends better than
  the null's twin by more than 0.001 on average over seeds 0–3.
- **P241** (~50%). At 10k under BPE 17k likewise.
- **P242** (~55%). At 10k under endings apart, case's twin ends better than the empty slot by more than
  0.001 on average.
- **P243** (~65%). Width 512 at 20k, seed 1: each twin ends within 0.002 of its original, and the lead
  (+0.0097 on the thesis machine) is reproduced within 0.002.
- **P244** (~70%). Item 5, removal: at each of seeds 0–3, from step 6,000 to the end, the switched
  model's deepest layer reads the word's part of speech within 1.5 points of the model never fed it
  (seed 4: level from step 6,000).
- **P245** (~60%). At the end the switched model reads the previous word's tag better at the deepest
  layer than the model never fed it, at three seeds of four or more (seed 4: +2.0 points).
- **P246** (~70%). Item 5, the target: at each of seeds 0–3 the real target's model reads part of
  speech better at layer 2 at step 2,000 than the shuffled target's, by more than 2 points (seed 4:
  +4.6).
- **P247** (~60%). It still reads it better at the end, at three seeds of four or more (seed 4: +1.6).
- **P248** (~65%). At full training under endings apart, on the thesis machine, the empty slot ends
  better than the null by more than 0.001 on average over seeds 0–3, paired by seed.
- **P249** (~55%). There, case ends better than the empty slot by more than 0.001 on average.
- **P250** (~60%). Item 7: `ucaseprep` under BPE 17k at full training ends within 0.0004 of `ucaselex`
  on average over the seeds run, i.e. the whole lead there (+0.0006 ± 0.0004) is the leak.

**The sentences these decide.**
- **G-78 (P240–P242, P248–P249).** The thesis states each Ukrainian gain against the same-shape null.
  Where an empty slot beats that null, the gain over the best model without the feature is smaller,
  and the text gives both.
- **Item 5 (P244–P247).** If the seeds agree, "one seed" goes from §3.7–§3.8; if not, the sentence
  gives their spread.

### 33.106 The machines agree, and the null is handicapped in Ukrainian: case's own gain sits on the endings

The first pieces of both machines' days (§33.103, §33.104) have finished. Analyses:
`device_twins_20261017T082521`, `paired_cells_20261017T081844`, `loss_by_kind_20261017T081901`.

**The twins.** Fifteen runs of the thesis machine were repeated on the second with the same arguments
and seeds: the calibration's three at 10k, and twelve at 20k under endings apart (the null, case and
the empty slot, seeds 0–3).
- **Overall:** the second machine minus the thesis machine is −0.0004 ± 0.0002 on average. Six pairs
  end above and nine below, which is no consistent sign (sign test p = 0.61). The typical difference
  is 0.0006; the largest is 0.0025, the empty slot at seed 3.
- **Against the seed spread:** per cell, the twins spread 0.2 to 4.5 times as widely as the thesis
  machine's seeds do. The null's four seeds there lie within 0.0004 of each other, closer than its
  twins.
- **The effects agree.** Case over the null is +0.0038 ± 0.0008 on the thesis machine and +0.0037 ±
  0.0008 on the second, seed by seed within 0.0018. Width 512's tokenizer lead (§33.102) is +0.0106
  and +0.0098 at different seeds.

**Why twins differ.** A twin and its original evaluate to the same bits per byte at step 0, within
2 × 10⁻⁹, so the initial weights and the evaluation's arithmetic are the same on both machines. From
the first batch on, the training losses differ. Dropout draws its masks on the GPU, and by kernel
layout the two GPUs draw them differently. A twin is therefore a run with a different dropout draw,
and its difference behaves like a smaller part of a seed's. The second machine repeats itself bit
for bit: the empty slot run on two days, 26 hours apart, gives the same bits per byte at every
evaluation of every seed. (§33.105's tests measure the masks and the rounding directly.) **For the
thesis:** a run on the second machine is a valid run of its cell, and so is an average over both
machines of comparisons made within one. A single run is not reproduced across machines; a pair is
still compared within one machine.

**G-78 at 20k, under endings apart, on each machine** (each seed paired from identical initial
weights; case's table has the null's shape, so case starts from the null's weights too):

| | thesis machine | second machine |
|---|---|---|
| the empty slot over the null | **+0.0025 ± 0.0004 (4/4)** | **+0.0034 ± 0.0009 (4/4)** |
| case over the empty slot | +0.0013 ± 0.0007 (3/4) | +0.0003 ± 0.0006 (2/4) |
| case over the null | +0.0038 ± 0.0008 (4/4) | +0.0037 ± 0.0008 (4/4) |

The same-shape null is handicapped in Ukrainian: a slot held at zero ends better. In English the two
are level (§33.102). Case still beats the empty slot, by about a third of what it gains over the null
on the thesis machine, and by little on the second machine.

**Where.** The thesis machine's twelve models were re-scored token by token on the CPU, all within
1.1 × 10⁻⁶ of their records. Each gain over the null is split by the kind of target it predicts
(`loss_by_kind.py`, `--facts mconst@OFF0 ucase`):

| target | share | the empty slot | case |
|---|---|---|---|
| a word's first token, uncut / first of a cut word | 21% / 20% | +0.0003 / +0.0007 | +0.0006 / +0.0007 |
| **the ending token** | 20% | **+0.0001 ± 0.0001** | **+0.0024 ± 0.0002 (4/4)** |
| the rest of a cut word / of an uncut word | 8% / 11% | +0.0004 / +0.0001 | +0.0001 / −0.0002 |
| **non-word** (punctuation, digits, line breaks) | 20% | **+0.0009 ± 0.0001 (4/4)** | +0.0003 ± 0.0001 |

The null's handicap is not morphological. It sits at units' first tokens and at non-words, where the
null's slot says whether the previous unit was a word. Case's slot says the same, plus the case, and
pays the same cost there. **What case carries is on the endings:** +0.0024 there (64% of its gain on
20% of the targets), where the empty slot gains nothing.

**Predictions.** P222 wrong (the null's seed-0 twin ends 0.0012 away; the rest of P222 waits for
BPE 17k's twins); P224 right (+0.0034); P225 right (+0.0003); P233 right (+0.0025); P234 wrong (seed 3's
twin −0.0025); P235 wrong (+0.0013). The tally is 127 right, 8 half right, 81 wrong, 8 never tested, 1
withdrawn, 25 pending, of 250.

**What it changes.** Every Ukrainian gain in the thesis is stated against the same-shape null. Where
that null loses to an empty slot, part of the stated gain is the slot's boundary cost, shared by every
slotted model, not information. At 20k under endings apart case's information is the endings' +0.0024,
and its net gain over an empty slot is +0.0013 (thesis machine). G-78 stays open for other budgets,
sizes and tokenizers. Today's next pieces measure 10k under both tokenizers, BPE 17k at 20k, and full
training (§33.105).

### 33.107 In English too the sentence's lead over the word form is the tagger's leak; full training's twins drift apart

**Item 2** (the thesis machine, `paired_cells_20261017T103000`). English part of speech from the
causal tagger of §33.99 (every word tagged from its left context alone), at 10k and 20k, seeds 0–3.
It starts from `pos`'s initial weights at each seed:

| | 10k | 20k |
|---|---|---|
| the causal tag over the null | +0.0105 ± 0.0003 (4/4) | +0.0054 ± 0.0003 (4/4) |
| **the causal tag over the word form** (`poslex`) | **+0.0006 ± 0.0004 (3/4)** | **+0.0004 ± 0.0003 (3/4)** |
| `pos` over the causal tag: **the leak** | +0.0025 ± 0.0004 (4/4) | +0.0021 ± 0.0005 (4/4) |
| `pos` over the word form (as recorded) | +0.0031 ± 0.0005 | +0.0025 ± 0.0003 |

What the tagged sentence adds over the word form is four fifths the tagger's right-context leak:
81% at 10k and 84% at 20k. The context left once the leak is gone is small and not established at
four seeds. **P229, P230, P231 and P232 right.** In both languages, then, what a context-aware
feature adds over the lexical one is mostly not context: the thesis's list gave this sentence for
this outcome.

**English twins at full training** (`device_twins_20261017T103034`). Both plain baselines at seeds
0–1 were repeated on the second machine:
- BPE 17k's twins end +0.0004 and −0.0009 away;
- endings apart's end +0.0063 and +0.0023 away.

So endings apart's lead at those seeds, +0.0051 and +0.0055 on the thesis machine, is −0.0009 and
+0.0022 on the second. Seed 0's twin drifts from early on: +0.0010 at step 5,000, +0.0027 at 20,000,
+0.0063 at the end. This is §33.92's drift of one run in four. The same initial weights and batches
drift on one machine and not on the other, so the drift is a chance event within training, not a
property of the seed. **P223 wrong.**

Over all nineteen twins the second machine minus the thesis machine is +0.0001 ± 0.0004 (9 above,
10 below). The largest difference grows with training: 0.0005 at 10k, 0.0025 at 20k, 0.0063 at 50k.
At full training a single English pair is not reproducible, which is why the thesis reports the
median of eight seeds (§33.102).

**The grammar tests** (`blimp_eval_20261017T094816`, `multiblimp_eval_20261017T094841`; data sent,
not downloaded):
- **BLiMP, English seeds 4–7 at full training:** endings apart judges better overall at every seed,
  +2.0, +0.6, +0.5 and +1.0 points. With seeds 0–3: **+1.0 ± 0.2 (8/8)**. Morphology is +1.1 ± 0.6
  (6/8); seeds 4–7 alone are mixed (2/4). **P227 right.**
- **MultiBLiMP, width 512 at 20k, seeds 2–3:** +5.6 and +6.1 points. With seeds 0–1: **+5.9 ± 0.2
  (4/4)**. **P228 right.**

**Fiction's data at 0.4**, rebuilt on the second machine with §33.102's fix (`fiction_full.sh`):
- both conditions built and gated, endings apart's training split at 153,925,966 tokens, tagged in
  2 h 13 min;
- the Bible's line tagged whole;
- both validation streams and null codes byte for byte the thesis machine's at 0.1;
- `FULL_VERIFIED` at 05:01 local time.

The fiction runs follow in the second machine's chain.

**Predictions.** P223 wrong; P227, P228, P229, P230, P231 and P232 right. The tally is 133 right,
8 half right, 82 wrong, 8 never tested, 1 withdrawn, 18 pending, of 250.

### 33.108 Fiction at full training: the tokenizer's gain holds, and reaches Wikipedia's share

The list's item 3, on the second machine (`tokenizer_table_20261017T124204`): both fiction nulls at
fraction 0.4 (153.9M training tokens under endings apart), 50k steps, seeds 0–3, each seed's pair on
one machine.

| | BPE 17k | endings apart | lead |
|---|---|---|---|
| seed 0 | 1.0663 | 1.0487 | +0.0176 |
| seed 1 | 1.0657 | 1.0510 | +0.0147 |
| seed 2 | 1.0656 | 1.0527 | +0.0129 |
| seed 3 | 1.0665 | 1.0498 | +0.0167 |

The lead is **+0.0155 ± 0.0010 (1.45%, 4/4)**.

Along training it is +0.0425 at step 500, +0.0254 at 1,000 and +0.0145 at 5,000. At 10,000 it dips
to +0.0112, where the learning rate is still high. From 20,000 on it is flat: +0.0153, +0.0149,
+0.0149, +0.0155 at the end.

At a 10k budget on a quarter of this text, fiction's lead was +0.0097 (0.8%), half its share on
Wikipedia (§33.88). At full training it is 1.45%, close to Wikipedia's 1.75%. **P216 right
(4/4); P217 wrong:** the lead grew instead of staying within 0.003 of the 10k lead. Of the list's
two sentences, the first holds: on both corpora the gain does not fade with training. Fiction's
smaller share at 10k belonged to the short budget. MultiBLiMP on these models runs in the second
machine's follow-up.

**Predictions.** P216 right, P217 wrong. The tally is 134 right, 8 half right, 83 wrong, 8 never
tested, 1 withdrawn, 16 pending, of 250.

### 33.109 Why twins differ: only dropout's masks — and the null's handicap is endings apart's

The second machine's follow-up began at 09:48 local time, after its chain completed. Its first
pieces and BPE 17k's twins: `device_twins_20261017T135807`, `dropout_masks_20261017T103028`
(thesis machine), `dropout_masks_20261017T134856` (second machine),
`multiblimp_eval_20261017T134827`.

**The mechanism, measured.** Two tests answer what §33.105 asked:
- **The arithmetic is the same.** A matrix product, a softmax and a layer norm on the same inputs
  give bitwise-identical results on both GPUs (equal SHA-256).
- **Dropout's masks are not.** Seeded alike, the two GPUs draw different masks for every activation
  shape of the model. The CPU draws the same masks on both machines. Each GPU keeps 90% of the units,
  as it should.
- **Without dropout the machines train identically.** BPE 17k's null at 10k with dropout off, seeds
  0 and 1, run on both machines, ends 9.9 × 10⁻⁹ and 5.7 × 10⁻⁹ apart. No evaluation along the
  10,000 steps differs by more than 2 × 10⁻⁸. The small differences in the GPUs' other kernels do
  not grow in training.

**P236, P237, P238, P239 right.** A twin is the same run with dropout drawn differently, and
nothing else. With dropout, the 23 twins differ by +0.0001 ± 0.0003 (10 above, 13 below; mean
|difference| 0.0009, largest 0.0063 at 50k). With dropout off the model is a different one (BPE
17k's null ends at 1.0528 and 1.0543 at 10k, against 1.0891 and 1.0910 with it), so the thesis's
runs keep dropout. **For the thesis:** the two machines run the same computation, and a run on
either is a run of its cell. Results differ between them only as a dropout draw would make them
differ: within a machine a seed fixes everything, across machines it fixes everything but dropout.

**G-78 under BPE 17k** (the second machine; the null as a twin, and its empty slot, 20k, seeds 0–3).
The null's twins end −0.0002, +0.0008, −0.0005 and −0.0004 from the thesis machine's. The empty slot
leads the null by +0.0005, 0.0000, +0.0002 and +0.0012: **+0.0005 ± 0.0003**. Under endings apart
it is +0.0025 and +0.0034 (§33.106). The null's handicap is therefore mainly endings apart's. Both
tokenizers mark the same words (`lexical.token_class`: an ending is letters without a leading
space, and so continues its word), so it is not a boundary bug. Its cause is not yet known.
**P226 wrong.** P222 stays wrong: of its twelve twins, one ends 0.0012 away, though their mean is
−0.0001.

**MultiBLiMP on fiction at full training** (item 3's grammar test): BPE 17k's models judge 88.1% of
the pairs right, and endings apart's do better by **+2.3 ± 0.3 points (4/4)**. By agreement type
that is number +3.4, gender +3.6 and person +0.7, all 4/4. At 10k on a quarter of the text it was
+5.3 ± 0.6, from a lower base.

**Predictions.** P236–P239 right; P226 wrong. The tally is 138 right, 8 half right, 84 wrong, 8 never
tested, 1 withdrawn, 11 pending, of 250.

### 33.110 G-78 mapped: the null's handicap is endings apart's at every budget — and it confounds "exposure makes case more useful"

The rest of 17 October's runs (`device_twins_20261018T033029`, `paired_cells_20261017T142732`,
`loss_by_kind_20261017T142734`). The thesis machine lost power at 18:02 and was shut down once its
records were committed, with item 7's first run unfinished. The second machine finished its
follow-up alone.

**The empty slot against the null** (each seed paired from identical initial weights; ~38M tokens
unless noted):

| | endings apart | BPE 17k |
|---|---|---|
| 10k (second machine) | **+0.0022 ± 0.0007 (4/4)** | −0.0006 ± 0.0004 (1/4) |
| 20k (thesis / second machine) | +0.0025 ± 0.0004 / +0.0034 ± 0.0009 (4/4 each) | +0.0005 ± 0.0003 (4/4, second) |
| 151M, 50k (thesis machine) | +0.0020 ± 0.0014 (3/4; one seed +0.0060) | — |

**Case against the empty slot, under endings apart:**
- **10k: +0.0022 ± 0.0007 (4/4)**, against +0.0044 over the null on the same machine, and +0.0048
  on the thesis machine;
- 20k: +0.0013 ± 0.0007 (3/4) and +0.0003 ± 0.0006 (2/4) on the two machines (§33.106);
- **full training: +0.0010 ± 0.0007 (3/4)**, against +0.0030 ± 0.0009 over the null.

At full training, case without the leak (prepositions from their form) and case from the word form
end level with the empty slot: −0.0005 ± 0.0006 (1/4) and −0.0004 ± 0.0008 (2/4). So against the
empty slot, what case keeps at full training is about the tagger's leak (+0.0015 ± 0.0001, §33.81).
Split by target, case's gain over the null sits on the endings at full training too: +0.0021 ±
0.0001 (4/4), 68% of it, where the empty slot gains −0.0001. The empty slot's own advantage is
spread thinly over word starts and continuations, and is noisy.

**P240 right** (+0.0022), **P241 wrong** (−0.0006), **P242 right** (+0.0022), **P248 right**
(+0.0020), **P249 wrong** on its boundary (+0.000992, not above 0.001).

**What it changes for the thesis.**
1. Under BPE 17k the null is as good as an empty slot (−0.0006 at 10k, +0.0005 at 20k), so its
   gains stand against both references. Under endings apart the null carries a cost of about 0.002
   at every budget, so its gains over the null are about 0.002 larger than over the best model
   without the feature.
2. This confounds a cross-tokenizer comparison the thesis makes. "Exposed endings make case more
   useful" compares case's gain over each tokenizer's own null (§33.58, §33.65), and that null's
   handicap depends on the tokenizer. At 10k case gains +0.0032 over BPE 17k's null, which is
   about the same as over an empty slot there. Under endings apart it gains +0.0048 over that
   tokenizer's null but +0.0022 over an empty slot. Against the empty slot, then, case gains less
   where endings are exposed: the hypothesis the comparison was built on (case helps where the
   tokenizer hides endings) would stand after all. Unigram 17k and BPE 4k, 8k and 32k have no
   empty slot yet. Until they do, the exposure claim is under revision (G-78), not withdrawn.
3. At full training under endings apart, what case adds over the empty slot is the tagger's leak,
   not the word's own case.

### 33.111 The probes at five seeds, width 512's twins, and fifty-two twins in all

**Item 5** (the thesis's list: the probes it cites at one seed). The cited one-seed runs were repeated
on the second machine at seeds 0–3, with a checkpoint every 1,000 steps. Each was probed as seed 4 was:
every layer, every checkpoint, the word's tag and, after removal, the previous word's. All 24 probes
ran on its GPU, about 30 seconds each; on the CPU the probe reproduces seed 4's GPU record within 6 ×
10⁻⁸ (§33.105).
- **After removal, the word's tag at the deepest layer** (switched minus never fed): 0.5–1.3
  points behind at the switch (step 5,000), then level within 0.3 points from step 6,000 to the
  end, at every seed (seed 4: level from 6,000). **P244 right.**
- **After removal, the previous word's tag at the end** (deepest layer): +1.8, +2.4, +1.9, +1.7;
  with seed 4's +2.0, **+2.0 ± 0.1 points (5/5)**. **P245 right.**
- **The auxiliary target, layer 2** (real minus shuffled): +5.1, +5.0, +4.3, +4.8 at step 2,000, and
  +2.0, +1.5, +1.7, +1.5 at the end; with seed 4's, **+4.8 ± 0.1 and +1.7 ± 0.1 (5/5 each)**. **P246,
  P247 right.**

For the thesis: §3.8's two probe sentences no longer rest on one seed; the probes across widths
(§3.7) still do.

**Width 512's twins at 20k, seed 1:** BPE 17k's null ends +0.0005 *(+0.00045 in the record `device_twins_20261018T033029`: +0.0004 to four decimals; annotated 19 October 2026)* and endings apart's +0.0001 from
the thesis machine's, so seed 1's lead is reproduced within 0.0004 (+0.0101 against +0.0097). The
second machine's seeds 1–3 give +0.0099 ± 0.0006. **P243 right.**

**All twins** (`device_twins_20261018T033029`): 52 runs of the thesis machine repeated on the second,
two of them with dropout off. They differ by +0.0001 ± 0.0002 (27 above, 25 below). The mean
|difference| is 0.0007, and the largest grows with training: 0.0011 at 10k, 0.0025 at 20k, 0.0063 at
50k. Per seed, effects measured within each machine agree: case over the null at 10k is +0.0048 and
+0.0044, at 20k +0.0038 and +0.0037.

**Predictions** (§33.110–§33.111). P240, P242, P243, P244, P245, P246, P247 and P248 right; P241 and
P249 wrong. P250 waits for item 7. The tally is 146 right, 8 half right, 86 wrong, 8 never tested, 1
withdrawn, 1 pending, of 250.

### 33.112 One long chain on the second machine: G-78 across tokenizers and sizes, the width probes, items 7 and 8 (prepared, 18 October)

The two machines are equivalent up to dropout's draw (§33.109), and the second is free for the day. One
chain of about fifteen hours runs there unattended (`scripts/chains/second_2026-10-18.sh`), each
comparison within it. Where a cell is the thesis machine's, it is repeated as a twin. In order of what
the results change in the text:

1. **G-78 across tokenizers** (the exposure claim, §33.110). At ~38M tokens and 10k, seeds 0–3, under
   Unigram 17k and BPE 4k, 8k and 32k: the null and case as twins, and the empty slot (48 runs). Under
   BPE 17k, case as a twin (4 runs): its null and empty slot are on that machine already.
2. **G-78 on less data:** under BPE 17k at ~1.5M and ~7.6M tokens, 10k, the null and case as twins,
   and the empty slot (24 runs). FINDINGS states case's gains on less data against the null.
3. **The width probes** (the list's item 5, last part; §3.7). The English null at widths 128 and 256,
   10k and 20k, seeds 0–3, with a checkpoint every 1,000 steps, each probed as seed 4 was: the word's
   tag and the previous word's, every layer (16 runs, 32 probes).
4. **Item 7:** case and its word form as twins, and case with prepositions from their form. 151M
   tokens, BPE 17k, 50k, seeds 0–3 (12 runs). The thesis machine's seed 0 of the last, run this
   morning, is one more twin.
5. **Item 8:** the null as a twin, frequency band and part of speech from the word form. English, 50k,
   seeds 2–3 (6 runs).

**Data.** The streams went over the direct route, compressed, checked by SHA-256 on arrival: 3.5 GB
in under three minutes. Where a stream's codes file holds many facts, the one a run reads was
written to a file of its own and read back equal (`scripts/ops/slim_codes.py`). Before the chain, a
30-step run through the real loader passed for each kind of cell on the second machine (12 of 12).
Rehearsed with the stub in seven scenarios, all passing.

Predictions, before any run:

- **P251** (~55%). Under Unigram 17k at 10k the empty slot ends better than the null's twin by more
  than 0.001 on average.
- **P252** (~60%). Under each of BPE 4k, 8k and 32k at 10k the empty slot ends within 0.001 of the
  null's twin on average.
- **P253** (~60%). Against the empty slot, case at 10k gains less under endings apart than under BPE
  17k, by more than 0.001 on average (both on the second machine).
- **P254** (~50%). Against the empty slot, case at 10k gains no more under Unigram 17k than under BPE
  17k: the difference is below +0.0005.
- **P255** (~70%). The new twins at 10k (the null and case under five tokenizers) end within 0.0015 of
  their originals each, and within ±0.0005 on average.
- **P256** (~55%). Under BPE 17k at ~1.5M and at ~7.6M the empty slot ends within 0.001 of the null's
  twin on average, at each size.
- **P257** (~70%). At each of seeds 0–3, at matched null BPB between 1.70 and 1.45, the 256-wide null's
  deepest layer reads the previous word's tag better than the 128-wide null's, by more than 3 points
  somewhere (seed 4: +3.9 to +10.4).
- **P258** (~55%). On the probe's axis (`probe_axis.py --previous`), the widths' part-of-speech gains
  at 10k differ by at most half as much as on the null-BPB axis, over five seeds (seed 4: 0.0035
  against 0.0098).
- **P259** (~70%). At 50k, seeds 2–3, frequency band and part of speech from the word form each end
  within 0.002 of the null's twin at both seeds.

P250 (item 7) stands as recorded in §33.105.

**The sentences these decide.**
- **G-78 (P251–P256).** If the handicap follows exposure, the exposure claim is restated against the
  empty slot: exposed endings make case partly redundant, as first hypothesised. If only endings
  apart carries it, the claim stands for Unigram and falls for endings apart. If BPE 17k has none on
  less data, the size claims stand against both references.
- **The width probes (P257–P258).** If they agree, §3.7's half is stated with its spread; if not,
  the sentence says how far the seeds differ.

### 33.113 G-78 across tokenizers and corpus sizes: the slot costs where endings are exposed, helps under BPE — and exposure makes case partly redundant

The chain of §33.112 ran on the second machine on 18 October from 00:12 to 14:08 (its clock),
unattended: 110 runs and 32 probes. Every piece's first run passed its twin check, within 0.002 of
the thesis machine's, and nothing was skipped or failed. 79 runs repeat the thesis machine's (twins,
kept apart in `results/audits/device_twins_20261018/`); 31 are new. The effects below are in
`device_twins_20261018T182550`, per seed on each machine.

**The empty slot against the null, and case against both** (the second machine; ~38M tokens, 10k;
seeds 0–3, each pair from identical initial weights; positive: the first named ends better):

| tokenizer | the empty slot over the null | case over the empty slot | case over the null |
|---|---|---|---|
| endings apart (§33.110) | **+0.0022 ± 0.0007** (4/4) | +0.0022 ± 0.0007 (4/4) | +0.0044 ± 0.0008 (4/4) |
| Unigram 17k | **+0.0039 ± 0.0016** (3/4) | +0.0018 ± 0.0012 (3/4) | +0.0056 ± 0.0007 (4/4) |
| BPE 4k | **−0.0037 ± 0.0014** (0/4) | +0.0076 ± 0.0012 (4/4) | +0.0039 ± 0.0007 (4/4) |
| BPE 8k | **−0.0016 ± 0.0006** (0/4) | +0.0065 ± 0.0014 (4/4) | +0.0049 ± 0.0012 (4/4) |
| BPE 17k | −0.0006 ± 0.0004 (1/4) | +0.0038 ± 0.0007 (4/4) | +0.0032 ± 0.0006 (4/4) |
| BPE 32k | **−0.0025 ± 0.0006** (0/4) | +0.0055 ± 0.0005 (4/4) | +0.0030 ± 0.0004 (4/4) |

**On less data, under BPE 17k** (10k):

| train tokens | the empty slot over the null | case over the empty slot | case over the null |
|---|---|---|---|
| ~1.5M | **−0.0032 ± 0.0013** (0/4) | +0.0097 ± 0.0005 (4/4) | +0.0065 ± 0.0015 (4/4) |
| ~7.6M | **−0.0015 ± 0.0005** (0/4) | +0.0040 ± 0.0006 (4/4) | +0.0025 ± 0.0005 (4/4) |
| ~38M | −0.0006 ± 0.0004 (1/4) | +0.0038 ± 0.0007 (4/4) | +0.0032 ± 0.0006 (4/4) |

On the thesis machine case over the null was +0.0076, +0.0031 and +0.0032 at the three sizes
(§33.56).

**The null's slot is not neutral, and its sign follows exposure.** Under the two tokenizers that
expose endings, the empty slot ends better than the null: by 0.0022 under endings
apart, the handicap of §33.106, and by 0.0039 under Unigram 17k. Under BPE the null ends better:
by 0.0037 at 4k, 0.0016 at 8k and 0.0025 at 32k (0/4 each), and under BPE 17k by 0.0032 on ~1.5M
tokens and 0.0015 on ~7.6M. On ~38M under BPE 17k the two are level (0.0006 ± 0.0004 here, the
empty slot 0.0005 better at 20k, §33.109). The null's slot says whether the previous unit was a
word (§33.106). That costs where endings are exposed and helps, or does nothing, under BPE. Where
the help sits under BPE was not re-scored.

**Verdicts.** **P251 right** (+0.0039). **P252 wrong:** under BPE 4k, 8k and 32k the empty slot is
not level with the null but worse, by 0.0016 to 0.0037. **P253 right:** against the empty slot case
gains +0.0022 under endings apart and +0.0038 under BPE 17k, 0.0016 less. **P254 right:** under
Unigram 17k it gains +0.0018, 0.0020 less than under BPE 17k. **P256 wrong:** on less data the
empty slot ends 0.0032 and 0.0015 worse than the null. P255, on the twins, is in §33.114.

**What it changes for the thesis.**
1. **The best model without the feature is not one model.** Where endings are exposed it is the
   empty slot; under BPE it is the null (at 17k on ~38M the two are level). Against it, case gains
   **+0.0022 under endings apart and +0.0018 under Unigram 17k**, against **+0.0030 to +0.0049
   under BPE** (32k +0.0030, 17k +0.0032, 4k +0.0039, 8k +0.0049; 4/4 each). Where the tokenizer
   exposes endings case adds about half as much. **Exposed endings make case partly redundant**: the
   hypothesis the comparison was built on (§33.58), as §33.112 set out for this outcome. Against
   the empty slot alone the gap is wider (+0.0038 to +0.0076 under BPE), since there the empty model
   lacks a slot that helps.
2. **Over each tokenizer's own null the order turns.** There case gains +0.0044 and +0.0056 where
   endings are exposed, and +0.0032 under BPE at the same vocabulary size (17k; +0.0030 to +0.0049
   across BPE's four). That is the measurement behind FINDINGS' "exposed endings make case more
   useful" (§33.58, §33.65), and it stays true as stated. Case's slot says what the null's says and
   pays the same cost (measured under endings apart, §33.106), so over the null it is case's own
   values that gain more where endings are exposed. But the slot that carries them costs there,
   and net the feature adds less. The thesis can state both: case's information is larger where
   endings are exposed, and what adding case buys is smaller.
3. **The size claims stand against both references.** On less data under BPE 17k the null is the
   better reference. So case's gains over it (+0.0076 at ~1.5M and +0.0031 at ~7.6M on the thesis
   machine) are gains over the best model without the feature. Over the empty slot they are larger
   (+0.0097, +0.0040). More gain on less data holds either way.
4. The statements comparing tokenizers over their own nulls at 20k and at full training (§33.68,
   §33.76) carry the same caveat. Under endings apart the null costs about 0.002 at every budget,
   and under BPE 17k it costs nothing (§33.110). Against the best model without the feature, case at
   full training keeps +0.0010 under endings apart, about the tagger's leak (§33.110). Under BPE 17k
   it ends below its null (−0.0013 ± 0.0008).

### 33.114 The width probes at five seeds, items 7 and 8, and 131 twins

**The width probes** (item 5's last part; `probe_axis_20261018T182434` on the previous word's tag,
`…T182436` on the word's own). The English null at widths 128 and 256, at 10k and 20k, seeds 0–3,
was repeated on the second machine with a checkpoint every 1,000 steps and probed as seed 4 was.
The gains are the thesis machine's, over seeds 0–3 of each width (§33.50). The probe is each seed's
own, at the deepest layer. `probe_axis.py` now finds a probed twin by its id among the twin copies.
It compares the widths at each seed, then once more on the accuracy averaged over the five seeds,
at the steps they share (seed 4 has a checkpoint every 500 steps, the others every 1,000).

| largest gap between the widths' gains at a common level | 10k | 20k |
|---|---|---|
| the previous word's tag, per seed | 0.0041, 0.0034, 0.0039, 0.0042, 0.0025: **0.0036 ± 0.0003** | 0.0041, 0.0037, 0.0037, 0.0043, 0.0035: **0.0039 ± 0.0001** |
| the same, on the accuracy averaged over seeds | 0.0038 | 0.0037 |
| the word's own tag, per seed | 0.0063, 0.0053, 0.0082, 0.0068, 0.0052: 0.0063 ± 0.0005 | 0.0066, 0.0056, 0.0084, 0.0068, 0.0063: 0.0068 ± 0.0005 |
| the null's BPB (the gains alone, recomputed as in §33.50) | 0.0099 | 0.0098 |

Seed 4's values are §33.50's, reproduced. At every matched null BPB from 1.75 to 1.45 the wider
null reads the previous word's tag better: by +2.5 to +10.4 points over the five seeds, each
seed's largest lead 8.7 to 10.4 at 20k (seed 4: +3.9 to +10.4). On the word's own tag the lead is
+0.4 to +1.5. At the end of the 20k budget the base null reads the previous word's tag at **0.629 ±
0.004** (seed 4: 0.619) and the wider at 0.732 ± 0.002. On the word's own tag the two read 0.883 ±
0.001 and 0.895 ± 0.001. The wider model's gain has collapsed by step 3,000 (+0.0033). By then its
deepest layer reads the previous word's tag at 0.678 to 0.686, at every seed.

**P257 right:** each of seeds 0–3 leads by more than 3 points (8.8, 10.4, 8.7 and 9.2 at its
largest). **P258 right:** at 10k the widths' gains differ by at most 0.0036 ± 0.0003 per seed (0.0042
at most) on the probe's axis, against 0.0099 on the null's BPB, under half. For the thesis (§3.7)
"about half" now holds at five seeds. On the previous word's tag the widths' gains differ by at most
0.0039 ± 0.0001 (20k), against 0.0068 ± 0.0005 on the word's own tag and 0.0098 on the null's BPB.
The axis closest to what the feature carries removes about three fifths of the widest gap. It also
halves the mean gap: +0.0034 on the null's BPB (§33.50), +0.0019 on the averaged accuracy.

**Item 7: case at full training under BPE 17k, without the leak** (the second machine; 151M tokens,
50k; seeds 0–3, each against the case and word-form twins there):
- case 0.9430, 0.9451, 0.9433, 0.9470; case with prepositions read from their form 0.9435, 0.9459,
  0.9437, 0.9475; case from the word form 0.9422, 0.9475, 0.9441, 0.9466;
- case over its word form +0.0005 ± 0.0007 (2/4; the thesis machine +0.0006 ± 0.0004, 3/4);
- the leak, case over prepositions from their form: **+0.0006 ± 0.0001 (4/4)** (the thesis
  machine's seed 0: +0.0005);
- prepositions from their form over the word form: **−0.0001 ± 0.0007 (2/4)** (seed 0 on the thesis
  machine +0.0004).

**P250 right** (within 0.0004 on average). Under BPE 17k too, all of case's lead over its word form
at full training is the tagger's leak. That is the list's outcome «під обома токенізаторами».

**Item 8: the frequency band and part of speech from the word form at 50k, seeds 2–3** (English;
the second machine, each against the null's twin there):
- the null 1.3432, 1.3388; the frequency band 1.3468, 1.3385; part of speech from the word form
  1.3439, 1.3370;
- the frequency band over the null −0.0036, +0.0003; part of speech from the word form −0.0008,
  +0.0018;
- with seeds 0–1 from the thesis machine: the frequency band **−0.0005 ± 0.0011** (3/4: +0.0004,
  +0.0009, −0.0036, +0.0003); part of speech from the word form **−0.0001 ± 0.0014** (2/4: +0.0023,
  −0.0037, −0.0008, +0.0018).

At seed 2 the second machine's three runs end 0.004 to 0.008 above the thesis machine's null (its
twin +0.0043). They drift apart from step 20,000, with no spike, on the draw they share. **P259 half
right:** part of speech from the word form stays within 0.002 at both seeds; the frequency band falls
0.0036 below the null at seed 2. For the thesis every row of the full-training table now has four
seeds, and none beats its null by more than 0.002 on average. That is the list's outcome: the
sentence stands on four seeds for every row.

**The twins** (`device_twins_20261018T182550`): 131 runs repeated across the two machines, two of
them with dropout off. They differ by −0.0001 ± 0.0001 (65 above, 66 below; mean |difference|
0.0010). The largest difference is 0.0031 at 10k (BPE 4k's null, seed 2), 0.0044 at 20k and 0.0063
at 50k. The 36 new twins at 10k are the null and case under Unigram 17k and BPE 4k, 8k and 32k, and
case under BPE 17k. They differ by −0.00001 ± 0.00018, but four exceed 0.0015: BPE 4k's null
−0.0031, BPE 8k's null +0.0025, and BPE 8k's case −0.0018 and +0.0018. **P255 half right:** level on
average, not within 0.0015 each. Effects measured within each machine agree. Case over the null at
10k, thesis machine against second: Unigram 17k +0.0054 / +0.0056; BPE 4k +0.0044 / +0.0039; 8k
+0.0040 / +0.0049; 17k +0.0032 / +0.0032; 32k +0.0029 / +0.0030; ~1.5M +0.0076 / +0.0065; ~7.6M
+0.0031 / +0.0025.

**Pairs across machines, found and left out.** With the second machine's new cells beside the
thesis machine's in `results/`, the exported comparisons paired 23 seeds across machines against the
thesis machine's plain baselines. Eight were item 4a's 50k runs; fifteen came from today's pull (the
empty slots on less data, item 7's control, item 8's two features). Refreshed as they were, the
figures would also have paired item 8's seeds 2–3 with the thesis machine's null.
`export_results.py`, `demo_figures.py`, `learning_law.py`, `thesis_figures.py` and `probe_axis.py`
now pair within one machine and name the seeds left out. The figures come out unchanged, and no
number in FINDINGS used those pairs.
*Corrected 19 October 2026: only `export_results.py` named the seeds it left out, and eight other
scripts did not yet pair within one machine; all were brought under one rule on 19 October (an audit
of that day).*

**Predictions** (§33.113–§33.114). P250, P251, P253, P254, P257 and P258 right; P255 and P259 half
right; P252 and P256 wrong. The tally is 152 right, 10 half right, 88 wrong, 8 never tested and 1
withdrawn, of 259.

### 33.115 Two follow-ups: the exposure comparison at 20k, and where the null's help sits (prepared, 18 October)

The queue review of 18 October (DIRECTIONS, "The queue") put two questions first. Both run on the
second machine's records:

1. **The exposure comparison at twice the budget.** The null and the empty slot under BPE 17k at
   20k ran on the second machine on 17 October, and so did endings apart's three cells (§33.109).
   Case under BPE 17k at 20k did not. Four runs, as twins of the thesis machine's, about 35 minutes
   (`scripts/chains/second_followup_2026-10-18.sh`; rehearsed with the stub in seven scenarios, all
   passing). At 20k the empty slot is the better reference under both tokenizers: by 0.0005 under
   BPE 17k and 0.0034 under endings apart on that machine. So case is compared with it. Under
   endings apart there case gains +0.0003 over it.
2. **Where the null's help sits under BPE.** The second machine's 48 models of the null and the
   empty slot under the six tokenizers, at ~38M and 10k, are re-scored token by token
   (`loss_by_kind.py --facts mconst@OFF0`). The thesis machine does it once it holds the models.
   Evaluation draws no dropout, so either machine gives the same number.

Every run's saved model and checkpoints are being copied to the thesis machine, so that it keeps
everything the records need.

Predictions, before any run:

- **P260** (~65%). On the second machine at 20k, case's gain over the empty slot under BPE 17k is
  within 0.001 of its gain under endings apart (+0.0003): against the best model without the
  feature, the two tokenizers end level at twice the budget.
- **P261** (~70%). Case over the null under BPE 17k at 20k on the second machine is within 0.001
  of the thesis machine's (+0.0010 ± 0.0008).
- **P262** (~55%). Under BPE 4k, 8k and 32k, more than half of the null's advantage over the empty
  slot sits on units' first tokens and on non-word targets.
- **P263** (~70%). Under endings apart and Unigram 17k at 10k, more than half of the empty slot's
  advantage over the null sits on units' first tokens and on non-word targets, as under endings
  apart at 20k (§33.106).

**The sentences these decide.**
- **P260.** *Level:* the redundancy is an early-training property; by 20k case adds about as much
  under either tokenizer, against the best model without it. *BPE 17k clearly ahead:* exposed
  endings keep case partly redundant at twice the budget.
- **P262–P263.** *Units' first tokens under every tokenizer:* method point 10 gains its mechanism:
  the slot's signal acts where a unit starts, and the tokenizer sets its sign. *Later tokens under
  BPE:* the point says the null helps BPE continue a word, and costs where endings are exposed at
  word starts.

### 33.116 The follow-ups: by 20k the two tokenizers are level; the slot costs at word starts or helps on a word's ending, by tokenizer

Both follow-ups of §33.115 ran on 18 October.

**The exposure comparison at 20k** (`device_twins_20261018T193030`; the second machine; ~38M tokens,
seeds 0–3). Case under BPE 17k at 20k ended at 1.0130, 1.0156, 1.0109 and 1.0142; its first seed
was within 0.0003 of the thesis machine's. Against the empty slot, the better reference at 20k under
both tokenizers, case gains:
- under BPE 17k +0.0004, −0.0001, +0.0016, +0.0003: **+0.0005 ± 0.0004 (3/4)**;
- under endings apart +0.0003 ± 0.0006 (2/4).

Over the null under BPE 17k it gains +0.0010 ± 0.0004 (3/4), as on the thesis machine (+0.0010 ±
0.0008). **P260 right, P261 right.** At 10k case gained 0.0016 less where endings are exposed
(+0.0022 against +0.0038); by 20k the difference is gone, and under either tokenizer case adds about
+0.0005 over the best model without it. The redundancy is an early-training effect. The queue's
third candidate, G-78 at 20k under the other tokenizers, ran only if the redundancy lasted, so it is
dropped.

**Where the slot's effect sits** (`loss_by_kind_20261018T192042`). The second machine's 48 models of
the null and the empty slot under six tokenizers (~38M, 10k, seeds 0–3) were re-scored token by token
on the thesis machine. Each re-scored within 1.5 × 10⁻⁸ of its record. In the table, the empty slot
over the null: positive where the null's slot costs, negative where it helps.
*Annotated 19 October 2026: these second-machine models were read from a hand-made root of links
under `scratch/`, which is not kept; the same models are now under
`results/audits/device_twins_models/` (twins) and `results/<id>/` (new runs).*

| tokenizer | in all | units' first tokens | later tokens | of which the ending's token | non-words | first tokens and non-words |
|---|---|---|---|---|---|---|
| endings apart | +0.0022 ± 0.0007 | +0.0011 ± 0.0003 | +0.0006 ± 0.0003 | +0.0000 | +0.0005 ± 0.0002 | 72% |
| Unigram 17k | +0.0039 ± 0.0016 | +0.0024 ± 0.0006 | +0.0008 ± 0.0007 | +0.0004 | +0.0007 ± 0.0004 | 79% |
| BPE 4k | −0.0037 ± 0.0014 | −0.0006 ± 0.0003 | **−0.0032 ± 0.0012** | −0.0019 ± 0.0010 | +0.0001 | 13% |
| BPE 8k | −0.0016 ± 0.0006 | +0.0001 | **−0.0019 ± 0.0006** | −0.0011 ± 0.0005 | +0.0002 | none |
| BPE 17k | −0.0006 ± 0.0004 | −0.0002 | −0.0006 ± 0.0002 | −0.0002 | +0.0002 | none |
| BPE 32k | −0.0025 ± 0.0006 | −0.0011 ± 0.0003 | −0.0010 ± 0.0004 | −0.0004 | −0.0004 ± 0.0002 | 62% |

Where the tokenizer exposes endings, the slot's cost sits at units' first tokens and non-words, as
under endings apart at 20k (§33.106). Under BPE at 4k and 8k its help sits on a word's later tokens,
half and three fifths of it on the token that holds the ending's start (50% and 60%). Under 32k it
is spread, three fifths at first tokens and non-words. **P262 wrong** (only 32k has more than half
there); **P263 right** (72%, 79%).

**What it changes for the thesis.**
1. **The exposure claim gains its time course.** Against the best model without the feature,
   exposed endings make case partly redundant early in training (10k). By 20k the two tokenizers are
   level. At full training case adds only the tagger's leak under endings apart (§33.110) and
   nothing under BPE 17k (§33.114).
2. **Method point 10 gains where the slot acts.** The null's slot says whether the previous unit was
   a word. Where the tokenizer makes endings tokens of their own, that costs at units' first tokens
   and non-words. Where BPE at a small vocabulary spreads a word over several tokens, it helps on the
   word's later tokens, mostly its ending. The tokenizer sets the sign of the slot's effect and where
   it acts.

**The second machine's models on the thesis machine.** Every saved model and checkpoint of the second
machine is being copied to the thesis machine, so that its records keep their models: a new run's
next to its record, a twin's in `results/audits/device_twins_models/`. In the first copy the three
calibration reruns of 16 October, which share the thesis machine's run ids, were taken for new runs.
Their models overwrote the thesis machine's own (G-79). The records are intact; those three models are
to be re-made by retraining when there is time.

**Predictions** (§33.115). P260, P261 and P263 right; P262 wrong. The tally is 155 right, 10 half
right, 89 wrong, 8 never tested and 1 withdrawn, of 263.

