# Documentation — start here

> **Kind:** index. **Authoritative for:** where each kind of information lives,
> and in what order to read.


## How the documents are organized

Every document is exactly one **kind**, named in its first lines:

| kind | what it holds | how it changes |
|---|---|---|
| **claims** | what is true now | rewritten in place; one statement per claim |
| **record** | what happened, in order | append-only; corrections annotated, never silently rewritten |
| **reference** | how something works now | rewritten whenever the thing changes; no dated status sections |

Each fact has **one home**, and every other page links to it: claims in [`docs/FINDINGS.md`](FINDINGS.md), numbers in the [results record](record/results.md), the narrative in the [journey](record/journey/README.md), open questions in [`docs/GAPS.md`](GAPS.md).
A summary page may quote a number only as it appears in its home — `scripts/ops/check_docs.py` checks that, and every link.

```
docs/
  README.md               this page — index
  FINDINGS.md             claims: what the thesis claims, with numbers and firmness, one heading per claim
  predictions.md          record: every prediction, made before its result, with its outcome
  GAPS.md                 claims: questions open, closed, out of scope
  DIRECTIONS.md           claims: what next — the programme and the queue
  reference/              reference: apparatus.md · conditions.md · ukrainian_pipeline.md · related_work.md (every finding from other work) · literature_audit.md · literature_analysis.md
  record/                 record: journey/ (parts, appendices and an index) · results.md (every number) · bugs.md · packs.md (what each results pack held)
```

## Reading order — for a report or a presentation

1. [`docs/FINDINGS.md`](FINDINGS.md) — the claims. **Authoritative for what the thesis claims.**
2. [`docs/GAPS.md`](GAPS.md), [Scope](GAPS.md#scope--declared-not-chased) — the limitations to state with them.
3. [`docs/predictions.md`](predictions.md) — every prediction, made before its result, with its outcome.
4. [`reference/related_work.md`](reference/related_work.md) — every finding from other work the project uses, with its reference and how it was checked; [`reference/literature_audit.md`](reference/literature_audit.md) — the four fairness conditions; no audited paper meets them all.
5. [`record/results.md`](record/results.md) §0 — every headline number with run ids. **Authoritative for numbers.**
6. [`record/journey/README.md`](record/journey/README.md) — the narrative, part by part, with an index by question and of every section.
7. As needed: [`reference/apparatus.md`](reference/apparatus.md) (model, data, reproduction), [`reference/conditions.md`](reference/conditions.md) (new corpora, tokenizers, conditions), [`reference/ukrainian_pipeline.md`](reference/ukrainian_pipeline.md), [`docs/DIRECTIONS.md`](DIRECTIONS.md), [`record/bugs.md`](record/bugs.md).

## The project in one paragraph

A master's thesis asking whether injecting linguistic structure into a language model's **embedding layer** helps, and how that compares with morphology in the **tokenizer**.
A fixed small decoder-only transformer (≈5M parameters: 4 layers, 4 heads, width 128, FFN 512; batch 32 × 256 tokens; learning rate 3e-4, cosine schedule; 50k steps, about an hour on one laptop GPU; widths 64 to 512 where stated) is trained on **WikiText-103** (English, ~120M BPE tokens), on **Ukrainian Wikipedia** (20% of the 2023 dump, ~151M tokens; validation 598 articles) and, for the tokenizer, on UberText 2.0's **fiction**.
Only the tokenizer (Q1) or the embedding (Q2) varies.
Linguistic features come from spaCy taggers (`en_core_web_sm`, `uk_core_news_sm`) and enter as an 8-dim **slot** concatenated to a 120-dim token embedding (or **added** to a full 128-dim one).
The metric is **bits per byte (BPB)** of held-out text — comparable across tokenizers, *not* across languages.

## The method, in five rules

1. **Same-shape null.** Every feature is compared with a null cell of identical architecture carrying no information (`mconst`) — not with the plain baseline, which differs in architecture — and, since in the 5M Ukrainian model the null is not neutral, with an empty slot too (FINDINGS, method point 10).
2. **Causal shift.** Word-level features describe the *previous* word, with word boundaries from the tokenizer, so a feature never reveals the word being predicted (the within-word lookahead leak, journey §30).
3. **Matched initialization.** Cells of identical parameter shapes share their initial weights at the same seed; comparisons are paired by seed, within one machine (method point 11).
4. **Seeds.** Four per comparison as a rule (one to eight; each claim names its count); reported as mean ± standard error over seeds, with how many seeds share the sign.
5. **Predictions first.** Every experiment's expected outcome was written down before it ran, with the result that would make it wrong.

## The claims

Each claim's statement, numbers and evidence are in [`docs/FINDINGS.md`](FINDINGS.md), under its own heading; this table only points there.

| claim | firmness | in FINDINGS |
|---|---|---|
| **Q1 in English:** BPE beats segmentation into morphemes | firm | [morphemes against BPE](FINDINGS.md#q1-in-english-morphemes-against-bpe) |
| **Q1 in English:** kept inside BPE, morphology helps, and the gain does not fade; grammar agrees | four seeds at 10k and 20k, eight at full training; BLiMP at eight seeds | [morphology inside BPE](FINDINGS.md#q1-in-english-morphology-inside-bpe) |
| **Q1 in Ukrainian:** BPE that keeps inflectional endings apart beats plain BPE, and the gain lasts to full training; it peaks at width 256, shows in grammar, and holds on fiction | two corpora, two corpus sizes, three budgets, three widths; four seeds each | [morphology inside BPE](FINDINGS.md#q1-in-ukrainian-morphology-inside-bpe) |
| **Q2:** at the end of training no injected feature beats its same-shape null, in either language | four seeds for every row | [no feature beats its null](FINDINGS.md#q2-no-feature-beats-its-null-at-the-end-of-training) |
| **Before the model has learned it, features help** | four seeds per condition, both languages | [before the model has learned it](FINDINGS.md#before-the-model-has-learned-it-features-help) |
| What reading the sentence adds over the word form is mostly not context; at full training it is the tagger's leak | four seeds; two tokenizers; English part of speech at two budgets | [what the sentence adds](FINDINGS.md#what-the-sentence-adds-over-the-word-form) |
| **Early help:** every informative feature leads its null early in training; the lead is gone by the end | firm as a pattern | [early help](FINDINGS.md#early-help-in-every-informative-feature) |
| Removing English part of speech leaves a lasting deficit, adding it late does not; Ukrainian case leaves none | firm; Ukrainian case at one size and budget | [removing a feature](FINDINGS.md#removing-a-feature-and-adding-it-late) |
| As a target, morphology teaches, and the knowledge stops paying | firm | [morphology as a target](FINDINGS.md#morphology-as-a-target) |
| No benefit hides in the rare tail | firm; by BPE-token frequency | [the rare tail](FINDINGS.md#no-benefit-in-the-rare-tail) |
| **The law:** a feature's gain depends on how much its null has learned — one curve per feature | firm in English; in Ukrainian within the seed spread | [one curve per feature](FINDINGS.md#one-curve-per-feature-within-one-model-size) |
| Within one model size only: the previous word's tag explains about half of the width difference | the probes at five seeds per width, the gains at four | [across model sizes](FINDINGS.md#across-model-sizes) |
| Morphology in the tokenizer and case in the embedding add up | four seeds under each of six tokenizers | [they add up](FINDINGS.md#tokenizer-and-case-add-up) |
| Where the tokenizer exposes endings, case gains less for a stretch of training, at 5M and 30M parameters | four seeds per tokenizer | [exposed endings](FINDINGS.md#exposed-endings-case-gains-less-for-a-stretch) |
| Over each tokenizer's own null the order turns in the 5M model | four seeds; without the leak at three of four | [over each tokenizer's own null](FINDINGS.md#over-each-tokenizers-own-null) |
| In the 5M model the same-shape null is not neutral; at 30M parameters it costs nothing | six tokenizers and three corpus sizes at 10k; endings apart at three budgets; width 512 at 10k | [the null is not neutral](FINDINGS.md#the-same-shape-null-is-not-neutral) |
| Case helps by predicting endings | four seeds at 10k, three at 20k | [case on the endings](FINDINGS.md#case-helps-by-predicting-endings) |
| What the models build for themselves: the form's grammar, more accurately with endings apart; context, no better | four seeds, one corpus, linear probes | [what the model builds](FINDINGS.md#what-the-model-builds-for-itself) |
| **Why the null must have the same shape:** one learned vector shared by every input helps by itself | firm under rotary positions; how it helps is parked, outside the thesis | [one learned vector](FINDINGS.md#one-learned-vector-shared-by-every-input); [learned positions](FINDINGS.md#with-learned-positions-a-position-channel); [with training, width, language](FINDINGS.md#it-builds-with-training-steadies-training-and-survives-width-and-language) |
| **Measurement:** eleven method points, each found in this project's own numbers or tools | firm | [how to measure this](FINDINGS.md#how-to-measure-this--the-methodological-contribution) |

Withdrawn claims — do not present them as findings: [FINDINGS, "Withdrawn along the way"](FINDINGS.md#withdrawn-along-the-way).

## Figures

Drawn by `python scripts/analysis/demo_figures.py --out exports/figures` from every `result.json` (PNGs in `exports/figures/`; regenerate after new runs).
Positive = the feature (or slot) is better; error bars = standard error over seeds.
`bash scripts/ops/refresh_exports.sh` redraws them with every table in `exports/`.

- [**fig1_q2_end**](../exports/figures/fig1_q2_end.png) — each feature against its same-shape null at the end of training. The grey bar is a *control* (structure shuffled across words), not a linguistic feature.
- [**fig2_early_help**](../exports/figures/fig2_early_help.png) — feature minus null over training steps (log axis): the early lead and its fade. Compare shapes across languages, not heights.
- **fig3_budget** — Ukrainian case's gain against train tokens (10k steps), with longer training on the same 151M tokens (20k, 50k steps). Not drawn: its 151M nulls at 10k and 20k, quarantined under G-73, were re-scored but not re-run (the re-runs were dropped, journey §33.83); its numbers are in the results record, §0.
- [**fig4_architecture**](../exports/figures/fig4_architecture.png) — the slot over the plain baseline (left); the position test against the zero slot, English (right).

## Glossary

**Terms.** *BPB* — bits per byte of UTF-8 text; lower is better.
*Null* / *same-shape null* — a cell with the feature cell's architecture and no information.
*Slot* — the 8 input dims a feature occupies.
*Zero slot* — a slot held at exactly zero; *empty slot* — the null switched off from step 0 (`-OFF0`), from its initial weights.
*Additive* (`-ADD`) — the feature added to a full-width embedding instead of a slot.
*Causal* — the feature describes the previous word.
*Matched / paired* — same seed, same initial weights where the shapes allow.
*Early help* — the feature's lead over its null early in training.
*Within-word lookahead* — a whole-word feature visible at the word's later fragments, revealing them.
*Endings apart* — BPE 17k that never joins a stem to its inflectional ending.

**Run labels**: `bpe_<feature>-LEARN-CAUSAL[-ADD][-OFF<step>]_<tokenizer>[_<corpus><fraction>][_st<steps>k][_w<width>]`.

| label part | meaning |
|---|---|
| `baseline` | plain model, full 128-dim token embedding |
| `mconst` | **the null**: the slot with one class for every word |
| `mzero` | zero slot; `mzeronopos` without position in it; `mzeroposonly` position only there |
| `mflat` | one learned vector at every position (additive null) |
| `mconst16/32/64` | the null with a 16/32/64-dim slot |
| `mstruct` | word structure (8 classes); `mstructshuf` shuffled across words of similar frequency |
| `mfreq` | frequency band of the previous word |
| `pos` / `poslex` / `poscausal` | part of speech from the tagger / each word's majority tag (no context) / a tagger reading the left context alone |
| `mfeat` / `mfeatlex` | inflection bundle from the tagger / majority bundle |
| `ucase` / `ucaselex` / `ucaseprep` | Ukrainian case from the tagger / each form's majority case / the tagger's case with prepositions read from their form (the leak's control) |
| `mzero,ucase` | two slots: a zero slot beside case |
| `-OFF<step>` / `-ON<step>` | the feature switched off / on at that step; `-OFF0` is the empty slot |
| `bpe_17k` / `bpe_uk_17k` | English / Ukrainian 17k BPE tokenizer |
| `bpe_uk_4k` … `bpe_uk_32k`, `unigram_uk_17k` | Ukrainian BPE at 4k, 8k and 32k; Unigram at 17k |
| `bpeend_uk_17k`, `bpeend_17k` | Ukrainian / English 17k BPE that keeps inflectional endings apart |
| `bpe_ukf_17k`, `bpeend_ukf_17k` | the same two, trained on UberText fiction |
| `ukwiki0.2` | Ukrainian corpus fraction (0.002 ≈ 1.5M, 0.01 ≈ 7.6M, 0.05 ≈ 38M, 0.2 ≈ 151M train tokens) |
| `ukfiction0.1`, `ukfiction0.4` | UberText fiction fraction (0.1 ≈ 33.4M BPE tokens; 0.4 for full training) |
| `_st10k` / `_st20k` | 10k / 20k training steps (cosine over the same length); none = 50k |
| `_w64` … `_w512` | width 64, 256 or 512; none = 128 |

## Say out loud, and do not claim

- Small models (about 5M parameters; widths up to 512, 30M); English on one corpus, Ukrainian on two for the tokenizer (Wikipedia, fiction) and on one for the features; four seeds per comparison as a rule.
- English and Ukrainian BPB are on different scales (Cyrillic letters take two bytes): compare directions and shapes across languages, never heights.
- Do **not** claim "morphology in the embedding helps language models" without the condition (before the model has learned it); morphology in the tokenizer is the gain that lasts.
- Do **not** claim the architecture effect is novel before checking TUPE and DeBERTa-style position/content separation.
- Do **not** compare a feature with the plain baseline as evidence of information — only with its null, and in the 5M Ukrainian model with an empty slot too.

## The journey, section by section

Every section of the record, linked: [`record/journey/README.md`](record/journey/README.md).
