# Documentation — start here

> **Kind:** index. **Authoritative for:** where each kind of information lives,
> and in what order to read. Updated 8 October 2026.


## How the documents are organized

Every document is exactly one **kind**, named in its first lines:

| kind | what it holds | how it changes |
|---|---|---|
| **claims** | what is true now | rewritten in place; one statement per claim |
| **record** | what happened, in order | append-only; corrections annotated, never silently rewritten |
| **reference** | how something works now | rewritten whenever the thing changes; no dated status sections |

Each fact has **one home**, and every other page links to it: claims in
[`docs/FINDINGS.md`](FINDINGS.md), numbers in the [results record](record/results.md),
the narrative in the [journey](record/journey/README.md), open questions in
[`docs/GAPS.md`](GAPS.md). A summary page may quote a number only as it appears in its
home — `scripts/ops/check_docs.py` checks that, and every link.

```
docs/
  README.md               this page — index
  docs/FINDINGS.md             claims: what the thesis claims, with numbers and firmness
  docs/predictions.md          record: every prediction, made before its result, with its outcome
  docs/GAPS.md                 claims: questions open, closed, out of scope
  docs/DIRECTIONS.md           claims: what next
  reference/              reference: apparatus.md · conditions.md · ukrainian_pipeline.md · docs/reference/related_work.md (every finding from other work) · docs/reference/literature_audit.md · docs/reference/literature_analysis.md
  record/                 record: journey/ (9 parts + index) · results.md (every number) · bugs.md · packs.md (what each results pack held)
```

## Reading order — for a report or a presentation

1. [`docs/FINDINGS.md`](FINDINGS.md) — the claims. **Authoritative for what the thesis claims.**
3. [`docs/predictions.md`](predictions.md) — every prediction, made before its result, with its outcome.
4. [`reference/related_work.md`](reference/related_work.md) — every finding from other work the project uses, with its reference and how it was checked; [`reference/literature_audit.md`](reference/literature_audit.md) — the four fairness conditions; no audited paper meets them all.
5. [`record/results.md`](record/results.md) §0 — every headline number with run ids. **Authoritative for numbers.**
6. [`record/journey/README.md`](record/journey/README.md) — the narrative, part by part, with an index of every section.
7. As needed: [`reference/apparatus.md`](reference/apparatus.md) (model, data, reproduction),
   [`reference/conditions.md`](reference/conditions.md) (new corpora, tokenizers, conditions),
   [`reference/ukrainian_pipeline.md`](reference/ukrainian_pipeline.md),
   [`docs/GAPS.md`](GAPS.md), [`docs/DIRECTIONS.md`](DIRECTIONS.md), [`record/bugs.md`](record/bugs.md).

## The project in one paragraph

A master's thesis asking whether injecting linguistic structure into a
language model's **embedding layer** helps. A fixed small decoder-only
transformer (≈5M parameters: 4 layers, 4 heads, width 128, FFN 512; batch 32
× 256 tokens; learning rate 3e-4, cosine schedule; 50k steps, about an hour on
one laptop GPU) is trained on **WikiText-103** (English, ~120M BPE tokens) and
on **Ukrainian Wikipedia** (20% of the 2023 dump, ~151M tokens; validation 598
articles). Only the tokenizer (Q1) or the embedding (Q2) varies. Linguistic
features come from spaCy taggers (`en_core_web_sm`, `uk_core_news_sm`) and
enter as an 8-dim **slot** concatenated to a 120-dim token embedding (or
**added** to a full 128-dim one). The metric is **bits per byte (BPB)** of
held-out text — comparable across tokenizers, *not* across languages.

## The method, in five rules

1. **Same-shape null.** Every feature is compared with a null cell of
   identical architecture carrying no information (`mconst`) — not with the
   plain baseline, which differs in architecture.
2. **Causal shift.** Word-level features describe the *previous* word, with
   word boundaries from the tokenizer, so a feature never reveals the word
   being predicted (the within-word lookahead leak, journey §30).
3. **Matched initialization.** Cells of identical parameter shapes share
   their initial weights at the same seed; comparisons are paired by seed.
4. **Seeds.** 2–4 per comparison; reported as mean ± standard error over
   seeds, with how many seeds share the sign.
5. **Predictions first.** Every experiment's expected outcome was written down
   before it ran, with the result that would make it wrong.

## The claims (full statements: `docs/FINDINGS.md`)

| claim | key numbers | firmness |
|---|---|---|
| **Q1** BPE beats morpheme segmentation | 13.5% BPB (12.9% held out); morphemes win only on the rarest words (+9%) | firm |
| **Q1 in Ukrainian**: BPE that keeps inflectional endings apart beats plain BPE of the same size — and, unlike a feature's gain, the lead lasts | +0.0174 ± 0.0014 BPB (1.6%), 4/4 seeds, though it reads 11% less text in the same steps; +0.0169 ± 0.0007 at twice the budget; +0.0254 ± 0.0012 (2.6%) in a 4× wider model; Unigram's early lead fades to nothing; case adds +0.0048 on top | firm at ~38M tokens (two budgets, two widths); every tokenizer's stream the same text (G-75) |
| **Q2** at the end of training, no feature beats its same-shape null | every feature within 0.002 BPB: English structure +0.0004 ± 0.0005, part of speech +0.0012 ± 0.0007, Ukrainian case −0.0013 ± 0.0008; inflection below (−0.0046 ± 0.0013) | firm, both languages |
| **Early help**: informative features lead their null early in training | step 2,000: English part of speech +0.0140 ± 0.0011, inflection +0.0077 ± 0.0011, Ukrainian case +0.0056 ± 0.0005 (against a null constant on every word) — every seed; gone by the end; under five tokenizers too (seed 0) | firm, both languages |
| **Before the model has learned it, features help** | Ukrainian case at a 10k-step budget, against a null constant on every word: +0.0076 ± 0.0016 (1.5M tokens) … +0.0026 ± 0.0004 (151M), every seed; on 151M: +0.0055 ± 0.0006 at 5k steps, −0.0003 ± 0.0004 at 20k; word form alone keeps 43–66%. English part of speech at 10k: +0.0130 ± 0.0004 | firm, both languages; Ukrainian restated (G-72, G-73) |
| **The law**: a feature's gain depends on how much its null has learned | one curve per feature across 5k–50k schedules (part of speech: worst deviation 0.0010; four more features within 0.0004 at three schedules); within one model size — across sizes, how early the model reads the previous word's tag from context explains about half | firm in English; approximate in Ukrainian |
| **Removing a feature leaves a lasting deficit** | switched off at 5,000 of a 20k budget: −0.0039 ± 0.0009 (5/5); not lost knowledge of part of speech, nor of the previous word's tag the feature carried (probes); a feature added late is integrated, given time | firm |
| **As a target, morphology teaches, then stops paying** | +0.0094 at step 1k, +0.0009 ± 0.0008 at the end (5 seeds); the model knows part of speech better (+4.6 probe points) without predicting better | firm |
| **Why the null must have the same shape**: the slot helps by itself — one learned vector shared by every input beats the plain baseline | rotary positions: +0.0064 ± 0.0004 in 8 dims of its own, +0.0072 ± 0.0008 added to all; empty dims +0.0013 ± 0.0015. Learned positions (dims filled by position): English +0.0111 ± 0.0025, Ukrainian +0.0028 ± 0.0010 | firm; not the window-start marker (82–89% of the gain without it, seed 0; G-71 closed); how a shared vector helps is parked — outside the thesis |
| **Measurement**: most large numbers in this area are artifacts | a 5.1% "gain" was 83% within-word lookahead; nine pitfalls, each found in this project's own numbers or tools | firm |

## Figures

Drawn by `python scripts/analysis/demo_figures.py --out exports/figures` from every
`result.json` (PNGs in `exports/figures/`; regenerate after new runs). Positive
= the feature (or slot) is better; error bars = standard error over seeds.
`bash scripts/ops/refresh_exports.sh` redraws them with every table in `exports/`.
fig3 and fig4 are not drawn until the Ukrainian nulls quarantined under G-73 are re-run
([`DIRECTIONS.md`](DIRECTIONS.md), the queue); their copies of 1 October showed numbers
since restated.

- **fig1_q2_end** — each feature against its same-shape null at the end of
  training. The grey bar is a *control* (structure shuffled across words),
  not a linguistic feature.
- **fig2_early_help** — feature minus null over training steps (log axis):
  the early lead and its fade. Compare shapes across languages, not heights.
- **fig3_budget** — Ukrainian case's gain against train tokens (10k steps),
  with longer training on the same 151M tokens (20k, 50k steps) at 151M.
- **fig4_architecture** — the slot over the plain baseline (left); the
  position test against the zero slot, English (right).

## Glossary

**Terms.** *BPB* — bits per byte of UTF-8 text; lower is better. *Null* /
*same-shape null* — a cell with the feature cell's architecture and no
information. *Slot* — the 8 input dims a feature occupies. *Zero slot* — a
slot held at exactly zero. *Additive* (`-ADD`) — the feature added to a
full-width embedding instead of a slot. *Causal* — the feature describes the
previous word. *Matched / paired* — same seed, same initial weights where the
shapes allow. *Early help* — the feature's lead over its null early in
training. *Within-word lookahead* — a whole-word feature visible at the word's
later fragments, revealing them.

**Run labels**: `bpe_<feature>-LEARN-CAUSAL[-ADD]_<tokenizer>[_<corpus><fraction>][_st<steps>k]`.

| label part | meaning |
|---|---|
| `baseline` | plain model, full 128-dim token embedding |
| `mconst` | **the null**: the slot with one class for every word |
| `mzero` | zero slot; `mzeronopos` without position in it; `mzeroposonly` position only there |
| `mflat` | one learned vector at every position (additive null) |
| `mconst16/32/64` | the null with a 16/32/64-dim slot |
| `mstruct` | word structure (8 classes); `mstructshuf` shuffled across words of similar frequency |
| `mfreq` | frequency band of the previous word |
| `pos` / `poslex` | part of speech from the tagger / each word's majority tag (no context) |
| `mfeat` / `mfeatlex` | inflection bundle from the tagger / majority bundle |
| `ucase` / `ucaselex` | Ukrainian case from the tagger / each form's majority case |
| `mzero,ucase` | two slots: a zero slot beside case |
| `bpe_17k` / `bpe_uk_17k` | English / Ukrainian 17k BPE tokenizer |
| `bpe_uk_4k` … `bpe_uk_32k`, `unigram_uk_17k` | Ukrainian BPE at 4k, 8k and 32k; Unigram at 17k |
| `bpeend_uk_17k` | Ukrainian 17k BPE that keeps inflectional endings apart |
| `ukwiki0.2` | Ukrainian corpus fraction (0.002 ≈ 1.5M, 0.01 ≈ 7.6M, 0.05 ≈ 38M, 0.2 ≈ 151M train tokens) |
| `_st10k` / `_st20k` | 10k / 20k training steps (cosine over the same length); none = 50k |

## Withdrawn claims — do not present these as findings

Each was believed at some point and later refuted (details in `docs/FINDINGS.md`):
a 5.1% gain from leaky word-identity features; "word structure helps 0.85%"
(architecture); "the headline is a floor"; the anaphor BLiMP gain (a
preference); "inflection behaves like structure"; "position only in the slot
keeps half its gain"; "case gains beside a zero slot" (one seed); "little data,
as such, is what makes case useful" (it is training progress).

## Say out loud, and do not claim

- One small model; one corpus per language; 2–4 seeds per comparison.
- English and Ukrainian BPB are on different scales (Cyrillic letters take
  two bytes): compare directions and shapes across languages, never heights.
- Do **not** claim "morphology helps language models" without the condition
  (before the model has learned it).
- Do **not** claim the architecture effect is novel before checking TUPE and
  DeBERTa-style position/content separation.
- Do **not** compare a feature with the plain baseline as evidence of
  information — only with its null.

## The journey, section by section

Every section of the record, linked: [`record/journey/README.md`](record/journey/README.md).
