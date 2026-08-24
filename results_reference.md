# Results Reference — Complete Experimental Record

*Companion to `research_journey.md`. The journey records DECISIONS and their
reasoning in narrative form; this document is the DATA: every run, every number,
every configuration, every audit result, with run IDs for traceability.*

*Written so that a thesis chapter can be drafted from this file plus the journey
without needing to re-derive or re-run anything.*

---

## 1. Reproduction

### 1.1 Environment

- Python 3.10 (conda env `morph`)
- `torch` (CUDA), `datasets`, `tokenizers`, `spacy` + `en_core_web_sm`,
  `morfessor`, `pandas`, `numpy`, `openpyxl`
- Training on a single CUDA GPU; ~14–18.5 optimizer steps/second depending on
  vocabulary size and whether context codes are loaded

### 1.2 Data assets

| asset | provenance | size |
|---|---|---|
| `data/MorphoLEX_en.xlsx` | hand-curated, external, NOT regenerable | 68,624 rows → 68,608 parsed words |
| `data/entity_vocab.csv` | `cli build entities`, spaCy NER over 6,000 articles | 15,374 raw → 1,259 after split threshold |
| `data/our_final_vocab.csv` | `cli build vocab` | **17,151 entries** |
| `data/bpe_tokenizer.json` | legacy, Whitespace pre-tokenizer, vocab 10,000 | RETIRED — not lossless |
| `data/bpe_bytelevel.json` | `cli build bpe`, ByteLevel, vocab 10,000 | lossless |
| `data/bpe_17k.json` | `cli build bpe --bpe-vocab 17151`, ByteLevel | **parameter-matched baseline** |

### 1.3 Final vocabulary composition (17,151 entries, 12 categories)

| category | count | layer | source |
|---|---|---|---|
| root | 15,105 | 3 | MorphoLex |
| entity | 1,259 | 5 | spaCy NER, split threshold |
| byte | 256 | 12 | UTF-8 fallback |
| suffix | 211 | 2 | MorphoLex |
| prefix | 127 | 1 | MorphoLex |
| function | 112 | 4 | hardcoded closed class |
| punct | 32 | 7 | ASCII |
| character | 26 | 6 | a–z |
| digit | 10 | 8 | 0–9 |
| infl | 8 | 9 | `<plur> <3sg> <past> <pastpart> <prog> <cmp> <sup> <poss>` |
| space | 3 | 10 | `<sp> <nl> <tab>` |
| case | 2 | 11 | `<cap> <upper>` |

Root-identity inventory for the `mroot` feature = 15,105 roots + 1,259 entities
+ 1 `none` = **16,365 classes**.

### 1.4 Corpus

WikiText-103-raw-v1 via HuggingFace `datasets`.

| split | bytes | articles processed |
|---|---|---|
| train | 535,800,393 | ~1,164,000 non-blank non-header lines |
| validation | 1,136,862 | ~2,000 lines |

Token counts differ per pipeline (see §3).

---

## 2. The fixed apparatus

Identical across every cell. Only the embedding and the tokenizer vary.

### 2.1 Model (`ModelConfig`)

| field | value |
|---|---|
| `d_model` | 128 |
| `n_layers` | 4 |
| `n_heads` | 4 (head_dim 32) |
| `d_ff` | 512 (4 × d_model) |
| `dropout` | 0.1 |
| `tie_output` | `'none'` (untied output head, no bias) |
| positional | learned absolute, 256 positions |
| blocks | pre-norm, GELU, textbook GPT-2 |

Transformer core (identical in every cell): **826,112 parameters**.

### 2.2 Training (`TrainConfig`)

| field | value |
|---|---|
| `batch_size` | 32 |
| `seq_len` | 256 (8,192 tokens per step) |
| `lr` | 3e-4 peak |
| schedule | linear warmup 100 steps, then cosine to 10% of peak |
| `max_steps` | 50,000 |
| `lr_horizon` | 50,000 (**separate field** — cosine denominator) |
| optimizer | AdamW, `weight_decay` 0.01 |
| `grad_clip` | 1.0 |
| `eval_every` | 500 steps (101 evaluations per run) |
| `patience` | 6 evaluations (3,000 steps) |
| `seed` | 0 unless stated |
| reported | **best-val**, never last-val |

`lr_horizon` is separate from `max_steps` because a single field doing both jobs
made runs of different lengths incomparable — see journey §32.

### 2.3 Parameter counts

`core + vocab×(d_model − T) + d_model×vocab + factor_table`

| configuration | vocab | T | trainable |
|---|---|---|---|
| BPE 10k baseline | 10,004 | 0 | 3,387,136 |
| BPE 17k baseline | 17,155 | 0 | **5,217,792** |
| morph/det baseline | 17,155 | 0 | 5,217,792 |
| morph/det + dep (frozen) | 17,155 | 8 | 5,080,552 |
| morph/det + dep (learned) | 17,155 | 8 | 5,080,616 (+64) |
| BPE + mstruct (frozen) | 17,155 | 8 | 5,080,552 |
| BPE + mroot (frozen projection) | 17,155 | 16 | 4,943,312 |
| BPE + mroot (learned) | 17,155 | 16 | 5,205,152 |
| BPE + all three (learned) | 17,155 | 29 | 4,982,226 |

**Every feature-injection cell has FEWER trainable parameters than its baseline.**

---

## 3. Complete run inventory

23 training runs in 9 batches. Perplexity is per-token and comparable ONLY
within a pipeline; BPB is the cross-pipeline metric.

### 3.1 Calibration (pre-refactor, not saved as JSON)

| run | result |
|---|---|
| BPE baseline, 3,000 articles | val loss bottomed at step 750 then climbed — overfitting, a data-quantity signal |
| BPE baseline, full corpus, 40k ceiling | val loss 4.17, **ppl 64.9, BPB 1.438** at step 39,500, still improving |

### 3.2 Batch 1 — first matrix (LOSSY pipeline, retired)

Legacy Whitespace BPE; morph/det streams not information-preserving. **These
numbers are void** and retained only because they are what the audits refuted.

| run_id | label | ppl | BPB | val_loss | val_tokens | best_step | wall |
|---|---|---|---|---|---|---|---|
| `6cc3e3789d4d` | bpe_baseline | 60.95 | 1.4162 | 4.1100 | 271,360 | 49,999 | 44.2 min |
| `7b28f2074be6` | det_baseline | 25.75 | 1.3271 | 3.2485 | 321,792 | 49,000 | 54.6 min |
| `7b9dd0fe08ef` | morph_baseline | 25.96 | 1.3268 | 3.2565 | 320,768 | 49,999 | 54.4 min |

Apparent result: morph beat BPE by **6.31% BPB**. Refuted — see §4.

### 3.3 Batch 2 — information-fair rerun

ByteLevel BPE; morph/det with all five surface channels (`Spdiwc`).

| run_id | label | ppl | BPB | val_tokens | bytes/token |
|---|---|---|---|---|---|
| `03b9423bfe83` | bpe_baseline_bpe_bytelevel | 57.69 | **1.3946** | 270,848 | 4.195 |
| `0e7cd8838192` | det_baseline_Spdiwc | 6.25 | 1.5246 | 655,360 | 1.734 |
| `735559832ec1` | morph_baseline_Spdiwc | 6.30 | 1.5292 | 654,592 | 1.737 |

**The sign flipped: morph now 9.7% WORSE.** Implied validation bytes agree to
0.05% across all three (1,136,212 / 1,136,663 / 1,136,794), confirming the same
denominator.

### 3.4 Batch 3 — Q2, frozen dependency role

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `51d794b3f24e` | morph_dep_Spdiwc | 6.02 | 1.4913 | 49,500 |
| `11e9fa7d35b3` | det_dep_Spdiwc | 5.97 | 1.4856 | 49,000 |

vs baseline: morph −4.46%, det −4.58%. Ahead in **91/91** evaluations from step
5,000.

### 3.5 Batch 4 — shuffled-codes control

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `ccd320c0a0a0` | morph_dep-SHUF_Spdiwc | 6.51 | 1.5560 | 49,999 |
| `88b6afa1e676` | det_dep-SHUF_Spdiwc | 6.45 | 1.5507 | 49,000 |

**Shuffled codes are WORSE than baseline** (+3.28% / +3.19%), so the gain is
linguistic content, not the presence of structure.

### 3.6 Batch 5 — learned factors and matched vocabulary

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `96ebe413e6bf` | morph_dep-LEARN_Spdiwc | 5.88 | 1.4721 | 49,999 |
| `873a52464643` | det_dep-LEARN_Spdiwc | 5.87 | 1.4716 | 49,500 |
| `a7175857fdf6` | bpe_baseline_bpe_17k | 67.36 | **1.3477** | 49,999 |

Learned beats frozen. BPE **improved** with the larger vocabulary
(1.3946 → 1.3477), widening the Q1 gap to **13.5%**. `bpe_17k` val_tokens =
252,160 (985 windows × 256).

### 3.7 Batch 6 — factor scale control

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `55c394dba2e3` | morph_dep-S0.22_Spdiwc | 6.00 | 1.4883 | 49,999 |
| `c69345978d4b` | det_dep-S0.22_Spdiwc | 5.94 | 1.4824 | 49,000 |

Rescaling the frozen one-hot to match the trained slice's norm closed only
**16% (morph) / 23% (det)** of the gap to learned. Trainability genuinely wins.

### 3.8 Batches 7–8 — Level 2, morphology as features on BPE

All against `bpe_baseline_bpe_17k` = 1.3477 BPB.

| run_id | label | ppl | BPB | vs base | params vs base |
|---|---|---|---|---|---|
| `1e57b24ed4f9` | bpe_mroot-SHUF-LEARN_bpe_17k | 68.26 | 1.3519 | **+0.31%** | −0.24% |
| `eed5d2a11a8a` | bpe_mstruct-SHUF_bpe_17k | 66.60 | 1.3440 | −0.27% | −2.63% |
| `3860406fd2c1` | bpe_mroot_bpe_17k (frozen projection) | 63.66 | 1.3296 | −1.34% | −5.26% |
| `34a9e9b342da` | bpe_mstruct_bpe_17k (frozen one-hot) | 62.25 | 1.3225 | −1.87% | −2.63% |
| `56239ccfaf84` | bpe_mroot-LEARN_bpe_17k | 59.42 | 1.3076 | −2.98% | −0.24% |
| `1fb14ea6f45d` | bpe_mroot+mstruct+mrole-LEARN_bpe_17k | **56.72** | **1.2927** | **−4.08%** | −4.51% |

Wall clock: 55–59 minutes per run (14.0–15.1 steps/s).

### 3.9 Batch 9 — seed replication

| run_id | label | seed | ppl | BPB |
|---|---|---|---|---|
| `a12b91da4049` | bpe_baseline_bpe_17k | 1 | 67.47 | 1.3482 |
| `e6b206f9a34a` | bpe_mroot+mstruct+mrole-LEARN_bpe_17k | 1 | 55.15 | 1.2837 |

| cell | seed 0 | seed 1 | spread | mean |
|---|---|---|---|---|
| baseline | 1.3477 | 1.3482 | **0.0005** | 1.3480 |
| all three facts | 1.2927 | 1.2837 | 0.0090 | 1.2882 |

- seed noise (max spread): **0.0090 BPB**
- effect (mean difference): **0.0598 BPB**
- **effect / noise = 6.6×**
- mean improvement **4.44% BPB**
- all four cross-seed pairings hold

---

## 4. Audit results

### 4.1 Reconstruction audit (`reconstruction_audit.py`)

det pipeline, 500 validation lines, 29,333 words.

| stage | exact reconstruction | failures | byte recovery |
|---|---|---|---|
| before multi-root fix | 24,036 (81.94%) | 5,297 (18.06%) | 72.14% |
| after multi-root fix | 24,356 (83.03%) | 4,977 (16.97%) | 73.99% |

Failure classes at the second measurement: inflection dropped ~78%,
orthographic ~22%, vanished morphemes 0% (fixed).

**Superseded by the collision audit** — reconstruction failure overstates the
problem, because non-colliding spelling variation is recoverable in principle.

### 4.2 Collision audit (`collision_audit.py`)

det, 2,000 validation lines. Correction = Σ −log₂ P(surface | token sequence),
an upper bound (context-free).

| stage | occurrences | colliding groups | in collision | BPB correction |
|---|---|---|---|---|
| original (lossy) | 142,732 | 2,064 | 28.46% | **0.0440** |
| after surface channels | 157,935 | 370 | 14.50% | 0.1571 ← *audit artifact* |
| after audit fixes | 165,594 | 111 | 1.24% | 0.0028 |
| after byte fallback | 167,025 | 103 | **0.75%** | **0.0008** |

The 0.1571 reading was measurement error: entity `word`-field inconsistency plus
the audit skipping digit/punct tokens. ~70% of it was artifact.

Residual collision classes (all genuine limitations, not bugs):
dialectal normalisation `{honor, honour}` `{colour, color}` `{travelled,
traveled}`; homograph conflation `{found:57, founded:10}` `{surveillance,
survey}`; multi-word entity spans.

### 4.3 Coverage audit (`coverage_audit.py`)

1,000 validation lines, 422,168 bytes.

| class | in corpus | BPE covered | det BEFORE | det AFTER |
|---|---|---|---|---|
| alpha | 319,600 | 95.0%* | 94.5% | 97.4% |
| digits | 8,098 | 99.7% | 5.5% | **99.9%** |
| punctuation | 15,207 | 99.5% | 1.9% | **100.0%** |
| whitespace | 79,263 | 0.0% | 0.0% | 0.0% |

\* BPE's alpha figure is an audit artifact (the script lowercases the original
while BPE preserves case); the digit and punctuation rows are case-insensitive
and reliable.

Asymmetry: **22,480 bytes (5.32% of corpus)** before → **−89 bytes (−0.02%)**
after. Whitespace is dropped by both, so it never biased Q1 — but it is 18.8% of
corpus bytes, so neither BPB was a true compression rate until ByteLevel.

### 4.4 Multi-root audit (`multiroot_audit.py`)

| metric | value |
|---|---|
| MorphoLex entries parsed | 68,623 |
| entries with >1 root | **8,453 (12.32%)** |
| roots per entry | 0: 38 · 1: 60,132 · 2: 8,375 · 3: 77 · 4: 1 |
| new roots added to vocabulary | 83 (15,022 → 15,105) |

Most frequently recovered morphemes: `log` 267, `man` 212, `men` 162,
`graph` 158, `house` 96, `head` 90, `board` 84, `way` 77, `work` 75, `line` 72.

For comparison, the format-B fix of journey §16 affected 4,941 entries.

### 4.5 OOV audit (`oov_audit.py`)

det, 1,000 validation lines, 43,050 alphabetic words.

| metric | value |
|---|---|
| fell to character fallback | 5,625 (**13.07%**) |
| bytes in those words | 55,481 (**21.58%** of word bytes) |

| reason | count | share |
|---|---|---|
| capitalised (likely entity) | 4,360 | **77.5%** |
| very short | 430 | 7.6% |
| alphanumeric | 416 | 7.4% |
| plain lowercase OOV | 315 | 5.6% |
| non-ascii | 68 | 1.2% |
| very long / compound | 36 | 0.6% |

Most frequent unsegmentable: `'s` 240, `Meridian` 61, `Tikal` 57,
`Mississippi` 28, `km` 27, `Mongol` 27, `Radcliffe` 26, `Richmond` 25, `US` 23.

### 4.6 Level-2 preflight (`preflight_level2.py`)

BPE 17k + morphological analyser, 300 validation lines, 26,315 tokens.

| check | result |
|---|---|
| alignment rate | **99.99%** (start-offset bug would show ~20%) |
| tokens carrying a root | **42.9%** |
| not a word (punct/digit) | 19.4% |
| word with no root | 37.6% |
| distinct roots observed | 2,020 of 16,365 |

Within-token-id variance — the property that makes a fact *contextual*:

| fact | ids receiving >1 code | share |
|---|---|---|
| dep | 1,913 | **38.4%** |
| ectx | 849 | 17.0% |
| pos | 739 | 14.8% |
| mroot | 553 | 11.1% |
| mstruct | 540 | 10.8% |
| mrole | 381 | 7.6% |

Full-corpus cache build: **4h37m**, 120,625,719 train tokens, alignment 99.98%.
Cached code inventories: dep 8, pos 18, ectx 12, mstruct 8, mrole 5,
mroot 10,873 distinct (max index 16,364).

---

## 5. Derived analyses

### 5.1 Category bit accounting (`category_bits.py`)

Where each pipeline spends its bits. Information-fair runs, sampled validation.

**BPE (ByteLevel 10k)** — 270,848 tokens, 1,584,556 bits, sampled BPB 1.3946

| category | tokens | % tok | bits | % bits | bits/token |
|---|---|---|---|---|---|
| word-initial (carries space) | 211,300 | 78.0% | 1,341,007 | 84.6% | 6.346 |
| word-internal | 59,548 | 22.0% | 243,549 | 15.4% | 4.090 |

**det (`Spdiwc`)** — 327,680 tokens, 855,208 bits, sampled BPB 1.5048

| category | tokens | % tok | bits | % bits | bits/token |
|---|---|---|---|---|---|
| root | 37,049 | 11.3% | 314,878 | 36.8% | 8.499 |
| character | 80,813 | 24.7% | 211,799 | **24.8%** | 2.621 |
| function | 34,561 | 10.5% | 137,795 | 16.1% | 3.987 |
| punct | 18,653 | 5.7% | 43,538 | 5.1% | 2.334 |
| digit | 10,221 | 3.1% | 34,505 | 4.0% | 3.376 |
| space | 100,844 | **30.8%** | 24,960 | 2.9% | **0.245** |
| case | 14,281 | 4.4% | 24,837 | 2.9% | 1.739 |
| prefix | 4,804 | 1.5% | 23,303 | 2.7% | 4.851 |
| infl | 17,281 | 5.3% | 13,904 | 1.6% | **0.807** |
| entity | 1,522 | 0.5% | 12,006 | 1.4% | 7.888 |
| suffix | 6,052 | 1.8% | 9,415 | 1.1% | **1.556** |
| byte | 1,599 | 0.5% | 4,269 | 0.5% | 2.670 |

BPB excluding whitespace tokens: det 1.4609, morph 1.4656.

Two findings: **inflection and suffix tokens are the cheapest in the stream**
(0.807 / 1.556 bits), so morphology is demonstrably being exploited; and
**character fallback consumes 24.8% of all bits**, which is where Q1 is lost.

A prior hypothesis that whitespace explained the Q1 gap was **refuted by this
measurement**: `<sp>` costs 0.245 bits, not the 0.721 the hypothesis required.

### 5.2 Coverage economics (`coverage_economics.py`)

det vs BPE-ByteLevel-10k, 400 validation lines.

**Part 1 — tokenization economics**

| word class | words | % | bytes | morph tok/word | BPE tok/word |
|---|---|---|---|---|---|
| analysed cleanly | 19,736 | 87.8% | 93,698 | 1.51 | **1.18** |
| character fallback | 2,744 | 12.2% | 26,117 | **9.54** | 3.20 |

Fallback words: 12.2% of words but **21.8% of word bytes**. morph spends
**2.98×** the tokens BPE does on them.

**Part 2 — restricted bits-per-byte**

| region | bytes | morph BPB | BPE BPB | delta | share of gap |
|---|---|---|---|---|---|
| words analysed cleanly | 113,240 | 1.2672 | 1.1937 | +0.0735 | 35.5% |
| words in char fallback | 28,774 | 2.4076 | 2.0172 | +0.3904 | **47.9%** |
| punct / digits | 13,809 | 1.8004 | 1.5189 | +0.2815 | 16.6% |
| **ALL** | 155,823 | **1.5250** | **1.3746** | **+0.1505** | 100% |

Extrapolations: optimistic (fallback words as cheap as clean ones) → morph
1.3145, **wins** by 0.0601; realistic (fallback words merely match BPE's rate)
→ morph 1.4529, **still loses** by 0.0783.

**Part 3 — clean words stratified by training frequency** *(the U-shape)*

| word frequency | bytes | % clean bytes | morph BPB | BPE BPB | delta | verdict |
|---|---|---|---|---|---|---|
| 1–10 | 5,475 | 5.8% | 1.8925 | 2.1707 | **−0.2782** | **morph wins 12.8%** |
| 10–100 | 16,552 | 17.7% | 1.6822 | 1.6731 | +0.0091 | tie |
| 100–1k | 33,732 | 36.0% | 1.4418 | 1.3049 | +0.1369 | morph loses 10.5% |
| 1k–10k | 18,697 | 20.0% | 1.4687 | 1.4038 | +0.0649 | morph loses |
| 10k–100k | 13,338 | 14.2% | 1.5668 | 1.5660 | +0.0008 | tie |
| 100k+ | 5,904 | 6.3% | 0.7280 | 0.7451 | −0.0171 | morph wins slightly |

Frequency bands from a 40,000-line training sample (~1/30 of corpus), so
full-corpus counts are ~30× higher.

### 5.3 Per-frequency-bucket Q2 effect (`compare_buckets.py`)

Frozen `dep` vs baseline, scored from checkpoints, 491,520 validation tokens.

| train frequency | val tokens | % tok | morph delta | det delta |
|---|---|---|---|---|
| 1–100 | 291 | 0.06% | −15.89% | −5.00%* |
| 100–1,000 | 2,212 | 0.45% | **−14.53%** | **−14.66%** |
| 1,000–10,000 | 14,317 | 2.91% | −14.80% | −15.19% |
| 10,000–100,000 | 47,472 | 9.66% | −13.14% | −14.27% |
| 100,000–1,000,000 | 53,809 | 10.95% | −10.13% | −11.07% |
| 1,000,000+ | 373,419 | **75.97%** | **−1.95%** | −1.87% |
| OVERALL | 491,520 | 100% | −4.47% | −4.64% |

\* 291 tokens — noise, disregard.

**Rare tokens benefit ~7.5× more than frequent ones.** Because 76% of tokens sit
in the highest-frequency bucket, the headline 4.5% substantially understates the
mechanism.

### 5.4 Convergence speed

Steps to first reach a given per-token perplexity.

| cell | ppl ≤ 6.5 | ppl ≤ 6.3 |
|---|---|---|
| det baseline | 29,000 | 41,500 |
| det + frozen dep | **19,500** | **25,500** |
| morph baseline | 30,500 | never reached |
| morph + frozen dep | **21,000** | **27,000** |

**33–39% fewer steps to the same perplexity.**

### 5.5 Level-2 composition and controls

Gains over `bpe_17k` baseline (1.3477 BPB):

| feature set | BPB gain |
|---|---|
| mroot alone | 0.0401 |
| mstruct alone | 0.0252 |
| sum if independent | 0.0653 |
| mroot+mstruct+mrole actual | **0.0550** (84% of additive) |

Control asymmetry:

| | shuffled vs baseline | real vs shuffled |
|---|---|---|
| mroot (learned) | **+0.31%** worse | −3.28% |
| mstruct (frozen) | −0.27% slightly better | −1.61% |

Explained by the scale finding: a frozen one-hot enters at L2 norm 1.0 against a
trained slice of 0.219, so even random one-hots inject a loud
constant-magnitude signal. A learned table starts at std 0.02 and is quiet.

### 5.6 Factor signal scale (journey §37.3)

| component | L2 norm at initialisation |
|---|---|
| trained embedding slice (120 dims, std 0.02) | 0.219 |
| positional embedding (128 dims, std 0.02) | 0.226 |
| frozen factor (one-hot, 8 dims) | **1.000** |
| learned factor (8 dims, std 0.02) | 0.057 |

### 5.7 Fixed random projection quality

16,365 root classes into 16 dimensions, unit-normalised rows:
max off-diagonal cosine among 2,000 sampled classes **0.878**, mean |cosine|
**0.203**. Near-orthogonal on average, but 16 dimensions cannot keep 16k classes
cleanly separated — a stated limitation of the frozen projection arm.

---

## 6. Script inventory

All under `scripts/`. Every one was written to answer a specific question that
arose during the project.

| script | question it answers | key output |
|---|---|---|
| `verify.py` | is the codebase sound? | 7 checks incl. the freeze guarantee and frozen-vs-learned control |
| `reconstruction_audit.py` | can the token stream rebuild the text? | word-level exact reconstruction rate |
| `collision_audit.py` | do two texts share a token sequence? | exact BPB correction (upper bound) |
| `coverage_audit.py` | does each pipeline encode the same bytes? | coverage by character class, asymmetry |
| `multiroot_audit.py` | how many MorphoLex entries had roots dropped? | 8,453 entries (12.32%) |
| `oov_audit.py` | what falls to character fallback, and why? | failure classes, 77.5% capitalised |
| `category_bits.py` | where does each pipeline spend its bits? | bits by token category |
| `coverage_economics.py` | is the Q1 deficit coverage or segmentation? | restricted BPB, frequency stratification |
| `compare_buckets.py` | does the Q2 gain concentrate in rare tokens? | per-frequency-bucket comparison |
| `preflight_level2.py` | would a Level-2 run be valid? | alignment, variance, parameter arithmetic |
| `probe_dep.py` | is the fact legible in the representation? | linear probe per layer (**not yet run**) |

---

## 7. Known limitations

Each is measured, not speculative.

| limitation | magnitude | status |
|---|---|---|
| `dep` is an ORACLE feature (full-sentence parse) | −3.75% loss with lookahead vs −2.98% for `mroot` without | ~80% of the effect survives without lookahead |
| single seed on most cells | baseline spread 0.0005, treatment 0.0090 BPB | replicated only for baseline and best Level-2 cell |
| no downstream task evaluation | — | every comparable paper has one |
| residual collisions | 0.0008 BPB | dialectal, homograph, entity-span whitespace |
| entity vocabulary coverage | 77.5% of OOV is capitalised | built on 6k articles, `freq≥10` single-word gate |
| frozen projection separability | max cosine 0.878 at 16 dims | inherent to 16k classes in 16 dims |
| no convergence at the ceiling | all runs still improving at 50k | fixed-compute comparison, not converged |
| English only | — | literature shows English is the weakest case |
| model scale | 3.4–5.2M params | MorphPiece uses 117M, Hou et al. 132–979M |

---

## 8. Literature reference numbers

Full analysis in `literature_analysis.md`. Numbers most likely to be cited:

**Hou et al. (2023), GPT perplexity — the key comparison**

| language | BPE | morphological | outcome |
|---|---|---|---|
| English, vocab 151k | 19.82 | Morfessor **20.05** | BPE wins |
| English, vocab 107k | 17.92 | StateMorph-pruned 17.55 | morph wins 2% |
| Finnish, vocab 78k | 23.69 | StateMorph-pruned **16.65** | morph wins **30%** |
| Turkish, vocab 60k | 12.93 | StateMorph-pruned 10.67 | morph wins 17% |

**MorphPiece (Jabbar, 2023)**: MorphyNet 346,340 words → 134,943 entries after
freq≥5 trim; 18,304 morpheme tokens + 32,000 BPE = 50,006 vocabulary; GPT-2 Base
117M on OpenWebText; Penn Treebank ppl 61.859 → 38.251; **17% more tokens than
BPE**, listed as a limitation.

**Calibration reference points (journey §29)**: word-level WikiText-103 SOTA
Transformer-XL 18.3 test ppl (Dai et al. 2019) — **not comparable to our
numbers**, different unit. GPT-2 Small (117M, zero-shot) ≈ 29 BPE ppl; GPT-2 XL
(1.5B) ≈ 16.

---

## 9. Bugs found and fixed — a chronological record

Useful for a methods or limitations section, since several would have silently
invalidated results.

| # | bug | impact if unfixed | journey |
|---|---|---|---|
| 1 | MorphoLex format-B prefixes lost | 4,941 entries missing prefixes | §16 |
| 2 | multi-root compounds: `re.search` not `findall` | 8,453 entries (12.32%) losing roots | §33 |
| 3 | dead frozen dimensions | 12 of 23 columns constant; 6 distinct vectors total | §30 |
| 4 | `max_steps` doubling as LR horizon | runs of different length incomparable | §32 |
| 5 | no seeding anywhere | results not reproducible, no noise floor | §32 |
| 6 | results keyed without seed | second seed silently overwrote the first | §32 |
| 7 | O(n) per-token category scan | 2.18 ms/token → ~93 h per pipeline | §33 |
| 8 | pandas destroying `na`/`nan`/`null` morphemes | three real morphemes → UNK | §33 |
| 9 | five information channels dropped | 5.32% of corpus unencoded; 6.3% "win" was an artifact | §33 |
| 10 | UNK as a collision sink | non-ASCII punctuation all collapsed together | §33 |
| 11 | collision audit skipping digits/punct | inflated the measured correction ~70% | §33 |
| 12 | region attribution by start offset | BPE reported at 0.1354 BPB on words | §35 |
| 13 | frozen one-hot 4.6× louder than trained slice | confounded trainability with scale | §37 |
| 14 | BPE context alignment by start offset | **78% of the stream would have received `none`** | §39 |
| 15 | `freq_bucket` sending unseen words to the last bucket | would have inverted the frequency analysis | §35 |

Bugs 12, 14 and 15 were caught by numbers that were *impossible* rather than
merely surprising. **Alignment logic is the recurring failure mode in this
codebase.**
