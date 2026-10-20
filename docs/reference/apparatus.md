# Apparatus — model, data, training, and how to reproduce a result

> **Kind:** reference — how the fixed apparatus works now; rewritten when it changes.
> **Authoritative for:** the model, the data, the training setup, reproduction, and the
> apparatus's known limitations. The Ukrainian pipeline has its own reference:
> [`docs/reference/ukrainian_pipeline.md`](ukrainian_pipeline.md).

*Sections keep their numbers from `results_reference.md` (§1, §2, §7).*

## 1. Reproduction

### 1.1 Environment

- Python 3.10 (conda env `morph`)
- `torch` (CUDA), `datasets`, `tokenizers`, `spacy` + `en_core_web_sm` and
  `uk_core_news_sm`, `pymorphy3` with its Ukrainian dictionary (the Ukrainian ending rule),
  `morfessor`, `pandas`, `numpy`, `openpyxl`
- Training on a single CUDA GPU; on the RTX 4060 ~14–18.5 optimizer steps/second at width
  128, depending on vocabulary size and whether context codes are loaded
- Two machines, each training on one GPU: an RTX 4060 Laptop GPU (8 GB; every run until 15 October)
  and, from 16 October, an RTX 5090 Laptop GPU (24 GB), the same software (torch 2.11, CUDA 13.0,
  everything in fp32). Each record names its GPU (`environment.gpu`). The same weights evaluate alike
  on both; a run repeated on the other machine differs like a dropout draw, so every comparison is
  made within one machine (journey §33.101, §33.106). The RTX 5090 is about 2.8 times faster: a
  width-128 run of 10k steps takes 245 seconds alone, and three at once 787 seconds each, so runs
  go one at a time (journey §33.101)

### 1.2 Data assets

| asset | provenance | size |
|---|---|---|
| `data/MorphoLEX_en.xlsx` | hand-curated, external, NOT regenerable | 68,624 rows → 68,608 parsed words |
| `data/entity_vocab.csv` | `cli build entities`, spaCy NER over 6,000 articles | 15,374 raw → 1,259 after split threshold |
| `data/our_final_vocab.csv` | `cli build vocab` | **17,151 entries** |
| `data/bpe_tokenizer.json` | legacy, Whitespace pre-tokenizer, vocab 10,000 | RETIRED — not lossless |
| `data/bpe_bytelevel.json` | `cli build bpe`, ByteLevel, vocab 10,000 | lossless |
| `data/bpe_17k.json` | `cli build bpe --bpe-vocab 17151`, ByteLevel | **parameter-matched baseline** |
| `data/unigram_17k.json` | `cli build bpe --algorithm unigram --bpe-vocab 17151`, ByteLevel: Q1's second tokenizer family (results §5.20) | 17,151 |
| `data/bpeend_17k.json` | `scripts/data/train_tokenizer.py --algorithm bpe-endings`, WikiText-103's train split, rule `en_endings_v1` | 17,000 |
| `data/bpe_uk_17k.json` | `cli build bpe --bpe-vocab 17000`, Ukrainian Wikipedia at 0.2 | 17,000 |
| `data/bpe_uk_4k.json`, `data/bpe_uk_8k.json`, `data/bpe_uk_32k.json` | `scripts/data/train_tokenizer.py`, Ukrainian Wikipedia at 0.2 | 4,000; 8,000; 32,000 |
| `data/unigram_uk_17k.json` | `scripts/data/train_tokenizer.py --algorithm unigram`, Ukrainian Wikipedia at 0.2 | 17,000 |
| `data/bpeend_uk_17k.json` | `scripts/data/train_tokenizer.py --algorithm bpe-endings`, Ukrainian Wikipedia at 0.2, rule `uk_endings_v1` | 17,000 |
| `data/bpe_ukf_17k.json`, `data/bpeend_ukf_17k.json` | `scripts/data/train_tokenizer.py`, plain and `bpe-endings`, UberText fiction at 0.4 | 17,000 each |

Every tokenizer but the first four has a sidecar `<name>.meta.json` with its word rule and,
for those trained by `train_tokenizer.py`, its recipe ([`conditions.md`](conditions.md)).

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

Token counts differ per pipeline (see [results §3](../record/results.md#3-complete-run-inventory)).

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

Transformer core, stated unambiguously because two project documents once
disagreed by exactly the size of the position table:

| component | parameters |
|---|---|
| 4 transformer blocks | 793,088 |
| final LayerNorm | 256 |
| **blocks + final norm** (`transformer_core`) | **793,344** |
| learned positional embedding, 256 x 128 | 32,768 |
| **blocks + norm + positional** (`transformer_core_with_positional`) | **826,112** |

Identical in every cell either way. `provenance.py` now records both fields, so
any figure quoted downstream carries its own definition.

### 2.1a Model variants

Every departure from the table above is a setting, and marks the run label, so
variants never join the standard cells:

| setting | values | label | since |
|---|---|---|---|
| `model.d_model`, `model.n_layers`, `model.n_heads` | any | `_w256`, `_L8`, `_h8` | §33.39 |
| `model.position` | `learned`, `rope` — rotary, nothing added at the input | `_rope` | §33.47 |
| `model.norm` | `layernorm`, `rmsnorm` — a gain, no mean subtraction or shift | `_rmsnorm` | §33.54 |
| `model.mlp` | `gelu`, `swiglu` — gated, hidden 8d/3 (344 at d=128), the same weight count | `_swiglu` | §33.54 |
| `model.dropout` | any; 0 switches it off — its masks are drawn on the GPU, differently on each GPU model | `_do<p>`, e.g. `_do0` | §33.105 |

With all three of rope, rmsnorm and swiglu the block is LLaMA-style (biases kept).
RMSNorm draws no random number, so at the same seed it starts from exactly the
weights of its LayerNorm twin. Before a new size is chained, `scripts/ops/fit_check.py`
measures it — parameters by part, peak GPU memory, steps per second, the
projected run time — over a grid of widths and depths in one call.

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
made runs of different lengths incomparable — see journey §16.

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

## 7. Known limitations

Each is measured, not speculative.

| limitation | magnitude | status |
|---|---|---|
| `dep` is an ORACLE feature (full-sentence parse) | −3.75% loss with lookahead vs −2.98% for `mroot` without | ~80% of the effect survives without lookahead |
| few seeds per cell | four for almost every claim's cells (Q1's morpheme arms two); eight for English with endings apart at full training; five for the probes | effects reported as a paired mean ± standard error with the count of seeds of the same sign (results §0, "Noise"); some cells' later seeds ran on the second machine, each compared within its machine (§1.1) |
| no downstream task evaluation | — | every comparable paper has one |
| residual collisions | 0.0008 BPB | dialectal, homograph, entity-span whitespace |
| entity vocabulary coverage | 77.5% of OOV is capitalised | built on 6k articles, `freq≥10` single-word gate |
| frozen projection separability | max cosine 0.878 at 16 dims | inherent to 16k classes in 16 dims |
| no convergence at the ceiling | all runs still improving at 50k | fixed-compute comparison, not converged |
| two languages | English (WikiText-103); Ukrainian (Wikipedia, and UberText fiction) | the literature finds morphology's gains largest in morphologically rich languages; Ukrainian is the test of that |
| model scale | widths 64–512, 1,819,168 to 30,017,760 parameters; most cells width 128, about 5M | MorphPiece uses 117M, Hou et al. 132–979M |

---

