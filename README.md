# Context-Aware Embeddings for Tokenizer-Free Language Models

Master's thesis project. It asks whether injecting linguistic structure into a
language model's **embedding layer** helps, and answers it with a matrix of
controlled comparisons on WikiText-103 — a fixed transformer throughout, with
only the tokenizer and the embedding varying.

## Finding

> **Morphological knowledge improves English language modeling as embedding
> features, not as segmentation.**

Two results, in the order they were measured:

| question | result |
|---|---|
| **Q1 — is morpheme segmentation better than BPE?** | No. Under a parameter-matched comparison, morphological segmentation loses to BPE by **13.5% BPB** in English. |
| **Q2 — do injected linguistic facts help?** | Yes. The same knowledge, delivered as embedding factors on a BPE-tokenized model, improves it by **4.44% BPB** (mean of two seeds, 6.6× seed noise) using **4.5% fewer trainable parameters**. |

Headline cells, all at `d_model=128`, 4 layers, 50k steps, best-val:

| cell | BPB | ppl | run id |
|---|---|---|---|
| `bpe_baseline_bpe_17k` (parameter-matched baseline) | 1.3477 | 67.36 | `a7175857fdf6` |
| `det_baseline_Spdiwc` (morphological segmentation) | 1.5246 | 6.25 | `0e7cd8838192` |
| `det_dep_Spdiwc` (frozen dependency fact) | 1.4856 | 5.97 | `11e9fa7d35b3` |
| `bpe_mroot+mstruct+mrole-LEARN_bpe_17k` | **1.2927** | 56.72 | `1fb14ea6f45d` |

A shuffled-code control is *worse* than baseline in both directions, so the gain
is linguistic content rather than the presence of extra structure. Full record in
[`results_reference.md`](results_reference.md).

**Reading the metrics.** Per-token perplexity is comparable *only within* a
tokenizer — the pipelines expand text differently, so the numbers above sit on
different units. **Bits-per-byte (BPB) is the cross-pipeline metric.** Never
compare these to the word-level WikiText-103 leaderboard (~18.3); that is a
different unit again.

## Layout

```
morphlm/            the package
  pipelines/        bpe · morpheme (+Morfessor) · deterministic
  vocab/            MorphoLex parser, NER entity extraction, vocab builder
  features/         frozen/learned factor blocks, morphological analysis
  modeling/         hybrid embedding, fixed transformer
  data/             corpus streaming, caching, byte tracking for BPB
  training/         loop, metrics, best-val + early stop
  cli.py            entry point
scripts/            audits — each answers one question that arose in the project
tests/              includes a regression test for the dead-frozen-dims bug
data/               lexicon, vocabularies, tokenizers
results/<hash>/     config.json + result.json per run
```

## Setup

```bash
conda create -n morph python=3.10 && conda activate morph
pip install torch datasets tokenizers spacy morfessor pandas numpy openpyxl
python -m spacy download en_core_web_sm

python scripts/verify.py --spacy     # 7 checks, incl. the freeze guarantee
```

`data/MorphoLEX_en.xlsx` is hand-curated, external, and **not regenerable** —
everything else under `data/` is rebuilt from it plus the corpus:

```bash
python -m morphlm.cli build all      # entities → vocabulary (17,151 entries) → BPE
```

## Running

```bash
python -m morphlm.cli run --pipeline bpe                          # a baseline cell
python -m morphlm.cli run --pipeline bpe --facts mroot,mstruct,mrole
python -m morphlm.cli run --pipeline det --facts dep -o train.seed=1
python -m morphlm.cli report                                      # comparison table
```

Any config field is reachable with `-o section.field=value`. Other subcommands:
`cache` (build the token cache alone), `inspect` (show the frozen layout without
training), `build`.

Budget roughly **55–60 min per cell** on one CUDA GPU, plus a one-off ~4.5 h to
build the annotated corpus cache. `data/cache/` reaches ~12 GB and checkpoints
are ~14 MB each; neither is in the repository. Pass `--no-checkpoint` to skip
saving weights.

## Documents

| file | what it is |
|---|---|
| [`research_journey.md`](research_journey.md) | 39 sections: every decision, fork, dead end, and bug, in the order they happened. The primary record. |
| [`results_reference.md`](results_reference.md) | The data: 23 runs, every configuration and audit result, with run ids. |
| [`literature_analysis.md`](literature_analysis.md) | Related work — MorphPiece, Hou et al. 2023, Bostrom & Durrett, FLOTA, factored LMs. |
| [`MIGRATION.md`](MIGRATION.md) | Why the flat layout became a package, and the old→new command mapping. |

Known limitations are measured rather than speculative and are listed in
`results_reference.md` §7 — chiefly: `dep` is an oracle feature, most cells run
at a single seed, no downstream task evaluation, English only, and no run reached
convergence at the 50k-step ceiling.
