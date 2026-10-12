# Context-Aware Morphological Embeddings for Language Models

Master's thesis project: does injecting linguistic structure into a language
model's embedding layer help? A fixed small transformer is trained on English
(WikiText-103) and Ukrainian (Wikipedia); only the tokenizer or the embedding
varies, and every feature is compared with a **null of identical shape** that
carries no information.

> **The finding.** Linguistic features help a small language model exactly
> while it has not yet learned what they encode — early in training, on a
> short budget, or with too little data to learn it — and add nothing once it
> has. At full scale, the gains that look like feature gains are architecture.
> In Ukrainian, morphology placed in the tokenizer — BPE that keeps inflectional
> endings apart — helps by itself (1.6% in bits per byte at the size tested), and
> unlike a feature's gain it lasts: it holds at twice the training budget and at
> full training, and is larger in a model twice as wide (2.4%). Case in the embedding adds on top of it.

**Start here: [`docs/README.md`](docs/README.md)** — what to read and in what
order. The claims, each with its evidence: [`docs/FINDINGS.md`](docs/FINDINGS.md).

## Repository layout

```
morphlm/            the package: pipelines, features, model, data, training, CLI
scripts/            data/ · eval/ · analysis/ · audits/ · ops/ · chains/ · rehearsal/
                    — every script, one line each: scripts/README.md
tests/              unit and regression tests: python tests/<file>.py — each file runs every test_ function it defines
docs/               the documentation — start at docs/README.md
data/               lexicon, vocabularies, tokenizers; caches in data/cache/ (not in the repository)
results/<run id>/   config.json + result.json per run (+ model.pt, never packed)
results_invalid/    runs quarantined as invalid, with the reason (scripts/ops/quarantine.py)
exports/            generated from results/: every table and figure — scripts/ops/refresh_exports.sh
manuscript/ · logs/ the thesis manuscript; block and chain logs (neither in the repository)
```

## Setup

```bash
conda create -n morph python=3.10 && conda activate morph
pip install torch datasets tokenizers spacy morfessor pandas numpy openpyxl matplotlib
python -m spacy download en_core_web_sm
pip install pymorphy3 pymorphy3-dicts-uk && python -m spacy download uk_core_news_sm   # Ukrainian

python scripts/ops/verify.py --spacy     # 7 checks, incl. the freeze guarantee
```

`data/MorphoLEX_en.xlsx` is hand-curated, external, and **not regenerable** —
everything else under `data/` is rebuilt from it plus the corpus:

```bash
python -m morphlm.cli build all      # entities → vocabulary (17,151 entries) → BPE
```

## Running

```bash
BPE='-o data.bpe_path=data/bpe_17k.json'
C='-o frozen.trainable=true -o frozen.causal_identity=true'

python -m morphlm.cli run --pipeline bpe $BPE                       # the plain baseline
python -m morphlm.cli run --pipeline bpe --facts mconst $BPE $C     # the null: the same slot, no information
python -m morphlm.cli run --pipeline bpe --facts pos $BPE $C        # a feature — compare it with the null
python -m morphlm.cli run --pipeline det -o train.seed=1            # morpheme segmentation (Q1)
python scripts/analysis/export_results.py --out exports/              # tables
python scripts/analysis/demo_figures.py --out exports/figures         # figures
```

The Ukrainian data and its features are built as described in
[`docs/reference/ukrainian_pipeline.md`](docs/reference/ukrainian_pipeline.md).

**Always pass `frozen.causal_identity=true` when injecting word-level features.**
Without it every BPE fragment receives the code of its *whole* word, naming
fragments not yet written; that once inflated an apparent gain
nearly sixfold (journey §30). `scripts/eval/lookahead_check.py` measures how much of any feature's
gain sits on the positions such a leak could reach. Shifting does not make a
*tagger*-derived feature causal (`pos`, `mfeat`, `magr`): the tagger reads a few
words to the right, which `scripts/audits/morph_causality_check.py` measures; lexical
features (`mstruct`, and `mfeatlex` from `scripts/data/derive_lexical_codes.py`) are
looked up from the word alone and are strictly causal (journey §33.3).

Before a chain that introduces new code, it is rehearsed on toy data from a
copy of the code (`scripts/rehearsal/`).

BLiMP minimal pairs, paired by seed against the baseline, on any checkpoints:
`python scripts/eval/blimp_eval.py --all --labels bpe_baseline_bpe_17k 'bpe_mstruct-LEARN-CAUSAL_bpe_17k'`.

Any config field is reachable with `-o section.field=value`. Other subcommands:
`cache` (build the token cache alone), `inspect` (show the frozen layout without
training), `build`.

Budget roughly **55–60 min per cell** on one CUDA GPU, plus a one-off ~4.5 h to
build the annotated corpus cache (gate a rebuild with `scripts/data/preflight_rebuild.py`). `data/cache/` reaches ~24 GB and checkpoints
are ~14 MB each; neither is in the repository. Pass `--no-checkpoint` to skip
saving weights.
