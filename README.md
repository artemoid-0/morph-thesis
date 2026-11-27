# Context-Aware Morphological Embeddings for Language Models

Master's thesis project: does injecting linguistic structure into a language
model's embedding layer help? A fixed small transformer is trained on English
(WikiText-103) and Ukrainian (Wikipedia); only the tokenizer or the embedding
varies, and every feature is compared with a **null of identical shape** that
carries no information.

> **The findings** ([`docs/FINDINGS.md`](docs/FINDINGS.md) states each, with its evidence and firmness).
> Linguistic features in the embedding help a small language model only while it has not yet learned what they encode — early in training, on a short budget, or with too little data to learn it.
> At the end of training none beats a null of the same shape, in English or Ukrainian (within 0.002 bits per byte on average, 0.003 at the one-sided 95% bound); what reading the sentence adds over the word form is mostly not context, and at full training it is the tagger's leak.
> Morphology placed in the tokenizer lasts.
> In Ukrainian, BPE that never joins a stem to its inflectional ending beats plain BPE of the same vocabulary by 1.6% in bits per byte at a 10k budget and by 1.75% at full training (1.3% on held-out test articles); by 0.97% at 30M parameters and 1.02% at 85M; at 30M, 1.44% at a peak learning rate of 1e-3, where plain BPE does not improve.
> Four fifths of the gain sits in the words the rule cuts, of middle and low frequency; words outside the morphological analyzer's dictionary gain nothing.
> It shows in grammar — +4.9 points on MultiBLiMP's subject–verb agreement at full training, and about half the errors on agreement inside the noun phrase and on government — and holds on fiction (1.45% at full training).
> It costs 4.5 CPU-minutes to mark the training text and about 12% more tokens for the same text; plain BPE needs about 15% more parameters or 17% more training to match it.
> In English the same kind of tokenizer gains a fifth as much, and that lasts too (0.39% at full training, on the median seed).
> Segmenting English into morphemes instead of BPE loses, by 13.5%.

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

python scripts/ops/verify.py --spacy     # 8 checks, incl. the freeze guarantee
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
bash scripts/ops/refresh_exports.sh                                  # every table and figure in exports/, from results/
for t in tests/test_*.py; do python "$t"; done                        # the tests
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

## Data and its sources

Tracked under `data/`: the tokenizers trained here (on WikiText-103, Ukrainian Wikipedia and UberText 2.0 fiction),
the English vocabulary and entity lists, `MorphoLEX_en.xlsx` (Sánchez-Gutiérrez et al., 2018, under its authors'
terms), and the Ukrainian minimal pairs of `scripts/data/make_minimal_pairs_uk.py` — sentences of Ukrainian Wikipedia
(CC BY-SA 4.0) with a native speaker's check of a random hundred. Not tracked, and fetched or rebuilt as the documents
describe: the corpora (WikiText-103, CC BY-SA 3.0; Ukrainian Wikipedia; UberText 2.0 under its own terms), the caches,
the checkpoints, and the evaluation data kept in `external/` with their checksums — MultiBLiMP (CC BY 4.0), UD
Ukrainian-IU (CC BY-NC-SA 4.0) and the tokenizer files of other models.

Budget roughly **55–60 min per cell** on one CUDA GPU, plus a one-off ~4.5 h to
build the annotated corpus cache (gate a rebuild with `scripts/data/preflight_rebuild.py`). `data/cache/` reaches ~24 GB and checkpoints
are ~14 MB each; neither is in the repository. Pass `--no-checkpoint` to skip
saving weights.
