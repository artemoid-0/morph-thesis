# The Ukrainian pipeline

> **Kind:** reference — how the Ukrainian data and features are built now;
> rewritten when that changes. **Authoritative for:** the corpus, tokenizer,
> word rule, features and checks behind every Ukrainian run. How each piece was
> found and fixed is in the journey (Part 9).

## Corpus

- **Source:** `wikimedia/wikipedia`, configuration `20231101.uk` — 1,294,720
  articles, 2,689M characters. Selected with `data.corpus=ukwiki`.
- **Split by a hash of the title:** one article in 2,000 to validation (598
  articles, 1.1M characters — WikiText-103's validation size), one in 2,000 to
  test, the rest to train. `data.corpus_holdout` sets the 2,000.
- **Train size:** a second hash keeps a fraction of train articles,
  `data.corpus_fraction`. The fractions are **nested** — a smaller one's articles
  are a subset of a larger one's — and validation is the same at every size.

  | fraction | train tokens | use |
  |---|---|---|
  | 0.2 | 151.1M | the full Ukrainian runs (536M characters, sized to WikiText-103) |
  | 0.05 | ~38M | the training-budget curve |
  | 0.01 | 7.66M | little data |
  | 0.002 | ~1.5M | little data |

- **Paragraphs keep their newline** (G-64): the stream builder strips each row,
  so a Ukrainian row carries its separator and gets it back. Without it a
  heading ran into the next paragraph's first word. The cache key marks this
  with `:nl`.

## Tokenizer and word rule

- **ByteLevel BPE, 17,000 tokens,** trained on the 0.2 train split; about 3.6
  characters per token. `data/bpe_uk_17k.json`.
- **Word rule `decoded`,** recorded in the sidecar `data/bpe_uk_17k.meta.json`
  (English uses `symbol`, unchanged). A token is a letter token if its decoded
  text, less a leading space, is letters. A token holding only the continuation
  bytes of a split character takes the class of the token before it; one
  holding a split character's first byte is classed by its Unicode block
  (G-63). Tested: the words found are exactly the text's runs of letters
  (`tests/test_blocks.py`), and the model's word finder on the GPU agrees with
  numpy's at every position.

## Features

- **Annotation:** spaCy `uk_core_news_sm` — the morphologizer and parser; the
  NER and the lemmatizer are skipped for non-English models (identical codes,
  about 4× faster). 99.76% of train tokens aligned. Version 3.8.0 tagged every
  Ukrainian cache: installed on 28 September, before the first, and never
  reinstalled (`scripts/ops/tagger_record.py`, G-74).
- **Universal Dependencies blocks,** each padded to 8 classes so a feature cell
  and its null share their parameter shapes: `ucase`, `ugender`, `unumber`,
  `uperson`, `utense`, `uaspect`, `uanimacy`, plus `dep` and `pos`. Requesting
  `ucase` builds them all in one pass.
- **Derived facts** (`scripts/data/derive_lexical_codes.py`):
  - `mconst`, the null — `--fact ucase --transform const --keep ucase` (one class
    for every word, seen in train or not: G-73);
  - `mzero`, the zero slot — derived from the null's file, so one codes file
    holds `ucase`, `mconst` and `mzero`;
  - `ucaselex`, each word form's majority case — checked with `--verify`
    against nine words whose case is unambiguous. Majority and in-context case
    agree for 89.74% of occurrences. For a little-data stream,
    `--lexicon-fraction 0.2` takes the lexicon from the full corpus.

## Checks

- `scripts/data/check_cache.py` — per feature, how often it fires, whether it
  is constant within words, and how many tokens are fragments of a split
  character; `--require-facts` and `--require-constant` make it a gate.
- `scripts/audits/morph_causality_check.py --spacy-model uk_core_news_sm
  --corpus ukwiki` — how many feature values depend on the words to the
  right: 5.0% of case values, mostly prepositions.
- `scripts/ops/smoke_model.py` — each cell's real model, one batch, causality.

## Building it

One command now does all of this for any corpus, size and tokenizer —
`scripts/data/prepare_condition.py` ([`conditions.md`](conditions.md)). The steps it runs:

```bash
F=0.2
UK="-o data.corpus=ukwiki -o data.corpus_fraction=$F -o data.bpe_path=data/bpe_uk_17k.json -o data.spacy_model=uk_core_news_sm"
D="--corpus ukwiki --corpus-fraction $F --bpe-path data/bpe_uk_17k.json"
python -m spacy download uk_core_news_sm
python scripts/data/ukwiki_stats.py                                       # sizes; suggests the fraction
python -m morphlm.cli build bpe --bpe-vocab 17000 --bpe-out data/bpe_uk_17k.json --pre-tokenizer bytelevel -o data.corpus=ukwiki -o data.corpus_fraction=$F
python -m morphlm.cli cache --pipeline bpe --facts ucase --splits validation $UK   # a pilot, minutes
python scripts/data/check_cache.py --bpe-path data/bpe_uk_17k.json --corpus ukwiki --corpus-fraction $F
python -m morphlm.cli cache --pipeline bpe --facts ucase $UK                       # train: 5 h 22 min at 0.2
python scripts/data/derive_lexical_codes.py --fact ucase --out-fact mconst --transform const --keep ucase $D --verify
python scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaselex $D --verify
python scripts/data/derive_lexical_codes.py --fact mconst --out-fact mzero --transform const $D --verify
```

A matched family, one seed (`MORPHLM_NO_BUILD=1` makes a missing cache fail at once):

```bash
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"
for f in mconst ucase ucaselex; do $RUN --facts $f $UK $C -o train.seed=0; done
$RUN $UK -o train.seed=0                                                   # the plain baseline
```

## Known residuals

- After a double space, spaCy's whitespace token lends its labels to the next
  word's first token: part of speech `none`, dependency `function`. At 0.05 that
  is 0.24% of validation words and 0.31% of training words, the same words under
  every BPE; where such a word has several tokens, `dep` and `pos` disagree
  within it (99.84–99.93% constant, lowest under the tokenizer that splits most;
  journey §33.63). The Universal Dependencies features are unaffected. Left
  alone, because fixing it would change English alignment.
- A split `×` or `÷` can join a word: its first byte is shared with accented
  Latin letters. A 17k vocabulary keeps both whole.
- Lexicon files are named per corpus and size (G-66, closed 6 October); those
  derived before keep the historical name, which readers fall back to.
