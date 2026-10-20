# Conditions — corpora, tokenizers and the data a run needs

> **Kind:** reference — how a new condition is added and prepared; rewritten when
> the tools change. **Authoritative for:** corpus specs, tokenizer names and
> statistics, and the preparation recipe for any corpus × size × tokenizer. The
> Ukrainian specifics are in [`ukrainian_pipeline.md`](ukrainian_pipeline.md); how
> the tools came about is in the journey (§33.53).

A **condition** is the data a run trains on: a corpus, the share of its training
documents used (`data.corpus_fraction`), and a tokenizer (`data.bpe_path`). The tools
below make any condition a few commands away. The two the work began with — WikiText-103
with `bpe_17k`, Ukrainian Wikipedia with `bpe_uk_17k` — are unchanged by them: their
streams, caches and labels are byte-identical. The conditions runs use:

- **WikiText-103** under BPE 17k, and under BPE 17k with endings apart (`bpeend_17k`;
  journey §33.85); `unigram_17k` is Q1's second tokenizer family (results §5.20).
- **Ukrainian Wikipedia** under BPE 17k at fractions 0.002, 0.01, 0.05 and 0.2; at 0.05
  also BPE 4k, 8k and 32k, Unigram 17k and BPE 17k with endings apart (D7; journey
  §33.58–§33.65), the last also at 0.2 (journey §33.68).
- **UberText fiction** under `bpe_ukf_17k` and `bpeend_ukf_17k`, at fractions 0.1 and 0.4
  (journey §33.85, §33.102).

**A stream can be kept packed.** A machine that no longer trains from a stream may keep its
arrays compressed (`scripts/data/cache_pack.py`; each checked by SHA-256 before its original
goes to the trash). Nothing reads or rebuilds a packed stream: `morphlm/data/stream.py` stops
any run or script that needs one and names the command that unpacks it.

## Corpora

A corpus is a spec in `morphlm/data/corpora.py` — the dataset and its
configuration, the text field, language, tagger, word rule, a short code for
tokenizer names, and two rules:

- **Splitting.** `native` reads the dataset's own splits; `hash` assigns each
  document by a hash of its key (the title, or the text): 1 in
  `data.corpus_holdout` (default 2000) to validation, 1 to test, the rest to
  train. Either way `data.corpus_fraction` keeps a share of the train documents
  by a second hash, so sizes are nested — a smaller share is a subset of a larger
  one — and validation never changes with the size.
- **Serving.** `rows` gives each dataset row as it is (WikiText's lines);
  `paragraphs` gives each document's paragraphs, each with the newline that split
  it off (G-64).

WikiText-103 keeps its historical path and refuses a fraction: its cache key
carries none, so a sub-sample would share the full stream's files. Its headings
stay in the stream (G-37). A run naming an unknown corpus fails when its
configuration is checked; tagging a corpus with another language's model fails
before the first document.

**The corpora.**
- `wikitext103`: English Wikipedia articles, the historical corpus.
- `ukwiki`: Ukrainian Wikipedia.
- `ukfiction`: UberText 2.0's fiction subcorpus (novels, prose, some poetry; 25,565 texts,
  1.25 billion characters). It is read from a local file, `external/ubertext/fiction.jsonl`
  (`data_files`), built by `scripts/data/prepare_ubertext_fiction.py` from the downloaded
  export. Its runs use `data.corpus_holdout=500`, because the texts are long. Its tokenizers
  are trained at fraction 0.4 (498M characters, like ukwiki's 0.2), and its runs read 0.1,
  ~120M characters, like ukwiki's 0.05 (journey §33.85).

To add a corpus: look at the dataset on a machine that can reach Hugging Face —

    python scripts/data/corpus_info.py --dataset NAME --config CONFIG

— which prints its splits, sizes and fields, streams a few documents, and gives a
spec skeleton; add the spec, then `--corpus NAME` prints size and tagging-time
estimates per fraction. The rehearsal stand-in for `datasets` serves any corpus
from `$REHEARSAL_DATA/<dataset>[__<config>]/<split>.jsonl`.

## Tokenizers

    python scripts/data/train_tokenizer.py --corpus ukwiki --corpus-fraction 0.2 --vocab 8000
    python scripts/data/train_tokenizer.py --corpus ukwiki --corpus-fraction 0.2 --vocab 17000 --algorithm unigram
    python scripts/data/train_tokenizer.py --corpus ukwiki --corpus-fraction 0.2 --vocab 17000 --algorithm bpe-endings

- **Trained** on the corpus's train split at the given fraction, by the same code
  as `morphlm.cli build bpe` (`morphlm/data/tokenization.py`): ByteLevel, lossless,
  checked by round trip.
- **Named** `<algorithm>_<code>_<k>k` — `bpe_17k` and `unigram_17k` for WikiText,
  `bpe_uk_8k` for Ukrainian Wikipedia, `bpe_ukf_17k` for UberText fiction. The name is
  in every cache key and run label built from the tokenizer, so **a tokenizer is never
  overwritten**; the tool refuses.
  This matters most for Unigram, whose training is not deterministic: two
  trainings give the same pieces with scores differing by up to 0.02, so the
  saved file is the artifact.
- **The sidecar** `<name>.meta.json` holds the word rule (read by
  `lexical.word_rule_for`: `decoded` for Ukrainian, `symbol` for English) and the
  recipe: corpus, fraction, algorithm, size, training rows, the file's sha256.
- **Opened** only through `tokenization.load_tokenizer` (a test finds no other
  door).
- **`bpe-endings`** (`bpeend_uk_<k>k`, `bpeend_ukf_<k>k`, and `bpeend_<k>k` on
  WikiText-103): BPE that never joins a stem to its ending, by the rule of the corpus's
  language (`morphlm/data/endings.py`). Each rule reads the word alone, and cuts only where
  every reading of the form gives the same cut. The Ukrainian rule (`uk_endings_v1`) takes
  pymorphy3's readings of the form, the lemma minus its citation ending as the stem; the
  English one (`en_endings_v1`) the readings spaCy's English rule lemmatizer gives a noun,
  verb or adjective, the stem being what the form shares with the lemma (journey §33.85).
  Training text and encoded text are both marked, so every ending is its own token or
  tokens. The sidecar carries the rule and its fingerprint (the code's constants, the
  dictionary or the lemmatizer's tables, probe cuts); `load_tokenizer` refuses the
  tokenizer on a machine that would cut differently. `scripts/audits/ending_gold_check.py`
  checks the Ukrainian rule's cuts against two morpheme dictionaries (journey §33.61).

**What a tokenizer does to words** is measured, on held-out text with spaCy's
words —

    python scripts/data/train_tokenizer.py --stats data/bpe_uk_8k.json data/bpe_uk_17k.json --corpus ukwiki

— characters per token, tokens per word, the share of words split, and **ending
exposure**: among inflected words (lemma differs from form and shares a prefix
with it), how often a token boundary falls exactly between stem and ending
(школ|и, студент|ам, teacher|s), and how often the ending is a token by itself;
for all inflected words and for the case-bearing ones (nouns, proper nouns,
adjectives). For Ukrainian and English, also at the cut of the language's ending
rule (`cut|`), which in Ukrainian sees citation forms and places `книгами` at
`книг|ами` rather than the common prefix's `книга|ми`. One record per call in
`results/audits/`.

## Preparing a condition

    python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_8k.json --dry-run
    python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_8k.json --pilot
    python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_8k.json

The Ukrainian recipe as one command: the validation cache (a timed pilot, which
estimates the train build), its check, the train cache (the tagging pass), the
derived facts — the null `mconst`, the zero slot `mzero`, and each word-form
variant (`ucaselex` for Ukrainian, `poslex` for English, checked against known
words) — and a final check that every fact a run asks for is present in both
splits. It calls the commands the project already uses, skips every finished
step, and stops at the first failure; a second run of a ready condition takes
seconds. It ends by printing the overrides a run needs and a matched family.

**The same text under every tokenizer.** Bits per byte compare tokenizers only
if each stream is the same text; `scripts/audits/stream_roundtrip.py` decodes
each tokenizer's cached stream of a split and checks its length against the
bytes recorded at build time and its hash against the others' (seconds for
validation). Run it once a condition under a new tokenizer is ready (G-75); it
passed for the six Ukrainian tokenizers at 0.05 (journey §33.66).

Tagging follows the text's bytes, whatever the tokenizer: Ukrainian Wikipedia
at 0.05 (235.7 MB) took 58 min on the morning of 7 October and 1 h 14 min –
1 h 31 min during the night chain of 6 October, beside its GPU runs (five
tokenizers); at 0.2, 5 h 22 min. It runs on the CPU,
beside training on the GPU. The pilot's estimate takes the slice's bytes from
another tokenizer's cache of the same slice where one exists, and the time from
the latest build of the same text (caches record it from 7 October); otherwise it
extrapolates the pilot, which bounds the time from above — the pilot includes
reading through the corpus — and on ukwiki 0.05 put the tokens 15% low (journey
§33.63).

**Derived codes belong to their stream** (G-72): a corpus's validation split is
the same tokens at every fraction, so the validation streams of two fractions
have the same token count under two cache stems. Built codes (the tagger's) may
be shared between them; derived codes (the null, the zero slot, word-form
variants) are read only by the stream they were derived for, since each
fraction's lexicon marks different words unseen. `check_cache.py` names the
file each fact was read from; `scripts/audits/codes_audit.py` lists what every
recorded run read.

**The constant nulls are constant** (G-73): `mconst` and its widths give every
word one class at every split, seen in train or not; the codes file declares it
(`rules`), a run refuses codes that do not, and each run records the codes file
and rules it read. Constants derived before the fix are replayed by
`scripts/data/rederive_constants.py`.

**Lexicons** (G-66, closed): outside WikiText a word-form lexicon is named for
the corpus and size it was built from — `lexicon_ucase_bpe_uk_8k_ukwiki0.05.npz`
— so deriving at one size no longer overwrites another's; readers fall back to
the historical name.

## Runs on a condition

A run names its condition with overrides —
`-o data.corpus=ukwiki -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpe_uk_8k.json -o data.spacy_model=uk_core_news_sm`
— and its label carries it (`bpe_ucase-LEARN-CAUSAL_bpe_uk_8k_ukwiki0.05`), so
cells of different conditions never mix. Within a condition, a feature and its
null share their initial weights as everywhere else; across tokenizers the
vocabulary changes the embedding and output layers, so comparisons are made
within each condition, then between conditions' gains.
