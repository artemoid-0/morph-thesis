"""
corpora.py — every corpus a run can name in `data.corpus` (journey §33.53).

A corpus is a SPEC, not code: where its text comes from, how documents are
assigned to splits, how rows are served to the stream builder, its language and
tagger, and the word rule a tokenizer trained on it needs. `stream.load_corpus`
reads only the spec, so adding a corpus means adding an entry to CORPORA —
after checking the dataset with `scripts/data/corpus_info.py` on a machine
that can reach it.

SERVING
  rows        each dataset row is one stream row, exactly as stored — WikiText,
              whose rows are already lines
  paragraphs  each document's paragraphs, each carrying the newline that split it
              off; without it a heading runs into the next paragraph's first word
              (G-64)

SPLITTING
  native      the dataset's own splits (`native_splits` maps ours to its names)
  hash        documents assigned by a hash of their key: 1 in `corpus_holdout`
              to validation, 1 to test, the rest to train
  Either way `corpus_fraction` sub-samples TRAIN documents by a second hash of the
  key, so the sizes are nested: a smaller fraction is a subset of a larger one.

WikiText-103 keeps its historical path exactly — the one `load_dataset` call,
no sub-sampling — because its cache key carries no fraction: a sub-sampled
WikiText stream would share the full stream's cache files. It refuses a
fraction; sub-sampled English needs a corpus name of its own.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass, field
from typing import Optional

#: the repository root, which a spec's `data_files` is relative to
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class CorpusSpec:
    name: str                       # the data.corpus value
    language: str                   # ISO 639-1; the tagger must match it
    dataset: str                    # Hugging Face dataset path
    config: Optional[str]           # its configuration name
    split_rule: str                 # native | hash
    serve: str                      # rows | paragraphs
    spacy_model: str                # the tagger for this language
    word_rule: str                  # symbol | decoded — for tokenizers trained on it
    code: str                       # short name in tokenizer files: bpe_<code>_17k
    text_field: str = 'text'
    key_field: Optional[str] = 'title'   # what the hashes read; None = the text itself
    native_splits: dict = field(default_factory=lambda: {
        'train': 'train', 'validation': 'validation', 'test': 'test'})
    data_files: Optional[str] = None     # a local file, project-relative, read with dataset='json'
    note: str = ''


CORPORA = {
    'wikitext103': CorpusSpec(
        name='wikitext103', language='en', dataset='wikitext',
        config='wikitext-103-raw-v1', split_rule='native', serve='rows',
        spacy_model='en_core_web_sm', word_rule='symbol', code='',
        note='English Wikipedia articles, 103M words; the historical corpus'),
    'ukwiki': CorpusSpec(
        name='ukwiki', language='uk', dataset='wikimedia/wikipedia',
        config='20231101.uk', split_rule='hash', serve='paragraphs',
        spacy_model='uk_core_news_sm', word_rule='decoded', code='uk',
        note='Ukrainian Wikipedia; fraction 0.2 is sized to WikiText-103 '
             '(docs/reference/ukrainian_pipeline.md)'),
    'ukfiction': CorpusSpec(
        name='ukfiction', language='uk', dataset='json', config=None,
        split_rule='hash', serve='paragraphs', spacy_model='uk_core_news_sm',
        word_rule='decoded', code='ukf', key_field='key',
        data_files='external/ubertext/fiction.jsonl',
        note='UberText 2.0 fiction (cleansed): novels, prose, some poetry — 25,565 texts, '
             '1.25 billion characters; built by scripts/data/prepare_ubertext_fiction.py '
             '(DIRECTIONS D9; journey §33.85). Runs use data.corpus_holdout=500'),
}


def get_corpus(name: str) -> CorpusSpec:
    if name not in CORPORA:
        raise ValueError(f'unknown corpus {name!r}; known: {sorted(CORPORA)} '
                         f'(morphlm/data/corpora.py)')
    return CORPORA[name]


def spacy_language(model: str) -> str:
    """'uk_core_news_sm' -> 'uk'; spaCy's package names start with the language."""
    return model.split('_', 1)[0]


def check_tagger(corpus: str, spacy_model: str) -> None:
    """Refuse to tag a corpus with another language's model (no silent garbage)."""
    spec = get_corpus(corpus)
    if spacy_language(spacy_model) != spec.language:
        raise ValueError(
            f'corpus {corpus!r} is {spec.language!r}, but data.spacy_model is '
            f'{spacy_model!r} — set -o data.spacy_model={spec.spacy_model}')


def _hash(key: str) -> str:
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def assigned_split(key: str, holdout: int) -> str:
    """The hash split: 1 in `holdout` to validation, 1 to test, the rest to train."""
    h = int(_hash(key)[:8], 16) % holdout
    return 'validation' if h == 0 else 'test' if h == 1 else 'train'


def kept_in_train(key: str, fraction: float) -> bool:
    """The nested train sub-sample: kept if the second hash falls under the fraction."""
    return fraction >= 1.0 or int(_hash(key)[8:16], 16) % 10000 < fraction * 10000


def rows(spec: CorpusSpec, split: str, fraction: float, holdout: int, load_dataset):
    """
    Rows with a 'text' field (and, when served as paragraphs, a 'sep') for one
    split of a non-WikiText corpus. `load_dataset` is passed in so the offline
    stand-in used in rehearsals can replace it.
    """
    kw = {'data_files': os.path.join(ROOT, spec.data_files)} if spec.data_files else {}
    if spec.split_rule == 'hash':
        ds = load_dataset(spec.dataset, spec.config, split='train', **kw)
    elif spec.split_rule == 'native':
        ds = load_dataset(spec.dataset, spec.config, split=spec.native_splits[split])
    else:
        raise ValueError(f'{spec.name}: unknown split rule {spec.split_rule!r}')

    def gen():
        for doc in ds:
            text = doc[spec.text_field]
            key = doc[spec.key_field] if spec.key_field else text
            if spec.split_rule == 'hash' and assigned_split(key, holdout) != split:
                continue
            if split == 'train' and not kept_in_train(key, fraction):
                continue
            if spec.serve == 'paragraphs':
                for para in text.split('\n'):
                    yield {'text': para, 'sep': '\n'}
            else:
                yield {'text': text}
    return gen()
