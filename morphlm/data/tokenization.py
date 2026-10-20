"""
tokenization.py — train a tokenizer for any corpus, and measure what it does to words
(journey §33.53).

One training path: `cli build bpe` and `scripts/data/train_tokenizer.py` both call
`train_tokenizer`, so a tokenizer made either way from the same corpus, size and
algorithm is the same file.

NAMES. A tokenizer's file stem enters every cache key and every run label, so
names are standard and files are never overwritten: `<algorithm>_<code>_<k>k`,
where `code` is the corpus's (corpora.py) — empty for WikiText-103, which gives
the historical `bpe_17k` and `unigram_17k`; `uk` for Ukrainian Wikipedia, which
gives `bpe_uk_17k`. A sidecar `<stem>.meta.json` records the word rule (which
`lexical.word_rule_for` reads) and how the tokenizer was made.

ENDINGS. `bpe-endings` is BPE that never joins a stem to its ending, by the
rule of the corpus's language (endings.py: `uk_endings_v1`, `en_endings_v1`):
trained on text marked at every cut, saved with a pre-tokenizer
that splits on the mark, named `bpeend_<code>_<k>k`, its sidecar carrying the
rule and the rule's fingerprint. Every tokenizer is opened with `load_tokenizer`,
which wraps such a tokenizer so that what it encodes is marked the same way —
`Tokenizer.from_file` appears nowhere else (tests/test_endings.py).

STATISTICS. How a tokenizer cuts words is the variable a tokenization study
turns on, so it is measured, not guessed — on held-out text, with spaCy's words:
  tokens per word, and the share of words split into two or more tokens;
  ending exposure — among inflected words (lemma differs from form and shares a
    prefix with it), how often a token boundary falls exactly where the stem
    ends and the ending begins: школ|и, студент|ам, teacher|s; and how often the
    ending is a token of its own. Reported for all inflected words and for the
    case-bearing ones (nouns, proper nouns, adjectives).
  for Ukrainian and English, ending exposure by the rule of endings.py as well — the common
    prefix with the tagger's lemma puts `книгами` at `книга|ми` and does not see
    citation forms (journey §33.60); the rule reads the word alone, as the
    `bpe-endings` tokenizer does, so every tokenizer is measured by its cut too.
"""

from __future__ import annotations

import os
import json
import datetime


#: the name each algorithm gives its files; `bpe-endings` is BPE trained and
#: applied on text marked at stem|ending cuts (endings.py)
NAME_OF = {'bpe': 'bpe', 'unigram': 'unigram', 'bpe-endings': 'bpeend'}


def standard_name(corpus: str, algorithm: str, vocab: int) -> str:
    """bpe_17k, unigram_17k (WikiText-103); bpe_uk_8k, bpeend_uk_17k (ukwiki); <algo>_<code>_<k>k."""
    from .corpora import get_corpus
    code = get_corpus(corpus).code
    k = f'{vocab / 1000:g}k'
    return '_'.join(p for p in (NAME_OF.get(algorithm, algorithm), code, k) if p)


def sidecar_path(path: str) -> str:
    """Where a tokenizer's sidecar lives: `<stem>.meta.json` beside it."""
    return (path[:-5] if path.endswith('.json') else path) + '.meta.json'


def tokenizer_meta(path: str) -> dict:
    """The sidecar `<stem>.meta.json` beside a tokenizer, or {} if it has none."""
    meta = sidecar_path(path)
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)
    return {}


def load_tokenizer(path: str, vocab_only: bool = False):
    """
    Open a tokenizer file — THE way, everywhere. A plain one is returned as it
    is. One trained on ending-marked text (its sidecar names the rule) comes
    wrapped, so the text it encodes is marked exactly as its training text was;
    if this machine's rule — the code and pymorphy3's dictionary — has another
    fingerprint than the one the tokenizer was trained with, it refuses rather
    than cut words differently. `vocab_only=True` returns the bare tokenizer for
    code that reads only its vocabulary (ids to strings): no rule is needed.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    meta = tokenizer_meta(path)
    presplit = meta.get('presplit')
    if vocab_only or not presplit:
        return tok
    from .endings import RULES, EndingTokenizer
    if presplit not in RULES:
        raise RuntimeError(f'{path} was trained with the cut rule {presplit!r}; this code has {sorted(RULES)}. '
                           f'A tokenizer is never re-cut under another rule: train a new one.')
    rule = RULES[presplit]()
    want, have = meta.get('presplit_fingerprint'), rule.fingerprint()
    if want != have:
        raise RuntimeError(
            f'{path} was trained with cut-rule fingerprint {want}; this machine gives {have} '
            f'(endings.py constants, or the dictionary the rule reads — pymorphy3 and its Ukrainian '
            f'dictionary, spaCy and its English tables — differ). Its words would be '
            f'cut differently from its training text — install the versions it was trained with, '
            f'or train a new tokenizer.')
    return EndingTokenizer(tok, rule)


def training_texts(cfg, max_lines=0):
    """The corpus's TRAIN rows, stripped, empty ones dropped — exactly what `cli build` trains on."""
    from .stream import load_corpus
    ds = load_corpus(cfg, 'train')
    n = 0
    for row in ds:
        t = row['text'].strip()
        if not t or t.startswith(' ='):     # never fires after strip(): headings stay, as in the stream
            continue
        yield t
        n += 1
        if max_lines and n >= max_lines:
            break


def train_tokenizer(cfg, vocab, out, algorithm='bpe', pre_tokenizer='bytelevel',
                    max_lines=0, sidecar='legacy'):
    """
    Train on `cfg`'s corpus, save to `out`. sidecar='legacy' writes exactly what
    `cli build` always wrote (nothing for WikiText; word rule and corpus otherwise);
    'full' writes the word rule and the whole recipe, for every corpus.
    algorithm='bpe-endings' trains BPE on text marked at every stem|ending cut
    (endings.py, the rule of the corpus's language) and always writes the full sidecar, with the
    rule and its fingerprint — the tokenizer cannot be opened without them.
    """
    from .corpora import get_corpus
    from ..pipelines.bpe import BPEPipeline
    corpus = getattr(cfg.data, 'corpus', 'wikitext103')
    spec = get_corpus(corpus)
    counter = {'texts': 0, 'cuts': 0, 'chars': 0}
    rule = None
    if algorithm == 'bpe-endings':
        from .endings import RULES, RULE_OF_LANGUAGE, mark_text
        if spec.language not in RULE_OF_LANGUAGE:
            raise ValueError(f'no ending rule for {spec.language!r} (corpus {corpus!r}); '
                             f'rules exist for {sorted(RULE_OF_LANGUAGE)}')
        if pre_tokenizer != 'bytelevel':
            raise ValueError('bpe-endings is byte-level only')
        rule = RULES[RULE_OF_LANGUAGE[spec.language]]()
        sidecar = 'full'

    def counted():
        for t in training_texts(cfg, max_lines):
            counter['texts'] += 1
            counter['chars'] += len(t)
            if rule is not None:
                t, cuts = mark_text(t, rule)
                counter['cuts'] += len(cuts)
            yield t

    bpe = BPEPipeline(tokenizer_path=None, vocab_size=vocab, verbose=True)
    bpe.train(counted(), save_path=out, pre_tokenizer=pre_tokenizer,
              algorithm='bpe' if rule is not None else algorithm, presplit=rule)
    meta_path = sidecar_path(out)
    meta = None
    if sidecar == 'full':
        meta = {'word_rule': spec.word_rule, 'corpus': corpus,
                'corpus_fraction': float(getattr(cfg.data, 'corpus_fraction', 1.0)),
                'corpus_holdout': int(getattr(cfg.data, 'corpus_holdout', 2000)),
                'algorithm': algorithm, 'vocab_size': vocab,
                'actual_vocab_size': bpe.tokenizer.get_vocab_size(),
                'pre_tokenizer': pre_tokenizer, 'training_texts': counter['texts'],
                'max_lines': max_lines,
                'trained_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        if rule is not None:
            from .endings import MARK
            meta.update({'presplit': rule.name, 'presplit_fingerprint': rule.fingerprint(),
                         'presplit_mark': f'U+{ord(MARK):04X}', 'training_cuts': counter['cuts'],
                         'training_characters': counter['chars']})
    elif corpus != 'wikitext103':
        meta = {'word_rule': spec.word_rule, 'corpus': corpus}
    if meta is not None:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  word rule: {spec.word_rule}  ->  {meta_path}')
    return counter['texts']


CASE_BEARING = ('NOUN', 'PROPN', 'ADJ')


def tokenizer_stats(tok_path, texts, nlp):
    """
    What a tokenizer does to words, measured on `texts` with spaCy's words.
    Only alphabetic spaCy tokens count as words.
    """
    tok = load_tokenizer(tok_path)
    rule = None
    from .endings import RULES, RULE_OF_LANGUAGE, common_prefix
    if getattr(nlp, 'lang', '') in RULE_OF_LANGUAGE:
        rule = RULES[RULE_OF_LANGUAGE[nlp.lang]]()
    chars = tokens = words = split = toks_in_split = 0
    infl = {'all': [0, 0, 0], 'case': [0, 0, 0]}      # inflected, boundary at stem|ending, ending a token of its own
    cut = {'all': [0, 0, 0], 'case': [0, 0, 0]}       # the same, at the ending rule's cut (Ukrainian)
    for doc, text in zip(nlp.pipe(texts, batch_size=64), texts):
        enc = tok.encode(text)
        chars += len(text)
        tokens += len(enc.ids)
        spans = enc.offsets
        starts = {s for s, _ in spans}
        exact = set(spans)
        for w in doc:
            if not w.is_alpha:
                continue
            ws, we = w.idx, w.idx + len(w.text)
            n = sum(1 for s, e in spans if s < we and e > ws)
            words += 1
            if n >= 2:
                split += 1
                toks_in_split += n
            if rule is not None:
                k = rule.cut(w.text)
                if k:
                    for kk in ('all', 'case') if w.pos_ in CASE_BEARING else ('all',):
                        cut[kk][0] += 1
                        cut[kk][1] += (ws + k) in starts
                        cut[kk][2] += (ws + k, we) in exact
            form, lemma = w.text.lower(), w.lemma_.lower()
            p = common_prefix(form, lemma)
            if form == lemma or p == 0 or p >= len(form):
                continue
            hit = (ws + p) in starts
            own = (ws + p, we) in exact
            for k in ('all', 'case') if w.pos_ in CASE_BEARING else ('all',):
                infl[k][0] += 1
                infl[k][1] += hit
                infl[k][2] += own
    out = {'tokenizer': os.path.basename(tok_path), 'vocab_size': tok.get_vocab_size(),
           'characters': chars, 'tokens': tokens, 'words': words,
           'chars_per_token': chars / max(tokens, 1),
           'tokens_per_word': None, 'words_split': split / max(words, 1),
           'tokens_per_split_word': toks_in_split / max(split, 1)}
    # tokens per word over words only: unsplit words count 1, split ones their pieces
    out['tokens_per_word'] = ((words - split) + toks_in_split) / max(words, 1)
    for k, (n, hit, own) in infl.items():
        out[f'inflected_{k}'] = n
        out[f'ending_boundary_{k}'] = hit / max(n, 1)
        out[f'ending_own_token_{k}'] = own / max(n, 1)
    if rule is not None:
        out['cut_rule'] = rule.name
        out['cut_fingerprint'] = rule.fingerprint()
        for k, (n, hit, own) in cut.items():
            out[f'cut_words_{k}'] = n
            out[f'cut_boundary_{k}'] = hit / max(n, 1)
            out[f'cut_own_token_{k}'] = own / max(n, 1)
    return out
