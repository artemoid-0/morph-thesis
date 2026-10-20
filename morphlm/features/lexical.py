"""
lexical.py — per-WORD lexicons over the BPE token stream.

A contextual feature (a tagger's POS or morphological bundle) is computed with
a window that extends to the RIGHT of the word, so even shifted to the previous
word it can carry information about the word being predicted. A LEXICAL
feature depends on the word's own form only — the way `mstruct` is looked up
from MorphoLEX — and after the causal shift depends only on words already
written.

This module turns a contextual feature into a lexical one: each word type gets
its MOST FREQUENT contextual value in the training corpus. It is shared by
`scripts/data/derive_lexical_codes.py`, which builds the lexicon and the cached
codes, and by any block that must compute codes for new text (BLiMP), so the
two cannot drift apart.

WORDS are the same units every other tool uses: a maximal alphabetic run of BPE
tokens, split again at each token carrying the ByteLevel space marker — the
rule in `derive_prefix_codes.py`, `lookahead_check.py` and the embedding's
causal boundaries. A word's KEY is a 64-bit rolling hash of its token ids;
collisions over ~10^5 word types are negligible at 2^64.
"""

from __future__ import annotations

import os
import json

import numpy as np

GDOT = '\u0120'
_P = np.uint64(1099511628211)       # FNV-64 prime; wraps mod 2^64 by design


def _bytelevel_decoder():
    """ByteLevel symbol -> byte: the inverse of GPT-2's bytes_to_unicode."""
    bs = (list(range(ord('!'), ord('~') + 1)) + list(range(ord('\u00a1'), ord('\u00ac') + 1))
          + list(range(ord('\u00ae'), ord('\u00ff') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def word_rule_for(bpe_path):
    """
    How a tokenizer's tokens are classified as word-internal (G-60).

    'symbol' — the original rule, `isalpha()` on the ByteLevel symbol string.
    Right for English, whose letters map to themselves, and it must stay the
    English rule: English tokens with accents (é is stored as `Ã©`) would change
    class under any other rule, and with them the codes behind every recorded
    English result. 'decoded' — classify the token's decoded bytes; needed for
    Cyrillic, whose letters are two UTF-8 bytes that ByteLevel maps partly to
    non-letters (the symbol rule calls 77% of Cyrillic tokens non-alphabetic).
    The rule is read from the sidecar written beside the tokenizer when it is
    trained (`tokenization.tokenizer_meta`); with none, 'symbol'. It travels
    with the tokenizer, so no command can forget it.
    """
    from ..data.tokenization import tokenizer_meta
    return tokenizer_meta(bpe_path).get('word_rule', 'symbol')


def _lead_is_letter(b0, b1=None):
    """
    Whether a character is a letter, from its first UTF-8 byte (and second,
    where the token holds it) — for a token holding only the START of a split
    character. By Unicode block: Latin, Greek, Cyrillic and the other scripts
    are letters; Latin-1 symbols (C2), combining marks (CC, CD<B0), general
    punctuation and symbols (E2, except Glagolitic/Coptic/Cyrillic Ext-A),
    CJK punctuation (E3 80), private use and emoji are not.
    """
    if b0 == 0xC2 or b0 == 0xCC or b0 == 0xEE or b0 == 0xF4:
        return False
    if b0 == 0xCD:
        return b1 is not None and b1 >= 0xB0          # Greek; below it, combining
    if b0 == 0xE2:
        return b1 is not None and 0xB0 <= b1 <= 0xB7  # U+2C00-2DFF; else punctuation
    if b0 == 0xE3:
        return b1 != 0x80                             # U+3000-303F CJK punctuation
    if b0 == 0xEF:
        return b1 is not None and 0xA4 <= b1 <= 0xB7  # compatibility ideographs, Arabic forms
    if b0 == 0xF0:
        # U+1F000-1FFFF (emoji, symbols) behind 9F; historic scripts otherwise.
        # Unknown second byte: emoji, by far the likelier in modern text
        return b1 is not None and b1 != 0x9F
    return b0 >= 0xC3


def token_class(symbol, rule, decoder=None):
    """
    (alpha, opens, fragment) of one token. A FRAGMENT holds only continuation
    bytes of a character begun in the token before: it has no class of its
    own, and `resolve_fragments` gives it that token's (G-63). Its `alpha`
    here, True, stands in only where there is no token before.
    """
    if symbol is None:
        return False, False, False
    if rule == 'symbol':
        core = symbol.lstrip(GDOT)
        return bool(core) and core.isalpha(), symbol.startswith(GDOT), False
    if rule != 'decoded':
        raise ValueError(f'unknown word rule {rule!r}')
    decoder = decoder or _bytelevel_decoder()
    raw = bytes(decoder[ch] for ch in symbol if ch in decoder)
    opens = raw[:1] == b' '
    core = raw.lstrip(b' ')
    if not core:
        return False, opens, False
    text = core.decode('utf-8', errors='replace').replace('\ufffd', '')
    if text:
        # BPE never merges across the pre-tokenizer's letter / non-letter
        # split, so a token's complete characters are all of one kind
        return text.isalpha(), opens, False
    if all(0x80 <= b < 0xC0 for b in core):
        return True, opens, True
    i = next(k for k, b in enumerate(core) if b >= 0xC0)
    return _lead_is_letter(core[i], core[i + 1] if i + 1 < len(core) else None), opens, False


def classify_token(symbol, rule, decoder=None):
    """(alpha, opens) of one token under a word rule — `token_class` without the fragment flag."""
    return token_class(symbol, rule, decoder)[:2]


class WordTables(tuple):
    """
    (alpha, opens) over token ids — unpacking as the pair it always was — plus
    `.frag`, the fragment flags (`token_class`). Under the symbol rule no token
    is a fragment, so English words are what they always were.
    """
    def __new__(cls, alpha, opens, frag):
        t = super().__new__(cls, (alpha, opens))
        t.frag = frag
        return t


FRAGMENT_STEPS = 3      # a UTF-8 character has at most three continuation bytes


def resolve_fragments(a, frag):
    """Per position: each fragment takes the class of the position before it —
    in running text, its own character's. Position 0 keeps its stand-in."""
    if frag is None or not frag.any():
        return a
    a = a.copy()
    for _ in range(FRAGMENT_STEPS):
        a[1:] = np.where(frag[1:], a[:-1], a[1:])
    return a


def tokenizer_tables(bpe_path, rule=None):
    """(alpha, opens) lookup tables over token ids, under the tokenizer's rule."""
    from ..data.tokenization import load_tokenizer
    tok = load_tokenizer(bpe_path, vocab_only=True)        # the vocabulary is all it reads
    rule = rule or word_rule_for(bpe_path)
    decoder = _bytelevel_decoder() if rule == 'decoded' else None
    n = tok.get_vocab_size()
    alpha = np.zeros(n, dtype=bool)
    opens = np.zeros(n, dtype=bool)
    frag = np.zeros(n, dtype=bool)
    for i in range(n):
        alpha[i], opens[i], frag[i] = token_class(tok.id_to_token(i), rule, decoder)
    return WordTables(alpha, opens, frag)


def word_spans(ids, tables):
    """
    Per position: the index of its word (-1 outside words), and the word's key.

    Returns (wid, key, starts) where `starts` marks each word's first position.
    Non-word positions (punctuation, digits, anything non-alphabetic) get
    wid -1 and key 0.
    """
    alpha_lut, opens_lut = tables
    frag_lut = getattr(tables, 'frag', None)
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    if n == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.uint64),
                np.zeros(0, bool))
    a = alpha_lut[ids]
    if frag_lut is not None:
        a = resolve_fragments(a, frag_lut[ids])
    pa = np.empty(n, dtype=bool)
    pa[0] = False
    pa[1:] = a[:-1]
    starts = a & (opens_lut[ids] | ~pa)

    idx = np.arange(n, dtype=np.int64)
    start_of = np.maximum.accumulate(np.where(starts, idx, -1))
    valid = a & (start_of >= 0)
    offset = np.where(valid, idx - start_of, 0)

    h = np.zeros(n, dtype=np.uint64)
    vals = (ids + 1).astype(np.uint64)
    max_off = int(offset.max()) if n else 0
    with np.errstate(over='ignore'):
        for k in range(max_off + 1):
            m = valid & (offset >= k)
            if not m.any():
                break
            h[m] = h[m] * _P + vals[start_of[m] + k]

    # a word ends where the next position is not a continuation of it
    nxt_cont = np.zeros(n, dtype=bool)
    nxt_cont[:-1] = valid[1:] & ~starts[1:]
    ends = valid & ~nxt_cont

    wid = np.where(valid, np.cumsum(starts) - 1, -1)
    n_words = int(starts.sum())
    key_of = np.zeros(max(n_words, 1), dtype=np.uint64)
    key_of[wid[ends]] = h[ends]
    key = np.where(valid, key_of[np.maximum(wid, 0)], np.uint64(0))

    # CAUSALITY of the key: a word's key is a function of that word's tokens
    # only. Asserted on the structure, not assumed.
    assert bool((start_of[valid] <= idx[valid]).all())
    return wid, key, starts


def build_lexicon(keys, codes):
    """
    Majority value per key. `keys`, `codes`: one entry per word occurrence.
    Ties go to the smaller code, so the result is deterministic.
    Returns (lex_keys sorted ascending, lex_codes, lex_counts).
    """
    keys = np.asarray(keys, dtype=np.uint64)
    codes = np.asarray(codes, dtype=np.int64)
    order = np.lexsort((codes, keys))
    ks, cs = keys[order], codes[order]
    change = np.ones(len(ks), dtype=bool)
    change[1:] = (ks[1:] != ks[:-1]) | (cs[1:] != cs[:-1])
    at = np.flatnonzero(change)
    pk, pc = ks[at], cs[at]
    pn = np.diff(np.append(at, len(ks)))
    o2 = np.lexsort((pc, -pn, pk))
    pk, pc, pn = pk[o2], pc[o2], pn[o2]
    first = np.ones(len(pk), dtype=bool)
    first[1:] = pk[1:] != pk[:-1]
    tot = np.add.reduceat(pn, np.flatnonzero(first)) if len(pn) else pn
    return pk[first], pc[first].astype(np.int16), tot


def lookup(keys, lexicon, none_code):
    lex_keys, lex_codes = lexicon[0], lexicon[1]
    keys = np.asarray(keys, dtype=np.uint64)
    if len(lex_keys) == 0:
        return np.full(len(keys), none_code, dtype=np.int16)
    i = np.searchsorted(lex_keys, keys)
    i = np.minimum(i, len(lex_keys) - 1)
    found = lex_keys[i] == keys
    return np.where(found, lex_codes[i], none_code).astype(np.int16)


def lexical_codes(ids, lexicon, tables, none_code):
    """
    Each word's lexicon entry, at every position of the word. Positions outside
    words get `none_code`; words absent from the lexicon get `none_code` too —
    never their contextual code, which would readmit the right-context
    information the lexicon exists to remove.
    """
    wid, key, _ = word_spans(ids, tables)
    out = lookup(key, lexicon, none_code)
    out[wid < 0] = none_code
    return out


def frequency_bins(counts, bin_size=64, max_ratio=2.0):
    """
    Consecutive runs of word types by descending frequency. A bin closes at
    `bin_size` types, or before its most frequent member would exceed
    `max_ratio` times its least frequent one — Zipfian frequencies span orders
    of magnitude, and a bin of 64 at the top would mix "the" (~6% of tokens)
    with words a hundred times rarer.
    """
    counts = np.asarray(counts)
    order = np.argsort(-counts, kind='stable')
    bins, cur = [], []
    for i in order:
        if cur and (len(cur) >= bin_size or counts[cur[0]] > max_ratio * max(counts[i], 1)):
            bins.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        bins.append(np.array(cur))
    return bins


def shuffle_within_frequency(codes, counts, seed, bin_size=64, max_ratio=2.0):
    """
    Reassign codes at random among word types of SIMILAR FREQUENCY — the
    control for "is it the content of a partition, or any partition?".

    Codes are permuted within each frequency bin (frequency_bins): every bin
    keeps its multiset of classes, so class sizes and the class-frequency
    profile survive, and a class only ever moves between words of similar
    frequency, so token-level class shares barely move. Which word carries
    which class does not survive. Deterministic by seed.
    """
    codes = np.asarray(codes)
    rng = np.random.RandomState(seed)
    new = codes.copy()
    for chunk in frequency_bins(counts, bin_size, max_ratio):
        new[chunk] = codes[chunk][rng.permutation(len(chunk))]
    return new


def frequency_bands(counts, n):
    """
    Each word type's frequency band: types ranked by corpus frequency, cut so
    each of `n` bands covers ~1/n of all word TOKENS (band 0 = the most
    frequent). The `mfreq` control: the only information is how common the
    previous word is.
    """
    counts = np.asarray(counts, dtype=np.float64)
    order = np.argsort(-counts, kind='stable')
    before = np.cumsum(counts[order]) - counts[order]
    band_sorted = np.minimum((before / max(counts.sum(), 1.0) * n).astype(np.int64), n - 1)
    band = np.empty(len(counts), dtype=np.int64)
    band[order] = band_sorted
    return band


def class_shares(codes, counts, n_classes):
    """Token-weighted share of each class."""
    w = np.bincount(np.asarray(codes, dtype=np.int64), weights=counts,
                    minlength=n_classes)
    return w / max(float(np.sum(counts)), 1.0)


def lexicon_condition(corpus, fraction):
    """
    The corpus and size a lexicon was built from, for its file name (G-66).

    '' for WikiText-103 — its lexicons keep their historical names. For any other
    corpus, '<corpus><fraction>': deriving at one corpus size used to overwrite the
    lexicon built at another under the same tokenizer.
    """
    if corpus in (None, 'wikitext103'):
        return ''
    return f'{corpus}{float(fraction):g}'


def lexicon_path(bpe_path, fact, cache_dir='data/cache', condition=''):
    stem = os.path.basename(bpe_path).replace('.json', '')
    suffix = f'_{condition}' if condition else ''
    return os.path.join(cache_dir, f'lexicon_{fact}_{stem}{suffix}.npz')


def save_lexicon(path, lexicon, meta):
    np.savez(path, keys=lexicon[0], codes=lexicon[1], counts=lexicon[2])
    with open(path.replace('.npz', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def load_lexicon(bpe_path, fact, cache_dir='data/cache', condition=''):
    """The lexicon named for this corpus and size; failing that, the historical name (G-66)."""
    paths = ([lexicon_path(bpe_path, fact, cache_dir, condition)] if condition else []) + \
        [lexicon_path(bpe_path, fact, cache_dir)]
    for path in paths:
        if os.path.exists(path):
            z = np.load(path)
            return z['keys'], z['codes'], z['counts']
    raise FileNotFoundError(
        f'{paths[0]} missing: run scripts/data/derive_lexical_codes.py --fact {fact}')
