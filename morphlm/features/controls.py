"""
controls.py — the derived facts and the controls.

A DERIVED fact's codes are made after the cache build, from cached codes or the
token stream, by the script its block names (`DerivedCodesMixin`); per-token
coding refuses. Most are derived per word type through a lexicon
(`LexicalLookupMixin`), which also codes new text (BLiMP). The controls keep a
fact's table and width and remove its content: the nulls, the shuffles, the
zero and flat slots, the frequency bands.

`blocks` imports this module once its base blocks exist, so `registry._load()`
registers these too, and every class stays importable from `blocks`.
"""

from __future__ import annotations

import numpy as np

from ..registry import register_fact
from .blocks import (FactBlock, MorphFeatureBlock, MorphRootBlock,
                     MorphStructBlock, PosBlock, UCaseBlock, UGenderBlock, UNumberBlock)


class DerivedCodesMixin:
    """
    A fact whose codes are derived after the cache build: per-token coding
    refuses (it would fill a cache with empty codes) and names the derivation,
    `derive_hint`. Put it first in the bases so its `code` wins.
    """
    derive_hint = None

    @classmethod
    def derivation(cls) -> str:
        """How this fact's codes are made — the message `code` raises."""
        return cls.derive_hint

    def code(self, token: dict) -> int:
        raise RuntimeError(self.derivation())


class LexicalLookupMixin(DerivedCodesMixin):
    """
    For facts DERIVED per word type by scripts/data/derive_lexical_codes.py: they
    refuse per-token coding and code new text through the lexicon the derivation
    saved under `lexicon_name`. Put it first in the bases so its methods win.
    """
    lexicon_name = None

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._cfg = cfg
        self._lex = None

    @classmethod
    def derivation(cls) -> str:
        return cls.derive_hint or (
            f'{cls.lexicon_name} is derived per WORD, not per token: run '
            f'scripts/data/derive_lexical_codes.py with --out-fact {cls.lexicon_name}')

    def codes_for(self, ids, tokens):
        """Codes for a new token sequence, via the saved lexicon."""
        from .lexical import load_lexicon, lexical_codes, tokenizer_tables, lexicon_condition
        if self._lex is None:
            data = getattr(self._cfg, 'data', None)
            path = getattr(data, 'bpe_path', None)
            self._tables = tokenizer_tables(path)
            self._lex = load_lexicon(path, self.lexicon_name,
                                     getattr(data, 'cache_dir', 'data/cache'),
                                     condition=lexicon_condition(getattr(data, 'corpus', None),
                                                                 getattr(data, 'corpus_fraction', 1.0)))
        return lexical_codes(ids, self._lex, self._tables,
                             none_code=len(self.classes) - 1).tolist()


# ── strictly causal variants: a word's majority value, from its form alone ────

@register_fact('mfeatlex')
class MorphFeatureLexicalBlock(LexicalLookupMixin, MorphFeatureBlock):
    """
    STRICTLY CAUSAL `mfeat`: each word's MOST FREQUENT bundle in the training
    corpus, looked up from the word alone — the way `mstruct` is looked up
    from MorphoLEX. No tagger context, so after the causal shift it depends
    only on words already written. It loses the tagger's disambiguation
    (`reading` always gets its majority reading); what it keeps is leak-free.

    Derived from cached `mfeat` codes by scripts/data/derive_lexical_codes.py,
    which also writes the lexicon that `codes_for` uses on new text (BLiMP).
    """
    lexicon_name = 'mfeat'
    derive_hint = ('mfeatlex is derived per WORD, not per token: run '
                   'scripts/data/derive_lexical_codes.py after the cache holds mfeat. '
                   'Building it token by token would fill the cache with empty codes.')


@register_fact('poslex')
class PosLexicalBlock(LexicalLookupMixin, PosBlock):
    """
    STRICTLY CAUSAL `pos`: each word's most frequent universal POS tag in the
    training corpus, looked up from the word alone. The tagger's `pos` depends
    on words to its right for ~7% of words (journey §33.6), so shifted `pos`
    can still carry a little of the word being predicted; this cannot. Width 8
    like `pos`, so the two compare like for like.

    Derived from cached `pos` codes: scripts/data/derive_lexical_codes.py --fact pos
    """
    lexicon_name = 'pos'
    derive_hint = ('poslex is derived per WORD, not per token: run '
                   'scripts/data/derive_lexical_codes.py --fact pos after the cache holds pos.')


@register_fact('ucaselex')
class UCaseLexicalBlock(LexicalLookupMixin, UCaseBlock):
    """
    STRICTLY CAUSAL case: each word form's most frequent case in the training
    corpus, looked up from the form alone. The tagger's case reads the right
    context — a preposition's case is set by the noun after it (на + Acc for a
    direction, на + Loc for a place), and that noun is exactly the word the
    shifted feature is read before. Same 8 classes as `ucase`, so the two share
    initial weights at the same seed.
    Derived: derive_lexical_codes.py --fact ucase --out-fact ucaselex --corpus ukwiki
    """
    lexicon_name = 'ucase'      # a majority lexicon is saved under its source fact
    derive_hint = ('ucaselex is derived per WORD, not per token: run '
                   'scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaselex --corpus ukwiki')


@register_fact('ugenderlex')
class UGenderLexicalBlock(LexicalLookupMixin, UGenderBlock):
    """
    STRICTLY CAUSAL gender (DIRECTIONS, queue item 2): each word form's most frequent gender in
    the training corpus, looked up from the form alone. The morphologizer's gender reads the right
    context, spread over adjectives, proper nouns, nouns and determiners (journey §33.125), so the
    shifted feature can carry part of the next word. Same 8 classes as `ugender`, so the two share
    initial weights at the same seed.
    Derived: derive_lexical_codes.py --fact ugender --out-fact ugenderlex --keep ugender --corpus ukwiki
    """
    lexicon_name = 'ugender'
    derive_hint = ('ugenderlex is derived per WORD, not per token: run '
                   'scripts/data/derive_lexical_codes.py --fact ugender --out-fact ugenderlex --corpus ukwiki')


@register_fact('unumberlex')
class UNumberLexicalBlock(LexicalLookupMixin, UNumberBlock):
    """
    STRICTLY CAUSAL number: each word form's most frequent number in the training corpus, looked up
    from the form alone; see `ugenderlex`. Same 8 classes as `unumber`.
    Derived: derive_lexical_codes.py --fact unumber --out-fact unumberlex --keep unumber --corpus ukwiki
    """
    lexicon_name = 'unumber'
    derive_hint = ('unumberlex is derived per WORD, not per token: run '
                   'scripts/data/derive_lexical_codes.py --fact unumber --out-fact unumberlex --corpus ukwiki')


# ── controls for the tagger's right-context leak ──────────────────────────────

@register_fact('ucaseprep')
class UCasePrepositionFormBlock(DerivedCodesMixin, UCaseBlock):
    """
    CONTROL for the tagger's right-context leak in `ucase` (G-38; journey §33.73):
    case in context everywhere except at prepositions — the word types whose
    majority part of speech in the training corpus is ADP — which carry their word
    form's majority case at every occurrence. A preposition's case in context is
    the case of the noun after it (на + Acc for a direction, на + Loc for a place),
    which the tagger reads from that very noun: the word the shifted feature is
    read before. Recoded, a preposition says only what its form says; every other
    word keeps its case in context. Same 8 classes as `ucase`, so the two share
    initial weights at the same seed.
    Derived: derive_lexical_codes.py --fact ucase --out-fact ucaseprep --recode-pos ADP
             --keep ucase --corpus ukwiki
    """
    derive_hint = ('ucaseprep is derived per WORD, not per token: run '
                   'scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaseprep '
                   '--recode-pos ADP --keep ucase --corpus ukwiki')


@register_fact('poscausal')
class PosCausalBlock(DerivedCodesMixin, PosBlock):
    """
    CONTROL for the tagger's right-context leak in `pos` (journey §33.98): part of speech from a
    strictly causal tagger — each word tagged from its left context only, the tag it would get
    if the text ended at that word. `pos` reads the whole paragraph, so a word's tag can depend
    on the next word ("to" before a verb or a noun), which the shifted feature is read before.
    In English the dependence is spread over thousands of words, so no word list removes it, as
    the prepositions did for Ukrainian case (`ucaseprep`). Same 19 classes and width as `pos`,
    so the two share initial weights at the same seed.
    Derived: scripts/data/derive_causal_pos.py
    """
    derive_hint = ('poscausal is derived from the text by a causal tagger, not per token: run '
                   'scripts/data/derive_causal_pos.py first')


@register_fact('ugendercausal')
class UGenderCausalBlock(DerivedCodesMixin, UGenderBlock):
    """
    CONTROL for the right-context leak in `ugender` (journey §33.125): gender from a strictly causal
    tagger, each word tagged from its left context only — the value it would get if the text ended
    at that word. The leak is spread over hundreds of word types (1,635 for gender in 10,000
    sentences) and four classes (adjectives, proper nouns, nouns, determiners), so no word list
    removes it, as the prepositions did for case (`ucaseprep`); this
    removes it at every word, as `poscausal` does for English part of speech. Same 8 classes as
    `ugender`, so the two share initial weights at the same seed.
    Derived: scripts/data/derive_causal_ud.py
    """
    derive_hint = ('ugendercausal is derived from the text by a causal tagger, not per token: run '
                   'scripts/data/derive_causal_ud.py first')


@register_fact('unumbercausal')
class UNumberCausalBlock(DerivedCodesMixin, UNumberBlock):
    """
    CONTROL for the right-context leak in `unumber`: number from a strictly causal tagger; see
    `ugendercausal`. Same 8 classes as `unumber`.
    Derived: scripts/data/derive_causal_ud.py
    """
    derive_hint = ('unumbercausal is derived from the text by a causal tagger, not per token: run '
                   'scripts/data/derive_causal_ud.py first')


# ── causal word identity ──────────────────────────────────────────────────────

@register_fact('mprefix')
class MorphPrefixBlock(DerivedCodesMixin, MorphRootBlock):
    """
    CAUSAL word identity: a hash of the word's fragments SO FAR — causal at
    every position, and at a word's final fragment the true word identity,
    where the previous-word shift still gives the word before (journey §30.8):

        recon            -> hash("recon")
        recon fig        -> hash("reconfig")
        recon fig uring  -> hash("reconfiguring")

    Same inventory and width as the other identity facts, so the comparison
    isolates causality. DERIVED from the cached ids alone, by
    scripts/data/derive_prefix_codes.py.
    """
    derive_hint = ('mprefix codes are derived from the cached token stream, not built '
                   'by the pipeline. Run scripts/data/derive_prefix_codes.py first, or the '
                   'cache would be filled with empty codes.')


# ── controls for `mstruct` and `pos`: the same table, the content removed ─────

@register_fact('mstructshuf')
class MorphStructShuffledBlock(LexicalLookupMixin, MorphStructBlock):
    """
    CONTROL for `mstruct`: its eight classes reassigned at random among word
    types of similar frequency (lexical.shuffle_within_frequency). Class sizes
    and the class-frequency profile are kept; the linguistic content — which
    word is prefixed, suffixed, compound, a name — is destroyed.

    If this helps as much as `mstruct`, the gain comes from ANY consistent
    coarse grouping of word types, not from morphology (journey §33.8).

    Derived: scripts/data/derive_lexical_codes.py --fact mstruct
             --out-fact mstructshuf --shuffle-types 0 --keep mstruct
    """
    lexicon_name = 'mstructshuf'
    derive_hint = ('mstructshuf is derived per WORD: run scripts/data/derive_lexical_codes.py '
                   '--fact mstruct --out-fact mstructshuf --shuffle-types 0')


@register_fact('posshuf')
class PosShuffledBlock(LexicalLookupMixin, PosBlock):
    """
    CONTROL for `pos`: its classes reassigned among word types of similar
    frequency, as `mstructshuf` is for `mstruct` (journey §33.9).
    Derived: derive_lexical_codes.py --fact pos --out-fact posshuf
             --shuffle-types 0 --keep pos
    """
    lexicon_name = 'posshuf'


@register_fact('mfreq')
class FrequencyBandBlock(LexicalLookupMixin, FactBlock):
    """
    FREQUENCY CONTROL: the previous word's frequency band, seven bands of equal
    token mass plus 'none' — the same table size as `mstruct`. The only
    information is how common the word is (journey §33.9).
    Derived: derive_lexical_codes.py --fact mstruct --out-fact mfreq
             --transform freq --keep mstruct
    """
    source = 'context'
    lexicon_name = 'mfreq'
    classes = tuple(f'f{i}' for i in range(7)) + ('none',)


# ── the nulls: the slot without information ───────────────────────────────────

@register_fact('mconst')
class MorphConstBlock(LexicalLookupMixin, MorphStructBlock):
    """
    THE NULL CONTROL: `mstruct`'s architecture — same 8-class table, same
    8 dims, same causal shift — with every word given the same class. It says
    only whether the previous token belonged to a word, which the model sees
    anyway. A gain here is the feature SLOT, not information (journey §33.9).
    Derived: derive_lexical_codes.py --fact mstruct --out-fact mconst
             --transform const --keep mstruct
    """
    lexicon_name = 'mconst'

    def codes_for(self, ids, tokens):
        """New text (BLiMP): class 0 on every word, `none` elsewhere — no lexicon,
        so a word unseen in train is not marked as a non-word (G-73)."""
        from .lexical import tokenizer_tables, word_spans
        if getattr(self, '_tables', None) is None:
            data = getattr(self._cfg, 'data', None)
            self._tables = tokenizer_tables(getattr(data, 'bpe_path', None))
        wid, _, _ = word_spans(np.asarray(ids, dtype=np.int64), self._tables)
        return np.where(wid >= 0, 0, len(self.classes) - 1).tolist()


@register_fact('mconst16')
class MorphConst16Block(MorphConstBlock):
    """The null at 16 dims (token embedding 112). Width dose-response, §33.10."""
    lexicon_name = 'mconst16'
    width = 16


@register_fact('mconst32')
class MorphConst32Block(MorphConstBlock):
    """The null at 32 dims (token embedding 96). Width dose-response, §33.10."""
    lexicon_name = 'mconst32'
    width = 32


@register_fact('mconst64')
class MorphConst64Block(MorphConstBlock):
    """The null at 64 dims (token embedding 64). Width dose-response, §33.10."""
    lexicon_name = 'mconst64'
    width = 64


@register_fact('mzero')
class MorphZeroBlock(MorphConstBlock):
    """
    ZERO-SLOT control: `mconst`'s 8 dims held at exactly zero by the embedding
    (`zero_slot`). No learned vector and no word/non-word signal: the only
    difference from the baseline is a token embedding of 120 dims instead of
    128. If this gains like the null, the lever is simply input dims free of
    token identity (journey §33.11).
    Derived (the codes are ignored): derive_lexical_codes.py --fact mstruct
             --out-fact mzero --transform const --keep mstruct
    """
    lexicon_name = 'mzero'
    zero_slot = True


@register_fact('mzeronopos')
class MorphZeroNoPosBlock(MorphZeroBlock):
    """
    `mzero` with NO position in its slot: the slot's 8 dims are exactly zero
    at the input — no token, no position. If the zero slot's gain survives,
    the free dims help as workspace; if it vanishes, what helped was a CLEAN
    POSITION CHANNEL (journey §33.12).
    Derived (codes ignored): --out-fact mzeronopos --transform const
    """
    lexicon_name = 'mzeronopos'
    slot_position = 'none'


@register_fact('mzeroposonly')
class MorphZeroPosOnlyBlock(MorphZeroBlock):
    """
    `mzero` with position ONLY in its slot: the token dims carry pure token
    identity, the slot's 8 dims pure position — a fully disentangled input.
    If a clean position channel is the mechanism, this does at least as well
    as `mzero` (journey §33.12).
    Derived (codes ignored): --out-fact mzeroposonly --transform const
    """
    lexicon_name = 'mzeroposonly'
    slot_position = 'only'


@register_fact('mflat')
class MorphFlatBlock(MorphConstBlock):
    """
    FLAT control: one learned vector at EVERY position, word or punctuation.
    `mconst` still marks whether the previous unit was a word (`simple`) or
    not (`none`); this removes even that. If it gains like the null, that
    boundary signal is not the ingredient (journey §33.11).
    Derived: derive_lexical_codes.py --fact mstruct --out-fact mflat
             --transform flat --keep mstruct
    """
    lexicon_name = 'mflat'

    def codes_for(self, ids, tokens):
        return [0] * len(ids)
