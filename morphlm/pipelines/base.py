"""
base.py — shared preprocessor interface for the morpheme pipelines.

Ported from the previous common.py BasePreprocessor, with ONE addition:
encode() now stamps each emitted token with the dependency label and POS tag
of the spaCy token it came from.

WHY THIS IS ALMOST FREE
-----------------------
encode() already iterates over a parsed spaCy Doc — the loop is driven by
`doc[i]`, so every emitted unit already knows its source token. And
spacy.load() enables the tagger and parser by default, so those annotations
were being computed and discarded on every run. Restoring them costs a couple
of dict entries, not a second pass.

This is what makes context-dependent freezing possible, and it closes the gap
between journey section 3 ("preprocessor scope != unit scope": you may look at
the whole sentence to decide what to stamp on one word) and the implementation,
which had quietly reverted to static per-vocabulary-row facts.

THE STAMPING FORK
-----------------
A word maps to several morphemes, so a word-level fact must be distributed.
`stamp` controls how:
    'all'   every morpheme of the word carries the word's dep/pos
    'root'  only root-role morphemes carry it; affixes get None -> 'none' slot
Journey section 4 flagged this and never resolved it. It is a genuine fork and
a candidate for its own comparison, so it is config, not a hardcoded guess.
"""

from __future__ import annotations

from ..vocab.morpholex import CATEGORY_PRIORITY
from ..vocab.entities import NERTagger


class BasePreprocessor:
    """Subclasses implement _segment_word(word) -> [(morpheme, role), ...]."""

    def __init__(self, vocab_df, morpholex_path, spacy_model='en_core_web_sm',
                 stamp='all', surface=None, verbose=True):
        from ..config import SurfaceConfig
        self.vocab = vocab_df
        self.stamp = stamp
        self.surface = surface if surface is not None else SurfaceConfig()
        self.token2id = {row['token']: idx for idx, row in vocab_df.iterrows()}
        self.id2token = {idx: row['token'] for idx, row in vocab_df.iterrows()}
        self.categories = {row['token']: row['category']
                           for _, row in vocab_df.iterrows()}

        n = len(self.token2id)
        self.UNK_ID = n
        self.PAD_ID = n + 1
        self.BOS_ID = n + 2
        self.EOS_ID = n + 3

        self.function_words = {t for t, c in self.categories.items()
                               if c == 'function'}
        self.entity_tokens = {t for t, c in self.categories.items()
                              if c == 'entity'}

        # ── HOT PATH: category resolution, precomputed ────────────────────────
        # This used to be a pandas scan over the whole vocabulary PER TOKEN:
        #     self.vocab[self.vocab['token'] == tok]['category']
        # measured at 2.18 ms/token, i.e. ~60 hours per 100M tokens. It never
        # surfaced because the morpheme pipelines had only ever been run on the
        # 60-sentence test set; the only full-corpus run was BPE, which does not
        # use this class. Resolving once per vocabulary entry makes it a dict
        # lookup and preserves the priority rule exactly:
        #     entity > function > prefix > suffix > root > character
        _prio = vocab_df['category'].map(
            lambda c: CATEGORY_PRIORITY.get(c, 99))
        _ranked = (vocab_df.assign(_prio=_prio)
                   .sort_values('_prio', kind='stable')
                   .drop_duplicates(subset=['token'], keep='first'))
        self.resolved_category = dict(zip(_ranked['token'],
                                          _ranked['category']))

        # entity labels, if the vocabulary carries them
        if 'entity_label' in vocab_df.columns:
            self.entity_labels = dict(zip(vocab_df['token'],
                                          vocab_df['entity_label']))
        else:
            self.entity_labels = {}

        self.ner = NERTagger(spacy_model)
        if verbose:
            print(f'  Vocabulary: {self.vocab_size()} tokens '
                  f'({len(self.token2id)} entries + 4 special)')

    def vocab_size(self):
        return len(self.token2id) + 4

    # ── token construction ────────────────────────────────────────────────────

    def _resolve_category(self, tok, override=None):
        """O(1) dict lookup. See the note in __init__ on why this matters."""
        if override:
            return override
        return self.resolved_category.get(tok, 'character')

    def _make_token(self, morpheme, role, word, entity_label=None,
                    category_override=None, dep=None, pos=None):
        tok = morpheme.lower()
        token_id = self.token2id.get(tok, self.UNK_ID)
        # character fallback always wins over the vocabulary category
        category = ('character' if role == 'char'
                    else self._resolve_category(tok, category_override))
        # stamping fork: 'root' restricts word-level facts to root morphemes
        if self.stamp == 'root' and role not in ('root', 'entity'):
            dep, pos = None, None
        return {
            'token_id': token_id,
            'token': tok,
            'category': category,
            'role': role,
            'entity_label': entity_label,
            'word': word,
            'dep': dep,
            'pos': pos,
        }

    def _segment_word(self, word):
        raise NotImplementedError

    # ── encoding ──────────────────────────────────────────────────────────────

    # ── surface helpers (journey section 17) ──────────────────────────────────

    WS_MAP = {' ': '<sp>', '\n': '<nl>', '\t': '<tab>'}

    def _surface_token(self, tok, category, word, dep=None, pos=None):
        """Emit a non-morphological token (whitespace, punct, digit, case, infl)."""
        return {
            'token_id': self.token2id.get(tok, self.UNK_ID),
            'token': tok,
            'category': category,
            'role': category,
            'entity_label': None,
            'word': word,
            'dep': dep,
            'pos': pos,
        }

    @staticmethod
    def case_marker(surface):
        """
        Case as a marker token rather than capitalised vocabulary variants.

        Doubling the vocabulary to carry case would add ~16k rows and worsen the
        parameter asymmetry against BPE; one marker token costs a single row.
        """
        if not surface or not surface[0].isalpha():
            return None
        if surface.isupper() and len(surface) > 1:
            return '<upper>'
        if surface[0].isupper():
            return '<cap>'
        return None

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode(self, text):
        """
        Text -> list of token dicts.

        Information-preserving by default. Each channel is a SurfaceConfig flag
        so its contribution can be ablated:

          whitespace   the exact gap between spaCy tokens, as <sp>/<nl>/<tab>
          case         <cap> / <upper> before a word
          punct/digit  previously dropped by `if not is_alpha: continue`
          inflection   explicit <plur>/<past>/... from spaCy morph features

        Context facts (dep, pos, entity_label) ride along on every token so
        frozen fact blocks can encode occurrence-level information.
        """
        doc = self.ner.nlp(text)
        entity_spans, entity_indices = self.ner.get_entity_spans(doc)
        sfc = self.surface

        result = []
        prev_end = 0
        i = 0

        def emit_gap(upto):
            if not sfc.emit_whitespace:
                return
            for ch in text[prev_end:upto]:
                result.append(self._surface_token(
                    self.WS_MAP.get(ch, '<sp>'), 'space', ch))

        while i < len(doc):
            spacy_tok = doc[i]
            emit_gap(spacy_tok.idx)

            # ── multi-word entity span ────────────────────────────────────────
            if spacy_tok.i in entity_spans:
                surface, label = entity_spans[spacy_tok.i]
                dep, pos = spacy_tok.dep_, spacy_tok.pos_
                last = i
                while last < len(doc) and doc[last].i in entity_indices:
                    last += 1
                span_text = text[spacy_tok.idx:
                                 doc[last - 1].idx + len(doc[last - 1].text)]
                if sfc.emit_case:
                    cm = self.case_marker(span_text)
                    if cm:
                        result.append(self._surface_token(
                            cm, 'case', span_text, dep, pos))
                # `word` must be the ORIGINAL span text, not the lowercased
                # entity key. Passing the key split the case marker off into a
                # separate "word" for every capitalised entity, which made the
                # collision audit report one enormous phantom group.
                if surface in self.entity_tokens:
                    result.append(self._make_token(
                        surface, 'entity', span_text,
                        entity_label=label, category_override='entity',
                        dep=dep, pos=pos))
                else:
                    for w in surface.split('_'):
                        for ch in w:
                            result.append(self._make_token(
                                ch, 'char', span_text, entity_label=label,
                                dep=dep, pos=pos))
                prev_end = doc[last - 1].idx + len(doc[last - 1].text)
                i = last
                continue

            if spacy_tok.i in entity_indices:
                i += 1
                continue

            surface = spacy_tok.text
            dep, pos = spacy_tok.dep_, spacy_tok.pos_
            prev_end = spacy_tok.idx + len(surface)

            if spacy_tok.is_alpha:
                if sfc.emit_case:
                    cm = self.case_marker(surface)
                    if cm:
                        result.append(self._surface_token(
                            cm, 'case', surface, dep, pos))
                pieces = list(self._segment_word(surface))
                for morpheme, role in pieces:
                    result.append(self._make_token(morpheme, role, surface,
                                                   dep=dep, pos=pos))
                if sfc.emit_inflection:
                    from ..features.inflection import (
                        inflection_of, already_marked, residue_marker)
                    morphemes = [m for m, _ in pieces]
                    infl = inflection_of(spacy_tok)
                    if infl is None:
                        # spaCy's POS can HIDE an inflection: gerunds are
                        # tagged NOUN ('gaming'), participles ADJ ('seasoned'),
                        # and MorphoLex root mismatches are invisible to any
                        # feature at all ('news' -> root 'new'). Falling back to
                        # the trailing difference between the morphemes and the
                        # surface catches all three, and it is deterministic on
                        # the surface string, so injectivity is guaranteed
                        # rather than hoped for.
                        infl = residue_marker(''.join(morphemes),
                                              surface.lower())
                    if infl and not already_marked(infl, morphemes):
                        result.append(self._surface_token(
                            infl, 'infl', surface, dep, pos))
            else:
                # punctuation, digits, symbols, mixed tokens like "1990s"
                for ch in surface:
                    if ch.isdigit():
                        if sfc.emit_digits:
                            result.append(self._surface_token(
                                ch, 'digit', surface, dep, pos))
                    elif ch.isalpha():
                        result.append(self._make_token(
                            ch, 'char', surface, dep=dep, pos=pos))
                    elif sfc.emit_punctuation:
                        result.append(self._surface_token(
                            ch, 'punct', surface, dep, pos))
            i += 1

        emit_gap(len(text))
        return self._byte_fallback(result)

    def _byte_fallback(self, tokens):
        """
        Replace every UNK with its UTF-8 bytes.

        UNK is a collision sink: each distinct unknown string maps to the same
        id, so two different inputs become one stream and information is lost.
        The audit caught exactly this — en-dash, em-dash, degree and pound all
        collapsed together. Spelling unknowns out in bytes makes the encoding
        TOTAL, so injectivity no longer depends on the vocabulary happening to
        cover the input.
        """
        out = []
        for t in tokens:
            if t['token_id'] != self.UNK_ID:
                out.append(t)
                continue
            for b in str(t['token']).encode('utf-8'):
                bt = dict(t)
                bt['token'] = f'<b{b:02x}>'
                bt['token_id'] = self.token2id.get(bt['token'], self.UNK_ID)
                bt['category'] = 'byte'
                bt['role'] = 'byte'
                out.append(bt)
        return out

    def encode_ids(self, text):
        return [t['token_id'] for t in self.encode(text)]

    def pretty_print(self, text):
        units = self.encode(text)
        print(f'Input: {text}')
        print(f'  {"token":20} {"cat":10} {"role":8} {"dep":10} {"pos":6} word')
        print(f'  {"-" * 72}')
        for u in units:
            print(f'  {u["token"]:20} {u["category"]:10} {u["role"]:8} '
                  f'{str(u["dep"] or "-"):10} {str(u["pos"] or "-"):6} {u["word"]}')
        print()
