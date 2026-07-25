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
between journey section 5 ("preprocessor scope != unit scope": you may look at
the whole sentence to decide what to stamp on one word) and the implementation,
which had quietly reverted to static per-vocabulary-row facts.

THE STAMPING FORK
-----------------
A word maps to several morphemes, so a word-level fact must be distributed.
`stamp` controls how:
    'all'   every morpheme of the word carries the word's dep/pos
    'root'  only root-role morphemes carry it; affixes get None -> 'none' slot
Journey section 6 flagged this and never resolved it. It is a genuine fork and
a candidate for its own comparison, so it is config, not a hardcoded guess.
"""

from __future__ import annotations

from ..vocab.morpholex import CATEGORY_PRIORITY
from ..vocab.entities import NERTagger


class BasePreprocessor:
    """Subclasses implement _segment_word(word) -> [(morpheme, role), ...]."""

    def __init__(self, vocab_df, morpholex_path, spacy_model='en_core_web_sm',
                 stamp='all', verbose=True):
        self.vocab = vocab_df
        self.stamp = stamp
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

        self.ner = NERTagger(spacy_model)
        if verbose:
            print(f'  Vocabulary: {self.vocab_size()} tokens '
                  f'({len(self.token2id)} entries + 4 special)')

    def vocab_size(self):
        return len(self.token2id) + 4

    # ── token construction ────────────────────────────────────────────────────

    def _resolve_category(self, tok, override=None):
        if override:
            return override
        matches = self.vocab[self.vocab['token'] == tok]['category'].tolist()
        if not matches:
            return 'character'
        return min(matches, key=lambda c: CATEGORY_PRIORITY.get(c, 99))

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

    def encode(self, text):
        """
        Text -> list of token dicts.

        Each dict carries the context facts (dep, pos, entity_label) of the
        spaCy token it originated from, so downstream fact blocks can encode
        occurrence-level information that a static id->vector table cannot.
        """
        doc = self.ner.nlp(text)
        entity_spans, entity_indices = self.ner.get_entity_spans(doc)

        result = []
        i = 0
        while i < len(doc):
            spacy_tok = doc[i]

            if not spacy_tok.is_alpha:
                i += 1
                continue

            if spacy_tok.i in entity_spans:
                surface, label = entity_spans[spacy_tok.i]
                dep, pos = spacy_tok.dep_, spacy_tok.pos_
                if surface in self.entity_tokens:
                    result.append(self._make_token(
                        surface, 'entity', surface,
                        entity_label=label, category_override='entity',
                        dep=dep, pos=pos))
                else:
                    # unknown entity -> honest character fallback, but the
                    # context facts of the span head still apply
                    for w in surface.split('_'):
                        for ch in w:
                            result.append(self._make_token(
                                ch, 'char', w, entity_label=label,
                                dep=dep, pos=pos))
                while i < len(doc) and doc[i].i in entity_indices:
                    i += 1
                continue

            if spacy_tok.i in entity_indices:
                i += 1
                continue

            word = spacy_tok.text
            dep, pos = spacy_tok.dep_, spacy_tok.pos_
            for morpheme, role in self._segment_word(word):
                result.append(self._make_token(morpheme, role, word,
                                               dep=dep, pos=pos))
            i += 1

        return result

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
