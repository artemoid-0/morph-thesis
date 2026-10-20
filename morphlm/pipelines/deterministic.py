"""
deterministic.py — Pipeline 3: morpheme-based, no Morfessor.

Segmentation priority:
    entity vocab -> function words -> MorphoLex -> prefix strip
    -> compound split -> last resort (`_fallback`: characters)

Every output is either linguistically grounded or honestly unknown. The
morpheme pipeline (morpheme.py) is this one with Morfessor as its last resort.
"""

from __future__ import annotations

from .base import BasePreprocessor
from ..vocab.morpholex import (
    build_morpholex_lookup, extract_morpheme_sets, compound_split,
)


class DeterministicPipeline(BasePreprocessor):

    def __init__(self, vocab_df, morpholex_path,
                 spacy_model='en_core_web_sm', surface=None, verbose=True):
        super().__init__(vocab_df, morpholex_path, spacy_model,
                         surface=surface, verbose=verbose)
        if verbose:
            print('  Loading MorphoLex...')
        self.morpholex = build_morpholex_lookup(morpholex_path)
        if verbose:
            print(f'  {len(self.morpholex)} words loaded')
        _, prefixes, suffixes = extract_morpheme_sets(morpholex_path)
        self.known_prefixes = prefixes
        self.known_suffixes = suffixes
        self._sorted_prefixes = sorted(prefixes, key=len, reverse=True)
        self._prepare_fallback(verbose)
        if verbose:
            print('  Ready.')

    def _prepare_fallback(self, verbose):
        """Whatever the last resort needs; characters need nothing."""

    def _fallback(self, w):
        """The last resort, for a lowercased word nothing else segments."""
        return [(ch, 'char') for ch in w]

    def _prefix_strip(self, word):
        for prefix in self._sorted_prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                remainder = word[len(prefix):]
                if remainder in self.morpholex:
                    entry = self.morpholex[remainder]
                    return ([(prefix, 'prefix')]
                            + list(zip(entry['morphemes'], entry['roles'])))
        return None

    def _segment_word(self, word):
        w = word.lower()
        if w in self.entity_tokens:
            return [(w, 'entity')]
        if w in self.function_words:
            return [(w, 'root')]
        if w in self.morpholex:
            e = self.morpholex[w]
            return list(zip(e['morphemes'], e['roles']))
        stripped = self._prefix_strip(w)
        if stripped:
            return stripped
        compound = compound_split(w, self.morpholex, self.function_words)
        if compound:
            return compound
        return self._fallback(w)
