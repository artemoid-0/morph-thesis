"""
deterministic.py — Pipeline 3: morpheme-based, no Morfessor.

Segmentation priority:
    entity vocab -> function words -> MorphoLex -> prefix strip
    -> compound split -> character fallback

Every output is either linguistically grounded or honestly unknown.
"""

from __future__ import annotations

from .base import BasePreprocessor
from ..vocab.morpholex import (
    build_morpholex_lookup, extract_morpheme_sets, compound_split,
)


class DeterministicPipeline(BasePreprocessor):

    def __init__(self, vocab_df, morpholex_path,
                 spacy_model='en_core_web_sm', stamp='all', verbose=True):
        super().__init__(vocab_df, morpholex_path, spacy_model,
                         stamp=stamp, verbose=verbose)
        if verbose:
            print('  Loading MorphoLex...')
        self.morpholex = build_morpholex_lookup(morpholex_path)
        if verbose:
            print(f'  {len(self.morpholex)} words loaded')
        _, prefixes, _ = extract_morpheme_sets(morpholex_path)
        self.known_prefixes = prefixes
        self._sorted_prefixes = sorted(prefixes, key=len, reverse=True)
        if verbose:
            print('  Ready.')

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
        return [(ch, 'char') for ch in w]
