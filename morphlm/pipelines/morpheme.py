"""
morpheme.py — Pipeline 2: morpheme-based with Morfessor fallback.

Identical to the deterministic pipeline except for the last resort: instead of
falling straight to characters, Morfessor segments the word and ONLY known
affixes are accepted. Root-position guesses are never trusted, which is what
stops biden -> bi[prefix] + den[root] (journey section 18).
"""

from __future__ import annotations

from .base import BasePreprocessor
from ..vocab.morpholex import (
    build_morpholex_lookup, extract_morpheme_sets, compound_split,
)

VOWELS = set('aeiou')
CONS = set('bcdfghjklmnpqrstvwxyz')


def _normalize_root(root, suffix):
    """Undo the three English spelling alternations at morpheme boundaries."""
    if (len(root) >= 2 and root[-1] == root[-2]
            and root[-1] in CONS and suffix and suffix[0] in VOWELS):
        return root[:-1]
    if (root.endswith('i')
            and suffix and suffix[0] in VOWELS | {'f', 'l', 'n', 's'}):
        return root[:-1] + 'y'
    if (len(root) >= 3 and root[-1] in CONS
            and root[-2] in VOWELS and root[-3] in CONS
            and suffix and suffix[0] in VOWELS):
        return root + 'e'
    return root


def _morfessor_segment(word, model, known_prefixes, known_suffixes):
    """Affix-only acceptance: root-position segments always fall to characters."""
    segs = list(model.viterbi_segment(word)[0])
    if len(segs) <= 1:
        return [(ch, 'char') for ch in word]

    for i in range(len(segs) - 1):
        fixed = _normalize_root(segs[i], segs[i + 1])
        if fixed != segs[i]:
            segs[i] = fixed
            break

    roles = []
    found_root = False
    for seg in segs:
        if not found_root and seg in known_prefixes:
            roles.append('prefix')
        elif not found_root:
            roles.append('root')
            found_root = True
        else:
            roles.append('suffix')

    result = []
    for seg, role in zip(segs, roles):
        if role == 'prefix' and len(seg) >= 2 and seg in known_prefixes:
            result.append((seg, 'prefix'))
        elif role == 'suffix' and len(seg) >= 2 and seg in known_suffixes:
            result.append((seg, 'suffix'))
        else:
            for ch in seg:
                result.append((ch, 'char'))
    return result


class MorphemePipeline(BasePreprocessor):

    def __init__(self, vocab_df, morpholex_path, morfessor_seed_words=None,
                 spacy_model='en_core_web_sm', stamp='all',
                 surface=None, verbose=True):
        super().__init__(vocab_df, morpholex_path, spacy_model,
                         stamp=stamp, surface=surface, verbose=verbose)
        if verbose:
            print('  Loading MorphoLex...')
        self.morpholex = build_morpholex_lookup(morpholex_path)
        if verbose:
            print(f'  {len(self.morpholex)} words loaded')

        _, prefixes, suffixes = extract_morpheme_sets(morpholex_path)
        self.known_prefixes = prefixes
        self.known_suffixes = suffixes
        self._sorted_prefixes = sorted(prefixes, key=len, reverse=True)

        import morfessor
        seed = morfessor_seed_words or list(self.morpholex.keys())[:3000]
        if verbose:
            print('  Training Morfessor...')
        model = morfessor.BaselineModel()
        model.load_data([(1, w) for w in seed])
        model.train_batch()
        self.morfessor = model
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
        return _morfessor_segment(w, self.morfessor,
                                  self.known_prefixes, self.known_suffixes)
