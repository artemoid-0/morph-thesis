"""
morpheme.py — Pipeline 2: morpheme-based with Morfessor fallback.

The deterministic pipeline with another last resort: instead of falling
straight to characters, Morfessor segments the word and ONLY known affixes are
accepted — a segment in root position falls to characters (journey §8).

`biden` is still bi[prefix] + den[root], by prefix stripping, before any last
resort: a wrong parse kept deliberately (journey §8).
"""

from __future__ import annotations

from .deterministic import DeterministicPipeline

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


class MorphemePipeline(DeterministicPipeline):

    def __init__(self, vocab_df, morpholex_path, morfessor_seed_words=None,
                 spacy_model='en_core_web_sm', surface=None, verbose=True):
        self._seed_words = morfessor_seed_words
        super().__init__(vocab_df, morpholex_path, spacy_model=spacy_model,
                         surface=surface, verbose=verbose)

    def _prepare_fallback(self, verbose):
        """Train Morfessor on MorphoLex's first 3,000 words (or the seed words given)."""
        import morfessor
        seed = self._seed_words or list(self.morpholex.keys())[:3000]
        if verbose:
            print('  Training Morfessor...')
        model = morfessor.BaselineModel()
        model.load_data([(1, w) for w in seed])
        model.train_batch()
        self.morfessor = model

    def _fallback(self, w):
        return _morfessor_segment(w, self.morfessor,
                                  self.known_prefixes, self.known_suffixes)
