"""
pipeline_morpheme.py

Pipeline 2 — morpheme-based with Morfessor fallback.

Segmentation priority per word:
  1. Entity vocabulary (single-word named entities)
  2. Function words (closed class, atomic)
  3. MorphoLex lookup (66k words, normalized roots)
  4. Prefix stripping + MorphoLex on remainder
  5. Morfessor segmentation — only affixes accepted, roots → characters
  6. Character fallback (a–z)
"""

import morfessor
from common import (
    BasePreprocessor, build_morpholex_lookup,
    extract_morpheme_sets, FUNCTION_WORDS, compound_split,
)

VOWELS = set('aeiou')
CONS   = set('bcdfghjklmnpqrstvwxyz')


def _normalize_root(root, suffix):
    """Undo spelling alternations at morpheme boundaries."""
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


def _morfessor_segment(word, model, known_prefixes, known_suffixes,
                       morpholex_lookup):
    """
    Segment an OOV word using Morfessor.
    Only affix segments (known prefix or suffix) are accepted.
    Root-position segments from Morfessor are never trusted — characters only.
    This prevents spurious morpheme assignments like biden -> bi[prefix]+den[root].
    """
    segs = list(model.viterbi_segment(word)[0])
    if len(segs) <= 1:
        return [(ch, 'char') for ch in word]

    # normalize spelling alternations
    for i in range(len(segs) - 1):
        fixed = _normalize_root(segs[i], segs[i + 1])
        if fixed != segs[i]:
            segs[i] = fixed
            break

    # assign roles
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

    # accept only known affixes — root-position segments → characters
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
    """
    Pipeline 2: full morpheme pipeline with Morfessor as final fallback.
    """

    def __init__(self, vocab_df, morpholex_path,
                 morfessor_seed_words=None, spacy_model='en_core_web_sm'):
        super().__init__(vocab_df, morpholex_path, spacy_model)

        print('  Loading MorphoLex...')
        self.morpholex = build_morpholex_lookup(morpholex_path)
        print(f'  {len(self.morpholex)} words loaded')

        roots, prefixes, suffixes = extract_morpheme_sets(morpholex_path)
        self.known_prefixes = prefixes
        self.known_suffixes = suffixes

        seed = morfessor_seed_words or list(self.morpholex.keys())[:3000]
        print('  Training Morfessor...')
        model = morfessor.BaselineModel()
        model.load_data([(1, w) for w in seed])
        model.train_batch()
        self.morfessor = model
        print('  Ready.')

    def _prefix_strip(self, word):
        """
        Try stripping each known prefix (longest first).
        If the remainder is in MorphoLex, return prefix + MorphoLex result.
        Returns None if no match found.
        """
        for prefix in sorted(self.known_prefixes, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                remainder = word[len(prefix):]
                if remainder in self.morpholex:
                    entry = self.morpholex[remainder]
                    return ([(prefix, 'prefix')] +
                            list(zip(entry['morphemes'], entry['roles'])))
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

        # try compound splitting before giving up to characters
        compound = compound_split(w, self.morpholex, self.function_words)
        if compound:
            return compound

        return _morfessor_segment(
            w, self.morfessor,
            self.known_prefixes, self.known_suffixes,
            self.morpholex,
        )