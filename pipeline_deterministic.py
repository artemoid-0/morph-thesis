"""
pipeline_deterministic.py

Pipeline 3 — morpheme-based, fully deterministic (no Morfessor).

Segmentation priority per word:
  1. Entity vocabulary (single-word named entities)
  2. Function words (closed class, atomic)
  3. MorphoLex lookup (66k words, normalized roots)
  4. Prefix stripping + MorphoLex on remainder
  5. Character fallback (a–z)

No statistical segmentation. Every output is either linguistically
grounded (from MorphoLex) or honestly unknown (characters).
"""

from common import (
    BasePreprocessor, build_morpholex_lookup,
    extract_morpheme_sets, FUNCTION_WORDS, compound_split,
)


class DeterministicPipeline(BasePreprocessor):
    """
    Pipeline 3: deterministic morpheme pipeline, no Morfessor.
    """

    def __init__(self, vocab_df, morpholex_path, spacy_model='en_core_web_sm'):
        super().__init__(vocab_df, morpholex_path, spacy_model)

        print('  Loading MorphoLex...')
        self.morpholex = build_morpholex_lookup(morpholex_path)
        print(f'  {len(self.morpholex)} words loaded')

        _, prefixes, _ = extract_morpheme_sets(morpholex_path)
        self.known_prefixes = prefixes
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

        # honest character fallback — we don't know this word
        return [(ch, 'char') for ch in w]