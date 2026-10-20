"""
morph_analysis.py — word-level morphological facts, for attaching to ANY tokenizer.

Morphology as FEATURES rather than segmentation (journey §22.4-§23): keep the
BPE tokens, and tell the model what morphological structure the underlying word
has — BPE cannot know that `recon`, `fig` and `uring` belong to one word whose
root is `configure`. Nothing here changes the token stream.

These facts depend on the word alone, not on the sentence — but each names the
WHOLE word at every fragment of it, so they are causal only shifted to the
previous word (`frozen.causal_identity`; journey §30).

WHAT IS EXTRACTED, per word
---------------------------
    root     lexical identity: the lemma-normalised root for analysable words,
             the entity string for named entities (high cardinality, ~16k)
    struct   simple / prefixed / suffixed / both / compound / entity /
             unanalysed  (low cardinality, one-hot friendly)
    role     which region of the word a character falls in:
             prefix / root / suffix / whole

`role` needs character positions for morphemes, but MorphoLex stores NORMALISED
roots, so a morpheme need not appear literally in the surface (`beautiful` ->
`beauty` + `ful`). Prefixes and suffixes usually DO appear literally, so the
approximation used is: prefixes claim the leading characters, suffixes claim the
trailing characters, and whatever is left is the root region. This is exact for
affixes and approximate only for the root span, which is the part where
precision matters least.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache


#: Word-structure classes. 'none' covers non-words (punctuation, digits, space).
STRUCT_CLASSES = ('simple', 'prefixed', 'suffixed', 'both', 'compound',
                  'entity', 'unanalysed', 'none')

#: Which region of a word a character belongs to.
ROLE_CLASSES = ('prefix', 'root', 'suffix', 'whole', 'none')


class WordAnalyser:
    """
    Wraps the deterministic morpheme segmenter so any pipeline can ask
    'what is the morphology of this word?' without adopting its tokenization.
    Deterministic rather than Morfessor: Morfessor added nothing (det matched or
    beat morph in 41 of 41 tail evaluations, journey §8), and is not trained on
    every cache build.
    """

    def __init__(self, vocab_df, morpholex_path, spacy_model='en_core_web_sm',
                 verbose=False):
        from ..pipelines.deterministic import DeterministicPipeline
        self._pipe = DeterministicPipeline(
            vocab_df, morpholex_path, spacy_model=spacy_model, verbose=verbose)
        # Stable, deterministic lexical-identity inventory: its order MUST NOT
        # drift between cache build and model build, or cached codes would
        # decode to different roots. Entities are roots of their own (`london`
        # is its own lemma; journey §23.2); unanalysable words fall to 'none'.
        roots = sorted(str(t) for t, c in
                       zip(vocab_df['token'], vocab_df['category'])
                       if c == 'root')
        ents = sorted(str(t) for t, c in
                      zip(vocab_df['token'], vocab_df['category'])
                      if c == 'entity')
        self.root_classes = tuple(roots) + tuple(ents) + ('none',)
        self._root_index = {r: i for i, r in enumerate(self.root_classes)}
        self._none_root = len(self.root_classes) - 1

    # ── inventory, exposed so fact blocks can size themselves ────────────────

    @property
    def n_roots(self) -> int:
        return len(self.root_classes)

    def root_code(self, root) -> int:
        return self._root_index.get(root, self._none_root)

    def lemma_code(self, lemma) -> int:
        """
        LEXICAL IDENTITY AT THE INFLECTIONAL LEVEL — the linguistic middle term
        (journey §29.1). Three features share one inventory and width, and
        differ ONLY in what they merge:

            mroot   morphological families   running/runner/ran, beautiful -> beauty
            mlemma  inflectional variants    running -> run, cats -> cat
            mhash   nothing (random)         running with telephone

        Hashed into the same inventory, so cardinality, width and projection
        path are identical by construction.
        """
        if not lemma:
            return self._none_root
        h = hashlib.md5(str(lemma).lower().encode('utf-8')).hexdigest()
        return int(h[8:16], 16) % max(self._none_root, 1)

    def hash_code(self, word) -> int:
        """
        MATCHED-CARDINALITY RANDOM GROUPING — the control that separates
        morphology from word identity (journey §27.1). Each surface word hashed
        into the SAME inventory size: every fragment of a word still shares one
        code (word identity kept), while related words no longer do (morphological
        grouping destroyed). The shuffled control destroys both at once.

        md5 rather than Python's hash(), which is salted per process and would
        change between the cache build and any later run.
        """
        if not word:
            return self._none_root
        h = hashlib.md5(str(word).lower().encode('utf-8')).hexdigest()
        return int(h[:8], 16) % max(self._none_root, 1)

    # ── analysis ─────────────────────────────────────────────────────────────

    @lru_cache(maxsize=200_000)
    def _segment(self, word: str):
        """(morphemes, roles) for a lowercased word. Cached — the corpus has a
        very steep type-token curve, so the same words recur constantly."""
        try:
            pieces = list(self._pipe._segment_word(word))
        except Exception:
            return (), ()
        if not pieces:
            return (), ()
        return tuple(m for m, _ in pieces), tuple(r for _, r in pieces)

    def analyse(self, word: str):
        """
        Return (root_string_or_None, struct_class, [(start, end, role), ...])
        with character offsets RELATIVE to the word.
        """
        w = str(word).lower()
        morphemes, roles = self._segment(w)
        if not morphemes:
            return None, 'none', []

        if roles[0] == 'entity':
            # the entity string is its own lemma
            return morphemes[0], 'entity', [(0, len(w), 'whole')]
        if all(r == 'char' for r in roles):
            return None, 'unanalysed', [(0, len(w), 'whole')]

        roots = [m for m, r in zip(morphemes, roles) if r == 'root']
        n_pre = sum(1 for r in roles if r == 'prefix')
        n_suf = sum(1 for r in roles if r == 'suffix')

        if len(roots) > 1:
            struct = 'compound'
        elif n_pre and n_suf:
            struct = 'both'
        elif n_pre:
            struct = 'prefixed'
        elif n_suf:
            struct = 'suffixed'
        else:
            struct = 'simple'

        root = roots[0] if roots else None

        # character regions: affixes claim the ends, the root claims the middle
        pre_len = sum(len(m) for m, r in zip(morphemes, roles) if r == 'prefix')
        suf_len = sum(len(m) for m, r in zip(morphemes, roles) if r == 'suffix')
        n = len(w)
        pre_len = min(pre_len, n)
        suf_len = min(suf_len, max(n - pre_len, 0))
        regions = []
        if pre_len:
            regions.append((0, pre_len, 'prefix'))
        mid_end = n - suf_len
        if mid_end > pre_len:
            regions.append((pre_len, mid_end, 'root'))
        if suf_len:
            regions.append((mid_end, n, 'suffix'))
        if not regions:
            regions = [(0, n, 'whole')]
        return root, struct, regions


def build_analyser(cfg, verbose=False):
    """Construct a WordAnalyser from an ExperimentConfig."""
    from ..vocab.build import load_vocab
    vocab_df = load_vocab(cfg.data.vocab_csv)
    return WordAnalyser(vocab_df, cfg.data.morpholex_path,
                        spacy_model=cfg.data.spacy_model, verbose=verbose)
