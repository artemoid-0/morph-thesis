"""
morph_analysis.py — word-level morphological facts, for attaching to ANY tokenizer.

WHY THIS EXISTS
---------------
Q1 established that morpheme segmentation loses to BPE in English: BPE emits
~1.18 tokens for 88% of words and has already absorbed English's eight
productive inflectional affixes. Q2 established that injecting linguistic facts
into the embedding helps by 6.4%.

The synthesis is to stop using morphology for SEGMENTATION and use it for
FEATURES: keep BPE tokens, and tell the model what morphological structure the
underlying word has. BPE cannot know that `recon`, `fig` and `uring` belong to
one word whose root is `configure` — its merges are frequency-driven, not
lexical.

Nothing here changes the token stream. These are input-side codes only; the
model still predicts BPE tokens, so the losslessness machinery of section 17 is
irrelevant to this path.

NO LOOKAHEAD
------------
Unlike `dep`, which comes from a full-sentence parse and therefore leaks
information about tokens not yet predicted (section 21.6), the morphological
analysis of a word depends only on that word. A root is knowable from the
characters already emitted. So these facts are causally available in a way the
dependency role is not — which makes them a cleaner claim, and a partial answer
to whether the Q2 gain was an oracle artifact.

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

    Deliberately built on DeterministicPipeline rather than MorphemePipeline:
    section 8 found Morfessor contributes nothing beyond the deterministic
    path, and every subsequent comparison has confirmed it (det matched or beat
    morph in 41 of 41 tail evaluations). Using det also avoids training a
    Morfessor model on every cache build.
    """

    def __init__(self, vocab_df, morpholex_path, spacy_model='en_core_web_sm',
                 verbose=False):
        from ..pipelines.deterministic import DeterministicPipeline
        self._pipe = DeterministicPipeline(
            vocab_df, morpholex_path, spacy_model=spacy_model, verbose=verbose)
        # Stable, deterministic lexical-identity inventory.
        # Order MUST NOT drift between cache build and model build, or codes
        # stored in the cache would decode to different roots.
        #
        # ENTITIES ARE INCLUDED, and that is a measured decision. A preflight
        # over 300 validation lines found 39.1% of BPE tokens sitting in words
        # for which no root was returned — largely named entities, which the
        # segmenter treats as atomic and which therefore fell to the 'none'
        # sentinel. But an entity's lexical identity IS itself: `london` is its
        # own lemma. Excluding them threw away information for a large share of
        # Wikipedia tokens, so entity surface forms now occupy root classes of
        # their own. Genuinely unanalysable words still fall to 'none', which
        # is honest.
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
        LEXICAL IDENTITY AT THE INFLECTIONAL LEVEL — the linguistic middle term.

        Three features now share one 16,365-class inventory and one 16-dim
        width, differing ONLY in what they merge:

            mroot   merges morphological families    running/runner/ran, and
                                                     also beautiful -> beauty
            mlemma  merges inflectional variants     running -> run, cats -> cat,
                                                     but beautiful stays put
            mhash   merges nothing (random)          running with telephone

        Section 27.3 found mroot loses to mhash because it collapses
        DISTRIBUTIONALLY SIMILAR words that context cannot separate. Lemma
        collapses strictly less than root — it keeps derivational families
        apart — so if that mechanism is right, lemma should sit above root.
        Whether it also beats random grouping is the open question, and it is
        the difference between a thesis whose best feature is linguistic and
        one whose best feature is a hash.

        Hashed into the same inventory so cardinality, width and projection
        path are identical by construction; only the grouping differs.
        """
        if not lemma:
            return self._none_root
        h = hashlib.md5(str(lemma).lower().encode('utf-8')).hexdigest()
        return int(h[8:16], 16) % max(self._none_root, 1)

    def hash_code(self, word) -> int:
        """
        MATCHED-CARDINALITY RANDOM GROUPING — the control that separates
        morphology from word identity.

        `mroot` supplies two things at once: every BPE fragment of a word gets
        the same code (WORD IDENTITY, which BPE's frequency-driven merges
        genuinely lack), and morphologically related words share a code
        (MORPHOLOGICAL RELATEDNESS). The thesis claims the second; without a
        control the experiment cannot distinguish it from the first.

        This assigns each surface word a code by hashing it into the SAME
        inventory size. Per-word consistency is preserved — every fragment of a
        word still gets one code — while morphological grouping is destroyed.

            mroot ~= mhash              -> the gain is word identity, and the
                                           morphology claim fails
            mroot > mhash > shuffled    -> morphology contributes beyond word
                                           identity, and the gap between them
                                           IS the morphological effect

        The shuffled control cannot do this job: it permutes codes within a
        window, destroying word identity and morphology together.

        md5 rather than Python's hash() because the latter is salted per process
        and would change between the cache build and any later run.
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
