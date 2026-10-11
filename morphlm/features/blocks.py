"""
blocks.py — composable frozen fact blocks.

WHY THIS EXISTS
---------------
The previous design used monolithic schema classes with a fixed 23-dim layout
[category(6) | role(5) | entity_label(12)]. Measured against the real
vocabulary that layout carried:

    * 12 constant columns   (entity_label was never populated from vocab rows)
    *  5 redundant columns  (role is a deterministic function of category)
    *  6 distinct frozen vectors across all 16,757 tokens  (~2.6 bits total)

So 23 dimensions of embedding width bought one 6-way categorical, and 12 of
those dimensions were provably uninformative — a constant input dimension is
just a bias term the model learns to ignore.

The fix is structural, not a patch: a frozen vector is now a CONCATENATION OF
REQUESTED BLOCKS. A block exists only if you ask for it, so a block that cannot
be populated cannot silently occupy space. Each block declares its own
cardinality, so the frozen width T is derived, never asserted.

THE AXIS THAT MATTERS
---------------------
Blocks differ in `source`, and that difference is the experiment:

    source = 'vocab'    the fact is a function of the token id. A trained
                        embedding row could represent it already, so freezing
                        it is STRUCTURED PARAMETER TYING, not information
                        injection. (e.g. morpheme category)

    source = 'context'  the fact varies across occurrences of the same token
                        id. A static lookup table PROVABLY CANNOT represent it.
                        This is genuine information injection.
                        (e.g. dependency role, POS in context)

'Does static freezing do little while context freezing helps?' is a real,
measurable claim, and it is the one that makes the thesis title honest.
"""

from __future__ import annotations

import numpy as np

from ..registry import register_fact


# ── base ──────────────────────────────────────────────────────────────────────

class FactBlock:
    """
    One frozen fact. Subclasses declare:

        source      'vocab' | 'context'
        classes     ordered label inventory; the LAST entry is the 'none'
                    sentinel that unknown/unaligned values fall to
        code(token) -> int index into `classes`, given a pipeline token dict
    """
    name: str = 'unnamed'
    source: str = 'vocab'
    classes: tuple = ('none',)
    width: int = 0          # 0 means 'same as n_classes', i.e. a one-hot

    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def n_classes(self) -> int:
        """Size of the label inventory."""
        return len(self.classes)

    @property
    def dim(self) -> int:
        """
        Embedding dimensions this block occupies.

        CARDINALITY vs WIDTH. For small inventories the two coincide and the
        frozen representation is a one-hot: 8 dependency classes -> 8 dims.
        Root identity has ~15k classes and cannot one-hot into 16 dims, so
        `width` may be smaller than `n_classes`. When it is, the frozen
        representation becomes a FIXED RANDOM PROJECTION — every class still
        gets a distinct unchanging vector, in far fewer dimensions. That is
        Lever 3 from journey section 11, unbuilt until now.
        """
        return self.width or len(self.classes)

    @property
    def is_projection(self) -> bool:
        """True when a one-hot is impossible because width < n_classes."""
        return self.dim < self.n_classes

    def index(self, value) -> int:
        """Map a raw label to a class index, falling to the sentinel."""
        try:
            return self.classes.index(value)
        except ValueError:
            return len(self.classes) - 1

    def code(self, token: dict) -> int:
        raise NotImplementedError

    def onehot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        if idx < self.dim:
            v[idx] = 1.0
        return v

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name!r}, '
                f'source={self.source!r}, classes={self.n_classes}, '
                f'dim={self.dim})')


# ── static facts (function of token id) ───────────────────────────────────────

@register_fact('cat')
class CategoryBlock(FactBlock):
    """
    Morpheme category. STATIC: fully determined by the vocabulary row, so this
    is parameter tying rather than information injection. Kept because the
    sample-efficiency argument for rare tokens is real — a root seen 3 times
    gets a correct category flag from step 0 for free.

    Note the old `role` block is deliberately absent: role was a deterministic
    function of category (6 categories -> 5 roles), so it added width without
    adding information.
    """
    source = 'vocab'
    classes = ('prefix', 'suffix', 'root', 'function', 'entity', 'character',
               'punct', 'digit', 'infl', 'space', 'case', 'byte')

    def code(self, token: dict) -> int:
        return self.index(token.get('category'))


@register_fact('etype')
class EntityTypeBlock(FactBlock):
    """
    Named-entity type. STATIC in the sense that an entity string has one type,
    but it needs the type stored in the vocabulary — which the old
    entity_vocab.csv did not do (it had only `entity,count`), which is exactly
    why the old 12-dim entity block was constant.

    Requires a vocabulary with an `entity_label` column. Build one with:
        python -m morphlm.cli build entities
    which now captures spaCy's label alongside the surface form.
    """
    source = 'vocab'
    classes = ('ORG', 'PERSON', 'GPE', 'LOC', 'FAC', 'NORP', 'PRODUCT',
               'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'none')

    def code(self, token: dict) -> int:
        return self.index(token.get('entity_label'))


# ── context facts (vary per occurrence — the point of the redesign) ───────────

#: spaCy's ~45 dependency labels collapsed to a coarse, high-frequency set.
#: Coarse is the v1 default: each class is seen far more often, which matters
#: at d_model=128 and is far more legible in a probing evaluation.
#: Switch with frozen.dep_granularity='fine' — a one-line config change.
DEP_COARSE_MAP = {
    # subject-ish
    'nsubj': 'subject', 'nsubjpass': 'subject', 'csubj': 'subject',
    'csubjpass': 'subject', 'expl': 'subject',
    # object-ish
    'dobj': 'object', 'obj': 'object', 'iobj': 'object', 'dative': 'object',
    'pobj': 'object', 'attr': 'object', 'oprd': 'object',
    # modifiers
    'amod': 'modifier', 'advmod': 'modifier', 'nmod': 'modifier',
    'npadvmod': 'modifier', 'acomp': 'modifier', 'appos': 'modifier',
    'det': 'modifier', 'poss': 'modifier', 'nummod': 'modifier',
    'quantmod': 'modifier', 'neg': 'modifier', 'predet': 'modifier',
    'compound': 'modifier', 'prep': 'modifier', 'prt': 'modifier',
    'case': 'modifier',
    # clausal
    'ccomp': 'clausal', 'xcomp': 'clausal', 'advcl': 'clausal',
    'relcl': 'clausal', 'acl': 'clausal', 'pcomp': 'clausal',
    'conj': 'clausal', 'parataxis': 'clausal',
    # function/structural
    'aux': 'function', 'auxpass': 'function', 'cc': 'function',
    'mark': 'function', 'punct': 'function', 'dep': 'function',
    'intj': 'function', 'meta': 'function',
    # head
    'ROOT': 'root',
}

DEP_COARSE_CLASSES = ('subject', 'object', 'modifier', 'clausal',
                      'function', 'root', 'other', 'none')

#: Universal POS tags — already coarse, used as-is.
POS_CLASSES = ('NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV', 'PRON', 'DET',
               'ADP', 'NUM', 'CONJ', 'CCONJ', 'SCONJ', 'PART', 'INTJ',
               'PUNCT', 'SYM', 'X', 'none')


@register_fact('dep')
class DependencyBlock(FactBlock):
    """
    Dependency role of the word this token came from.

    THE headline context fact: whether a word is a subject or an object is a
    property of the word IN THIS SENTENCE, not of the word type. The same token
    id gets different codes in different occurrences, so no static embedding
    table — however well trained — can represent it. This is the fact that
    turns freezing from parameter tying into information injection.

    Cardinality depends on frozen.dep_granularity, so the fine-grained version
    is a config switch rather than a code change.
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        gran = getattr(getattr(cfg, 'frozen', None), 'dep_granularity', 'coarse')
        if gran == 'fine':
            labels = sorted(set(DEP_COARSE_MAP.keys()))
            self.classes = tuple(labels) + ('other', 'none')
            self._coarse = False
        elif gran == 'coarse':
            self.classes = DEP_COARSE_CLASSES
            self._coarse = True
        else:
            raise ValueError(f'dep_granularity must be coarse|fine, got {gran!r}')

    def code(self, token: dict) -> int:
        raw = token.get('dep')
        if raw is None:
            return self.index('none')
        if self._coarse:
            return self.index(DEP_COARSE_MAP.get(raw, 'other'))
        return self.index(raw)


@register_fact('pos')
class PosBlock(FactBlock):
    """
    Coarse part of speech of the source word, in context. Context-dependent for
    genuinely ambiguous types ('run' as NOUN vs VERB), which a static table
    cannot distinguish.
    """
    source = 'context'
    classes = POS_CLASSES
    # Width 8, like every other word-class fact (dep, mstruct, mfeat, magr):
    # feature dims are carved out of d_model, so a 19-dim pos would leave the
    # model 11 fewer token dims — ~190k fewer parameters at 17k vocabulary —
    # and confound its comparison with mstruct (content vs capacity). Nineteen
    # classes fit in an 8-dim learned table. No recorded run used pos as an
    # input before this was set (journey §33.5).
    width = 8

    def code(self, token: dict) -> int:
        return self.index(token.get('pos'))


@register_fact('ectx')
class EntityContextBlock(FactBlock):
    """
    Entity type as resolved BY THE PARSER IN THIS SENTENCE, rather than looked
    up from the vocabulary. Distinct from 'etype': this one is genuinely
    context-sensitive (spaCy can label the same surface string differently in
    different sentences) and is available even for entities absent from the
    entity vocabulary.
    """
    source = 'context'
    classes = ('ORG', 'PERSON', 'GPE', 'LOC', 'FAC', 'NORP', 'PRODUCT',
               'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'none')

    def code(self, token: dict) -> int:
        return self.index(token.get('entity_label'))


# ── morphological facts (attachable to ANY tokenizer, incl. BPE) ─────────────

@register_fact('mstruct')
class MorphStructBlock(FactBlock):
    """
    Morphological structure of the word a token belongs to: simple, prefixed,
    suffixed, both, compound, entity, unanalysed.

    CONTEXT-sourced even though it is a property of the word type, because for
    BPE the same token id appears inside different words — `ing` sits in
    `running` (suffixed) and in `thing` (simple). A static id->fact table
    therefore cannot express it.
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        from .morph_analysis import STRUCT_CLASSES
        self.classes = STRUCT_CLASSES

    def code(self, token: dict) -> int:
        return self.index(token.get('mstruct'))


@register_fact('minfl')
class MorphInflectionBlock(FactBlock):
    """
    INFLECTION of the word a token belongs to: plural, third-singular, past,
    past participle, progressive, comparative, superlative, possessive, or
    none. Nine classes.

    This is the other half of morphological structure. `mstruct` describes how
    a word is BUILT (derivation: prefixed, suffixed, compound); `minfl`
    describes how it is INFLECTED (Number=Plur, Tense=Past, ...). The thesis
    is titled "context-aware morphological embeddings", and until G-28 the
    inflection of the context had never been used as a feature — spaCy's
    morphologizer computes it on every parse, and the morpheme tokenizer's own
    inflection tokens (features/inflection.py) classify it, but the BPE model
    never saw it.

    Under `causal_identity` this supplies the PREVIOUS word's inflection:
    after a plural noun, after a past participle, after a comparative. Some of
    that is visible in the previous word's BPE fragments (`Ġcat` `s`); much of
    it is not — irregulars (`children`, `ran`, `taken`), and the ambiguity of
    `-s` between plural and third-singular.

    CONTEXT-sourced for the same reason as `mstruct`: the same BPE token id
    sits in differently inflected words.
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        from .inflection import INFLECTION_TOKENS
        self.classes = tuple(t.strip('<>') for t in INFLECTION_TOKENS) + ('none',)

    width = 8                   # the shared word-class budget (see PosBlock)

    def code(self, token: dict) -> int:
        v = token.get('minfl')
        return self.index(v.strip('<>') if v else None)


@register_fact('magr')
class MorphAgreementBlock(FactBlock):
    """
    AGREEMENT features of the word a token belongs to — UD Number and Person:
    Sing, Plur, 1Sing, 1Plur, 2, 3Sing, 3Plur, or none. Eight classes.

    `minfl` carries the inflectional morphemes a tokenizer would mark; `magr`
    carries what the NEXT word has to agree with. They overlap on nouns and
    finite verbs, and differ where it matters for agreement: determiners
    (`these` is Plur and has no inflectional morpheme), pronouns (`they`,
    `you`), and singular nouns (`minfl` marks only the plural). Under
    `causal_identity` it supplies the previous word's agreement class — at
    `sketch` in "these sketch", that is `Plur`.

    Both come from spaCy's morphological features on the parse the pipeline
    already runs (journey §32, G-28).
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        from .inflection import AGREEMENT_CLASSES
        self.classes = AGREEMENT_CLASSES + ('none',)

    def code(self, token: dict) -> int:
        return self.index(token.get('magr'))


#: Every morphological feature bundle en_core_web_sm 3.8 can assign to an
#: alphabetic word — read from its attribute_ruler table (177 patterns, 77
#: bundles, 70 once punctuation is excluded). English morph is set entirely by
#: that table from the tagger's fine-grained tag, so the inventory is CLOSED:
#: no hashing, no collisions. A bundle outside it (another spaCy version) maps
#: to 'other' rather than silently aliasing a real class.
MORPH_BUNDLES = (
    'Aspect=Perf|Tense=Past|VerbForm=Part',
    'Aspect=Prog|Tense=Pres|VerbForm=Part',
    'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs',
    'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs',
    'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs',
    'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Plur|Person=1|PronType=Prs',
    'Case=Acc|Number=Plur|Person=1|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Plur|Person=3|PronType=Prs',
    'Case=Acc|Number=Plur|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Sing|Person=1|PronType=Prs',
    'Case=Acc|Number=Sing|Person=1|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Person=2|PronType=Prs',
    'Case=Acc|Person=2|PronType=Prs|Reflex=Yes',
    'Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs',
    'Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs',
    'Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs',
    'Case=Nom|Number=Plur|Person=1|PronType=Prs',
    'Case=Nom|Number=Plur|Person=3|PronType=Prs',
    'Case=Nom|Number=Sing|Person=1|PronType=Prs',
    'Case=Nom|Person=2|PronType=Prs',
    'ConjType=Cmp',
    'Definite=Def|PronType=Art',
    'Definite=Ind|PronType=Art',
    'Degree=Cmp',
    'Degree=Pos',
    'Degree=Sup',
    'Foreign=Yes',
    'Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs',
    'Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs',
    'Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs',
    'Gender=Neut|Number=Sing|Person=3|PronType=Prs',
    'Hyph=Yes',
    'Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin',
    'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'Mood=Ind|Tense=Past|VerbForm=Fin',
    'Mood=Ind|Tense=Pres|VerbForm=Fin',
    'NumType=Card',
    'NumType=Mult',
    'NumType=Ord',
    'Number=Plur',
    'Number=Plur|Person=1|Poss=Yes|PronType=Prs',
    'Number=Plur|Person=2|Poss=Yes|PronType=Prs',
    'Number=Plur|Person=3|Poss=Yes|PronType=Prs',
    'Number=Plur|PronType=Dem',
    'Number=Sing',
    'Number=Sing|Person=1|Poss=Yes|PronType=Prs',
    'Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'Number=Sing|PronType=Dem',
    'Number=Sing|PronType=Ind',
    'Person=2|Poss=Yes|PronType=Prs',
    'Person=2|PronType=Prs',
    'Polarity=Neg',
    'Poss=Yes',
    'Poss=Yes|PronType=Prs',
    'PronType=Dem',
    'PronType=Prs',
    'PronType=Rel',
    'Tense=Past|VerbForm=Fin',
    'Tense=Past|VerbForm=Part',
    'Tense=Pres|VerbForm=Fin',
    'VerbForm=Fin',
    'VerbForm=Ger',
    'VerbForm=Inf',
    'VerbType=Mod',
    '_',
)


@register_fact('mfeat')
class MorphFeatureBlock(FactBlock):
    """
    The full INFLECTIONAL feature bundle of the word, as Universal Dependencies
    features: `Number=Plur`, `Case=Nom|Number=Plur|Person=3|PronType=Prs`,
    `Mood=Ind|Tense=Past|VerbForm=Fin`, ...  71 classes plus 'none'.

    Why not `minfl`: the nine-class classifier built for the morpheme tokenizer
    returns nothing for determiners and pronouns, yet those carry exactly the
    cues agreement turns on — `this`/`these` differ only in Number, `he`/`they`
    in Number and Person. `mfeat` keeps them.

    CONTEXTUAL, and therefore not strictly causal even when shifted. English
    morph comes from the TAGGER, whose receptive field extends about four
    tokens to the right: `reading` is VBG in "reading books" but NN in "reading
    is fun". Shifted to the previous word, the bundle can still carry a little
    of the word being predicted. `mfeatlex` is the strictly causal variant,
    and `morph_causality_check.py` measures the difference.

    Width 8, the same budget as `mstruct`, so the two compare like for like.
    """
    source = 'context'
    width = 8

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.classes = MORPH_BUNDLES + ('other', 'none')

    def code(self, token: dict) -> int:
        v = token.get('mfeat')
        if v is None:
            return len(self.classes) - 1
        try:
            return self.classes.index(v)
        except ValueError:
            return len(self.classes) - 2          # 'other'


@register_fact('mfeatlex')
class MorphFeatureLexicalBlock(MorphFeatureBlock):
    """
    STRICTLY CAUSAL `mfeat`: each word's MOST FREQUENT bundle in the training
    corpus, looked up from the word alone — the way `mstruct` is looked up
    from MorphoLEX. No tagger context, so after the causal shift it depends
    only on words already written. It loses the tagger's disambiguation
    (`reading` always gets its majority reading); what it keeps is leak-free.

    Derived from cached `mfeat` codes by scripts/data/derive_lexical_codes.py,
    which also writes the lexicon that `codes_for` uses on new text (BLiMP).
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._cfg = cfg
        self._lex = None

    def code(self, token: dict) -> int:
        raise RuntimeError(
            'mfeatlex is derived per WORD, not per token: run '
            'scripts/data/derive_lexical_codes.py after the cache holds mfeat. '
            'Building it token by token would fill the cache with empty codes.')

    def codes_for(self, ids, tokens):
        """Codes for a new token sequence, via the saved lexicon."""
        from .lexical import load_lexicon, lexical_codes, tokenizer_tables, lexicon_condition
        if self._lex is None:
            data = getattr(self._cfg, 'data', None)
            path = getattr(data, 'bpe_path', None)
            self._tables = tokenizer_tables(path)
            self._lex = load_lexicon(path, 'mfeat',
                                     getattr(data, 'cache_dir', 'data/cache'),
                                     condition=lexicon_condition(getattr(data, 'corpus', None), getattr(data, 'corpus_fraction', 1.0)))
        ctx = [MorphFeatureBlock.code(self, t) for t in tokens]
        return lexical_codes(ids, ctx, self._lex, self._tables,
                             none_code=len(self.classes) - 1).tolist()


@register_fact('poslex')
class PosLexicalBlock(PosBlock):
    """
    STRICTLY CAUSAL `pos`: each word's most frequent universal POS tag in the
    training corpus, looked up from the word alone. The tagger's `pos` depends
    on words to its right for ~7% of words (journey §33.6), so shifted `pos`
    can still carry a little of the word being predicted; this cannot. Width 8
    like `pos`, so the two compare like for like.

    Derived from cached `pos` codes: scripts/data/derive_lexical_codes.py --fact pos
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._cfg = cfg
        self._lex = None

    def code(self, token: dict) -> int:
        raise RuntimeError(
            'poslex is derived per WORD, not per token: run '
            'scripts/data/derive_lexical_codes.py --fact pos after the cache holds pos.')

    def codes_for(self, ids, tokens):
        from .lexical import load_lexicon, lexical_codes, tokenizer_tables, lexicon_condition
        if self._lex is None:
            data = getattr(self._cfg, 'data', None)
            path = getattr(data, 'bpe_path', None)
            self._tables = tokenizer_tables(path)
            self._lex = load_lexicon(path, 'pos',
                                     getattr(data, 'cache_dir', 'data/cache'),
                                     condition=lexicon_condition(getattr(data, 'corpus', None), getattr(data, 'corpus_fraction', 1.0)))
        return lexical_codes(ids, None, self._lex, self._tables,
                             none_code=len(self.classes) - 1).tolist()


@register_fact('mstructshuf')
class MorphStructShuffledBlock(MorphStructBlock):
    """
    CONTROL for `mstruct`: its eight classes reassigned at random among word
    types of similar frequency (lexical.shuffle_within_frequency). Class sizes
    and the class-frequency profile are kept; the linguistic content — which
    word is prefixed, suffixed, compound, a name — is destroyed.

    If this helps as much as `mstruct`, the gain comes from ANY consistent
    coarse grouping of word types, not from morphology (journey §33.8).

    Derived: scripts/data/derive_lexical_codes.py --fact mstruct
             --out-fact mstructshuf --shuffle-types 0 --keep mstruct
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._cfg = cfg
        self._lex = None

    def code(self, token: dict) -> int:
        raise RuntimeError(
            'mstructshuf is derived per WORD: run scripts/data/derive_lexical_codes.py '
            '--fact mstruct --out-fact mstructshuf --shuffle-types 0')

    def codes_for(self, ids, tokens):
        from .lexical import load_lexicon, lexical_codes, tokenizer_tables, lexicon_condition
        if self._lex is None:
            data = getattr(self._cfg, 'data', None)
            path = getattr(data, 'bpe_path', None)
            self._tables = tokenizer_tables(path)
            self._lex = load_lexicon(path, 'mstructshuf',
                                     getattr(data, 'cache_dir', 'data/cache'),
                                     condition=lexicon_condition(getattr(data, 'corpus', None), getattr(data, 'corpus_fraction', 1.0)))
        return lexical_codes(ids, None, self._lex, self._tables,
                             none_code=len(self.classes) - 1).tolist()


class LexicalLookupMixin:
    """
    For facts DERIVED per word type by scripts/data/derive_lexical_codes.py: they
    refuse per-token coding (which would fill a cache with empty codes) and
    code new text through the lexicon the derivation saved under
    `lexicon_name`. Put it first in the bases so these two methods win.
    """
    lexicon_name = None

    def _lex_init(self, cfg):
        self._cfg = cfg
        self._lex = None

    def code(self, token: dict) -> int:
        raise RuntimeError(
            f'{self.lexicon_name} is derived per WORD, not per token: run '
            f'scripts/data/derive_lexical_codes.py with --out-fact {self.lexicon_name}')

    def codes_for(self, ids, tokens):
        from .lexical import load_lexicon, lexical_codes, tokenizer_tables, lexicon_condition
        if self._lex is None:
            data = getattr(self._cfg, 'data', None)
            path = getattr(data, 'bpe_path', None)
            self._tables = tokenizer_tables(path)
            self._lex = load_lexicon(path, self.lexicon_name,
                                     getattr(data, 'cache_dir', 'data/cache'),
                                     condition=lexicon_condition(getattr(data, 'corpus', None), getattr(data, 'corpus_fraction', 1.0)))
        return lexical_codes(ids, None, self._lex, self._tables,
                             none_code=len(self.classes) - 1).tolist()


@register_fact('mconst')
class MorphConstBlock(LexicalLookupMixin, MorphStructBlock):
    """
    THE NULL CONTROL: `mstruct`'s architecture — same 8-class table, same
    8 dims, same causal shift — with every word given the same class. It says
    only whether the previous token belonged to a word, which the model sees
    anyway. A gain here is the feature SLOT, not information (journey §33.9).
    Derived: derive_lexical_codes.py --fact mstruct --out-fact mconst
             --transform const --keep mstruct
    """
    lexicon_name = 'mconst'

    def __init__(self, cfg=None):
        MorphStructBlock.__init__(self, cfg)
        self._lex_init(cfg)

    def codes_for(self, ids, tokens):
        """New text (BLiMP): class 0 on every word, `none` elsewhere — no lexicon,
        so a word unseen in train is not marked as a non-word (G-73)."""
        from .lexical import tokenizer_tables, word_spans
        if getattr(self, '_tables', None) is None:
            data = getattr(self._cfg, 'data', None)
            self._tables = tokenizer_tables(getattr(data, 'bpe_path', None))
        wid, _, _ = word_spans(np.asarray(ids, dtype=np.int64), self._tables)
        return np.where(wid >= 0, 0, len(self.classes) - 1).tolist()


@register_fact('mconst16')
class MorphConst16Block(MorphConstBlock):
    """The null at 16 dims (token embedding 112). Width dose-response, §33.10."""
    lexicon_name = 'mconst16'
    width = 16


@register_fact('mconst32')
class MorphConst32Block(MorphConstBlock):
    """The null at 32 dims (token embedding 96). Width dose-response, §33.10."""
    lexicon_name = 'mconst32'
    width = 32


@register_fact('mconst64')
class MorphConst64Block(MorphConstBlock):
    """The null at 64 dims (token embedding 64). Width dose-response, §33.10."""
    lexicon_name = 'mconst64'
    width = 64


@register_fact('mzero')
class MorphZeroBlock(MorphConstBlock):
    """
    ZERO-SLOT control: `mconst`'s 8 dims held at exactly zero by the embedding
    (`zero_slot`). No learned vector and no word/non-word signal: the only
    difference from the baseline is a token embedding of 120 dims instead of
    128. If this gains like the null, the lever is simply input dims free of
    token identity (journey §33.11).
    Derived (the codes are ignored): derive_lexical_codes.py --fact mstruct
             --out-fact mzero --transform const --keep mstruct
    """
    lexicon_name = 'mzero'
    zero_slot = True


@register_fact('mzeronopos')
class MorphZeroNoPosBlock(MorphZeroBlock):
    """
    `mzero` with NO position in its slot: the slot's 8 dims are exactly zero
    at the input — no token, no position. If the zero slot's gain survives,
    the free dims help as workspace; if it vanishes, what helped was a CLEAN
    POSITION CHANNEL (journey §33.12).
    Derived (codes ignored): --out-fact mzeronopos --transform const
    """
    lexicon_name = 'mzeronopos'
    slot_position = 'none'


@register_fact('mzeroposonly')
class MorphZeroPosOnlyBlock(MorphZeroBlock):
    """
    `mzero` with position ONLY in its slot: the token dims carry pure token
    identity, the slot's 8 dims pure position — a fully disentangled input.
    If a clean position channel is the mechanism, this does at least as well
    as `mzero` (journey §33.12).
    Derived (codes ignored): --out-fact mzeroposonly --transform const
    """
    lexicon_name = 'mzeroposonly'
    slot_position = 'only'


@register_fact('mflat')
class MorphFlatBlock(MorphConstBlock):
    """
    FLAT control: one learned vector at EVERY position, word or punctuation.
    `mconst` still marks whether the previous unit was a word (`simple`) or
    not (`none`); this removes even that. If it gains like the null, that
    boundary signal is not the ingredient (journey §33.11).
    Derived: derive_lexical_codes.py --fact mstruct --out-fact mflat
             --transform flat --keep mstruct
    """
    lexicon_name = 'mflat'

    def codes_for(self, ids, tokens):
        return [0] * len(ids)


@register_fact('mfreq')
class FrequencyBandBlock(LexicalLookupMixin, FactBlock):
    """
    FREQUENCY CONTROL: the previous word's frequency band, seven bands of equal
    token mass plus 'none' — the same table size as `mstruct`. The only
    information is how common the word is (journey §33.9).
    Derived: derive_lexical_codes.py --fact mstruct --out-fact mfreq
             --transform freq --keep mstruct
    """
    source = 'context'
    lexicon_name = 'mfreq'

    def __init__(self, cfg=None):
        FactBlock.__init__(self, cfg)
        self.classes = tuple(f'f{i}' for i in range(7)) + ('none',)
        self._lex_init(cfg)


@register_fact('posshuf')
class PosShuffledBlock(LexicalLookupMixin, PosBlock):
    """
    CONTROL for `pos`: its classes reassigned among word types of similar
    frequency, as `mstructshuf` is for `mstruct` (journey §33.9).
    Derived: derive_lexical_codes.py --fact pos --out-fact posshuf
             --shuffle-types 0 --keep pos
    """
    lexicon_name = 'posshuf'

    def __init__(self, cfg=None):
        PosBlock.__init__(self, cfg)
        self._lex_init(cfg)


class UDFeatureBlock(FactBlock):
    """
    One Universal Dependencies feature of the word, by name, from spaCy's
    morphologizer — language-neutral (docs/reference/ukrainian_pipeline.md). Every inventory
    is PADDED to 8 classes, the shape of `mstruct` and the nulls, so each
    feature cell starts from the same weights as its null at the same seed
    (journey §33.15). A value outside the inventory — or a multi-valued one
    such as `Acc,Gen` — maps to 'none': these are closed UD inventories, and
    the causality and preflight checks count how often that happens.
    CONTEXTUAL: the morphologizer reads the right context (§33.3).
    """
    source = 'context'
    feature = None
    values = ()

    def __init__(self, cfg=None):
        super().__init__(cfg)
        pad = tuple(f'pad{i}' for i in range(7 - len(self.values)))
        self.classes = tuple(self.values) + pad + ('none',)
        assert len(self.classes) == 8, (self.feature, len(self.classes))

    def code(self, token: dict) -> int:
        v = (token.get('udfeats') or {}).get(self.feature)
        try:
            return self.classes.index(v) if v in self.values else len(self.classes) - 1
        except ValueError:
            return len(self.classes) - 1


def _ud_block(name, feature, values):
    cls = type(f'UD{feature}Block', (UDFeatureBlock,),
               {'feature': feature, 'values': tuple(values),
                '__doc__': f'The UD feature {feature} ({", ".join(values)}); see UDFeatureBlock.'})
    return register_fact(name)(cls)


UCaseBlock = _ud_block('ucase', 'Case', ('Nom', 'Gen', 'Dat', 'Acc', 'Ins', 'Loc', 'Voc'))
_ud_block('ugender', 'Gender', ('Masc', 'Fem', 'Neut'))
_ud_block('unumber', 'Number', ('Sing', 'Plur'))
_ud_block('uperson', 'Person', ('0', '1', '2', '3'))
_ud_block('utense', 'Tense', ('Past', 'Pres', 'Fut'))
_ud_block('uaspect', 'Aspect', ('Imp', 'Perf'))
_ud_block('uanimacy', 'Animacy', ('Anim', 'Inan'))


@register_fact('ucaselex')
class UCaseLexicalBlock(LexicalLookupMixin, UCaseBlock):
    """
    STRICTLY CAUSAL case: each word form's most frequent case in the training
    corpus, looked up from the form alone. The tagger's case reads the right
    context — a preposition's case is set by the noun after it (на + Acc for a
    direction, на + Loc for a place), and that noun is exactly the word the
    shifted feature is read before. Same 8 classes as `ucase`, so the two share
    initial weights at the same seed.
    Derived: derive_lexical_codes.py --fact ucase --out-fact ucaselex --corpus ukwiki
    """
    lexicon_name = 'ucase'      # a majority lexicon is saved under its source fact

    def __init__(self, cfg=None):
        UCaseBlock.__init__(self, cfg)
        self._lex_init(cfg)

    def code(self, token: dict) -> int:
        raise RuntimeError(
            'ucaselex is derived per WORD, not per token: run '
            'scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaselex --corpus ukwiki')


@register_fact('ucaseprep')
class UCasePrepositionFormBlock(UCaseBlock):
    """
    CONTROL for the tagger's right-context leak in `ucase` (G-38; journey §33.73):
    case in context everywhere except at prepositions — the word types whose
    majority part of speech in the training corpus is ADP — which carry their word
    form's majority case at every occurrence. A preposition's case in context is
    the case of the noun after it (на + Acc for a direction, на + Loc for a place),
    which the tagger reads from that very noun: the word the shifted feature is
    read before. Recoded, a preposition says only what its form says; every other
    word keeps its case in context. Same 8 classes as `ucase`, so the two share
    initial weights at the same seed.
    Derived: derive_lexical_codes.py --fact ucase --out-fact ucaseprep --recode-pos ADP
             --keep ucase --corpus ukwiki
    """

    def code(self, token: dict) -> int:
        raise RuntimeError(
            'ucaseprep is derived per WORD, not per token: run '
            'scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaseprep '
            '--recode-pos ADP --keep ucase --corpus ukwiki')


@register_fact('mrole')
class MorphRoleBlock(FactBlock):
    """
    Which region of its word a token covers: prefix, root, suffix or whole.
    Tells a BPE model that `uring` is suffix material while `fig` is root
    material — a distinction its frequency-driven merges never encode.
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        from .morph_analysis import ROLE_CLASSES
        self.classes = ROLE_CLASSES

    def code(self, token: dict) -> int:
        return self.index(token.get('mrole'))


@register_fact('mroot')
class MorphRootBlock(FactBlock):
    """
    Lemma-normalised ROOT of the word a token belongs to. The strongest
    morphological fact available and the one BPE structurally cannot have:
    its merges are frequency-driven, so `running`, `runner` and `ran` are
    unrelated symbols to it.

    ~15k classes, so one-hot is impossible; width comes from
    `frozen.root_width` (default 16). Frozen mode uses a fixed random
    projection; learned mode uses an nn.Embedding of the same shape.

    Unlike `dep` this fact has NO LOOKAHEAD — the root of a word depends only
    on that word, so it is causally available during generation.
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.width = int(getattr(getattr(cfg, 'frozen', None),
                                 'root_width', 16) or 16)
        # COARSENING, for the identity-precision sweep.
        #
        # `root_classes` folds the inventory down modulo N at model-build time.
        # It needs no cache rebuild: the cached codes stay as they are and the
        # block reduces them on the way in. That makes a cardinality sweep
        # nearly free, which matters because the question it answers —
        # is the gain proportional to how PRECISELY a word is identified? —
        # is the mechanism behind the whole word-identity finding.
        #
        # Monotonic improvement with N means identity precision is what helps,
        # and every linguistic grouping is a lossy approximation of it. A peak
        # at some intermediate N would mean there is an optimal granularity,
        # and the interesting question becomes whether a linguistic grouping
        # sits near it.
        self.coarsen = int(getattr(getattr(cfg, 'frozen', None),
                                   'root_classes', 0) or 0)
        n = int(getattr(getattr(cfg, 'frozen', None), 'n_roots', 0) or 0)
        if not n:
            from ..vocab.build import load_vocab
            v = load_vocab(cfg.data.vocab_csv)
            # MUST match WordAnalyser.root_classes exactly: roots + entities +
            # none. A mismatch here would silently shift every cached code.
            n = (int((v['category'] == 'root').sum())
                 + int((v['category'] == 'entity').sum()) + 1)
        # placeholder labels: only the COUNT matters for sizing, and materialising
        # 15k strings on every block construction would be wasteful
        # n_full is the inventory the CACHE was built with. It must be kept,
        # because the model reads cached codes and must fold them from THIS
        # range — not from the coarsened one.
        self.n_full = n
        if self.coarsen:
            n = min(self.coarsen, n)
        self.classes = tuple(range(n))
        assert n < 32767, ('root inventory exceeds int16, which is the cache '
                           'code dtype')

    def index(self, value) -> int:
        if value is None:
            return self.n_classes - 1
        try:
            v = int(value)
        except (TypeError, ValueError):
            return self.n_classes - 1
        if self.coarsen:
            # fold the cached code into the reduced inventory; the last class
            # stays reserved as the `none` sentinel
            return v % max(self.n_classes - 1, 1)
        return v if 0 <= v < self.n_classes else self.n_classes - 1

    def code(self, token: dict) -> int:
        return self.index(token.get('mroot'))


@register_fact('mlemma')
class MorphLemmaBlock(MorphRootBlock):
    """
    Lexical identity at the INFLECTIONAL level: the linguistic middle term
    between `mroot` (merges derivational families too) and `mhash` (merges
    nothing). Same inventory, width and projection path as both, by
    construction — only the grouping principle differs.
    """

    def code(self, token: dict) -> int:
        return self.index(token.get('mlemma'))


@register_fact('mprefix')
class MorphPrefixBlock(MorphRootBlock):
    """
    CAUSAL word identity: a hash of the word's fragments SO FAR.

    `mroot`, `mlemma` and `mhash` give every fragment the code of the WHOLE
    word, which names fragments the model has not produced yet — 86-97% of
    their gain sat on exactly those positions (journey §30).

    `causal_shift` fixed that by handing over the PREVIOUS word's code, but it
    is conservative: at a word's final fragment the whole word has been seen,
    yet the model still receives the word before it, because it cannot know
    that fragment is the last.

    A prefix hash sidesteps that. It asks "what is the current word so far?",
    which is always answerable, and at a word's final fragment "so far" IS the
    whole word. The model never needs to know the word has ended — it simply
    receives the prefix, which happens to be complete.

        recon            -> hash("recon")
        recon fig        -> hash("reconfig")
        recon fig uring  -> hash("reconfiguring")   <- the true word identity

    Same inventory and width as the other identity facts, so the comparison
    isolates causality and nothing else.

    Codes are DERIVED from the cached ids by scripts/data/derive_prefix_codes.py,
    not produced by the pipeline: they depend only on the token stream, so
    re-running spaCy would be wasted work.
    """

    def code(self, token: dict) -> int:
        raise RuntimeError(
            'mprefix codes are derived from the cached token stream, not built '
            'by the pipeline. Run scripts/data/derive_prefix_codes.py first, or the '
            'cache would be filled with empty codes.')


@register_fact('mhash')
class MorphHashBlock(MorphRootBlock):
    """
    CONTROL for `mroot`: same cardinality, same per-word consistency, random
    grouping instead of morphological grouping. See WordAnalyser.hash_code.

    Subclasses MorphRootBlock so width, inventory size and the projection path
    are identical by construction — the two cells must differ in exactly one
    thing, and inheriting guarantees it rather than hoping for it.
    """

    def code(self, token: dict) -> int:
        return self.index(token.get('mhash'))


# ── assembly ──────────────────────────────────────────────────────────────────

def build_blocks(cfg):
    """Instantiate the requested blocks, in the order named by the config."""
    from ..registry import get_fact
    return [get_fact(name)(cfg) for name in cfg.frozen.facts]


def frozen_width(cfg) -> int:
    """
    Total frozen width T for a config, BEFORE amplification.
    Derived from the blocks, never asserted — the old code hardcoded 23 and
    nobody noticed 17 of those dims carried nothing.
    """
    return sum(b.dim for b in build_blocks(cfg))


def total_frozen_width(cfg) -> int:
    """Frozen width after Lever-2 amplification (tiling)."""
    return frozen_width(cfg) * max(1, cfg.frozen.amplify)


def describe_blocks(cfg) -> str:
    blocks = build_blocks(cfg)
    if not blocks:
        return 'baseline (T=0, plain trained embedding)'
    parts = [f'{b.name}[{b.source}]:{b.dim}' for b in blocks]
    t = frozen_width(cfg)
    amp = max(1, cfg.frozen.amplify)
    return (f'{" + ".join(parts)} = T:{t}'
            + (f' x{amp} = {t * amp}' if amp > 1 else ''))
