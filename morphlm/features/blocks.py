"""
blocks.py — composable frozen fact blocks.

A frozen vector is a CONCATENATION OF REQUESTED BLOCKS: a block exists only if
asked for, and declares its own cardinality, so the frozen width T is derived,
never asserted. The fixed 23-dim layout it replaced carried 6 distinct vectors,
12 of its columns constant (bugs.md #3, journey §14).

Blocks differ in `source`, and that difference is the experiment (journey §15):

    source = 'vocab'    a function of the token id, which a trained embedding
                        row could represent already: freezing it is STRUCTURED
                        PARAMETER TYING (e.g. morpheme category)
    source = 'context'  varies across occurrences of one token id, which a
                        static table CANNOT represent: genuine information
                        injection (e.g. dependency role, POS in context)

The label inventories are inventories.py's; the derived facts and the controls
are controls.py's, imported at the end — every name stays importable from here.
"""

from __future__ import annotations

import numpy as np

from ..registry import register_fact
from .inventories import (DEP_COARSE_CLASSES, DEP_COARSE_MAP, ENTITY_CLASSES,  # noqa: F401
                          MORPH_BUNDLES, POS_CLASSES)


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
        Embedding dimensions this block occupies: `width`, or one per class (a
        one-hot). Root identity has ~15k classes in 16 dims, which a frozen
        table can only hold as a FIXED RANDOM PROJECTION (embedding.py; §11).
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
    is parameter tying rather than information injection — kept for rare
    tokens, which get a correct flag from step 0. There is no `role` block: role
    was a function of category (journey §14).
    """
    source = 'vocab'
    classes = ('prefix', 'suffix', 'root', 'function', 'entity', 'character',
               'punct', 'digit', 'infl', 'space', 'case', 'byte')

    def code(self, token: dict) -> int:
        return self.index(token.get('category'))


@register_fact('etype')
class EntityTypeBlock(FactBlock):
    """
    Named-entity type. STATIC: an entity string has one type, stored in the
    vocabulary's `entity_label` column (`python -m morphlm.cli build entities`
    records spaCy's label; without it the block is constant, journey §14).
    """
    source = 'vocab'
    classes = ENTITY_CLASSES

    def code(self, token: dict) -> int:
        return self.index(token.get('entity_label'))


# ── context facts (vary per occurrence — the point of the redesign) ───────────

@register_fact('dep')
class DependencyBlock(FactBlock):
    """
    Dependency role of the word this token came from — a property of the word
    IN THIS SENTENCE, so the same token id gets different codes in different
    occurrences, which no static table can represent. Coarse (8 classes) or
    fine by frozen.dep_granularity.
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
    classes = ENTITY_CLASSES

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
    none — the classifier of the morpheme tokenizer's inflection tokens
    (features/inflection.py), as a feature (G-28, journey §32.4). `mstruct` is
    how a word is BUILT; this is how it is INFLECTED. Shifted, it gives the
    previous word's inflection, much of which its BPE fragments do not show
    (`children`, `ran`, the `-s` of plural or third-singular).

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

    What the NEXT word has to agree with, where `minfl` sees no morpheme:
    determiners (`these` is Plur), pronouns (`they`, `you`), singular nouns.
    Shifted, at `sketch` in "these sketch" it gives `Plur` (journey §33.2).
    """
    source = 'context'

    def __init__(self, cfg=None):
        super().__init__(cfg)
        from .inflection import AGREEMENT_CLASSES
        self.classes = AGREEMENT_CLASSES + ('none',)

    def code(self, token: dict) -> int:
        return self.index(token.get('magr'))


@register_fact('mfeat')
class MorphFeatureBlock(FactBlock):
    """
    The full INFLECTIONAL feature bundle of the word, as Universal Dependencies
    features: `Number=Plur`, `Case=Nom|Number=Plur|Person=3|PronType=Prs`, ...
    — 71 classes plus 'none' (MORPH_BUNDLES). Unlike `minfl` it keeps what
    determiners and pronouns carry (`this`/`these`; journey §33.2). Width 8,
    the budget of `mstruct`.

    CONTEXTUAL: the tagger reads about four tokens to the right, so even
    shifted the bundle can carry a little of the word being predicted (§33.3);
    `mfeatlex` is the strictly causal variant.
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
        return self.classes.index(v) if v in self.values else len(self.classes) - 1


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
        # COARSENING (`root_classes`), for the identity-precision sweep: the
        # inventory folded modulo N, at model time from the cached codes, so a
        # cardinality sweep needs no cache rebuild (journey §29.3-§29.6)
        self.coarsen = int(getattr(getattr(cfg, 'frozen', None),
                                   'root_classes', 0) or 0)
        from ..vocab.build import load_vocab
        v = load_vocab(cfg.data.vocab_csv)
        # MUST match WordAnalyser.root_classes exactly: roots + entities +
        # none. A mismatch here would silently shift every cached code.
        n = (int((v['category'] == 'root').sum())
             + int((v['category'] == 'entity').sum()) + 1)
        # n_full: the inventory the CACHE was built with, which the model folds
        # cached codes from (embedding.fold_codes). The labels are placeholders:
        # only their count sizes the block.
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


# The derived facts and the controls (controls.py) build on the blocks above and
# register themselves when imported; their classes stay importable from here.
from . import controls as _controls  # noqa: E402


def __getattr__(name):
    try:
        return getattr(_controls, name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
