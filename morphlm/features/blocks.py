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

    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def dim(self) -> int:
        return len(self.classes)

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
        v[idx] = 1.0
        return v

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name!r}, '
                f'source={self.source!r}, dim={self.dim})')


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
    classes = ('prefix', 'suffix', 'root', 'function', 'entity', 'character')

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
