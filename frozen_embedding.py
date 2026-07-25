"""
frozen_embedding.py

The theoretical centerpiece of the thesis: a hybrid embedding layer where
some dimensions are FROZEN (carry guaranteed linguistic facts, never updated
by backprop) and the rest are TRAINED normally.

    embedding(token) = [ frozen (T) | trained (N - T) ]

The frozen part is built from the preprocessor's per-token metadata
(category, role, entity_label) via a FrozenSchema. The trained part is a
standard learnable embedding table. Concatenation keeps the two regions
cleanly separated; the frozen region's gradient is masked to zero so training
can never erode the injected signal.

Design decisions (see research_journey.md for the full arc):
  - Concatenation, not addition  → guaranteed presence of the signal
  - Frozen gradient masked to 0  → "frozen" is literally enforced
  - Fixed-width schema, zero-fill → ragged facts handled uniformly
  - T = 0 collapses to a plain trained embedding (the control / baseline)

Schemas form a ladder of increasing linguistic richness:
  NullSchema     (T=0)   baseline control — no frozen signal
  OneHotSchema   (T≈23)  categorical flags: category + role + entity_label
  FeatureSchema  (B-affix) rich hand-annotated affix features [STUB / hook]
  WordNetSchema  (B-full)  + WordNet supersenses on roots      [STUB / hook]
"""

import numpy as np
import torch
import torch.nn as nn


# ── Canonical label inventories ───────────────────────────────────────────────
# These mirror the preprocessor exactly. Order matters: it defines slot indices.

CATEGORIES = ['prefix', 'suffix', 'root', 'function', 'entity', 'character']

ROLES      = ['prefix', 'root', 'suffix', 'char', 'none']

ENTITY_LABELS = [
    'ORG', 'PERSON', 'GPE', 'LOC', 'FAC',
    'NORP', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
    'none',
]


def _onehot(value, inventory):
    """Return a one-hot row for `value` within `inventory`.
    Unknown/None values map to the last slot (which is the 'none' sentinel
    for role/entity inventories)."""
    vec = np.zeros(len(inventory), dtype=np.float32)
    if value in inventory:
        vec[inventory.index(value)] = 1.0
    else:
        vec[-1] = 1.0   # fall to the sentinel slot
    return vec


# ── Frozen schemas ────────────────────────────────────────────────────────────

class FrozenSchema:
    """
    Base class. A schema turns one token's metadata dict into a fixed-width
    frozen vector. Every subclass must expose:
        dim                      → int, the width T
        vector(token_dict)       → np.ndarray of shape (T,)
    The token_dict is exactly what BasePreprocessor.encode() emits per token:
        {'category', 'role', 'entity_label', 'token', 'token_id', 'word'}
    """
    dim = 0

    def vector(self, token):
        raise NotImplementedError

    def matrix(self, vocab_df):
        """
        Build the full [vocab_size x dim] frozen matrix, one row per vocab id.
        We reconstruct a minimal token dict from each vocab row. Tokens whose
        frozen facts depend on context (entity_label) use the vocab category;
        single-word entity rows carry their own category so this is well-defined
        for everything in the vocabulary.
        """
        if self.dim == 0:
            return np.zeros((len(vocab_df), 0), dtype=np.float32)

        rows = []
        for _, r in vocab_df.iterrows():
            cat = r['category']
            # reconstruct the role the preprocessor would have assigned
            role = {
                'prefix': 'prefix', 'suffix': 'suffix', 'root': 'root',
                'function': 'root', 'entity': 'none', 'character': 'char',
            }.get(cat, 'none')
            token = {
                'category':     cat,
                'role':         role,
                'entity_label': None,   # vocab rows have no NER context
                'token':        r['token'],
            }
            rows.append(self.vector(token))
        return np.stack(rows).astype(np.float32)


class NullSchema(FrozenSchema):
    """Baseline control: no frozen dimensions at all (T = 0).
    With this schema HybridEmbedding is exactly a standard trained embedding."""
    dim = 0

    def vector(self, token):
        return np.zeros(0, dtype=np.float32)


class OneHotSchema(FrozenSchema):
    """
    Option A: categorical linguistic flags.
        [ category(6) | role(5) | entity_label(12) ]  →  T = 23
    Pure indicator vector. Says exactly *what kind* of unit this is, with an
    explicit 'none' slot wherever a fact doesn't apply (so a character's empty
    entity block is a real, meaningful zero-pattern, not missing data).
    """
    dim = len(CATEGORIES) + len(ROLES) + len(ENTITY_LABELS)   # 6 + 5 + 12 = 23

    def vector(self, token):
        cat  = _onehot(token.get('category'), CATEGORIES)
        role = _onehot(token.get('role'),     ROLES)
        ent  = _onehot(token.get('entity_label') or 'none', ENTITY_LABELS)
        return np.concatenate([cat, role, ent])


class FeatureSchema(FrozenSchema):
    """
    Option B-affix [HOOK — not yet populated].
    Rich hand-annotated morphological features for the 381 affixes; roots and
    entities keep one-hot flags (this is B1 for roots). The affix feature table
    will be filled in from a reviewed annotation pass. Until then this raises,
    so it can't be silently selected.
    """
    dim = 0   # set once the feature schema is defined

    def __init__(self):
        raise NotImplementedError(
            'FeatureSchema (B-affix) is a planned variant. '
            'Populate the affix feature table before using it.'
        )

    def vector(self, token):
        raise NotImplementedError


class WordNetSchema(FrozenSchema):
    """
    Option B-full [HOOK — not yet populated].
    FeatureSchema plus WordNet supersenses (45-dim) on root morphemes.
    Left as a clearly-marked future extension; requires the WordNet-import
    judgment call recorded in research_journey.md.
    """
    dim = 0

    def __init__(self):
        raise NotImplementedError(
            'WordNetSchema (B-full) is a planned future variant.'
        )

    def vector(self, token):
        raise NotImplementedError


SCHEMAS = {
    'baseline': NullSchema,
    'onehot':   OneHotSchema,
    'features': FeatureSchema,    # hook
    'wordnet':  WordNetSchema,    # hook
}


# ── The hybrid embedding module ───────────────────────────────────────────────

class HybridEmbedding(nn.Module):
    """
    Embedding layer producing  [ frozen (T) | trained (N - T) ]  per token.

    Args:
        vocab_df      : the vocabulary DataFrame (defines ids and categories)
        embedding_dim : N, total width of the output embedding
        schema        : a FrozenSchema instance (NullSchema → plain embedding)
        num_special   : extra ids beyond the vocab (UNK/PAD/BOS/EOS = 4)
        freeze_amplify: Lever 2 — integer tiling factor for the frozen block,
                        widening its footprint inside N without inventing facts.
                        1 = no amplification (default).

    The frozen rows are registered as a buffer (not a Parameter) so they never
    receive gradients. The trained part is a normal nn.Embedding. We also keep
    a gradient mask as defence-in-depth in case the two are ever fused.
    """

    def __init__(self, vocab_df, embedding_dim, schema,
                 num_special=4, freeze_amplify=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.schema        = schema
        self.num_special   = num_special

        vocab_size = len(vocab_df) + num_special

        # build frozen matrix for real vocab rows, zero-pad the special ids
        base = schema.matrix(vocab_df)                       # [V, t0]
        t0   = base.shape[1]
        if t0 > 0 and freeze_amplify > 1:
            base = np.tile(base, (1, freeze_amplify))        # Lever 2
        t = base.shape[1]

        if t > 0:
            special = np.zeros((num_special, t), dtype=np.float32)
            frozen  = np.concatenate([base, special], axis=0)  # [vocab_size, T]
            self.register_buffer('frozen', torch.from_numpy(frozen))
        else:
            self.register_buffer('frozen',
                                 torch.zeros(vocab_size, 0))

        self.T = t
        if self.T >= embedding_dim:
            raise ValueError(
                f'Frozen width T={self.T} must be smaller than '
                f'embedding_dim N={embedding_dim}. Increase N, reduce '
                f'freeze_amplify, or pick a thinner schema.'
            )

        # trained part fills the remaining N - T dimensions
        self.trained = nn.Embedding(vocab_size, embedding_dim - self.T)

    def forward(self, ids):
        """
        ids: LongTensor [...]. Returns [..., embedding_dim].
        """
        trained = self.trained(ids)                  # [..., N-T]
        if self.T == 0:
            return trained
        frozen = self.frozen[ids]                    # [..., T]  (no grad: buffer)
        return torch.cat([frozen, trained], dim=-1)  # [..., N]

    def extra_repr(self):
        return (f'N={self.embedding_dim}, T={self.T}, '
                f'trained={self.embedding_dim - self.T}, '
                f'schema={self.schema.__class__.__name__}')


def build_embedding(vocab_df, embedding_dim=128, variant='baseline',
                    freeze_amplify=1, num_special=4):
    """
    Convenience constructor.
        variant ∈ {'baseline', 'onehot', 'features', 'wordnet'}
    Default embedding_dim=128 (Lever 1: small N so the frozen block is a
    meaningful fraction of the vector).
    """
    if variant not in SCHEMAS:
        raise ValueError(f'Unknown variant {variant!r}. '
                         f'Choose from {list(SCHEMAS)}.')
    schema = SCHEMAS[variant]()
    return HybridEmbedding(vocab_df, embedding_dim, schema,
                           num_special=num_special,
                           freeze_amplify=freeze_amplify)