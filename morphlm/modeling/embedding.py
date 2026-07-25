"""
embedding.py — the hybrid frozen/trained embedding.

Produces  [ frozen (T) | trained (N - T) ]  by CONCATENATION. Addition was
rejected because it lets the trained part learn to cancel the frozen signal,
which defeats the entire premise of guaranteed presence (journey section 27).

The frozen part is assembled from the requested blocks:

    static block   one-hot table [vocab_size, dim], indexed by TOKEN ID
    context block  identity      [card, card],      indexed by PER-POSITION CODE

Both are registered buffers, never Parameters, so no gradient ever reaches
them. The trained part fills the remaining N - T dimensions.

WHAT CHANGED AND WHY
--------------------
The previous implementation built a single [vocab_size, 23] matrix from static
vocabulary rows. Measured on the real vocabulary that matrix had 12 constant
columns, 5 columns that were a deterministic function of another block, and
only 6 distinct rows in total. It was verified per-token (each row had exactly
3 nonzero bits, as designed) and never verified across the population, which is
the check that fails.

Two consequences are designed in here:

  * T is DERIVED from the blocks, never asserted. You cannot request a width
    the facts do not fill.
  * Context blocks make the frozen vector a function of POSITION, not just of
    token id. That is the only way freezing can inject information a trained
    embedding table could not already represent — otherwise it is structured
    parameter tying, which is a real but much weaker inductive bias.

`assert_blocks_informative()` is the population-level check that would have
caught the old bug. It is called at build time and raises rather than warning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class HybridEmbedding(nn.Module):

    def __init__(self, cfg, vocab_size, vocab_df=None, verbose=True):
        super().__init__()
        from ..features.blocks import build_blocks

        self.embedding_dim = cfg.model.d_model
        self.amplify = max(1, cfg.frozen.amplify)
        self.blocks = build_blocks(cfg)
        self.static_blocks = [b for b in self.blocks if b.source == 'vocab']
        self.context_blocks = [b for b in self.blocks if b.source == 'context']
        self.vocab_size = vocab_size

        # ── static side: one-hot lookup tables indexed by token id ────────────
        static_tables = []
        for block in self.static_blocks:
            if vocab_df is None:
                raise ValueError(
                    f"static fact {block.name!r} needs a vocabulary DataFrame; "
                    f"the BPE pipeline has none. Use context facts for BPE.")
            table = np.zeros((vocab_size, block.dim), dtype=np.float32)
            for idx, row in vocab_df.iterrows():
                if idx >= vocab_size:
                    break
                table[idx] = block.onehot(block.code(dict(row)))
            # special ids (UNK/PAD/BOS/EOS) keep an all-zero row, which is a
            # meaningful pattern rather than missing data
            static_tables.append(table)

        if static_tables:
            self.register_buffer('static_table',
                                 torch.from_numpy(np.concatenate(
                                     static_tables, axis=1)))
            self.static_dim = self.static_table.shape[1]
        else:
            self.register_buffer('static_table',
                                 torch.zeros(vocab_size, 0))
            self.static_dim = 0

        # ── context side: identity tables indexed by per-position codes ───────
        self.context_dims = [b.dim for b in self.context_blocks]
        for i, block in enumerate(self.context_blocks):
            self.register_buffer(f'ctx_eye_{i}',
                                 torch.eye(block.dim, dtype=torch.float32))
        self.context_dim = sum(self.context_dims)

        base_T = self.static_dim + self.context_dim
        self.T = base_T * self.amplify

        if self.T >= self.embedding_dim:
            raise ValueError(
                f'frozen width T={self.T} must be smaller than d_model='
                f'{self.embedding_dim}. Reduce facts, lower amplify, or raise '
                f'd_model.')

        self.trained = nn.Embedding(vocab_size, self.embedding_dim - self.T)

        if verbose:
            print(f'  embedding: {self.extra_repr()}')

    # ── the check the old design lacked ───────────────────────────────────────

    def assert_blocks_informative(self, strict=True):
        """
        Population-level sanity: every frozen COLUMN must actually vary.

        A constant input dimension is just a bias term — it consumes embedding
        width and carries nothing. The old 23-dim schema had 12 such columns and
        passed every per-token test. This is the check that catches it.

        Only the static table can be verified here; context columns depend on
        the corpus and are checked in tests/ against real code streams.
        """
        if self.static_dim == 0:
            return True
        table = self.static_table.cpu().numpy()
        dead = []
        offset = 0
        for block in self.static_blocks:
            sub = table[:, offset: offset + block.dim]
            for j in range(block.dim):
                col = sub[:, j]
                if col.min() == col.max():
                    dead.append(f'{block.name}:{block.classes[j]}')
            offset += block.dim
        if dead and strict:
            raise ValueError(
                f'{len(dead)} constant frozen column(s): {dead}. '
                f'A constant column is a bias term, not information. Remove '
                f'the fact, or populate it (e.g. rebuild the entity vocabulary '
                f'with labels so etype has content).')
        return not dead

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, ids, codes=None):
        """
        ids   : LongTensor [B, L]
        codes : LongTensor [B, n_context_facts, L] or None
        returns [B, L, d_model]
        """
        trained = self.trained(ids)
        if self.T == 0:
            return trained

        parts = []
        if self.static_dim:
            parts.append(self.static_table[ids])            # [B, L, static]

        if self.context_blocks:
            if codes is None or codes.numel() == 0:
                raise ValueError(
                    'context facts were configured but the batch carried no '
                    'codes. The dataset and the model disagree — check that '
                    'the cache was built with the same frozen.facts.')
            for i, block in enumerate(self.context_blocks):
                eye = getattr(self, f'ctx_eye_{i}')
                idx = codes[:, i, :].clamp(0, block.dim - 1)
                parts.append(eye[idx])                      # [B, L, dim]

        frozen = torch.cat(parts, dim=-1)
        if self.amplify > 1:
            frozen = frozen.repeat(1, 1, self.amplify)
        return torch.cat([frozen, trained], dim=-1)

    def extra_repr(self):
        names = ', '.join(f'{b.name}[{b.source}]:{b.dim}' for b in self.blocks)
        return (f'N={self.embedding_dim}, T={self.T} '
                f'(static {self.static_dim}, context {self.context_dim}, '
                f'x{self.amplify}), trained={self.embedding_dim - self.T}'
                + (f' | {names}' if names else ' | baseline'))
