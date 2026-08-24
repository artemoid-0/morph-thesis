"""
embedding.py — the hybrid frozen/trained embedding.

Produces  [ factor dims (T) | trained dims (N - T) ]  by CONCATENATION.
Addition was rejected because it lets the trained part learn to cancel the
injected signal, defeating the premise of guaranteed presence (journey §27).

FROZEN vs LEARNED FACTORS — the comparison this module now supports
-------------------------------------------------------------------
Factored Neural Language Models (Alexandrescu & Kirchhoff, 2006) are the direct
ancestor of this design: a word is represented as a set of features, each with
its own embedding, concatenated into the input vector. That architecture is
nineteen years old and well established.

The one thing it does NOT do is FREEZE those feature embeddings — and that is
this project's actual claim. To test it, both modes must be available and
exactly comparable, which is what the `frozen.trainable` flag provides:

    trainable=False   factor table is a fixed identity matrix (a registered
                      buffer). The factor contributes a one-hot pattern that no
                      gradient can alter. This is the thesis condition.

    trainable=True    factor table is an nn.Embedding of the same shape. The
                      model may reshape the factor space freely. This is the
                      2006 baseline.

Both index the SAME per-position codes and occupy the SAME T dimensions, so the
comparison isolates exactly one variable: whether the linguistic signal is
guaranteed or negotiable. The learned mode adds only n_classes x dim parameters
per block (64 for coarse dependency), so parameter counts stay within 0.01%.

HOW A CODE IS OBTAINED
----------------------
    source='vocab'    code = static_codes[token_id]      (function of identity)
    source='context'  code = codes[:, block_i, :]        (function of position)

The static path uses an int lookup buffer rather than a precomputed one-hot
matrix, so that frozen and learned modes share one indexing path.

WHY THE POPULATION CHECK EXISTS
-------------------------------
The previous design built a fixed [vocab_size, 23] matrix in which 12 columns
were constant and 5 were a deterministic function of another block: 23
dimensions carrying one 6-way categorical. It passed every per-token test and
failed only when the population was examined. `assert_blocks_informative` is
that check, and it raises rather than warns.
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
        self.trainable_factors = bool(getattr(cfg.frozen, 'trainable', False))
        self.factor_scale = float(getattr(cfg.frozen, 'scale', 1.0))
        self.blocks = build_blocks(cfg)
        self.static_blocks = [b for b in self.blocks if b.source == 'vocab']
        self.context_blocks = [b for b in self.blocks if b.source == 'context']
        self.vocab_size = vocab_size

        # ── static side: token id -> class code ───────────────────────────────
        for i, block in enumerate(self.static_blocks):
            if vocab_df is None:
                raise ValueError(
                    f"static fact {block.name!r} needs a vocabulary DataFrame; "
                    f"the BPE pipeline has none. Use context facts for BPE.")
            codes = np.zeros(vocab_size, dtype=np.int64)
            for idx, row in vocab_df.iterrows():
                if idx >= vocab_size:
                    break
                codes[idx] = block.code(dict(row))
            if block.is_projection:
                raise ValueError(
                    f'static fact {block.name!r} is a projection block; '
                    f'projections are only supported for context facts')
            # special ids (UNK/PAD/BOS/EOS) keep code 0; their pattern is a
            # real, meaningful position rather than missing data
            self.register_buffer(f'static_codes_{i}', torch.from_numpy(codes))

        # ── factor tables: buffer (frozen) or Embedding (learned) ─────────────
        #
        # Three cases, and the third is new:
        #   learned                      -> nn.Embedding(n_classes, dim)
        #   frozen, dim == n_classes     -> identity, i.e. a one-hot
        #   frozen, dim <  n_classes     -> FIXED RANDOM PROJECTION
        #
        # The third exists because root identity has ~15k classes and cannot
        # one-hot into 16 dims. A fixed random matrix gives every class a
        # distinct, unchanging vector with no parameters and no sparse-gradient
        # problem — journey section 27's Lever 3, unbuilt until now. Seeded so
        # the projection is reproducible across runs.
        self.factor_tables = nn.ModuleList()
        for i, block in enumerate(self.blocks):
            if self.trainable_factors:
                emb = nn.Embedding(block.n_classes, block.dim)
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
                self.factor_tables.append(emb)
            else:
                self.factor_tables.append(nn.Identity())
                if block.is_projection:
                    g = torch.Generator().manual_seed(
                        1234 + 17 * i + block.n_classes)
                    proj = torch.randn(block.n_classes, block.dim,
                                       generator=g, dtype=torch.float32)
                    # unit rows so every class enters at the same magnitude and
                    # `frozen.scale` remains the single volume control
                    proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    self.register_buffer(f'factor_eye_{i}', proj)
                else:
                    self.register_buffer(
                        f'factor_eye_{i}',
                        torch.eye(block.dim, dtype=torch.float32))

        self.static_dim = sum(b.dim for b in self.static_blocks)
        self.context_dim = sum(b.dim for b in self.context_blocks)
        base_T = self.static_dim + self.context_dim
        self.T = base_T * self.amplify

        if self.T >= self.embedding_dim:
            raise ValueError(
                f'factor width T={self.T} must be smaller than d_model='
                f'{self.embedding_dim}. Reduce facts, lower amplify, or raise '
                f'd_model.')

        self.trained = nn.Embedding(vocab_size, self.embedding_dim - self.T)

        if verbose:
            print(f'  embedding: {self.extra_repr()}')

    # ── the check the old design lacked ───────────────────────────────────────

    def assert_blocks_informative(self, strict=True):
        """
        Population-level sanity: every static block must actually USE more than
        one class across the vocabulary, and every class it declares must be
        reachable. A constant factor column is a bias term, not information.

        Context blocks depend on the corpus and are checked in tests/ and by
        scripts/collision_audit.py against real code streams.
        """
        dead = []
        for i, block in enumerate(self.static_blocks):
            codes = getattr(self, f'static_codes_{i}').cpu().numpy()
            used = set(int(c) for c in np.unique(codes))
            if len(used) <= 1:
                dead.append(f'{block.name}: only class '
                            f'{block.classes[list(used)[0]] if used else "?"}')
                continue
            # only meaningful for small inventories: a 15k-class root feature
            # will legitimately leave most classes unused in any one corpus
            if block.n_classes <= 32:
                missing = [block.classes[j] for j in range(block.n_classes)
                           if j not in used]
                if missing:
                    dead.append(f'{block.name}: unused classes {missing}')
        if dead and strict:
            raise ValueError(
                f'{len(dead)} uninformative factor block(s): {dead}. '
                f'An unused class is a constant column — a bias term, not '
                f'information. Remove the fact, or populate it (e.g. rebuild '
                f'the entity vocabulary with labels so etype has content).')
        return not dead

    # ── forward ───────────────────────────────────────────────────────────────

    def _factor(self, i, code):
        """code -> factor vector, via the frozen identity or the learned table."""
        if self.trainable_factors:
            # a learned table sets its own magnitude, so scale is not applied
            return self.factor_tables[i](code)
        eye = getattr(self, f'factor_eye_{i}')
        out = eye[code]
        if self.factor_scale != 1.0:
            out = out * self.factor_scale
        return out

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
        si = 0
        ci = 0
        for i, block in enumerate(self.blocks):
            if block.source == 'vocab':
                code = getattr(self, f'static_codes_{si}')[ids]
                si += 1
            else:
                if codes is None or codes.numel() == 0:
                    raise ValueError(
                        'context facts were configured but the batch carried '
                        'no codes. The dataset and the model disagree — check '
                        'that the cache was built with the same frozen.facts.')
                code = codes[:, ci, :].clamp(0, block.n_classes - 1)
                ci += 1
            parts.append(self._factor(i, code))

        factor = torch.cat(parts, dim=-1)
        if self.amplify > 1:
            factor = factor.repeat(1, 1, self.amplify)
        return torch.cat([factor, trained], dim=-1)

    # ── introspection ─────────────────────────────────────────────────────────

    def factor_parameters(self):
        """Trainable parameters belonging to factor tables (0 when frozen)."""
        if not self.trainable_factors:
            return 0
        return sum(p.numel() for p in self.factor_tables.parameters())

    def extra_repr(self):
        names = ', '.join(f'{b.name}[{b.source}]:{b.dim}' for b in self.blocks)
        mode = ('LEARNED' if self.trainable_factors
                else f'FROZEN@{self.factor_scale:g}')
        return (f'N={self.embedding_dim}, T={self.T} {mode} '
                f'(static {self.static_dim}, context {self.context_dim}, '
                f'x{self.amplify}), trained={self.embedding_dim - self.T}'
                + (f' | {names}' if names else ' | baseline'))
