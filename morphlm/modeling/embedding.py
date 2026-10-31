"""
embedding.py — the hybrid frozen/trained embedding.

Produces [ factor dims (T) | trained dims (N - T) ] by CONCATENATION, the
default (`frozen.inject='concat'`): journey §11 chose it because a trained part
ADDED to the signal can learn to cancel it. `inject='add'` gives each fact a
full-width learned table added to a full-width token embedding — no carved
slot, so a feature's effect over the baseline is its information alone (§33.13).

FROZEN vs LEARNED (`frozen.trainable`): a fixed identity table — the thesis
condition, a one-hot no gradient can alter — or an nn.Embedding of the same
shape, the Factored Neural Language Model baseline (Alexandrescu & Kirchhoff,
2006; journey §20, §21.2). Both index the same codes in the same T dims, so
the comparison isolates whether the signal is guaranteed or negotiable.

HOW A CODE IS OBTAINED
----------------------
    source='vocab'    code = static_codes[token_id]      (function of identity)
    source='context'  code = codes[:, block_i, :]        (function of position)

The static path uses an int lookup buffer rather than a precomputed one-hot
matrix, so that frozen and learned modes share one indexing path.

`assert_blocks_informative` refuses a constant factor column: the old 23-dim
layout had 12 and passed every per-token test (bugs.md #3, journey §14).
"""

from __future__ import annotations

import numpy as np
import os
import torch
import torch.nn as nn

from ..features.lexical import FRAGMENT_STEPS


def word_runs(is_start):
    """
    Given a word-start mask, the source index each position should read from.

    `src[i]` is the last position of the previous run; `start[i] == 0` marks
    positions with no previous run inside the window.

    Boundaries come from the tokenizer (`_word_starts`), never from where the
    codes change: that needs a fact constant across a word, and `mprefix` is
    not — the shift moved back one token and leaked (journey §30.8).
    """
    B, L = is_start.shape
    idx = torch.arange(L, device=is_start.device).unsqueeze(0).expand(B, L)
    start = torch.cummax(idx * is_start.long(), dim=1).values
    src = (start - 1).clamp(min=0)
    if not bool(((src < idx) | (start == 0)).all()):
        bad = ((src >= idx) & (start > 0)).nonzero()
        raise RuntimeError(f'word_runs read a non-past position: {bad[:3]}')
    return src, start


def causal_shift(code, src, start, sentinel):
    """
    Replace each position's code with the PREVIOUS word's code.

    A cached code names the WHOLE word containing its position, so at `recon`
    of recon/fig/uring it reveals fragments still to come — where 86-97% of the
    word-form features' gain sat (journey §30). The most recent word known to be
    complete is the one BEFORE the current one. Applied to every per-word fact:
    gated on the identity blocks alone, `mstruct` was left leaking (§30.5).
    """
    out = torch.gather(code, 1, src)
    return torch.where(start == 0, torch.full_like(out, sentinel), out)


def keep_struct(code, mode):
    """
    Collapse part of `mstruct` into its `simple` class, at model time (G-25:
    is its gain morphology, or name status? journey §31.7).

        mode='morphology'  entity, unanalysed                 -> simple
        mode='names'       prefixed, suffixed, both, compound -> simple

    Each removes one kind of information and keeps the other; `none` stays in
    both. Applied after the causal shift, so it acts on the previous word's
    class, which is what the model receives. Indices follow STRUCT_CLASSES:
        0 simple  1 prefixed  2 suffixed  3 both  4 compound
        5 entity  6 unanalysed  7 none
    """
    if mode == 'all':
        return code
    if mode == 'morphology':
        drop = (code == 5) | (code == 6)
    elif mode == 'names':
        drop = (code >= 1) & (code <= 4)
    else:
        raise ValueError(f'struct_keep must be all|morphology|names, not {mode!r}')
    return torch.where(drop, torch.zeros_like(code), code)


def fold_codes(code, block):
    """
    Fold cached codes into this block's coarsened inventory at model time, from
    the range the cache was built with (`n_full`), keeping `none` as the
    sentinel. Folding at build time with a model-side clamp put 96.9% of words
    in one class at k=512 (journey §29.4; results_invalid/coarsen_clamp).
    """
    coarsen = getattr(block, 'coarsen', 0)
    if not coarsen:
        return code.clamp(0, block.n_classes - 1)
    n_full = getattr(block, 'n_full', block.n_classes)
    sentinel = block.n_classes - 1
    is_none = code >= (n_full - 1)
    folded = torch.remainder(code, max(sentinel, 1))
    return torch.where(is_none, torch.full_like(code, sentinel), folded)


class HybridEmbedding(nn.Module):

    def __init__(self, cfg, vocab_size, vocab_df=None, verbose=True):
        super().__init__()
        from ..features.blocks import build_blocks

        self.embedding_dim = cfg.model.d_model
        self.amplify = max(1, cfg.frozen.amplify)
        self.causal_identity = bool(getattr(
            getattr(cfg, 'frozen', None), 'causal_identity', False))
        self.struct_keep = str(getattr(
            getattr(cfg, 'frozen', None), 'struct_keep', 'all'))

        self.trainable_factors = bool(getattr(cfg.frozen, 'trainable', False))
        self.factor_scale = float(getattr(cfg.frozen, 'scale', 1.0))
        self.blocks = build_blocks(cfg)
        # HOW facts enter (journey §33.13). 'concat' — every cell before this —
        # gives the facts T of the d_model dims, so the token embedding is T
        # narrower than the residual stream, and that narrowing alone improves
        # the model (§33.11–§33.12). 'add' gives each fact a full-width table
        # ADDED to a full-width token embedding: no carved slot, so a feature's
        # effect over the baseline is its information and nothing else.
        self.inject = str(getattr(getattr(cfg, 'frozen', None), 'inject', 'concat'))
        if self.inject not in ('concat', 'add'):
            raise ValueError(f'frozen.inject must be concat or add, not {self.inject!r}')
        if self.inject == 'add' and self.blocks:
            if not self.trainable_factors:
                raise ValueError('additive injection needs learned factors (frozen.trainable=true)')
            # a zero slot IS allowed here: added as zeros it leaves the
            # baseline's exact function, plus an unused table — the control for
            # everything a feature cell changes besides architecture (§33.14).
            # Positional masks need a carved slot and are refused.
            if any(getattr(b, 'slot_position', None) for b in self.blocks):
                raise ValueError('positional slot controls need a carved slot (concat)')

        # WORD BOUNDARIES for causal mode, taken from the tokenizer.
        # A token opens a word when it is alphabetic AND either carries the
        # ByteLevel space marker or follows a non-alphabetic token; a run of
        # non-alphabetic tokens opens its own unit. Same rule as
        # derive_prefix_codes and lookahead_check, so all three agree about
        # where words begin.
        if self.causal_identity:
            path = getattr(getattr(cfg, 'data', None), 'bpe_path', None)
            if not path or not os.path.exists(path):
                raise ValueError(
                    'causal_identity needs the tokenizer to locate word '
                    f'boundaries; data.bpe_path is {path!r}. Inferring them '
                    'from codes is what broke mprefix.')
            # the tokenizer's own word rule (lexical.word_rule_for) — one
            # definition shared with the lexicon, so the two cannot drift
            from ..features.lexical import tokenizer_tables
            _tb = tokenizer_tables(path)
            _al, _op = _tb
            al = torch.from_numpy(_al)
            op = torch.from_numpy(_op)
            self.register_buffer('tok_alpha', al, persistent=False)
            self.register_buffer('tok_opens', op, persistent=False)
            # fragments of split characters (G-63); none under the symbol rule
            _fr = getattr(_tb, 'frag', None)
            self.register_buffer(
                'tok_frag', torch.from_numpy(_fr) if _fr is not None and _fr.any() else None,
                persistent=False)
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
        #   learned                      -> nn.Embedding(n_classes, dim)
        #   frozen, dim == n_classes     -> identity, i.e. a one-hot
        #   frozen, dim <  n_classes     -> FIXED RANDOM PROJECTION: root identity
        #       has ~15k classes and cannot one-hot into 16 dims; a seeded random
        #       matrix gives every class a distinct, unchanging vector (§11, Lever 3)
        self.factor_tables = nn.ModuleList()
        for i, block in enumerate(self.blocks):
            if self.trainable_factors:
                emb = nn.Embedding(block.n_classes,
                                   self.embedding_dim if self.inject == 'add' else block.dim)
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

        if self.inject == 'add' and self.amplify != 1:
            raise ValueError('additive injection does not tile (amplify must be 1)')
        # the width the token embedding gives up: T in concat mode, none in add
        self.carved = self.T if self.inject == 'concat' else 0
        # The feature switch (frozen.feature_off_step, journey §33.33): a gate on
        # every feature's output, open until the training loop closes it at
        # step K. Registered only when set, so no other model or checkpoint
        # changes; persistent, so a checkpoint saved after K stays switched off.
        off = getattr(getattr(cfg, 'frozen', None), 'feature_off_step', None)
        on = getattr(getattr(cfg, 'frozen', None), 'feature_on_step', None)
        if off is not None and on is not None:
            raise ValueError('frozen.feature_off_step and frozen.feature_on_step are separate experiments; set one')
        if off is not None and self.T:
            self.register_buffer('feature_gate', torch.tensor(0.0 if off <= 0 else 1.0))
        if on is not None and self.T:
            self.register_buffer('feature_gate', torch.tensor(0.0 if on > 0 else 1.0))
        # The switch to a constant (frozen.feature_const_step, journey §33.144): from
        # step K every feature's value is its block's first class, 'none' kept, so
        # the information goes and a shared vector stays. A flag, registered only
        # when set, and persistent, as the gate is.
        const = getattr(getattr(cfg, 'frozen', None), 'feature_const_step', None)
        if const is not None and (off is not None or on is not None):
            raise ValueError('frozen.feature_const_step and the other switches are separate experiments; set one')
        if const is not None and self.T:
            self.register_buffer('feature_const', torch.tensor(1 if const <= 0 else 0))
        # Weaning (frozen.feature_dropout_until): the gate holds the KEEP rate;
        # in training each token's features survive with that probability, at
        # evaluation they are scaled by it (journey §33.47).
        drop = getattr(getattr(cfg, 'frozen', None), 'feature_dropout_until', None)
        if drop is not None and (off is not None or on is not None or const is not None):
            raise ValueError('feature dropout and the switches are separate experiments; set one')
        self.feature_dropout = drop is not None and bool(self.T)
        if self.feature_dropout:
            self.register_buffer('feature_gate', torch.tensor(1.0))
        self.trained = nn.Embedding(vocab_size, self.embedding_dim - self.carved)

        if verbose:
            print(f'  embedding: {self.extra_repr()}')

    # ── the check the old design lacked ───────────────────────────────────────

    def assert_blocks_informative(self, strict=True):
        """
        Population-level sanity: every static block must actually USE more than
        one class across the vocabulary, and every class it declares must be
        reachable. A constant factor column is a bias term, not information.

        Context blocks depend on the corpus and are checked in tests/ and by
        scripts/audits/collision_audit.py against real code streams.
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
        if getattr(self.blocks[i], 'zero_slot', False):
            # ZERO-SLOT control (`mzero`, journey §33.11): the slot's dims are
            # exactly zero — no learned vector, no information, no gradient.
            # What remains is a token embedding narrower than the residual
            # stream, the only difference from the baseline.
            width = self.embedding_dim if self.inject == 'add' else self.blocks[i].dim
            return torch.zeros(*code.shape, width, device=code.device,
                               dtype=self.trained.weight.dtype)
        if self.trainable_factors:
            # a learned table sets its own magnitude, so scale is not applied
            return self.factor_tables[i](code)
        eye = getattr(self, f'factor_eye_{i}')
        out = eye[code]
        if self.factor_scale != 1.0:
            out = out * self.factor_scale
        return out

    def _word_starts(self, ids):
        """Word-start mask from the tokenizer, for causal boundary detection."""
        alpha = self.tok_alpha[ids]
        opens = self.tok_opens[ids]
        frag_tab = getattr(self, 'tok_frag', None)
        if frag_tab is not None:
            # lexical.resolve_fragments, on the batch: a fragment takes the
            # class of the position before it; position 0 keeps its own
            frag = frag_tab[ids]
            for _ in range(FRAGMENT_STEPS):
                prev = torch.cat([alpha[:, :1], alpha[:, :-1]], dim=1)
                alpha = torch.where(frag, prev, alpha)
        prev_alpha = torch.cat(
            [torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1)
        new_word = alpha & (opens | ~prev_alpha)
        new_punct = (~alpha) & prev_alpha      # a punctuation run opens a unit
        st = new_word | new_punct
        st[:, 0] = True
        return st

    def switch_features_off(self):
        """Close the feature gate: from now on every feature contributes zeros."""
        if hasattr(self, 'feature_gate'):
            self.feature_gate.fill_(0.0)

    def switch_features_on(self):
        """Open the feature gate: from now on every feature contributes its embedding."""
        if hasattr(self, 'feature_gate'):
            self.feature_gate.fill_(1.0)

    def switch_features_const(self):
        """The switch to a constant: from now on every feature's value is its block's first class."""
        if hasattr(self, 'feature_const'):
            self.feature_const.fill_(1)

    def set_feature_keep(self, q):
        """Weaning: the probability that a token keeps its features."""
        if hasattr(self, 'feature_gate'):
            self.feature_gate.fill_(float(q))

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
        _runs = None
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
                code = codes[:, ci, :]
                if getattr(self, 'causal_identity', False):
                    if _runs is None:
                        _runs = word_runs(self._word_starts(ids))
                    code = causal_shift(code, _runs[0], _runs[1],
                                        block.n_classes - 1)
                if block.name == 'mstruct' and self.struct_keep != 'all':
                    code = keep_struct(code, self.struct_keep)
                code = fold_codes(code, block)
                ci += 1
            if hasattr(self, 'feature_const'):
                # after the switch to a constant every value but 'none' is class 0
                keep = (code == block.n_classes - 1) | (self.feature_const == 0)
                code = torch.where(keep, code, torch.zeros_like(code))
            parts.append(self._factor(i, code))

        if hasattr(self, 'feature_gate'):
            g = self.feature_gate
            if self.feature_dropout and self.training:
                g = (torch.rand(ids.shape[0], ids.shape[1], 1, device=ids.device) < g).to(parts[0].dtype)
            parts = [p * g for p in parts]
        if self.inject == 'add':
            return trained + torch.stack(parts, dim=0).sum(dim=0)
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
        inj = ' ADDED to the token embedding' if self.inject == 'add' else ''
        return (f'N={self.embedding_dim}, T={self.T} {mode}{inj} '
                f'(static {self.static_dim}, context {self.context_dim}, '
                f'x{self.amplify}), trained={self.embedding_dim - self.carved}'
                + (f' | {names}' if names else ' | baseline'))
