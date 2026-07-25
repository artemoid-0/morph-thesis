"""
model.py

A standard, deliberately-unclever decoder-only transformer. It is the FIXED
APPARATUS through which we measure the embedding hypothesis — identical across
every cell of the experiment matrix, with the ONLY difference being the
embedding layer.

Design stance (see research_journey.md §28):
  - Vanilla GPT-2-style block: pre-norm, multi-head causal attention, GELU FFN.
  - Everything configurable so the BPE-baseline calibration run can tell us
    whether the apparatus sits in the right regime (learns well, but with
    visible headroom) before we commit to a size.
  - d_model = N = the embedding width. The input layer is HybridEmbedding;
    for the baseline/BPE case that collapses to a plain trained embedding.
  - Output projection is configurable:
        tie_output='none'   → separate learned [d_model → vocab] projection
                              (Phase 1 default — clean, frozen role stays on input)
        tie_output='frozen' → HOOK for the later variant that ties the frozen
                              columns into the output (option C). Not yet active.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from frozen_embedding import HybridEmbedding, SCHEMAS


# ── Config ────────────────────────────────────────────────────────────────────

class ModelConfig:
    """Plain config object. Every field is a knob we can re-probe."""
    def __init__(self,
                 vocab_size,
                 d_model=128,
                 n_layers=4,
                 n_heads=4,
                 d_ff=None,
                 seq_len=256,
                 dropout=0.1,
                 tie_output='none',     # 'none' | 'frozen' (hook)
                 num_special=4):
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.d_ff        = d_ff or 4 * d_model
        self.seq_len     = seq_len
        self.dropout     = dropout
        self.tie_output  = tie_output
        self.num_special = num_special

        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'

    def __repr__(self):
        return (f'ModelConfig(vocab={self.vocab_size}, d_model={self.d_model}, '
                f'layers={self.n_layers}, heads={self.n_heads}, '
                f'd_ff={self.d_ff}, seq_len={self.seq_len}, '
                f'tie_output={self.tie_output!r})')


# ── Transformer block ─────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        # causal mask, registered so it moves with .to(device)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x)                                  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, heads, L, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:L, :L], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = att @ v                                      # [B, heads, L, hd]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2  = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only LM. The input embedding is a HybridEmbedding; everything above
    it is a standard transformer that never knows or cares whether some input
    dimensions were frozen.
    """

    def __init__(self, cfg, embedding):
        """
        cfg       : ModelConfig
        embedding : a HybridEmbedding instance (variant chosen by caller).
                    Its embedding_dim MUST equal cfg.d_model.
        """
        super().__init__()
        assert embedding.embedding_dim == cfg.d_model, \
            'embedding width must equal d_model'
        self.cfg = cfg
        self.embed = embedding
        self.pos   = nn.Embedding(cfg.seq_len, cfg.d_model)   # learned absolute
        self.drop  = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)

        # output projection
        if cfg.tie_output == 'none':
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        elif cfg.tie_output == 'frozen':
            # HOOK (option C): tie frozen columns into the output projection.
            # Deferred — see research_journey.md. Refuse rather than silently
            # do the wrong thing.
            raise NotImplementedError(
                "tie_output='frozen' (option C) is a planned later variant.")
        else:
            raise ValueError(f'unknown tie_output={cfg.tie_output!r}')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None):
        """
        ids     : LongTensor [B, L]
        targets : LongTensor [B, L] or None
        Returns (logits, loss). loss is None if targets is None.
        """
        B, L = ids.shape
        pos_ids = torch.arange(L, device=ids.device).unsqueeze(0)

        x = self.embed(ids) + self.pos(pos_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                              # [B, L, vocab]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def num_params(self, trainable_only=True):
        params = (p for p in self.parameters()
                  if (p.requires_grad or not trainable_only))
        return sum(p.numel() for p in params)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_model(vocab_df=None, vocab_size=None, variant='baseline',
                d_model=128, n_layers=4, n_heads=4, seq_len=256,
                dropout=0.1, tie_output='none', freeze_amplify=1,
                num_special=4):
    """
    Construct a TransformerLM with the chosen embedding variant.

    For morpheme pipelines: pass vocab_df (the vocabulary DataFrame). The
    HybridEmbedding builds its frozen matrix from it.

    For BPE: pass vocab_size (an int). BPE only ever uses variant='baseline'
    (T=0), so no frozen matrix is needed; we build a NullSchema embedding over
    a trivial index space of the right width.
    """
    if vocab_df is not None:
        schema = SCHEMAS[variant]()
        embedding = HybridEmbedding(
            vocab_df, d_model, schema,
            num_special=num_special, freeze_amplify=freeze_amplify)
        vsize = len(vocab_df) + num_special
    elif vocab_size is not None:
        if variant != 'baseline':
            raise ValueError('BPE/int-vocab path supports only baseline variant')
        # NullSchema over a vocab_size-row frame proxy: build a plain embedding
        import pandas as pd
        proxy = pd.DataFrame({'token': [str(i) for i in range(vocab_size)],
                              'category': ['bpe'] * vocab_size})
        schema = SCHEMAS['baseline']()
        embedding = HybridEmbedding(
            proxy, d_model, schema,
            num_special=num_special, freeze_amplify=1)
        vsize = vocab_size + num_special
    else:
        raise ValueError('provide either vocab_df or vocab_size')

    cfg = ModelConfig(
        vocab_size=vsize, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, seq_len=seq_len, dropout=dropout,
        tie_output=tie_output, num_special=num_special)

    return TransformerLM(cfg, embedding)
