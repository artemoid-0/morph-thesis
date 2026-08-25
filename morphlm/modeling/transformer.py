"""
transformer.py — the fixed apparatus.

Load-bearing but NOT the object of study. The thesis claim is about embeddings;
the transformer is the microscope. So the goal is the fairest, most standard,
most controlled model — identical across every cell, with only the embedding
varying. Textbook GPT-2: pre-norm blocks, GELU, learned absolute positions.

Small on purpose (journey section 12): a large model learns morphology for free
and washes the effect out, so small is the regime where the hypothesis is
testable at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, seq_len):
        super().__init__()
        self.n_heads = cfg.model.n_heads
        self.head_dim = cfg.model.d_model // cfg.model.n_heads
        d = cfg.model.d_model
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.model.dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:L, :L], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, ff = cfg.model.d_model, cfg.model.d_ff
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.drop = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg, seq_len):
        super().__init__()
        d = cfg.model.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg, seq_len)
        self.ln2 = nn.LayerNorm(d)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Decoder-only LM. Everything above the embedding is standard and never knows
    that some input dimensions were frozen.
    """

    def __init__(self, cfg, embedding, vocab_size):
        super().__init__()
        assert embedding.embedding_dim == cfg.model.d_model, \
            'embedding width must equal d_model'
        self.cfg = cfg
        self.vocab_size = vocab_size
        d = cfg.model.d_model
        L = cfg.data.seq_len

        self.embed = embedding
        self.pos = nn.Embedding(L, d)
        self.drop = nn.Dropout(cfg.model.dropout)
        self.blocks = nn.ModuleList([Block(cfg, L)
                                     for _ in range(cfg.model.n_layers)])
        self.ln_f = nn.LayerNorm(d)

        if cfg.model.tie_output == 'none':
            self.head = nn.Linear(d, vocab_size, bias=False)
        elif cfg.model.tie_output == 'frozen':
            raise NotImplementedError(
                "tie_output='frozen' (option C) is a planned later variant: "
                "tie the frozen columns into the output projection so frozen "
                "facts shape predictions directly.")
        else:
            raise ValueError(f'unknown tie_output={cfg.model.tie_output!r}')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, targets=None, codes=None):
        B, L = ids.shape
        pos_ids = torch.arange(L, device=ids.device).unsqueeze(0)

        x = self.embed(ids, codes) + self.pos(pos_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def num_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))


def build_model(cfg, verbose=True):
    """Construct the model for a config. Returns (model, vocab_size)."""
    from ..pipelines.registry import pipeline_vocab
    from .embedding import HybridEmbedding

    vocab_df, bpe_size = pipeline_vocab(cfg)
    num_special = 4
    if vocab_df is not None:
        vocab_size = len(vocab_df) + num_special
    else:
        vocab_size = bpe_size + num_special

    embedding = HybridEmbedding(cfg, vocab_size, vocab_df=vocab_df,
                                verbose=verbose)
    embedding.assert_blocks_informative(strict=True)

    model = TransformerLM(cfg, embedding, vocab_size)
    if verbose:
        print(f'  trainable params: {model.num_params():,}')
    return model, vocab_size
