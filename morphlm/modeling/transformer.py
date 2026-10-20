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
        # Rotary positions (model.position = 'rope', journey §33.47): queries and
        # keys are rotated by position, so attention sees only RELATIVE position
        # and nothing positional is added at the input. Non-persistent buffers.
        self.rope = getattr(cfg.model, 'position', 'learned') == 'rope'
        if self.rope:
            inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            ang = torch.outer(torch.arange(seq_len).float(), inv)
            self.register_buffer('rope_cos', ang.cos(), persistent=False)
            self.register_buffer('rope_sin', ang.sin(), persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            cos, sin = self.rope_cos[:L], self.rope_sin[:L]
            def rot(t):
                t1, t2 = t[..., 0::2], t[..., 1::2]
                return torch.stack((t1 * cos - t2 * sin, t1 * sin + t2 * cos), dim=-1).flatten(-2)
            q, k = rot(q), rot(k)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:L, :L], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.proj(out))


class RMSNorm(nn.Module):
    """
    Root-mean-square normalization (model.norm = 'rmsnorm', journey §33.54): a
    learned gain and no mean subtraction or shift, as in LLaMA-style blocks. Its
    gain starts at ones and draws no random number, so a model with it starts
    from exactly the weights of its LayerNorm twin at the same seed.
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg, d):
    kind = getattr(cfg.model, 'norm', 'layernorm')
    if kind == 'layernorm':
        return nn.LayerNorm(d)
    if kind == 'rmsnorm':
        return RMSNorm(d)
    raise ValueError(f'unknown model.norm {kind!r}')


def swiglu_hidden(d):
    """8d/3 rounded to a multiple of 8: three matrices with the GELU MLP's 8d² weights."""
    return int(round(8 * d / 3 / 8)) * 8


class SwiGLU(nn.Module):
    """
    The gated MLP of LLaMA-style blocks (model.mlp = 'swiglu', journey §33.54):
    w2(silu(w1 x) * w3 x). Its hidden width keeps the GELU MLP's weight count;
    biases are kept, like every other layer here.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.model.d_model
        h = swiglu_hidden(d)
        self.w1 = nn.Linear(d, h)
        self.w3 = nn.Linear(d, h)
        self.w2 = nn.Linear(h, d)
        self.drop = nn.Dropout(cfg.model.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


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
        self.ln1 = make_norm(cfg, d)
        self.attn = CausalSelfAttention(cfg, seq_len)
        self.ln2 = make_norm(cfg, d)
        mlp = getattr(cfg.model, 'mlp', 'gelu')
        if mlp not in ('gelu', 'swiglu'):
            raise ValueError(f'unknown model.mlp {mlp!r}')
        self.ff = SwiGLU(cfg) if mlp == 'swiglu' else FeedForward(cfg)

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
        # learned absolute positions — unless rotary positions replace them
        self.pos = None if getattr(cfg.model, 'position', 'learned') == 'rope' else nn.Embedding(L, d)
        # Where the positional embedding may go (journey §33.12). None — the
        # default, every existing cell — adds it across all d_model dims. A
        # zero-slot control can instead keep position OUT of its slot
        # ('none': those dims carry nothing at the input) or put it ONLY in
        # its slot ('only': token dims carry pure identity). The embedding
        # places fact dims first, so the slot is dims [0:T].
        # a NON-persistent buffer: no new state_dict key, so every existing
        # checkpoint loads unchanged; it follows .to(device) like any buffer
        self.register_buffer('pos_mask', None, persistent=False)
        modes = {getattr(b, 'slot_position', None) for b in embedding.blocks} - {None}
        if modes:
            if len(modes) > 1 or len(embedding.blocks) != 1:
                raise ValueError('slot_position controls need exactly one block')
            mode, T = modes.pop(), embedding.T
            mask = torch.ones(d)
            if mode == 'none':
                mask[:T] = 0.0
            elif mode == 'only':
                mask[T:] = 0.0
            else:
                raise ValueError(f'unknown slot_position {mode!r}')
            self.pos_mask = mask
        self.drop = nn.Dropout(cfg.model.dropout)
        self.blocks = nn.ModuleList([Block(cfg, L)
                                     for _ in range(cfg.model.n_layers)])
        self.ln_f = make_norm(cfg, d)

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

    def forward(self, ids, targets=None, codes=None, return_hidden=False):
        """
        return_hidden=True additionally returns the hidden state after EVERY
        block, which is what the legibility probe needs. Whether an injected
        fact SURVIVES into the deeper representation is a different question
        from whether it is present at the input, and it is the one place
        freezing can still win after section 21 refuted the perplexity claim.
        """
        B, L = ids.shape
        pos_ids = torch.arange(L, device=ids.device).unsqueeze(0)

        x = self.embed(ids, codes)
        if self.pos is not None:
            pe = self.pos(pos_ids)
            if self.pos_mask is not None:
                pe = pe * self.pos_mask
            x = x + pe
        x = self.drop(x)
        hidden = [x] if return_hidden else None
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden.append(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        # The auxiliary target (frozen.role = 'target'): predict the fact's class
        # of each token from the final hidden state. Kept OUT of the returned
        # loss — the training step adds it; evaluation never sees it (§33.47).
        self.last_aux_loss = None
        if getattr(self, 'aux_head', None) is not None and codes is not None:
            aux_t = codes[:, 0, :].clamp(0, self.aux_head.out_features - 1)
            aux = self.aux_head(x)
            self.last_aux_loss = F.cross_entropy(aux.reshape(-1, aux.size(-1)), aux_t.reshape(-1))
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def num_params(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))


def build_model(cfg, verbose=True):
    """Construct the model for a config. Returns (model, vocab_size)."""
    from ..pipelines.registry import pipeline_vocab
    from .embedding import HybridEmbedding

    vocab_df, bpe_size = pipeline_vocab(cfg)
    # Four rows for special tokens, for every pipeline — deliberate; never "fix"
    # it. BPE's vocabulary already holds its specials at ids 0-3, so its four
    # extra rows are never read, and they keep BPE-17k at 17,151 + 4 = 17,155
    # rows, the morpheme vocabulary's count. Every run's parameter count and
    # initial weights depend on it.
    num_special = 4
    if vocab_df is not None:
        vocab_size = len(vocab_df) + num_special
    else:
        vocab_size = bpe_size + num_special

    # The auxiliary target injects nothing: the embedding is a plain one, while
    # the data still carries the fact's codes — as targets (journey §33.47).
    target = getattr(cfg.frozen, 'role', 'input') == 'target'
    if target:
        import dataclasses
        from ..features.blocks import build_blocks
        specs = build_blocks(cfg)
        if len(specs) != 1 or any(getattr(cfg.frozen, k, None) is not None
                                  for k in ('feature_off_step', 'feature_on_step', 'feature_dropout_until')):
            raise ValueError('role=target takes exactly one fact and no switch or dropout')
        n_target = specs[0].n_classes
        embed_cfg = dataclasses.replace(cfg, frozen=dataclasses.replace(cfg.frozen, facts=()))
    else:
        embed_cfg = cfg
    embedding = HybridEmbedding(embed_cfg, vocab_size, vocab_df=vocab_df,
                                verbose=verbose)
    embedding.assert_blocks_informative(strict=True)

    model = TransformerLM(cfg, embedding, vocab_size)
    if target:
        # made LAST, so every other weight starts exactly as a baseline's
        model.aux_head = nn.Linear(cfg.model.d_model, n_target)
    if verbose:
        print(f'  trainable params: {model.num_params():,}')
    return model, vocab_size
