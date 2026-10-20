"""
windows.py — fixed-length next-token-prediction windows.

Concatenate-and-chop over the cached stream; document boundaries ignored
(standard GPT-2 practice, seam noise negligible at L=256).

Each item is (input_ids, target_ids, codes) where codes is
[n_context_facts, seq_len] or an empty tensor when no context facts are used.

The model owns the embedding and the dataset emits integers only (journey §12):
the embedding's trained part has gradients. The codes are integers too, with no
gradients, so the dataset still stays a pure int-in/int-out machine.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LMWindowDataset(Dataset):

    def __init__(self, stream_meta, seq_len=256, shuffle_codes=False, seed=0):
        self.ids = stream_meta['ids']
        self.codes = stream_meta.get('codes')
        self.fact_names = stream_meta.get('fact_names', [])
        # provenance: the codes file this stream was read from, and its rules
        self.codes_file = stream_meta.get('codes_file')
        self.codes_rules = stream_meta.get('codes_rules')
        self.shuffle_codes = shuffle_codes
        self.shuffle_seed = seed
        self.seq_len = seq_len
        self.num_tokens = stream_meta['num_tokens']
        self.num_bytes = stream_meta['num_bytes']
        self.num_chars = stream_meta['num_chars']
        self.pipeline = stream_meta.get('pipeline')
        self.num_windows = (self.num_tokens - 1) // seq_len
        if self.num_windows < 1:
            raise ValueError(
                f'a stream of {len(self.ids):,} tokens holds no window of '
                f'{seq_len + 1} — the split is (nearly) empty; for a small '
                f'corpus lower data.corpus_holdout')

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = np.asarray(self.ids[start: start + self.seq_len + 1],
                           dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        if self.codes is None:
            c = torch.zeros(0, self.seq_len, dtype=torch.long)
        else:
            block = np.asarray(self.codes[:, start: start + self.seq_len],
                               dtype=np.int64)
            if self.shuffle_codes:
                # ABLATION: permute codes within the window. Same multiset,
                # same frozen width, same parameter count — only the alignment
                # between a code and its token is destroyed. Seeded by window
                # index so the ablation is reproducible.
                rng = np.random.default_rng(self.shuffle_seed * 1_000_003 + idx)
                block = block[:, rng.permutation(block.shape[1])]
            c = torch.from_numpy(np.ascontiguousarray(block))
        return x, y, c

    # ── metrics ───────────────────────────────────────────────────────────────

    def bits_per_byte(self, total_loss_nats, num_target_tokens):
        """
        Summed cross-entropy (nats) -> bits per byte.

        BPB = total_bits / bytes_of_the_scored_portion, where the byte count is
        scaled by the fraction of the stream actually scored. The unscored
        remainder is at most one window, so the approximation is negligible.
        """
        total_bits = total_loss_nats / np.log(2)
        frac = num_target_tokens / max(self.num_tokens, 1)
        scored_bytes = self.num_bytes * frac
        return float(total_bits / max(scored_bytes, 1))

    def perplexity(self, mean_loss_nats):
        """Per-token perplexity — only comparable WITHIN a pipeline."""
        return float(np.exp(mean_loss_nats))


def make_val_dataset(pipeline, cfg, verbose=True):
    """
    Validation window dataset only.

    Post-hoc analysis (probing, bucketed comparison) needs validation data and
    nothing else. `make_datasets` also builds the TRAIN stream, which is a
    120M-token no-op on a cache hit and a four-hour spaCy rebuild on a miss —
    the exact failure that silently hung the first probe run.
    """
    from .stream import build_stream
    meta = build_stream(pipeline, cfg, split='validation',
                        max_articles=cfg.data.max_articles_val,
                        verbose=verbose)
    shuf = getattr(cfg.frozen, 'shuffle_codes', False)
    ds = LMWindowDataset(meta, seq_len=cfg.data.seq_len,
                         shuffle_codes=shuf, seed=cfg.train.seed)
    if verbose:
        print(f'  val: {len(ds):,} windows ({ds.num_tokens:,} tokens, '
              f'{ds.num_bytes:,} bytes)')
    return ds


def make_datasets(pipeline, cfg, verbose=True):
    """Build train and validation window datasets for a config."""
    from .stream import build_stream

    train_meta = build_stream(pipeline, cfg, split='train',
                              max_articles=cfg.data.max_articles_train,
                              verbose=verbose)
    val_meta = build_stream(pipeline, cfg, split='validation',
                            max_articles=cfg.data.max_articles_val,
                            verbose=verbose)

    shuf = getattr(cfg.frozen, 'shuffle_codes', False)
    train_ds = LMWindowDataset(train_meta, seq_len=cfg.data.seq_len,
                               shuffle_codes=shuf, seed=cfg.train.seed)
    val_ds = LMWindowDataset(val_meta, seq_len=cfg.data.seq_len,
                             shuffle_codes=shuf, seed=cfg.train.seed)
    if shuf and verbose:
        print('  ABLATION: context codes are SHUFFLED within each window '
              '(distribution preserved, alignment destroyed)')

    if verbose:
        print(f'  train: {len(train_ds):,} windows '
              f'({train_ds.num_tokens:,} tokens, {train_ds.num_bytes:,} bytes)')
        print(f'  val:   {len(val_ds):,} windows '
              f'({val_ds.num_tokens:,} tokens, {val_ds.num_bytes:,} bytes)')
    return train_ds, val_ds
