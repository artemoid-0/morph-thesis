"""
windows.py — fixed-length next-token-prediction windows.

Concatenate-and-chop over the cached stream; document boundaries ignored
(standard GPT-2 practice, seam noise negligible at L=256).

Each item is (input_ids, target_ids, codes) where codes is
[n_context_facts, seq_len] or an empty tensor when no context facts are used.

ARCHITECTURAL NOTE
------------------
Journey section 28 decided "the dataset emits integer IDs only; the model owns
the embedding" — the reasoning being that the embedding's trained part carries
gradients, and backpropagating into the dataloader would be wrong.

That reasoning does not extend to frozen metadata. Frozen facts have NO
gradients — that is their defining property. So the dataset can emit small
integer codes alongside the ids with no gradient problem at all, and the
principle is preserved: this remains a pure int-in/int-out machine. The earlier
design mistook the rule for a constraint and, in doing so, lost the ability to
express context-dependent facts.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LMWindowDataset(Dataset):

    def __init__(self, stream_meta, seq_len=256):
        self.ids = stream_meta['ids']
        self.codes = stream_meta.get('codes')
        self.fact_names = stream_meta.get('fact_names', [])
        self.seq_len = seq_len
        self.num_tokens = stream_meta['num_tokens']
        self.num_bytes = stream_meta['num_bytes']
        self.num_chars = stream_meta['num_chars']
        self.pipeline = stream_meta.get('pipeline')
        self.num_windows = (self.num_tokens - 1) // seq_len

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
            c = torch.from_numpy(block)
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


def make_datasets(pipeline, cfg, verbose=True):
    """Build train and validation window datasets for a config."""
    from .stream import build_stream

    train_meta = build_stream(pipeline, cfg, split='train',
                              max_articles=cfg.data.max_articles_train,
                              verbose=verbose)
    val_meta = build_stream(pipeline, cfg, split='validation',
                            max_articles=cfg.data.max_articles_val,
                            verbose=verbose)

    train_ds = LMWindowDataset(train_meta, seq_len=cfg.data.seq_len)
    val_ds = LMWindowDataset(val_meta, seq_len=cfg.data.seq_len)

    if verbose:
        print(f'  train: {len(train_ds):,} windows '
              f'({train_ds.num_tokens:,} tokens, {train_ds.num_bytes:,} bytes)')
        print(f'  val:   {len(val_ds):,} windows '
              f'({val_ds.num_tokens:,} tokens, {val_ds.num_bytes:,} bytes)')
    return train_ds, val_ds
