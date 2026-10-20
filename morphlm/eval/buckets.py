"""
buckets.py — per-frequency-bucket perplexity.

The mechanism by which freezing could help is SAMPLE EFFICIENCY for rare
tokens: a token seen rarely gets little gradient signal, so its trained row is
noise, while a frozen fact about it is correct from step zero. That predicts an
effect concentrated in the TAIL, which one corpus-wide number averages away;
per bucket it shows (journey §18, "The mechanism is the predicted one").
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch


DEFAULT_EDGES = (1, 10, 100, 1000, 10000, float('inf'))


def adaptive_edges(freqs, val_ids, n_buckets=6):
    """
    Frequency edges chosen so each bucket holds roughly EQUAL VALIDATION MASS.

    Fixed thresholds cannot work across pipelines. The morpheme pipelines emit
    2.4x more tokens than BPE over the same corpus, so a threshold set that
    spreads morph tokens evenly piles BPE tokens into the top bucket: measured
    on BPE-17k, the `10000-inf` bucket held 177,796 of 252,160 validation
    tokens (70.5%) while the two rarest held 1 and 16. The rare-token question
    — the one that decides whether morphological sharing helps where word
    identity cannot be learned — was simply unanswerable.

    Equal-mass buckets give equal statistical power at every frequency, which
    is what the question needs.

    IMPORTANT: the caller must compute edges ONCE and apply the same set to
    every model being compared. Edges derived per-model would silently compare
    different partitions.
    """
    f = np.asarray([freqs.get(int(t), 0) for t in np.asarray(val_ids)],
                   dtype=np.int64)
    if f.size == 0:
        return DEFAULT_EDGES
    order = np.sort(f)
    target = f.size / max(n_buckets, 1)

    edges = [int(order[0])]
    acc = 0
    for i in range(f.size):
        acc += 1
        if acc >= target and i + 1 < f.size:
            nxt = int(order[i + 1])
            if nxt > edges[-1]:          # strictly increasing edges only
                edges.append(nxt)
                acc = 0
    edges.append(float('inf'))
    # a heavily skewed distribution can collapse edges; fall back rather than
    # report a table with empty or duplicated bands
    if len(edges) < 3:
        return DEFAULT_EDGES
    return tuple(edges)


def token_frequencies(train_ds, chunk=1 << 24):
    """
    How often each token id appears in the training stream: a Counter of the ids
    that occur. Counted in chunks with np.bincount — a Counter over the stream as
    a Python list held ~4 GB at 150M tokens.
    """
    ids = np.asarray(train_ds.ids)
    counts = np.zeros(0, dtype=np.int64)
    for i in range(0, len(ids), chunk):
        part = np.bincount(np.asarray(ids[i:i + chunk], dtype=np.int64))
        if len(part) > len(counts):
            part[:len(counts)] += counts
            counts = part
        else:
            counts[:len(part)] += part
    seen = np.flatnonzero(counts)
    return Counter(dict(zip(seen.tolist(), counts[seen].tolist())))


def bucket_of(count, edges=DEFAULT_EDGES):
    for i in range(len(edges) - 1):
        if edges[i] <= count < edges[i + 1]:
            return i
    return len(edges) - 2


@torch.no_grad()
def bucketed_perplexity(model, loader, freqs, device, edges=DEFAULT_EDGES,
                        max_batches=None):
    """
    Per-bucket mean loss over the validation set, bucketed by TARGET token
    training frequency. Returns list of dicts with count / mean_loss / ppl.
    `max_batches` scores only the loader's first batches (None or 0: all).
    """
    model.eval()
    n_buckets = len(edges) - 1
    loss_sum = [0.0] * n_buckets
    tok_count = [0] * n_buckets

    # id -> bucket lookup, built once
    max_id = max(freqs) if freqs else 0
    lut = np.zeros(max_id + 2, dtype=np.int64)
    for tid, c in freqs.items():
        lut[tid] = bucket_of(c, edges)

    import torch.nn.functional as F
    for bi, (x, y, c) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        c = c.to(device) if c.numel() else None
        logits, _ = model(x, y, codes=c)
        per_tok = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
        flat_y = y.view(-1).cpu().numpy()
        flat_y = np.clip(flat_y, 0, max_id)
        b = lut[flat_y]
        per_tok = per_tok.cpu().numpy()
        for k in range(n_buckets):
            m = (b == k)
            if m.any():
                loss_sum[k] += float(per_tok[m].sum())
                tok_count[k] += int(m.sum())

    model.train()
    out = []
    for k in range(n_buckets):
        if tok_count[k] == 0:
            out.append({'bucket': k, 'range': f'{edges[k]}-{edges[k+1]}',
                        'tokens': 0, 'mean_loss': None, 'perplexity': None})
            continue
        ml = loss_sum[k] / tok_count[k]
        out.append({'bucket': k, 'range': f'{edges[k]}-{edges[k+1]}',
                    'tokens': tok_count[k], 'mean_loss': ml,
                    'perplexity': math.exp(ml)})
    return out
