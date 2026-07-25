"""
buckets.py — per-frequency-bucket perplexity.

WHY THIS MATTERS FOR THE THESIS
-------------------------------
The mechanism by which static freezing could help is SAMPLE EFFICIENCY for rare
tokens: a root seen three times gets almost no gradient signal, so its trained
row is noise, while its frozen category flag is correct from step zero.

That predicts an effect concentrated in the TAIL, not in the aggregate. A
single corpus-wide perplexity number averages the tail away — which is very
likely why 2.6 bits spread over 23 dimensions looked so unpromising when
measured as 'does overall perplexity drop'.

Reporting perplexity per frequency bucket turns an aggregate null result into a
real mechanistic finding: 'frozen flags help rare tokens by X% and leave
frequent tokens unchanged' survives even when the headline number does not move.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch


DEFAULT_EDGES = (1, 10, 100, 1000, 10000, float('inf'))


def token_frequencies(train_ds):
    """Count how often each token id appears in the training stream."""
    ids = np.asarray(train_ds.ids)
    counts = Counter(ids.tolist())
    return counts


def bucket_of(count, edges=DEFAULT_EDGES):
    for i in range(len(edges) - 1):
        if edges[i] <= count < edges[i + 1]:
            return i
    return len(edges) - 2


@torch.no_grad()
def bucketed_perplexity(model, loader, freqs, device, edges=DEFAULT_EDGES):
    """
    Per-bucket mean loss over the validation set, bucketed by TARGET token
    training frequency. Returns list of dicts with count / mean_loss / ppl.
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
    for x, y, c in loader:
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


def print_buckets(rows, label=''):
    print(f'\n--- per-frequency-bucket perplexity {label} ---')
    print(f'{"train freq":>14} {"val tokens":>12} {"ppl":>10}')
    for r in rows:
        ppl = f'{r["perplexity"]:.2f}' if r['perplexity'] else '-'
        print(f'{r["range"]:>14} {r["tokens"]:12,} {ppl:>10}')
