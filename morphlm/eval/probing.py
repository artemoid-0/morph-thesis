"""
probing.py — the mechanistic story [INTERFACE DEFINED, CLASSIFIER NOT WRITTEN].

Train a tiny classifier on the learned embeddings to predict a linguistic
property. If the property is encoded, a small classifier learns it from little
data. This tests whether frozen dimensions are ACTIVELY USED rather than merely
present — the difference between 'the information is there' and 'the model
reads it'.

Deliberately left as a defined interface rather than a guess: what to probe
depends on which facts Phase 1 shows to matter, and writing it before that is
premature. The extraction half IS implemented, since it is unambiguous.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_embeddings(model, loader, device, max_batches=50):
    """
    Collect (embedding, token_id, codes) triples from the input embedding layer.
    Returns X [n, d_model], ids [n], codes [n, n_facts] (empty if none).
    """
    model.eval()
    xs, tids, cds = [], [], []
    for i, (x, y, c) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        cc = c.to(device) if c.numel() else None
        emb = model.embed(x, cc)                       # [B, L, d]
        xs.append(emb.reshape(-1, emb.shape[-1]).cpu().numpy())
        tids.append(x.reshape(-1).cpu().numpy())
        if c.numel():
            cds.append(c.permute(0, 2, 1).reshape(-1, c.shape[1]).cpu().numpy())
    model.train()
    X = np.concatenate(xs, axis=0)
    ids = np.concatenate(tids, axis=0)
    codes = np.concatenate(cds, axis=0) if cds else np.zeros((len(ids), 0))
    return X, ids, codes


def probe(X, labels, test_frac=0.2, seed=0):
    """
    Fit a linear probe predicting `labels` from `X`, return accuracy.

    NOT YET IMPLEMENTED — the choice of probe (linear vs MLP, which layer,
    which property) should follow from Phase 1 results rather than precede them.
    """
    raise NotImplementedError(
        'probing classifier not written yet — see research_journey.md. '
        'extract_embeddings() is implemented and ready to feed it.')
