"""
probing.py — reading a linguistic property from a model's states.

A small classifier trained on a model's states predicts a property; if the
property is encoded, a linear probe learns it from little data. `fit_linear` is
the probe of scripts/analysis/probe.py (when a model comes to know a feature,
DIRECTIONS D2) and of probe_words.py (a word's grammar in context, journey
§33.84). `extract_embeddings` collects the input embedding layer's vectors.
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


def fit_linear(Xtr, ytr, n_cls, epochs=300, dev='cpu'):
    """
    A linear probe: multinomial logistic regression on standardised states (the
    training states' mean and standard deviation), fitted full-batch by Adam
    (learning rate 1e-2, weight decay 1e-4) for `epochs` steps from weights seeded
    with 0. Returns a predictor: states [n, d] -> classes [n], on the CPU.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs, ys = ((Xtr - mu) / sd).to(dev), ytr.to(dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xtr.shape[1], n_cls).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(Xs), ys).backward()
        opt.step()

    def predict(X):
        with torch.no_grad():
            return lin(((X - mu) / sd).to(dev)).argmax(-1).cpu()
    return predict
