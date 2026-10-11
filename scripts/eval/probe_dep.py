#!/usr/bin/env python3
"""
probe_dep.py — is the injected fact ACTUALLY IN the representation?

WHAT THIS CLOSES
----------------
Journey section 2 promised two things from freezing: lower perplexity AND
probing showing the injected dimensions are used. Section 21 refuted the first
half — learned factors beat frozen by 1.7-2.3% under matched capacity.

Legibility is the second half, and it is the one place freezing can still win by
construction: a frozen one-hot is trivially decodable, whereas a learned table
is free to collapse classes together if the loss does not punish it. If frozen
dimensions stay linearly readable at depth while learned ones do not, the
trade-off is ~2% perplexity for guaranteed interpretability — a defensible
result rather than a loss.

THREE MODELS, THREE DIFFERENT PREDICTIONS
-----------------------------------------
    baseline    never saw the labels. If a probe can already read them off its
                hidden states, the model infers the property unaided and
                injection is partly redundant — which would explain why the
                effect is small for frequent tokens.

    frozen      the input contains a literal one-hot, so the embedding layer
                should be ~100%. The interesting number is how much SURVIVES to
                the final block.

    learned     if the model preserves the distinction it probes high; if it
                collapses classes it probes LOW WHILE STILL having better
                perplexity, which is exactly the interpretability/performance
                trade-off the thesis originally hypothesised.

WHY A LINEAR PROBE
------------------
Linear probing asks whether the information is present in a LINEARLY READABLE
form, which is the form attention heads can actually use. A strong non-linear
probe can extract information the model itself cannot easily exploit, so it
answers a less useful question.

WHY A MAJORITY BASELINE IS REPORTED
-----------------------------------
Label distributions here are skewed. A probe scoring 0.90 on a label that is 90%
one class has learned nothing. Chance is the majority-class rate, not 1/n.

    python scripts/eval/probe_dep.py --runs <baseline_id> <frozen_id> <learned_id>
    python scripts/eval/probe_dep.py --runs <ids...> --fact mroot
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# Post-hoc analysis must never trigger a cache build: a miss here means the
# wrong config, not a job to start. Without this the first probe run began a
# silent four-hour rebuild.
os.environ.setdefault('MORPHLM_NO_BUILD', '1')


def plan_codes(model_facts, probe_fact):
    """
    Decide which code rows to load, and which of them feed the model versus
    supply the probe labels.

    Returns (request, model_rows, label_row):
      request     facts to load, in order: the model's own context facts first,
                  then the probed fact if the model does not already read it
      model_rows  indices into `request` that the MODEL reads, in the order its
                  context blocks expect them
      label_row   index into `request` holding the labels to predict

    WHY THIS IS SEPARATE, AND WHY IT MATTERS
    ----------------------------------------
    An earlier version loaded ONE dataset configured with the probed fact only,
    and fed those codes to the model as its input. That conflated what to
    PREDICT with what the model READS. It went unnoticed because every probe
    until then asked about the very fact the model was trained on, so the one
    loaded row happened to be exactly what the model needed.

    The first probe of a DIFFERENT fact — part of speech, on models trained
    with lemma features — exposed it. A two-fact model crashed, which was lucky.
    A one-fact model would not have: it would have read part-of-speech tags as
    though they were lemma codes (0-17 are valid indices into a 16,365-class
    table), produced meaningless hidden states, and reported a real-looking
    accuracy for them.
    """
    request = list(model_facts)
    if probe_fact not in request:
        request.append(probe_fact)
    model_rows = list(range(len(model_facts)))
    label_row = request.index(probe_fact)
    return request, model_rows, label_row


def collect(run_id, results_dir, fact, batches):
    """
    Run a checkpoint over validation windows and return per-layer hidden states
    plus the TRUE code for every position.

    The model is fed its OWN context facts; the labels come from `fact`, which
    may or may not be one of them. A baseline checkpoint reads no facts at all
    and can still be probed — that arm is the informative one.
    """
    import torch
    import dataclasses
    from torch.utils.data import DataLoader

    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_val_dataset
    from morphlm.registry import context_facts

    model, cfg = load_checkpoint(run_id, results_dir)
    device = get_device()
    model = model.to(device).eval()

    model_facts = list(context_facts(cfg.frozen.facts))
    request, model_rows, label_row = plan_codes(model_facts, fact)

    data_cfg = dataclasses.replace(
        cfg, frozen=dataclasses.replace(cfg.frozen, facts=tuple(request),
                                        shuffle_codes=False))
    pipeline = build_pipeline(data_cfg, verbose=False)
    val_ds = make_val_dataset(pipeline, data_cfg, verbose=False)
    loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    # Derive row positions from what was ACTUALLY loaded, by name — never from
    # the order we assumed we asked for. And verify by NAME, not by count: a
    # count check would have passed the silently-wrong case above (one row
    # expected, one row supplied).
    loaded = list(getattr(val_ds, 'fact_names', []) or [])
    missing = [f for f in model_facts + [fact] if f not in loaded]
    if missing:
        raise RuntimeError(f'{run_id}: loaded codes {loaded} lack {missing}')
    model_rows = [loaded.index(f) for f in model_facts]
    label_row = loaded.index(fact)
    fed = [loaded[i] for i in model_rows]
    if fed != model_facts:
        raise RuntimeError(
            f'{run_id}: the model reads {model_facts} but would be fed {fed}. '
            f'Refusing to probe — the hidden states would be meaningless.')
    if loaded[label_row] != fact:
        raise RuntimeError(f'{run_id}: label row holds {loaded[label_row]!r}, '
                           f'not {fact!r}')

    H, Y = None, []
    with torch.no_grad():
        for bi, (x, y, c) in enumerate(loader):
            if batches and bi >= batches:
                break
            x = x.to(device)
            cc = (c[:, model_rows, :].to(device) if model_facts else None)
            out = model(x, None, codes=cc, return_hidden=True)
            hidden = out[2]
            if H is None:
                H = [[] for _ in hidden]
            for li, h in enumerate(hidden):
                H[li].append(h.reshape(-1, h.shape[-1]).cpu().numpy())
            Y.append(c[:, label_row, :].reshape(-1).numpy())

    H = [np.concatenate(v, axis=0) for v in H]
    Y = np.concatenate(Y, axis=0)
    return cfg, H, Y


def linear_probe(X, y, n_classes, seed=0, epochs=80, lr=0.05):
    """Multinomial logistic regression, held-out accuracy."""
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = torch.from_numpy(((X[tr] - mu) / sd).astype(np.float32))
    Xte = torch.from_numpy(((X[te] - mu) / sd).astype(np.float32))
    ytr = torch.from_numpy(y[tr].astype(np.int64))
    yte = torch.from_numpy(y[te].astype(np.int64))

    torch.manual_seed(seed)
    clf = nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(-1) == yte).float().mean().item()
    return acc


def majority_baseline(y):
    c = Counter(y.tolist())
    return max(c.values()) / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='checkpoint ids to probe')
    ap.add_argument('--fact', default='dep',
                    help='which injected fact to probe for')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--batches', type=int, default=12)
    ap.add_argument('--sample', type=int, default=40000,
                    help='positions sampled per layer')
    args = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('probe_dep', args)

    from morphlm.registry import get_fact
    from morphlm.config import ExperimentConfig, apply_overrides
    probe_cfg = apply_overrides(ExperimentConfig(),
                                [f'frozen.facts={args.fact}'])
    n_classes = get_fact(args.fact)(probe_cfg).n_classes

    results = []
    for rid in args.runs:
        print(f'collecting {rid}...')
        cfg, H, Y = collect(rid, args.results_dir, args.fact, args.batches)
        rng = np.random.default_rng(0)
        sel = (rng.choice(len(Y), args.sample, replace=False)
               if len(Y) > args.sample else np.arange(len(Y)))
        accs = [linear_probe(X[sel], Y[sel], n_classes) for X in H]
        results.append((cfg.label(), majority_baseline(Y[sel]), accs))
        print('  ' + cfg.label() + ': ' +
              ' '.join(f'L{i}={a:.3f}' for i, a in enumerate(accs)))

    rec.add('fact', args.fact)
    rec.add('probes', [{'label': l, 'chance_majority': c, 'accuracy_by_layer': a}
                       for l, c, a in results])
    rec.note('chance is the MAJORITY-CLASS rate, not 1/n: the label distribution is skewed.')
    n_layers = max(len(a) for _, _, a in results)
    print()
    print('=' * 78)
    print(f'LINEAR PROBE — can `{args.fact}` be read off the representation?')
    print('=' * 78)
    hdr = f'  {"model":30} {"chance":>7}' + ''.join(
        f'{("emb" if i == 0 else "L" + str(i)):>8}' for i in range(n_layers))
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for label, chance, accs in results:
        print(f'  {label:30} {chance:7.3f}' +
              ''.join(f'{a:8.3f}' for a in accs))

    print()
    print('  READING THIS  ("chance" is the MAJORITY-CLASS rate, not 1/n)')
    print('    baseline high at the last layer -> the model infers the property')
    print('      unaided; injection adds precision, not information.')
    print('    frozen >> learned at depth      -> freezing buys GUARANTEED')
    print('      LEGIBILITY at a small perplexity cost. This is the second half')
    print('      of the original hypothesis (journey sections 2 and 11).')
    print('    frozen ~= learned               -> freezing buys nothing extra;')
    print('      restate the contribution around feature injection.')
    print()

    rec.write()


if __name__ == '__main__':
    main()
