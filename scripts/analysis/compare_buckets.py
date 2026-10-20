#!/usr/bin/env python3
"""
compare_buckets.py — does context freezing help RARE tokens, or everything?

WHY
---
The mechanistic prediction for freezing has always been that it buys SAMPLE
EFFICIENCY: a token seen rarely gets almost no gradient signal, so its trained
row is noise, while a frozen fact about it is correct from step 0. That
predicts an effect concentrated in the tail.

The Q2 result is a flat ~4.5% perplexity gain in both pipelines. Whether that
gain is uniform across the frequency range or concentrated in rare tokens is a
different claim, and it is the difference between:

    'freezing acts as useful structured input everywhere'
    'freezing rescues tokens the trained table cannot learn'

Only the second is the mechanism the thesis argues for.

This script scores TWO checkpoints on the same validation windows and reports
per-frequency-bucket perplexity side by side (morphlm/eval/buckets.py), so
baseline and frozen cells are directly comparable. The default bucket edges are
chosen for a ~300M-token corpus: the edges shipped in eval/buckets.py put 96.6%
of validation tokens in a single bucket, which hides exactly the structure we
are looking for.

    python scripts/analysis/compare_buckets.py --baseline <run_id> --treatment <run_id>
"""

import os
import sys
import math
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

# Post-hoc analysis must never trigger a cache build: a miss here means the
# wrong config, not a job to start. Without this the first probe run began a
# silent four-hour rebuild.
os.environ.setdefault('MORPHLM_NO_BUILD', '1')

#: Fallback only. Fixed thresholds cannot serve both pipelines: morph emits 2.4x
#: more tokens than BPE, so a set that spreads morph evenly piles 64-70% of BPE
#: tokens into the top bucket. `--adaptive` (the default) derives equal-mass
#: edges from the data instead.
EDGES = (1, 100, 1_000, 10_000, 100_000, 1_000_000, float('inf'))


def datasets(run_id, results_dir):
    """A saved model, its config, and its train and validation datasets — built once a run."""
    from morphlm.training.metrics import load_checkpoint
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_datasets
    model, cfg = load_checkpoint(run_id, results_dir)
    train_ds, val_ds = make_datasets(build_pipeline(cfg, verbose=False), cfg, verbose=False)
    return model, cfg, train_ds, val_ds


def score(run, edges, batches):
    """Each bucket's mean loss (nats; None when empty) and target count over `batches` validation
    batches (0: all), targets bucketed by their frequency in the run's own train stream."""
    from torch.utils.data import DataLoader
    from morphlm.training.loop import get_device
    from morphlm.eval.buckets import bucketed_perplexity, token_frequencies
    model, cfg, train_ds, val_ds = run
    device = get_device()
    rows = bucketed_perplexity(model.to(device), DataLoader(val_ds, batch_size=cfg.train.batch_size),
                               token_frequencies(train_ds), device, edges, max_batches=batches)
    return cfg, [r['mean_loss'] for r in rows], [r['tokens'] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--treatment', required=True)
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--batches', type=int, default=60,
                    help='validation batches (0 = all)')
    ap.add_argument('--fixed-edges', action='store_true',
                    help='use the legacy fixed thresholds instead of '
                         'equal-mass adaptive edges')
    ap.add_argument('--n-buckets', type=int, default=6)
    args = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('compare_buckets', args)

    # Edges are derived ONCE, from the baseline, and applied to BOTH models.
    # Per-model edges would silently compare different partitions.
    edges, base = EDGES, None
    if not args.fixed_edges:
        from morphlm.eval.buckets import adaptive_edges, token_frequencies
        print('deriving equal-mass bucket edges from the baseline...')
        base = datasets(args.baseline, args.results_dir)
        edges = adaptive_edges(token_frequencies(base[2]), np.asarray(base[3].ids), n_buckets=args.n_buckets)
        shown = [int(x) if x != float('inf') else 'inf' for x in edges]
        print(f'  edges: {shown}')
        print('  (equal validation mass per bucket, so every frequency band '
              'has comparable statistical power)')
        print()

    print(f'scoring baseline  {args.baseline}...')
    cb, lb, nb = score(base or datasets(args.baseline, args.results_dir), edges, args.batches)
    print(f'scoring treatment {args.treatment}...')
    ct, lt, nt = score(datasets(args.treatment, args.results_dir), edges, args.batches)

    print()
    print('=' * 78)
    print(f'BUCKETED COMPARISON')
    print(f'  baseline : {cb.label()}')
    print(f'  treatment: {ct.label()}')
    print('=' * 78)
    print(f'  {"train freq":>16} {"val tokens":>11} {"% tok":>7} '
          f'{"base ppl":>11} {"treat ppl":>11} {"delta":>9}')
    print('  ' + '-' * 74)

    total = sum(nb)
    rec.add('edges', [float(e) for e in edges])
    rec.add('baseline_run', args.baseline)
    rec.add('treatment_run', args.treatment)
    rec.add('buckets', [
        {'lo': float(edges[k]), 'hi': float(edges[k+1]),
         'tokens': int(nb[k]),
         'baseline_ppl': (math.exp(lb[k]) if nb[k] else None),
         'treatment_ppl': (math.exp(lt[k]) if nt[k] else None)}
        for k in range(len(edges) - 1)])
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        rng = f'{lo:,}-{"inf" if hi == float("inf") else format(int(hi), ",")}'
        if nb[k] == 0 or nt[k] == 0:
            print(f'  {rng:>16} {nb[k]:11,} {"":>7} {"-":>11} {"-":>11} {"-":>9}')
            continue
        pb = math.exp(lb[k])
        pt = math.exp(lt[k])
        d = 100.0 * (pt - pb) / pb
        print(f'  {rng:>16} {nb[k]:11,} {100.0*nb[k]/max(total,1):6.2f}% '
              f'{pb:11.3f} {pt:11.3f} {d:+8.2f}%')

    total_loss = lambda means, ns: sum(m * n for m, n in zip(means, ns) if n)
    pb_all = math.exp(total_loss(lb, nb) / max(sum(nb), 1))
    pt_all = math.exp(total_loss(lt, nt) / max(sum(nt), 1))
    print('  ' + '-' * 74)
    print(f'  {"OVERALL":>16} {total:11,} {"100.00%":>7} '
          f'{pb_all:11.3f} {pt_all:11.3f} '
          f'{100.0*(pt_all-pb_all)/pb_all:+8.2f}%')

    print()
    print('  READING THIS')
    print('    Roughly equal deltas across buckets  -> freezing acts as useful')
    print('      structured input generally, not as a rare-token rescue.')
    print('    Delta growing toward rare buckets    -> the sample-efficiency')
    print('      mechanism the thesis predicts.')
    print()

    rec.write()


if __name__ == '__main__':
    main()
