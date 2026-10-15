#!/usr/bin/env python3
"""
cells_ready.py — does this machine hold every cache a run needs? (a chain's readiness guard on a
machine that holds only the streams sent to it: journey §33.101)

Given a run's arguments, exactly as `morphlm.cli run` takes them: for the train and validation
splits, the stream's ids must be cached and, for each fact the run reads, a codes file of that
stream must hold it (`stream.stream_codes_files`, the loader's own rule). Exits 0 if so; prints
what is missing and exits 1 otherwise. Nothing is built.

    python scripts/ops/cells_ready.py --facts pos -o data.bpe_path=data/bpe_17k.json -o train.seed=0
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--facts', default='')
    ap.add_argument('--pipeline', default='bpe')
    ap.add_argument('-o', '--override', action='append', default=[], dest='over')
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    a = ap.parse_args()
    import numpy as np
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem, stream_codes_files
    cfg = apply_overrides(ExperimentConfig(), [f'data.pipeline={a.pipeline}']
                          + ([f'frozen.facts={a.facts}'] if a.facts else []) + a.over)
    facts = [f for f in a.facts.split('+') if f] if a.facts else []
    missing = []
    for split in ('train', 'validation'):
        stem = cache_stem(a.pipeline, split, None, cfg.surface_spec())
        ids = os.path.join(a.cache_dir, stem + '.npy')
        if not os.path.exists(ids):
            missing.append(f'{split} ids {stem}')
            continue
        n = int(np.load(ids, mmap_mode='r').shape[0])
        held = set()
        for _, _, meta in stream_codes_files(a.cache_dir, a.pipeline, split, None, stem, n):
            held |= set(meta.get('fact_names', []))
        missing += [f'{split} codes of {f} for {stem}' for f in facts if f not in held]
    if missing:
        print('  missing: ' + '; '.join(missing))
        sys.exit(1)


if __name__ == '__main__':
    main()
