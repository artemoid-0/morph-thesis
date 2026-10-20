#!/usr/bin/env python3
"""
backfill_results.py — recover the derivable provenance for runs that predate it.

Twenty-five runs completed before `result.json` recorded parameter counts,
corpus statistics or convergence indicators. Some of that is recoverable from
the saved config and the evaluation log; some is not.

    RECOVERABLE            parameter breakdown, frozen block detail,
                           convergence indicators, timing from the log
    NOT RECOVERABLE        library versions, GPU model, git commit,
                           per-frequency buckets without a checkpoint

Anything not recoverable is left absent rather than guessed. A field that says
nothing is honest; a field that says something invented is not.

    python scripts/ops/backfill_results.py            # dry run
    python scripts/ops/backfill_results.py --write
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def rebuild_cfg(cfg_dict):
    # a config without a `surface` section predates it: the legacy stream, every channel off
    from morphlm.config import ExperimentConfig
    return ExperimentConfig.from_dict(cfg_dict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--write', action='store_true',
                    help='actually modify result.json (default is a dry run)')
    args = ap.parse_args()

    from morphlm.provenance import frozen_detail, convergence

    touched, skipped = 0, 0
    for entry in sorted(os.listdir(args.results_dir)):
        rp = os.path.join(args.results_dir, entry, 'result.json')
        cp = os.path.join(args.results_dir, entry, 'config.json')
        if not (os.path.exists(rp) and os.path.exists(cp)):
            continue
        with open(rp) as f:
            r = json.load(f)
        with open(cp) as f:
            cfg_dict = json.load(f)

        added = []
        try:
            cfg = rebuild_cfg(cfg_dict)
        except Exception as e:
            print(f'  {entry}: cannot rebuild config — {e}')
            skipped += 1
            continue

        if 'frozen_blocks' not in r:
            # NOT wrapped in a bare `except: pass`. An earlier version was, and
            # it silently hid an ImportError in frozen_detail for all 25 runs —
            # the same bug would have crashed a real training run at save time.
            # Failures are reported, never swallowed.
            try:
                r['frozen_blocks'] = frozen_detail(cfg)
                added.append('frozen_blocks')
            except Exception as e:
                print(f'  {entry}: frozen_blocks FAILED — '
                      f'{type(e).__name__}: {e}')

        if 'convergence' not in r and r.get('log'):
            r['convergence'] = convergence(r['log'],
                                           r.get('max_steps', 50000))
            added.append('convergence')

        if 'parameters' not in r:
            # derivable from config alone: build the model WITHOUT weights
            try:
                from morphlm.modeling.transformer import build_model
                from morphlm.provenance import parameter_breakdown
                model, _ = build_model(cfg, verbose=False)
                r['parameters'] = parameter_breakdown(model, cfg)
                added.append('parameters')
            except Exception as e:
                print(f'  {entry}: parameters not recoverable — '
                      f'{type(e).__name__}')

        if 'timing' not in r and r.get('log'):
            last = max(r['log'], key=lambda e: e['step'])
            if last.get('elapsed'):
                r['timing'] = {
                    'wall_seconds': last['elapsed'],
                    'steps_completed': last['step'],
                    'steps_per_second': (last['step'] / last['elapsed']
                                         if last['elapsed'] else None),
                    'device': None,          # not recorded at the time
                    'reconstructed': True,
                }
                added.append('timing')

        r.setdefault('provenance_note',
                     'environment, GPU and per-frequency buckets are not '
                     'recoverable for runs predating provenance capture')

        if added:
            touched += 1
            print(f'  {entry}: + {", ".join(added)}')
            if args.write:
                with open(rp, 'w') as f:
                    json.dump(r, f, indent=2, default=str)

    print()
    print(f'{touched} run(s) would gain fields, {skipped} skipped')
    if not args.write:
        print('DRY RUN — pass --write to apply')
    else:
        print('written')
    print()
    print('Not recoverable for these runs, and deliberately left absent:')
    print('  library versions, GPU model, git commit, per-frequency buckets')


if __name__ == '__main__':
    main()
