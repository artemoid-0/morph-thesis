#!/usr/bin/env python3
"""
run_ids.py — the run id a command line WOULD produce, without running it.

Chains need the id of a run they just made (to probe it, say); computing it
from the same overrides avoids parsing logs. One id per line; --widths repeats
the command for each model width.

    python scripts/ops/run_ids.py --facts mconst -o train.seed=0 --widths 128 256

--done makes a chain's runs resumable pieces (journey §33.51): it exits 0 —
printing "[skip] ..." — if results/ already holds a FINISHED run (result.json)
of the same cell and seed, and 1 otherwise. "Same cell and seed" means the
same label and seed: the identity every analysis groups by. Run ids hash the
whole configuration, so an equivalent run recorded before a config field was
added has another id; matching on label and seed still finds it, and a re-run
that would only duplicate a seed is never started (journey §33.48).

    python scripts/ops/run_ids.py --done --facts mflat -o model.position=rope -o train.seed=0
"""
import os
import sys
import glob
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def finished_run(results_dir, label, seed):
    """The id of a finished run of this label and seed in results_dir, or None."""
    for p in sorted(glob.glob(os.path.join(results_dir, '*', 'result.json'))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue                    # unreadable or half-written: not finished
        if r.get('label') == label and r.get('seed') == seed:
            return os.path.basename(os.path.dirname(p))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--facts', default='')
    ap.add_argument('--pipeline', default='bpe')
    ap.add_argument('-o', '--override', action='append', default=[], dest='over')
    ap.add_argument('--widths', nargs='*', type=int, default=None)
    ap.add_argument('--done', action='store_true',
                    help='exit 0 if results/ holds a finished run of this cell and seed')
    ap.add_argument('--bpb', action='store_true',
                    help="print the finished run's bits per byte (a chain's guard); exit 1 if there is none")
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    a = ap.parse_args()
    from morphlm.config import ExperimentConfig, apply_overrides
    base = [f'data.pipeline={a.pipeline}'] + ([f'frozen.facts={a.facts}'] if a.facts else []) + a.over
    if a.done or a.bpb:
        if a.widths:
            sys.exit('--done and --bpb check one configuration; give the width as -o model.d_model=N')
        cfg = apply_overrides(ExperimentConfig(), base)
        rid = finished_run(a.results_dir, cfg.label(), cfg.train.seed)
        if rid and a.bpb:
            print(json.load(open(os.path.join(a.results_dir, rid, 'result.json')))['bits_per_byte'])
            sys.exit(0)
        if rid:
            print(f'  [skip] {cfg.label()} seed {cfg.train.seed}: already done ({rid})')
            sys.exit(0)
        sys.exit(1)
    for w in (a.widths or [None]):
        cfg = apply_overrides(ExperimentConfig(), base + ([f'model.d_model={w}'] if w else []))
        print(cfg.run_id())


if __name__ == '__main__':
    main()
