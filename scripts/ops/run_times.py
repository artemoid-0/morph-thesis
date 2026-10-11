#!/usr/bin/env python3
"""
run_times.py — how long runs of a kind took, from their records (`timing` in each
result.json): the estimate a plan is built from is read, not guessed
(the queue review in docs/DIRECTIONS.md).

One line per label matching every pattern given (regular expressions): how many
runs, which seeds, the shortest, median and longest wall time in minutes, and the
steps. With --sum, the total of the matching runs' longest times by label times a
count — what N more seeds of each matching label would take.

    python scripts/ops/run_times.py ukwiki0.2$ mconst
    python scripts/ops/run_times.py 'ukwiki0.05_st10k_w256'
    python scripts/ops/run_times.py 'bpeend_uk_17k_ukwiki0.2' --sum 4
"""
import os
import re
import sys
import json
import glob
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('patterns', nargs='*', help='regular expressions every label must match')
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--sum', type=int, default=None, metavar='N',
                    help='N runs of each matching label, at its longest recorded time')
    a = ap.parse_args()
    pats = [re.compile(p) for p in a.patterns]
    by = {}
    for p in glob.glob(os.path.join(a.results, '*', 'result.json')):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        w = (r.get('timing') or {}).get('wall_seconds')
        lab = r.get('label') or ''
        if not w or not all(q.search(lab) for q in pats):
            continue
        by.setdefault(lab, []).append((r.get('seed'), w / 60, r.get('max_steps')))
    if not by:
        sys.exit('no recorded run matches ' + ' '.join(a.patterns))
    total = 0.0
    for lab in sorted(by):
        v = by[lab]
        m = [x[1] for x in v]
        print(f'{lab:72} n={len(v):<2} seeds={",".join(str(x[0]) for x in sorted(v, key=lambda x: (x[0] is None, x[0])))}'
              f'  min {min(m):5.1f}  median {st.median(m):5.1f}  max {max(m):5.1f}  steps {v[0][2]}')
        if a.sum:
            total += a.sum * max(m)
    if a.sum:
        print(f'\n{a.sum} run(s) of each of {len(by)} label(s) at its longest time: {total:.0f} min ({total / 60:.1f} h)')


if __name__ == '__main__':
    main()
