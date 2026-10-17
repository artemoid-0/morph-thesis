#!/usr/bin/env python3
"""
paired_cells.py — one cell against another, paired by seed, each pair on one machine.

The project's comparisons are paired by seed: two cells at one seed start alike (or as alike as their
shapes allow) and see the same batches, so their difference is the effect at that seed. Since two
machines run cells (journey §33.101–§33.105), a pair is counted only when both runs come from the same
GPU; a seed whose runs come from two machines is reported and left out.

    python scripts/analysis/paired_cells.py \\
        --pair 'causal pos over the null, 10k' bpe_poscausal-LEARN-CAUSAL_bpe_17k_st10k bpe_mconst-LEARN-CAUSAL_bpe_17k_st10k

Each --pair is NAME A B: the effect is B's bits per byte minus A's at each seed (positive: A is
better). Reports each seed, the mean ± standard error and how many seeds are positive, and writes
results/audits/paired_cells_<timestamp>.json.
"""
import os
import sys
import glob
import json
import math
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def machine(r):
    gpu = (r.get('environment') or {}).get('gpu') or ''
    return '5090' if '5090' in gpu else ('4060' if ('4060' in gpu or not gpu) else gpu)


def load(results):
    runs = {}
    for p in glob.glob(os.path.join(results, '*', 'result.json')):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if 'label' in r and 'seed' in r and 'bits_per_byte' in r:
            runs.setdefault((r['label'], r['seed']), r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=3, action='append', required=True, metavar=('NAME', 'A', 'B'))
    ap.add_argument('--seeds', nargs='*', type=int, default=None, help='default: every seed both cells have')
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    a = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('paired_cells', a)
    runs = load(a.results_dir)
    out = []
    for name, la, lb in a.pair:
        seeds = sorted({s for (l, s) in runs if l == la} & {s for (l, s) in runs if l == lb})
        if a.seeds is not None:
            seeds = [s for s in seeds if s in a.seeds]
        per, skipped = {}, {}
        for s in seeds:
            ra, rb = runs[(la, s)], runs[(lb, s)]
            if machine(ra) != machine(rb):
                skipped[s] = [machine(ra), machine(rb)]
                continue
            per[s] = {'effect': rb['bits_per_byte'] - ra['bits_per_byte'], 'A': ra['bits_per_byte'],
                      'B': rb['bits_per_byte'], 'machine': machine(ra), 'runs': [ra['run_id'], rb['run_id']]}
        xs = [v['effect'] for v in per.values()]
        m = st.mean(xs) if xs else None
        se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
        row = {'name': name, 'A': la, 'B': lb, 'seeds': per, 'mean': m, 'se': se,
               'positive': sum(x > 0 for x in xs), 'n': len(xs), 'skipped_across_machines': skipped}
        out.append(row)
        print(f'\n  {name}\n    A {la}\n    B {lb}')
        if xs:
            print('    ' + '  '.join(f's{s} {v["effect"]:+.4f}' for s, v in per.items())
                  + f'   mean {m:+.4f}' + (f' ± {se:.4f}' if se is not None else '') + f'  ({row["positive"]}/{len(xs)} positive)')
        else:
            print('    no seed has both cells on one machine')
        if skipped:
            print(f'    left out, runs on two machines: {skipped}')
    rec.add('pairs', out)
    rec.note('effect = B minus A in bits per byte at each seed (positive: A better); pairs on one machine only')
    rec.write()


if __name__ == '__main__':
    main()
