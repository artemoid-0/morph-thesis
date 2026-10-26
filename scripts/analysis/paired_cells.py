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
results/audits/paired_cells_<timestamp>.json. --step reads each run's validation BPB at that
evaluation step instead of its record's final figure (a seed missing the step in either run is
left out and named). --machine keeps one machine's records, as
morphlm.runs.load(machine=...) reads them: where a cell has runs on both machines, the first record
of a label and seed otherwise wins, and a pair split across machines is left out.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

GPU = {'thesis': '4060', 'second': '5090'}       # how this script's records name the machines


def machine(where):
    """A machine (runs.pair's left-out entries, or a record) by its GPU, as this script's records name it."""
    if isinstance(where, dict):
        where = runs.machine(where) or runs.gpu(where)
    return GPU.get(where, where)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=3, action='append', required=True, metavar=('NAME', 'A', 'B'))
    ap.add_argument('--seeds', nargs='*', type=int, default=None, help='default: every seed both cells have')
    ap.add_argument('--machine', choices=('thesis', 'second'), default=None,
                    help="one machine's records only ('second' with its twin copies)")
    ap.add_argument('--step', type=int, default=None, help="each run's validation BPB at this evaluation step")
    runs.add_results_flag(ap)
    a = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('paired_cells', a, out_dir=runs.AUDITS)
    R = runs.load(runs.results_of(a), machine=a.machine, keep=lambda r: 'seed' in r and 'bits_per_byte' in r)
    out = []
    bpb = (lambda r: r['bits_per_byte']) if a.step is None else (lambda r: runs.curve(r).get(a.step))
    for name, la, lb in a.pair:
        pairs, left = runs.pair(R, la, lb)
        keep = lambda s: a.seeds is None or s in a.seeds
        lacking = [s for s, (ra, rb) in pairs.items() if keep(s) and (bpb(ra) is None or bpb(rb) is None)]
        per = {s: {'effect': bpb(rb) - bpb(ra), 'A': bpb(ra), 'B': bpb(rb),
                   'machine': machine(ra), 'runs': [ra['run_id'], rb['run_id']]}
               for s, (ra, rb) in pairs.items() if keep(s) and s not in lacking}
        skipped = {s: [machine(m) for m in ms] for s, ms in left.items() if keep(s)}
        xs = [v['effect'] for v in per.values()]
        sm = runs.summary(xs)
        m, se = (sm['mean'], sm['se']) if sm else (None, None)
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
        if lacking:
            row['lacking_step'] = lacking
            print(f'    left out, no evaluation at step {a.step}: seeds {lacking}')
    rec.add('pairs', out)
    rec.note('effect = B minus A in bits per byte at each seed (positive: A better); pairs on one machine only'
             + ('' if a.step is None else f"; each run's validation BPB at step {a.step}"))
    rec.write()


if __name__ == '__main__':
    main()
