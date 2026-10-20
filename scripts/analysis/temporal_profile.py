#!/usr/bin/env python3
"""
temporal_profile.py — when during training does a feature's gain appear? (journey §32.2)

Every run records its validation BPB every 500 steps, so a cell's gain over a baseline can be
read along training at no cost. For every cell whose own baseline — its label with the features
replaced by `baseline` (export_results.py's rule) — is the one given, paired by seed within one
machine (FINDINGS, method point 11; a seed whose runs ran on two machines is named and left out),
this prints the mean gain at the end of training (step 49,999), the share of it reached by 5k and
10k steps, and the peak; and it classes the profile by the share at 5k: FADES above 200%,
PLATEAUS above 80%, GROWS below. A cell whose final gain is under 0.002 BPB is not profiled. What
the three profiles mean, and the caveats that travel with them, are §32.2's. The baseline here is
the plain one: the same-shape nulls and the learning law (learning_law.py, journey §33) answer the
question since.

A run is read when its log holds more than 50 evaluations; of two runs of one cell and seed, the
later in path order is used. --pairs CELL:BASELINE prints both cells' mean BPB along training.

    python scripts/analysis/temporal_profile.py                 # all cells vs baseline
    python scripts/analysis/temporal_profile.py --pairs det_baseline_Spdiwc:bpe_baseline_bpe_17k
"""

import os
import sys
import argparse
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

STEPS = (500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 49999)


def load(roots):
    """{label: {seed: record}} — runs whose log holds more than 50 evaluations; of two runs of one
    cell and seed, the later in path order."""
    by = defaultdict(dict)
    for p, r in runs.records(roots):
        if r.get('log') and len(r['log']) > 50:
            by[r['label']][r['seed']] = r
    return by


def one_machine(cell, base):
    """The seeds both cells have, on one machine (runs.pair): (seeds, the seeds left out)."""
    R = {(k, s): r for k, d in (('base', base), ('cell', cell)) for s, r in d.items()}
    pairs, left = runs.pair(R, 'base', 'cell')
    return sorted(pairs), left


def bpb_at(r, step):
    for e in r['log']:
        if e['step'] == step:
            return e['bits_per_byte']
    return None


def profile(cell, base, seeds):
    if not seeds:
        return None
    out = {}
    for stp in STEPS:
        g = [bpb_at(base[s], stp) - bpb_at(cell[s], stp) for s in seeds]
        if any(x is None for x in g):
            continue
        out[stp] = st.mean(g)
    fin = out.get(49999)
    if fin is None or abs(fin) < 0.002:
        return None
    steps_all = [e['step'] for e in cell[seeds[0]]['log'] if e['step'] > 0]
    peak = max(st.mean(bpb_at(base[s], t) - bpb_at(cell[s], t) for s in seeds)
               for t in steps_all if all(bpb_at(cell[s], t) is not None and
                                          bpb_at(base[s], t) is not None
                                          for s in seeds))
    e5, e10 = out.get(5000, 0) / fin, out.get(10000, 0) / fin
    if e5 > 2.0:
        kind = 'FADES'
    elif e5 > 0.8:
        kind = 'PLATEAUS'
    else:
        kind = 'GROWS'
    return {'seeds': seeds, 'final': fin, 'at5k': e5, 'at10k': e10,
            'peak': peak / fin, 'kind': kind, 'curve': out}


def main():
    ap = argparse.ArgumentParser()
    runs.add_results_flag(ap)
    ap.add_argument('--baseline', default='bpe_baseline_bpe_17k')
    ap.add_argument('--pairs', nargs='*', default=[],
                    help='extra cell:baseline pairs, e.g. for Q1')
    args = ap.parse_args()

    from morphlm.record import AuditRecord
    rec = AuditRecord('temporal_profile', args, out_dir=runs.AUDITS)

    by = load(runs.results_of(args))
    base = by[args.baseline]
    print('=' * 88)
    print(f'TEMPORAL PROFILE — gain over training, against {args.baseline}')
    print('=' * 88)
    print(f'  {"cell":42} {"final":>8} {"@5k":>6} {"@10k":>6} {"peak":>6} n  profile')
    rows = {}
    for lab in sorted(by):
        # only cells whose OWN baseline is the one given: a 100k-step cell
        # against the 50k baseline would compare different learning-rate
        # schedules (the rule export_results.own_baseline applies)
        if 'baseline' in lab or runs.own_baseline(lab) != args.baseline:
            continue
        seeds, left = one_machine(by[lab], base)
        pr = profile(by[lab], base, seeds)
        if pr:
            rows[lab] = pr
            print(f'  {lab[:42]:42} {pr["final"]:+8.4f} {100*pr["at5k"]:5.0f}% '
                  f'{100*pr["at10k"]:5.0f}% {100*pr["peak"]:5.0f}% {len(pr["seeds"])}  '
                  f'{pr["kind"]}')
        if left:
            print(f'  {lab[:42]:42} {runs.left_out(left)}')
            if pr:
                pr['left_out'] = {str(s): list(m) for s, m in left.items()}
    rec.add('profiles', {k: {kk: vv for kk, vv in v.items() if kk != 'curve'}
                         for k, v in rows.items()})
    rec.add('curves', {k: v['curve'] for k, v in rows.items()})

    for pair in args.pairs:
        c, b = pair.split(':')
        cell, bb = by.get(c, {}), by.get(b, {})
        seeds, left = one_machine(cell, bb)
        if left:
            print(f'\n  {pair}: {runs.left_out(left)}')
        if not seeds:
            print(f'\n  {pair}: no shared seeds')
            continue
        print(f'\n  {c} minus {b} over training (seeds {seeds}):')
        print(f'  {"step":>6} {"base":>8} {"cell":>8} {"gap":>8} {"rel":>7}')
        curve = {}
        for stp in STEPS:
            vb = st.mean(bpb_at(bb[s], stp) for s in seeds)
            vc = st.mean(bpb_at(cell[s], stp) for s in seeds)
            curve[stp] = (vb, vc)
            print(f'  {stp:6} {vb:8.4f} {vc:8.4f} {vc-vb:+8.4f} {100*(vc-vb)/vb:6.2f}%')
        rec.add(f'pair:{c}:{b}', {str(k): v for k, v in curve.items()})

    print()
    print('  READING THIS')
    print('    FADES     peaks early, shrinks toward zero: the model learns the')
    print('              same information itself (causal word identity)')
    print('    PLATEAUS  ~100% by 5k steps: exploitable immediately — leaky')
    print('              features, but also directly predictive ones like dep')
    print('    GROWS     still rising at 50k: complementary to what the model')
    print('              learns (causal structure). The final gain is a floor.')
    print('    Frozen cells are slow regardless; read learned cells.')
    rec.note('The plateau profile is consistent with shortcut learning (a '
             'leak is a simple, perfectly predictive feature) but is not a '
             'leak test on its own: dep plateaus and is mostly legitimate.')
    rec.write()


if __name__ == '__main__':
    main()
