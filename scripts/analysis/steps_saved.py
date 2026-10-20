#!/usr/bin/env python3
"""
steps_saved.py — what a feature's early help is worth in training steps (journey §33.33,
restated in §33.91).

For a feature and its same-shape null at the same seeds, the feature's validation BPB at
step k is a level the null reaches later, at step k'. Interpolating the null's curve
linearly between its evaluations gives k'; k' − k is the training the feature saves at
step k, and (k' − k) / k its share. Computed per seed, then averaged over the seeds both
cells share on one machine (FINDINGS, method point 11) — the method of §33.33, which this
reproduces for English to the step.

The pairs are the 50k runs of §33.33: English part of speech and inflection against their
null, and Ukrainian case on 151M tokens against the null constant on every word (G-73).
A record goes to results/audits/steps_saved_<UTC timestamp>.json.

    python scripts/analysis/steps_saved.py
"""
import os
import sys
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs                     # noqa: E402
from morphlm.record import AuditRecord       # noqa: E402

STEPS = (1000, 2000, 5000, 10000)
PAIRS = (
    ('English part of speech', 'bpe_mconst-LEARN-CAUSAL_bpe_17k', 'bpe_pos-LEARN-CAUSAL_bpe_17k'),
    ('English inflection', 'bpe_mconst-LEARN-CAUSAL_bpe_17k', 'bpe_mfeatlex-LEARN-CAUSAL_bpe_17k'),
    ('Ukrainian case, 151M', 'bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2',
     'bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2'),
)


def reached(curve, level):
    """The step at which a falling curve {step: BPB} first reaches `level`, interpolated."""
    return runs.at([(curve[x], x) for x in sorted(curve)], level)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs.add_results_flag(ap)
    a = ap.parse_args()
    R = runs.load(runs.results_of(a))
    run_id = lambda r: os.path.basename(os.path.dirname(r['_path']))
    out = []
    for name, null, feat in PAIRS:
        pairs, left = runs.pair(R, null, feat)
        if left:
            print(f'  {name:24} {runs.left_out(left)}')
        seeds = sorted(pairs)
        if not seeds:
            sys.exit(f'{name}: no paired runs of {feat} and {null}')
        row = {'feature': name, 'null': null, 'feature_label': feat, 'seeds': seeds,
               'run_ids': {s: [run_id(R[(null, s)]), run_id(R[(feat, s)])] for s in seeds}, 'at': {}}
        if left:
            row['left_out'] = {str(s): list(m) for s, m in left.items()}
        for k in STEPS:
            per = []
            for s in seeds:
                n = {e['step']: e['bits_per_byte'] for e in R[(null, s)]['log']}
                f = {e['step']: e['bits_per_byte'] for e in R[(feat, s)]['log']}
                x = reached(n, f[k]) if k in f else None
                per.append(None if x is None else x - k)
            mean = None if None in per else st.mean(per)
            row['at'][k] = {'saved_steps': mean, 'share': None if mean is None else mean / k, 'per_seed': per}
        out.append(row)
        print(f'  {name:24} seeds {seeds}  ' + '  '.join(
            f'{k // 1000}k {v["saved_steps"]:5.0f} ({100 * v["share"]:.1f}%)' if v['saved_steps'] is not None
            else f'{k // 1000}k —' for k, v in row['at'].items()))
    rec = AuditRecord('steps_saved', a, out_dir=runs.AUDITS)
    rec.add('steps', STEPS).add('pairs', out)
    print(f'  [record] {rec.write(verbose=False)}')


if __name__ == '__main__':
    main()
