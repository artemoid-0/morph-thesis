#!/usr/bin/env python3
"""
steps_saved.py — what a feature's early help is worth in training steps (journey §33.33,
restated in §33.91).

For a feature and its same-shape null at the same seeds, the feature's validation BPB at
step k is a level the null reaches later, at step k'. Interpolating the null's curve
linearly between its evaluations gives k'; k' − k is the training the feature saves at
step k, and (k' − k) / k its share. Computed per seed, then averaged over the seeds both
cells share — the method of §33.33, which this reproduces for English to the step.

The pairs are the 50k runs of §33.33: English part of speech and inflection against their
null, and Ukrainian case on 151M tokens against the null constant on every word (G-73).
A record goes to results/audits/steps_saved_<UTC timestamp>.json.

    python scripts/analysis/steps_saved.py
"""
import os
import sys
import glob
import json
import datetime
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEPS = (1000, 2000, 5000, 10000)
PAIRS = (
    ('English part of speech', 'bpe_mconst-LEARN-CAUSAL_bpe_17k', 'bpe_pos-LEARN-CAUSAL_bpe_17k'),
    ('English inflection', 'bpe_mconst-LEARN-CAUSAL_bpe_17k', 'bpe_mfeatlex-LEARN-CAUSAL_bpe_17k'),
    ('Ukrainian case, 151M', 'bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2',
     'bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2'),
)


def load(root):
    runs = {}
    for p in glob.glob(os.path.join(root, '**', 'result.json'), recursive=True):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        runs.setdefault((r['label'], r['seed']), (os.path.basename(os.path.dirname(p)), r))
    return runs


def reached(curve, level):
    """The step at which a falling curve {step: BPB} first reaches `level`, interpolated."""
    xs = sorted(curve)
    for a, b in zip(xs, xs[1:]):
        if curve[a] >= level >= curve[b] and curve[a] != curve[b]:
            return a + (b - a) * (curve[a] - level) / (curve[a] - curve[b])
    return None


def main():
    runs = load(os.path.join(ROOT, 'results'))
    out = []
    for name, null, feat in PAIRS:
        seeds = sorted(s for (l, s) in runs if l == feat and (null, s) in runs)
        if not seeds:
            sys.exit(f'{name}: no paired runs of {feat} and {null}')
        row = {'feature': name, 'null': null, 'feature_label': feat, 'seeds': seeds,
               'run_ids': {s: [runs[(null, s)][0], runs[(feat, s)][0]] for s in seeds}, 'at': {}}
        for k in STEPS:
            per = []
            for s in seeds:
                n = {e['step']: e['bits_per_byte'] for e in runs[(null, s)][1]['log']}
                f = {e['step']: e['bits_per_byte'] for e in runs[(feat, s)][1]['log']}
                x = reached(n, f[k]) if k in f else None
                per.append(None if x is None else x - k)
            mean = None if None in per else st.mean(per)
            row['at'][k] = {'saved_steps': mean, 'share': None if mean is None else mean / k, 'per_seed': per}
        out.append(row)
        print(f'  {name:24} seeds {seeds}  ' + '  '.join(
            f'{k // 1000}k {v["saved_steps"]:5.0f} ({100 * v["share"]:.1f}%)' if v['saved_steps'] is not None
            else f'{k // 1000}k —' for k, v in row['at'].items()))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '+0000'
    rec = os.path.join(ROOT, 'results', 'audits', f'steps_saved_{ts}.json')
    json.dump({'script': 'scripts/analysis/steps_saved.py', 'timestamp_utc': ts, 'steps': STEPS,
               'results': out}, open(rec, 'w'), indent=1)
    print(f'  [record] {rec}')


if __name__ == '__main__':
    main()
