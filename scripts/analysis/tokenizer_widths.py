#!/usr/bin/env python3
"""
tokenizer_widths.py — the tokenizer's lead across model widths, at matched null BPB
(DIRECTIONS D8; journey §33.94).

For each condition — width, corpus size, budget — BPE 17k's null against the endings-apart
null, paired by seed: per evaluation step, the mean of BPE 17k's null BPB (how far the model
has got: the law's axis) and the mean lead (BPE 17k's null minus the endings-apart null), kept
to each run's best step. The lead is then read at common null BPB levels by linear
interpolation between evaluations, so that widths are compared at the same point of learning
rather than at the same step.

A record goes to results/audits/tokenizer_widths_<UTC timestamp>.json.

    python scripts/analysis/tokenizer_widths.py
"""
import os
import glob
import json
import datetime
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVELS = [1.30, 1.18, 1.11, 1.04, 1.006, 0.98, 0.95, 0.92, 0.905, 0.89, 0.88, 0.86, 0.84, 0.82]
NULL = lambda t, fr, x: f'bpe_mconst-LEARN-CAUSAL_{t}_ukwiki{fr}{x}'
CONDITIONS = [  # name, corpus fraction, label suffix
    ('width 128, ~38M, 10k', '0.05', '_st10k'),
    ('width 128, ~38M, 20k', '0.05', '_st20k'),
    ('width 128, 151M, 50k', '0.2', ''),
    ('width 256, ~38M, 10k', '0.05', '_st10k_w256'),
    ('width 256, ~38M, 20k', '0.05', '_st20k_w256'),
    ('width 512, ~38M, 10k', '0.05', '_st10k_w512'),
    ('width 512, ~38M, 20k', '0.05', '_st20k_w512'),
]


def load():
    runs = {}
    for p in glob.glob(os.path.join(ROOT, 'results', '**', 'result.json'), recursive=True):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        runs.setdefault((r['label'], r['seed']), r)
    return runs


def lead_curve(runs, ref, other):
    """[(mean reference null BPB, mean lead, step)] over the seeds both share, to each best step."""
    seeds = sorted(s for (l, s) in runs if l == other and (ref, s) in runs)
    logs = {(k, s): {e['step']: e['bits_per_byte'] for e in runs[(k, s)]['log'] if e['step'] > 0}
            for k in (ref, other) for s in seeds}
    stop = {s: min(runs[(ref, s)]['best_step'], runs[(other, s)]['best_step']) for s in seeds}
    steps = sorted(set.intersection(*[set(v) for v in logs.values()])) if seeds else []
    return [(st.mean(logs[(ref, s)][x] for s in seeds), st.mean(logs[(ref, s)][x] - logs[(other, s)][x] for s in seeds), x)
            for x in steps if all(x <= stop[s] for s in seeds)], seeds


def at(points, level):
    for (a0, g0, _), (a1, g1, _) in zip(points, points[1:]):
        if a0 >= level >= a1 and a0 != a1:
            return g0 + (g1 - g0) * (a0 - level) / (a0 - a1)
    return None


def main():
    runs = load()
    out = []
    print(f'  {"condition":22} n ' + ' '.join(f'{l:>7}' for l in LEVELS) + '   end: null BPB, lead')
    for name, fr, x in CONDITIONS:
        pts, seeds = lead_curve(runs, NULL('bpe_uk_17k', fr, x), NULL('bpeend_uk_17k', fr, x))
        if not pts:
            print(f'  {name:22} no paired runs')
            continue
        vals = {str(l): at(pts, l) for l in LEVELS}
        out.append({'condition': name, 'reference': NULL('bpe_uk_17k', fr, x), 'other': NULL('bpeend_uk_17k', fr, x),
                    'seeds': seeds, 'lead_at_null_bpb': vals,
                    'end': {'null_bpb': pts[-1][0], 'lead': pts[-1][1], 'step': pts[-1][2]},
                    'curve': [{'step': s, 'null_bpb': a, 'lead': g} for a, g, s in pts]})
        print(f'  {name:22} {len(seeds)} ' + ' '.join(f'{v:+.4f}' if v is not None else '      —' for v in vals.values())
              + f'   {pts[-1][0]:.4f}, {pts[-1][1]:+.4f}')
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '+0000'
    rec = os.path.join(ROOT, 'results', 'audits', f'tokenizer_widths_{ts}.json')
    json.dump({'script': 'scripts/analysis/tokenizer_widths.py', 'timestamp_utc': ts, 'levels': LEVELS,
               'results': out}, open(rec, 'w'), indent=1)
    print(f'  [record] {rec}')


if __name__ == '__main__':
    main()
