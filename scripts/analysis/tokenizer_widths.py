#!/usr/bin/env python3
"""
tokenizer_widths.py — the tokenizer's lead across model widths, at matched null BPB
(DIRECTIONS D8; journey §33.94, §33.120).

For each condition — width, corpus size, budget — BPE 17k's null against the endings-apart
null, paired by seed within one machine (FINDINGS, method point 11; a seed whose runs ran on
two machines is named and left out): per evaluation step, the mean of BPE 17k's null BPB (how far the model
has got: the law's axis) and the mean lead (BPE 17k's null minus the endings-apart null), kept
to each run's best step. The lead is then read at common null BPB levels by linear
interpolation between evaluations, so that widths are compared at the same point of learning
rather than at the same step. Each seed's own curve is read at the same levels too, for the
spread over seeds (mean ± standard error, seeds positive).

--machine keeps one machine's records, as morphlm.runs.load(machine=...) reads them; 'second'
takes the second machine's twin copies too (results/audits/device_*/), so that width 1024, run
there only, is compared with width 512's twins on the same machine.

A record goes to results/audits/tokenizer_widths_<UTC timestamp>.json.

    python scripts/analysis/tokenizer_widths.py
    python scripts/analysis/tokenizer_widths.py --machine second
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs                     # noqa: E402
from morphlm.record import AuditRecord       # noqa: E402

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
    ('width 1024, ~38M, 10k', '0.05', '_st10k_w1024'),
    ('width 1024, ~38M, 20k', '0.05', '_st20k_w1024'),
]


def lead_curve(R, ref, other):
    """[(mean reference null BPB, mean lead, step)] over the seeds both share on one machine, to each
    best step (runs.gain_curve); each seed's own [(null BPB, lead)] over the same steps, in seed
    order; the seeds; the seeds left out."""
    pts, seeds, left = runs.gain_curve(R, ref, other)
    own = [[v[k] for _, _, _, v in pts] for k in range(len(seeds))]
    return [(b, g, x) for x, b, g, _ in pts], own, seeds, left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--machine', choices=('thesis', 'second'), default=None,
                    help="one machine's records only ('second' with its twin copies); by default every "
                         'record, each condition paired within one machine')
    runs.add_results_flag(ap)
    a = ap.parse_args()
    R = runs.load(runs.results_of(a), machine=a.machine)
    out = []
    print(f'  {"condition":22} n ' + ' '.join(f'{l:>7}' for l in LEVELS) + '   end: null BPB, lead')
    for name, fr, x in CONDITIONS:
        pts, own, seeds, left = lead_curve(R, NULL('bpe_uk_17k', fr, x), NULL('bpeend_uk_17k', fr, x))
        if left:
            print(f'  {name:22} {runs.left_out(left)}')
        if not pts:
            print(f'  {name:22} no paired runs')
            continue
        vals = {str(l): runs.at(pts, l) for l in LEVELS}
        spread = {str(l): runs.summary([runs.at(c, l) for c in own]) for l in LEVELS}
        end = runs.summary([c[-1][1] for c in own])
        out.append({'condition': name, 'reference': NULL('bpe_uk_17k', fr, x), 'other': NULL('bpeend_uk_17k', fr, x),
                    'seeds': seeds, 'lead_at_null_bpb': vals, 'per_seed_at_null_bpb': spread,
                    'end': {'null_bpb': pts[-1][0], 'lead': pts[-1][1], 'step': pts[-1][2], 'per_seed': end},
                    'curve': [{'step': s, 'null_bpb': a, 'lead': g} for a, g, s in pts]})
        if left:
            out[-1]['left_out'] = {str(s): list(m) for s, m in left.items()}
        print(f'  {name:22} {len(seeds)} ' + ' '.join(f'{v:+.4f}' if v is not None else '      —' for v in vals.values())
              + f'   {pts[-1][0]:.4f}, {pts[-1][1]:+.4f}')
        if len(seeds) > 1:     # each seed's own curve at the levels: ± standard error, seeds positive
            print(f'  {"":22}   ' + ' '.join(f'±{s["se"]:.4f}' if s and s['se'] is not None else '      —'
                                             for s in spread.values())
                  + f'   ±{end["se"]:.4f}')
            print(f'  {"":22}   ' + ' '.join(f'{s["positive"]:>3}/{s["n"]:<3}' if s else '      —' for s in spread.values())
                  + f'   {end["positive"]}/{end["n"]}')
    rec = AuditRecord('tokenizer_widths', a, out_dir=runs.AUDITS)
    rec.add('levels', LEVELS).add('conditions', out)
    print(f'  [record] {rec.write(verbose=False)}')


if __name__ == '__main__':
    main()
