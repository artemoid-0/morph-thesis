#!/usr/bin/env python3
"""
practical_value.py — what the tokenizer's gain is worth, as training steps or parameters plain BPE would need
(the text's list of 30 October, item 4; journey §33.155).

WHY
---
A gain in bits per byte says little to a practitioner. Two equivalents are read from runs already made, each
a model trained to the end of its own schedule — never a step inside a cosine schedule, which is not a shorter
budget:
  * steps: BPE 17k's null at 10k and at 20k, on the same ~38M tokens at width 128, against endings apart's
    null at 10k — how many steps of its own schedule BPE 17k needs to reach that BPB;
  * parameters: BPE 17k's null at widths 128 and 256, at 10k, against endings apart's at width 128 — how many
    parameters BPE 17k needs.
Each is read per seed, paired, by interpolating BPE 17k's BPB linearly in the logarithm of steps or parameters
between its two runs; a seed whose target lies outside the two is reported as beyond them, never extrapolated.
The thesis machine's runs only.

    python scripts/analysis/practical_value.py
"""
import os
import sys
import math
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

N = 'bpe_mconst-LEARN-CAUSAL_'
B, E = f'{N}bpe_uk_17k_ukwiki0.05', f'{N}bpeend_uk_17k_ukwiki0.05'
FORMS = [
    {'name': 'training steps, width 128, ~38M tokens', 'axis': 'steps',
     'target': E + '_st10k', 'low': (B + '_st10k', 10000), 'high': (B + '_st20k', 20000)},
    {'name': 'parameters, 10k steps, ~38M tokens', 'axis': 'params',
     'target': E + '_st10k', 'low': (B + '_st10k', None), 'high': (B + '_st10k_w256', None)},
    {'name': 'parameters, 20k steps, ~38M tokens', 'axis': 'params',
     'target': E + '_st20k', 'low': (B + '_st20k', None), 'high': (B + '_st20k_w256', None)},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('practical_value', argparse.Namespace(), out_dir=runs.AUDITS)
    R = runs.load(machine='thesis', keep=lambda r: 'seed' in r and 'bits_per_byte' in r)
    out = []
    for f in FORMS:
        (lo, xlo), (hi, xhi) = f['low'], f['high']
        pairs, left = runs.pair(R, f['target'], lo, hi)
        per = {}
        for s, (rt, rl, rh) in sorted(pairs.items()):
            a = xlo if xlo is not None else rl['parameters']['trainable_total']
            b = xhi if xhi is not None else rh['parameters']['trainable_total']
            t, yl, yh = rt['bits_per_byte'], rl['bits_per_byte'], rh['bits_per_byte']
            entry = {'target_bpb': t, 'low_bpb': yl, 'high_bpb': yh, 'low_x': a, 'high_x': b,
                     'runs': [rt['run_id'], rl['run_id'], rh['run_id']]}
            if yh < t <= yl:
                frac = (yl - t) / (yl - yh)
                x = math.exp(math.log(a) + frac * (math.log(b) - math.log(a)))
                entry.update(equivalent=x, multiple=x / a)
            else:
                entry.update(equivalent=None, multiple=None, beyond=('above the low run' if t > yl else 'below the high run'))
            per[str(s)] = entry
        mult = [e['multiple'] for e in per.values() if e['multiple'] is not None]
        sm = runs.summary(mult)
        out.append(dict(f, seeds=per, summary=sm, left_out={str(k): list(v) for k, v in left.items()}))
        print(f'\n  {f["name"]}: endings apart {f["target"]} against BPE 17k {lo} and {hi}')
        for s, e in per.items():
            got = (f'{e["multiple"]:.2f}× ({e["equivalent"]:,.0f})' if e['multiple'] is not None else e['beyond'])
            print(f'    seed {s}: target {e["target_bpb"]:.4f}; BPE 17k {e["low_bpb"]:.4f} at {e["low_x"]:,} and '
                  f'{e["high_bpb"]:.4f} at {e["high_x"]:,} -> {got}')
        if sm:
            print(f'    mean {sm["mean"]:.2f}× ± {sm["se"] or 0:.2f} over {sm["n"]} seed(s)')
    rec.add('forms', out)
    rec.write()


if __name__ == '__main__':
    main()
