#!/usr/bin/env python3
"""
blimp_item_split.py — is a BLiMP gain agreement, or a shifted preference?

A model can score well on a minimal-pair paradigm without the knowledge it
tests. In anaphor gender agreement, a model that simply prefers "herself" to
"himself" scores the share of items whose GOOD sentence uses "herself" — near
chance if the paradigm is balanced, far from it if not. The baseline's accuracy
on that paradigm swings from 47.6% to 67.0% across seeds (journey §33.6), which
is what arbitrary preferences would do. A feature that nudged the preference
would then look like an agreement gain.

The test: group the items by WHAT CHANGES between the good and the bad sentence
(for anaphor gender: "herself > himself", "himself > herself", ...), and read
each group's paired effect separately.

  * AGREEMENT gains in every direction.
  * A shifted PREFERENCE gains in one direction and LOSES in the reverse one.

Reads a blimp_eval record and its saved per-item margins — nothing is re-scored.

Scope: the directional test needs paradigms whose changes come in both
directions (gender, number, determiner-noun). Ellipsis items MOVE a word
instead ("big > big"); their groups show where a gain sits, not whether it is
a preference.

    python scripts/eval/blimp_item_split.py                      # latest record
    python scripts/eval/blimp_item_split.py --record results/audits/blimp_eval_X.json \\
        --paradigms anaphor_gender_agreement ellipsis_n_bar_1
"""

import os
import re
import sys
import glob
import json
import argparse
import difflib
import statistics as st
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def change(good, bad):
    """The words that differ, as 'good-side > bad-side' (lower-cased)."""
    w = lambda s: re.findall(r"[\w']+", s.lower())
    g, b = w(good), w(bad)
    gs, bs = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=g, b=b).get_opcodes():
        if op != 'equal':
            gs += g[i1:i2]
            bs += b[j1:j2]
    return f"{' '.join(gs) or '∅'} > {' '.join(bs) or '∅'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', help='blimp_eval audit record (default: latest)')
    ap.add_argument('--paradigms', nargs='*',
                    default=['anaphor_gender_agreement', 'anaphor_number_agreement',
                             'ellipsis_n_bar_1', 'ellipsis_n_bar_2'])
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--reference', help='label to pair against (default: the record\'s)')
    ap.add_argument('--top', type=int, default=8, help='largest change groups shown')
    args = ap.parse_args()

    from blimp_eval import load_pairs
    from morphlm.record import AuditRecord
    rec_out = AuditRecord('blimp_item_split', args)

    path = args.record or sorted(glob.glob(os.path.join(
        args.results_dir, 'audits', 'blimp_eval_*.json')))[-1]
    blob = json.load(open(path))
    R, a = blob['results'], blob['args']
    items = np.load(path[:-len('.json')] + '_items.npz')
    owner = items['owner']
    runs, ref = R['runs'], (args.reference or R['reference'])
    by = defaultdict(dict)
    for rid, r in sorted(runs.items()):
        by[r['label']].setdefault(r['seed'], rid)
    if ref not in by:
        raise SystemExit(f'reference {ref!r} not in this record')
    print(f'  record {os.path.basename(path)}: {len(runs)} checkpoints, reference {ref}')

    wanted = [p for p in args.paradigms if (owner == p).any()]
    missing = [p for p in args.paradigms if p not in wanted]
    if missing:
        print(f'  not in this record: {missing}')
    pairs = load_pairs(wanted, a.get('local'))
    if a.get('limit'):
        pairs = {k: v[:a['limit']] for k, v in pairs.items()}

    out = {}
    for p in wanted:
        idx = np.flatnonzero(owner == p)
        rows = pairs[p]
        if len(rows) != len(idx):
            raise SystemExit(f'{p}: record has {len(idx)} items, dataset {len(rows)} — '
                             f'cannot align; refusing to guess')
        keys = [change(r['sentence_good'], r['sentence_bad']) for r in rows]
        groups = defaultdict(list)
        for k, i in zip(keys, idx):
            groups[k].append(i)
        top = sorted(groups, key=lambda k: -len(groups[k]))[:args.top]
        n = len(idx)
        print(f'\n  {p}  ({n} items, {len(groups)} kinds of change)')
        hdr = f'    {"change (good > bad)":28} {"items":>6}'
        labs = [ref] + sorted(l for l in by if l != ref)
        short = lambda l: l.replace('bpe_', '').replace('_bpe_17k', '').replace('-LEARN', '')[:20]
        print(hdr + ''.join(f'{short(l):>24}' for l in labs))
        out[p] = {}
        for k in top:
            sel = np.array(groups[k])
            row = f'    {k[:28]:28} {len(sel):6}'
            out[p][k] = {'items': int(len(sel))}
            for l in labs:
                accs = {s: float((items[rid][sel] > 0).mean()) for s, rid in by[l].items()}
                cell = f'{100 * st.mean(accs.values()):5.1f}%'
                if l != ref:
                    d = [100 * (accs[s] - float((items[by[ref][s]][sel] > 0).mean()))
                         for s in sorted(accs) if s in by[ref]]
                    if len(d) >= 2:
                        agree = max(sum(x > 0 for x in d), sum(x < 0 for x in d))
                        cell += f' {st.mean(d):+5.1f} {agree}/{len(d)}'
                    elif d:
                        cell += f' {d[0]:+5.1f} n1'
                    out[p][k][l] = d
                row += f'{cell:>24}'
            print(row)
    print()
    print('  READING THIS')
    print('    Agreement gains in EVERY direction of change. A preference shift gains')
    print('    in one direction and LOSES in its reverse (herself > himself up,')
    print('    himself > herself down). Only the first is grammatical knowledge.')
    rec_out.add('record', os.path.basename(path))
    rec_out.add('groups', out)
    rec_out.write()


if __name__ == '__main__':
    main()
