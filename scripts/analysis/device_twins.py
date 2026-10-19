#!/usr/bin/env python3
"""
device_twins.py — do the two machines give the same results? (journey §33.101, §33.103, §33.104)

A TWIN is a run of one machine repeated on the other, with the same arguments and seed: the same
cell, the same initial weights, the same data order. Both machines run the same software in single
precision; only the GPU and its driver differ (§33.104). A twin's difference is therefore each
GPU's own rounding, amplified by training. The question is how far it goes, against the spread
between seeds.

Every record is read — `results/*/result.json` and the twin copies in `results/audits/device_*/`
— classed by the GPU its environment names (a record naming none is the thesis machine's, from
before records named it), and keyed by (label, seed, steps). A key with a record on each machine
is a twin pair.

Reported:
  pairs     the difference at the end (second machine minus thesis machine, bits per byte at the
            best step), and along training: at the first evaluation, the largest, and the first
            step at which it passes 1e-4 and 1e-3
  groups    per cell: mean ± SE of the differences, their spread against the cell's spread between
            seeds on the thesis machine, and their signs
  overall   mean ± SE, the signs, the largest difference
  effects   for chosen pairs of cells, the effect per seed on each machine, and the two machines'
            effects compared seed by seed
  repeats   one run repeated on one machine

    python scripts/analysis/device_twins.py
"""
import os
import sys
import glob
import json
import math
import argparse
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

UKE = 'bpeend_uk_17k_ukwiki0.05_st20k'
UKB = 'bpe_uk_17k_ukwiki0.05_st20k'
UKE10 = 'bpeend_uk_17k_ukwiki0.05_st10k'
UKB10 = 'bpe_uk_17k_ukwiki0.05_st10k'
# (name, A, B, steps): the effect of A over B, B's bits per byte minus A's (positive: A is better)
EFFECTS = [
    ('case over the null, endings apart, 20k', f'bpe_ucase-LEARN-CAUSAL_{UKE}', f'bpe_mconst-LEARN-CAUSAL_{UKE}', 20000),
    ('the empty slot over the null, endings apart, 20k', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKE}', f'bpe_mconst-LEARN-CAUSAL_{UKE}', 20000),
    ('case over the empty slot, endings apart, 20k', f'bpe_ucase-LEARN-CAUSAL_{UKE}', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKE}', 20000),
    ('case off from the start over the null, endings apart, 20k', f'bpe_ucase-LEARN-CAUSAL-OFF0_{UKE}', f'bpe_mconst-LEARN-CAUSAL_{UKE}', 20000),
    ('the empty slot over the null, BPE 17k, 20k', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKB}', f'bpe_mconst-LEARN-CAUSAL_{UKB}', 20000),
    ('case over the empty slot, BPE 17k, 20k', f'bpe_ucase-LEARN-CAUSAL_{UKB}', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKB}', 20000),
    ('case over the null, BPE 17k, 20k', f'bpe_ucase-LEARN-CAUSAL_{UKB}', f'bpe_mconst-LEARN-CAUSAL_{UKB}', 20000),
    ('case over the null, endings apart, 10k', f'bpe_ucase-LEARN-CAUSAL_{UKE10}', f'bpe_mconst-LEARN-CAUSAL_{UKE10}', 10000),
    ('the empty slot over the null, endings apart, 10k', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKE10}', f'bpe_mconst-LEARN-CAUSAL_{UKE10}', 10000),
    ('case over the empty slot, endings apart, 10k', f'bpe_ucase-LEARN-CAUSAL_{UKE10}', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKE10}', 10000),
    ('the empty slot over the null, BPE 17k, 10k', f'bpe_mconst-LEARN-CAUSAL-OFF0_{UKB10}', f'bpe_mconst-LEARN-CAUSAL_{UKB10}', 10000),
    ('the empty slot over the null, endings apart, 151M, 50k', 'bpe_mconst-LEARN-CAUSAL-OFF0_bpeend_uk_17k_ukwiki0.2',
     'bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2', 50000),
    ('case over the empty slot, endings apart, 151M, 50k', 'bpe_ucase-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2',
     'bpe_mconst-LEARN-CAUSAL-OFF0_bpeend_uk_17k_ukwiki0.2', 50000),
    ('endings apart over BPE 17k, English, 50k', 'bpe_baseline_bpeend_17k', 'bpe_baseline_bpe_17k', 50000),
    ('endings apart over BPE 17k, Ukrainian, width 512, 20k', f'bpe_mconst-LEARN-CAUSAL_{UKE}_w512',
     f'bpe_mconst-LEARN-CAUSAL_{UKB}_w512', 20000),
]
# G-78 across tokenizers and corpus sizes at 10k (journey §33.112–§33.113): per stream, the empty slot
# over the null, case over the empty slot, case over the null
G78 = [('Unigram 17k', 'unigram_uk_17k_ukwiki0.05_st10k'), ('BPE 4k', 'bpe_uk_4k_ukwiki0.05_st10k'),
       ('BPE 8k', 'bpe_uk_8k_ukwiki0.05_st10k'), ('BPE 32k', 'bpe_uk_32k_ukwiki0.05_st10k'),
       ('BPE 17k', UKB10), ('BPE 17k at ~1.5M', 'bpe_uk_17k_ukwiki0.002_st10k'),
       ('BPE 17k at ~7.6M', 'bpe_uk_17k_ukwiki0.01_st10k')]
for _name, _s in G78:
    for _what, _a, _b in (('the empty slot over the null', 'mconst-LEARN-CAUSAL-OFF0', 'mconst-LEARN-CAUSAL'),
                          ('case over the empty slot', 'ucase-LEARN-CAUSAL', 'mconst-LEARN-CAUSAL-OFF0'),
                          ('case over the null', 'ucase-LEARN-CAUSAL', 'mconst-LEARN-CAUSAL')):
        if (_name, _what) != ('BPE 17k', 'the empty slot over the null'):     # listed above
            EFFECTS.append((f'{_what}, {_name}, 10k', f'bpe_{_a}_{_s}', f'bpe_{_b}_{_s}', 10000))
# item 7: case, case with prepositions from their form, the word form (151M, BPE 17k, 50k); item 8 (English, 50k)
F7 = 'bpe_uk_17k_ukwiki0.2'
EFFECTS += [
    ('case over its word form, BPE 17k, 151M, 50k', f'bpe_ucase-LEARN-CAUSAL_{F7}', f'bpe_ucaselex-LEARN-CAUSAL_{F7}', 50000),
    ('case over prepositions from their form (the leak), BPE 17k, 151M, 50k', f'bpe_ucase-LEARN-CAUSAL_{F7}',
     f'bpe_ucaseprep-LEARN-CAUSAL_{F7}', 50000),
    ('prepositions from their form over the word form, BPE 17k, 151M, 50k', f'bpe_ucaseprep-LEARN-CAUSAL_{F7}',
     f'bpe_ucaselex-LEARN-CAUSAL_{F7}', 50000),
    ('frequency band over the null, English, 50k', 'bpe_mfreq-LEARN-CAUSAL_bpe_17k', 'bpe_mconst-LEARN-CAUSAL_bpe_17k', 50000),
    ('part of speech from the word form over the null, English, 50k', 'bpe_poslex-LEARN-CAUSAL_bpe_17k',
     'bpe_mconst-LEARN-CAUSAL_bpe_17k', 50000),
]
# item 4 (journey §33.117–§33.118): the same three effects at width 512, ~38M, 10k, under both tokenizers
for _name, _s in (('BPE 17k', UKB10), ('endings apart', UKE10)):
    for _what, _a, _b in (('the empty slot over the null', 'mconst-LEARN-CAUSAL-OFF0', 'mconst-LEARN-CAUSAL'),
                          ('case over the empty slot', 'ucase-LEARN-CAUSAL', 'mconst-LEARN-CAUSAL-OFF0'),
                          ('case over the null', 'ucase-LEARN-CAUSAL', 'mconst-LEARN-CAUSAL')):
        EFFECTS.append((f'{_what}, {_name}, width 512, 10k', f'bpe_{_a}_{_s}_w512', f'bpe_{_b}_{_s}_w512', 10000))
# the tokenizer's gain over each tokenizer's null and over its empty slot, at widths 128 and 512 (§33.118)
for _w in ('', '_w512'):
    for _ref, _a in (('the nulls', 'mconst-LEARN-CAUSAL'), ('the empty slots', 'mconst-LEARN-CAUSAL-OFF0')):
        EFFECTS.append((f'endings apart over BPE 17k over {_ref}, Ukrainian, width {512 if _w else 128}, 10k',
                        f'bpe_{_a}_{UKE10}{_w}', f'bpe_{_a}_{UKB10}{_w}', 10000))


def machine(r):
    gpu = (r.get('environment') or {}).get('gpu') or ''
    if '5090' in gpu:
        return 'second'
    if '4060' in gpu or not gpu:
        return 'thesis'
    return None


def curve(r):
    return {e['step']: e['bits_per_byte'] for e in r.get('log', []) if e.get('step', 0) > 0}


def mse(xs):
    if not xs:
        return None
    m = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    return m, se


def fmt(x, nd=5):
    return '—' if x is None else f'{x:+.{nd}f}'


def sign_p(k, n):
    """Two-sided sign test: the chance of a split at least this uneven, with no offset."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(root):
    paths = glob.glob(os.path.join(root, 'results', '*', 'result.json')) + \
        glob.glob(os.path.join(root, 'results', 'audits', 'device_*', '*.json'))
    recs = defaultdict(lambda: defaultdict(list))
    for p in sorted(paths):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if not isinstance(r, dict) or 'label' not in r or 'log' not in r or 'bits_per_byte' not in r:
            continue
        m = machine(r)
        if m is None:
            continue
        key = (r['label'], r['seed'], r.get('max_steps'))
        recs[key][m].append((p, r))
    return recs


def one(rs):
    """One record per machine and key; repeats that differ are kept for the repeat table."""
    first = rs[0][1]
    differ = [(p, r) for p, r in rs[1:] if abs(r['bits_per_byte'] - first['bits_per_byte']) > 1e-12]
    return first, differ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    a = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('device_twins', a, out_dir=os.path.join(a.root, 'results', 'audits'))
    recs = load(a.root)

    pairs, repeats = [], []
    for key, by in sorted(recs.items()):
        for m in ('thesis', 'second'):
            if len(by.get(m, [])) > 1:
                first, differ = one(by[m])
                for p, r in differ:
                    repeats.append({'label': key[0], 'seed': key[1], 'steps': key[2], 'machine': m,
                                    'first': first['bits_per_byte'], 'repeat': r['bits_per_byte'],
                                    'difference': r['bits_per_byte'] - first['bits_per_byte'], 'path': os.path.relpath(p, a.root)})
        if not by.get('thesis') or not by.get('second'):
            continue
        t, _ = one(by['thesis'])
        s, _ = one(by['second'])
        ct, cs = curve(t), curve(s)
        steps = sorted(set(ct) & set(cs))
        d = {k: cs[k] - ct[k] for k in steps}
        past = lambda lim: next((k for k in steps if abs(d[k]) > lim), None)
        pairs.append({'label': key[0], 'seed': key[1], 'steps': key[2],
                      'thesis_run': t['run_id'], 'second_run': s['run_id'],
                      'thesis_bpb': t['bits_per_byte'], 'second_bpb': s['bits_per_byte'],
                      'difference': s['bits_per_byte'] - t['bits_per_byte'],
                      'best_step': [t.get('best_step'), s.get('best_step')],
                      'first_eval': {'step': steps[0], 'difference': d[steps[0]]} if steps else None,
                      'largest_along_training': max((abs(v) for v in d.values()), default=None),
                      'past_1e-4_at': past(1e-4), 'past_1e-3_at': past(1e-3),
                      'curve': [{'step': k, 'difference': d[k]} for k in steps]})

    print(f'\n  TWINS — second machine minus thesis machine, bits per byte at the best step ({len(pairs)} pairs)\n')
    print(f'  {"cell":66} seed  steps   thesis    second   difference  first eval  largest  >1e-4 at  >1e-3 at')
    for q in pairs:
        fe = q['first_eval']['difference'] if q['first_eval'] else None
        print(f'  {q["label"][:66]:66} {q["seed"]:>4} {q["steps"]:>6}  {q["thesis_bpb"]:.5f}  {q["second_bpb"]:.5f}  {q["difference"]:+.6f}'
              f'  {fmt(fe, 6):>10}  {q["largest_along_training"]:.6f}  {str(q["past_1e-4_at"]):>8}  {str(q["past_1e-3_at"]):>8}')

    # groups: one per cell and budget; the cell's seed spread from the thesis machine
    groups = defaultdict(list)
    for q in pairs:
        groups[(q['label'], q['steps'])].append(q)
    gout = []
    print(f'\n  GROUPS — the twins of a cell against its spread between seeds on the thesis machine\n')
    print(f'  {"cell":66} steps   n   mean ± SE              twins SD   seeds SD   ratio  signs (+/−)')
    for (lab, steps), qs in sorted(groups.items()):
        ds = [q['difference'] for q in qs]
        m, se = mse(ds)
        sd = st.stdev(ds) if len(ds) > 1 else None
        seeds = [one(by['thesis'])[0]['bits_per_byte'] for (l, s_, n), by in recs.items()
                 if l == lab and n == steps and by.get('thesis')]
        ssd = st.stdev(seeds) if len(seeds) > 1 else None
        ratio = (sd / ssd) if sd is not None and ssd else None
        pos = sum(x > 0 for x in ds)
        gout.append({'label': lab, 'steps': steps, 'n': len(ds), 'mean': m, 'se': se, 'twins_sd': sd,
                     'seeds_sd_thesis': ssd, 'seeds_n': len(seeds), 'ratio': ratio, 'positive': pos, 'negative': len(ds) - pos})
        print(f'  {lab[:66]:66} {steps:>6} {len(ds):>3}   {m:+.5f} ± {se if se is not None else float("nan"):.5f}'
              f'   {sd if sd is not None else float("nan"):.5f}   {ssd if ssd is not None else float("nan"):.5f}'
              f'   {ratio if ratio is not None else float("nan"):5.2f}  {pos}/{len(ds) - pos}')

    ds = [q['difference'] for q in pairs]
    overall = {}
    if ds:
        m, se = mse(ds)
        pos = sum(x > 0 for x in ds)
        by_steps = {}
        for n in sorted({q['steps'] for q in pairs}):
            dn = [q['difference'] for q in pairs if q['steps'] == n]
            mn, sen = mse(dn)
            by_steps[str(n)] = {'n': len(dn), 'mean': mn, 'se': sen, 'largest': max(abs(x) for x in dn)}
        overall = {'n': len(ds), 'mean': m, 'se': se, 'positive': pos, 'negative': len(ds) - pos,
                   'sign_test_p': sign_p(pos, len(ds)), 'largest': max(abs(x) for x in ds),
                   'mean_abs': st.mean(abs(x) for x in ds), 'by_steps': by_steps}
        print(f'\n  OVERALL — {len(ds)} pairs: mean {m:+.6f} ± {se if se is not None else float("nan"):.6f}; '
              f'{pos} above, {len(ds) - pos} below (sign test p = {overall["sign_test_p"]:.3f}); '
              f'mean |difference| {overall["mean_abs"]:.6f}, largest {overall["largest"]:.6f}')
        for n, b in by_steps.items():
            print(f'    at {n:>6} steps: {b["n"]:>2} pairs, mean {b["mean"]:+.6f} ± {b["se"] if b["se"] is not None else float("nan"):.6f}, largest {b["largest"]:.6f}')

    # effects on each machine
    eout = []
    print(f'\n  EFFECTS — B minus A in bits per byte (positive: A better), per seed on each machine\n')
    for name, la, lb, steps in EFFECTS:
        row = {'effect': name, 'A': la, 'B': lb, 'steps': steps, 'per_machine': {}, 'per_seed_difference': {}}
        per = {}
        for m in ('thesis', 'second'):
            vals = {}
            for s in range(16):
                ka, kb = recs.get((la, s, steps), {}), recs.get((lb, s, steps), {})
                if ka.get(m) and kb.get(m):
                    vals[s] = one(kb[m])[0]['bits_per_byte'] - one(ka[m])[0]['bits_per_byte']
            per[m] = vals
            if vals:
                mm, se = mse(list(vals.values()))
                row['per_machine'][m] = {'seeds': vals, 'mean': mm, 'se': se, 'positive': sum(v > 0 for v in vals.values())}
        both = sorted(set(per['thesis']) & set(per['second']))
        row['per_seed_difference'] = {s: per['second'][s] - per['thesis'][s] for s in both}
        eout.append(row)
        if not any(per.values()):
            continue
        print(f'  {name}')
        for m in ('thesis', 'second'):
            if per[m]:
                pm = row['per_machine'][m]
                print(f'    {m:7} seeds {sorted(per[m])}: ' + ' '.join(f'{per[m][s]:+.4f}' for s in sorted(per[m]))
                      + f'   mean {pm["mean"]:+.4f}' + (f' ± {pm["se"]:.4f}' if pm['se'] is not None else '')
                      + f' ({pm["positive"]}/{len(per[m])} positive)')
        if both:
            dd = [row['per_seed_difference'][s] for s in both]
            print(f'    the same seeds, second minus thesis: ' + ' '.join(f'{x:+.5f}' for x in dd)
                  + f'   largest {max(abs(x) for x in dd):.5f}')

    if repeats:
        print(f'\n  REPEATS on one machine\n')
        for r in repeats:
            print(f'    {r["machine"]:7} {r["label"]} seed {r["seed"]}: {r["first"]:.9f} then {r["repeat"]:.9f} ({r["difference"]:+.2e})')

    rec.add('pairs', pairs).add('groups', gout).add('overall', overall).add('effects', eout).add('repeats', repeats)
    rec.note('difference = second machine (RTX 5090 Laptop) minus thesis machine (RTX 4060 Laptop), '
             'bits per byte at the best step; effects are B minus A (positive: A better)')
    rec.write()


if __name__ == '__main__':
    main()
