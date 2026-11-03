#!/usr/bin/env python3
"""
bounds.py — 95% intervals for the thesis's two tables of effects, one-sided bounds, and the minimum
detectable effect at their seeds (the text's request of 30 October, item 5; journey §33.151).

WHY
---
Each effect in the text is a mean ± standard error over four seeds, with how many seeds agree in sign.
Four of four is p = 1/16 by a one-sided sign test, and a mean says nothing of how large the effect could
be. So for every row of the two tables:
  * the tokenizer's gain (the text's table of the tokenizer's lead): its two-sided 95% t-interval, and the
    one-sided 95% LOWER bound — the gain is at least this;
  * a feature against its null at the end of training (the text's table of features): the one-sided 95%
    UPPER bound — the gain is at most this; the text's "within 0.002" is read from the means;
  * each row's two-sided t-test p, and its minimum detectable effect: the true effect a paired t-test at
    these seeds finds with 80% power at α = 0.05, given the seeds' observed spread
    ((t₀.₉₇₅ + t₀.₈₀) · sd / √n, morphlm.runs.interval).

HOW
---
Each row is a cell against its reference, paired by seed within one machine (FINDINGS, method point 11),
from the run records; a row whose seeds ran on two machines names which seeds each machine gives, and
pairs each seed on its own machine (the second machine's twins included). The gain is the reference's
BPB minus the cell's (positive: the cell is better).

A check that could fail: each row's mean and standard error, recomputed here, must match the figures the
text prints (ROWS' `printed`) within rounding, or the tool names the row and writes nothing — a wrong label
cannot pass silently.

    python scripts/analysis/bounds.py
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

N = 'bpe_mconst-LEARN-CAUSAL_'
UK, END = 'bpe_uk_17k_ukwiki', 'bpeend_uk_17k_ukwiki'
EN = 'bpe_17k'


def tok(name, suffix, printed, machines=None):
    """The tokenizer's gain: endings apart's null against BPE 17k's, same corpus, width and budget."""
    return {'table': 'tokenizer', 'name': name, 'ref': f'{N}{UK}{suffix}', 'cell': f'{N}{END}{suffix}',
            'printed': printed, 'machines': machines or {'thesis': None}}


def feat(name, cell, ref, printed, machines=None):
    return {'table': 'features', 'name': name, 'ref': ref, 'cell': cell, 'printed': printed,
            'machines': machines or {'thesis': None}}


SECOND = {'second': None}
ROWS = [
    tok('Wikipedia, 38M tokens, width 128, 10k', '0.05_st10k', (0.0174, 0.0014)),
    tok('Wikipedia, 38M tokens, width 128, 20k', '0.05_st20k', (0.0169, 0.0007)),
    tok('Wikipedia, 151M tokens, width 128, 50k', '0.2', (0.0165, 0.0008)),
    tok('Wikipedia, 38M tokens, width 256, 10k', '0.05_st10k_w256', (0.0254, 0.0012)),
    tok('Wikipedia, 38M tokens, width 256, 20k', '0.05_st20k_w256', (0.0218, 0.0029)),
    tok('Wikipedia, 38M tokens, width 512, 10k', '0.05_st10k_w512', (0.0125, 0.0009)),
    tok('Wikipedia, 38M tokens, width 512, 20k (seeds 2-3: second machine)', '0.05_st20k_w512', (0.0102, 0.0006),
        {'thesis': [0, 1], 'second': [2, 3]}),
    tok('Wikipedia, 151M tokens, width 512, 50k (second machine)', '0.2_w512', (0.0072, 0.0003), SECOND),
    tok('Wikipedia, 38M tokens, width 1024, 10k, warm-up 1,000 (second machine)', '0.05_wu1000_st10k_w1024',
        (0.0110, 0.0011), SECOND),
    tok('Wikipedia, 38M tokens, width 1024, 20k, warm-up 1,000 (second machine)', '0.05_wu1000_st20k_w1024',
        (0.0105, 0.0004), SECOND),
    tok('Wikipedia, 151M tokens, width 1024, 20k, warm-up 1,000 (second machine)', '0.2_wu1000_st20k_w1024',
        (0.0082, 0.0005), SECOND),
    tok('Wikipedia, 151M tokens, width 1024, 50k, warm-up 1,000 (second machine)', '0.2_wu1000_w1024',
        (0.0071, 0.0004), SECOND),
    dict(tok('Fiction, 33.4M tokens, width 128, 10k', '', (0.0097, 0.0010)),
         ref=f'{N}bpe_ukf_17k_ukfiction0.1_st10k', cell=f'{N}bpeend_ukf_17k_ukfiction0.1_st10k'),
    dict(tok('Fiction, four times the text, width 128, 50k (second machine)', '', (0.0155, 0.0010), SECOND),
         ref=f'{N}bpe_ukf_17k_ukfiction0.4', cell=f'{N}bpeend_ukf_17k_ukfiction0.4'),
    dict(tok('Unigram 17k against BPE 17k, Wikipedia, 38M tokens, width 128, 10k', '', (-0.0005, 0.0004)),
         ref=f'{N}{UK}0.05_st10k', cell=f'{N}unigram_uk_17k_ukwiki0.05_st10k'),
    feat('English: word structure, its own input dims', f'bpe_mstruct-LEARN-CAUSAL_{EN}', f'{N}{EN}',
         (0.0004, 0.0005)),
    feat('English: word structure, added to a full-width embedding (against the added flat vector, §33.19)',
         f'bpe_mstruct-LEARN-CAUSAL-ADD_{EN}', f'bpe_mflat-LEARN-CAUSAL-ADD_{EN}', (-0.0013, None)),
    feat('English: structure shuffled among words of similar frequency', f'bpe_mstructshuf-LEARN-CAUSAL_{EN}',
         f'{N}{EN}', (0.0015, 0.0005)),
    feat('English: real structure against its shuffled version', f'bpe_mstruct-LEARN-CAUSAL_{EN}',
         f'bpe_mstructshuf-LEARN-CAUSAL_{EN}', (-0.0012, 0.0008)),
    feat('English: frequency band (seeds 2-3: second machine)', f'bpe_mfreq-LEARN-CAUSAL_{EN}', f'{N}{EN}',
         (-0.0005, 0.0011), {'thesis': [0, 1], 'second': [2, 3]}),
    feat('English: part of speech (tagger)', f'bpe_pos-LEARN-CAUSAL_{EN}', f'{N}{EN}', (0.0012, 0.0007)),
    feat('English: part of speech, strictly causal (seeds 2-3: second machine)', f'bpe_poslex-LEARN-CAUSAL_{EN}',
         f'{N}{EN}', (-0.0001, 0.0014), {'thesis': [0, 1], 'second': [2, 3]}),
    feat('English: inflection, strictly causal', f'bpe_mfeatlex-LEARN-CAUSAL_{EN}', f'{N}{EN}', (-0.0046, 0.0013)),
    feat('Ukrainian: case (tagger)', f'bpe_ucase-LEARN-CAUSAL_{UK}0.2', f'{N}{UK}0.2', (-0.0013, 0.0008)),
    feat('Ukrainian: case, strictly causal', f'bpe_ucaselex-LEARN-CAUSAL_{UK}0.2', f'{N}{UK}0.2', (-0.0020, 0.0010)),
    feat('Ukrainian at 30M parameters: case (tagger; second machine)', f'bpe_ucase-LEARN-CAUSAL_{UK}0.2_w512',
         f'{N}{UK}0.2_w512', (0.0015, 0.0005), SECOND),
]


def gains(row, loaded):
    """{seed: (gain, machine, [ref run, cell run])} over the row's machines and seeds."""
    out = {}
    for m, seeds in row['machines'].items():
        if m not in loaded:
            loaded[m] = runs.load(machine=m, keep=lambda r: 'seed' in r and 'bits_per_byte' in r)
        pairs, _ = runs.pair(loaded[m], row['ref'], row['cell'])
        for s, (ra, rb) in sorted(pairs.items()):
            if (seeds is None or s in seeds) and s not in out:
                out[s] = (ra['bits_per_byte'] - rb['bits_per_byte'], m, [ra['run_id'], rb['run_id']])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('bounds', argparse.Namespace(), out_dir=runs.AUDITS)
    loaded, out, wrong = {}, [], []
    for row in ROWS:
        g = gains(row, loaded)
        iv = runs.interval([v[0] for v in g.values()])
        pm, pse = row['printed']
        if iv is None or abs(iv['mean'] - pm) > 0.000051 or (pse is not None and abs(iv['se'] - pse) > 0.000051):
            wrong.append((row['name'], pm, pse, iv and round(iv['mean'], 5), iv and round(iv['se'], 5), sorted(g)))
        out.append(dict(row, seeds={str(s): {'gain': v[0], 'machine': v[1], 'runs': v[2]} for s, v in g.items()},
                        interval=iv))
    if wrong:
        for w in wrong:
            print('  MISMATCH', w)
        raise SystemExit(f'{len(wrong)} row(s) do not reproduce the printed figures — nothing written')
    for table, title, side in (('tokenizer', 'the tokenizer\'s gain', 'lower'), ('features', 'features against their null', 'upper')):
        print(f'\n  {title}: mean ± se, 95% interval, one-sided 95% {side} bound, t-test p, minimum detectable effect')
        for r in (r for r in out if r['table'] == table):
            iv = r['interval']
            print(f'  {r["name"][:74]:74} n={iv["n"]} {iv["mean"]:+.4f} ± {iv["se"]:.4f}  '
                  f'[{iv["low"]:+.4f}, {iv["high"]:+.4f}]  {side} {iv[side]:+.4f}  p={iv["p"]:.4f}  mde {iv["mde"]:.4f}')
    rec.add('rows', out)
    rec.write()


if __name__ == '__main__':
    main()
