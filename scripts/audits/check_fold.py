#!/usr/bin/env python3
"""
check_fold.py — does coarsening actually reach the model? On real data.

WHY THIS EXISTS
---------------
The first cardinality sweep was invalid because coarsening was verified by a
test that checked the CACHE SPEC — true and irrelevant — while the model read
cached codes through clamp() and put 96.9% of words into one class.

The fix was then verified by a NUMPY MIRROR of the torch code, plus a check that
the source routes through fold_codes. Neither runs the actual torch function on
actual cached codes. A check that could not have failed is not yet a closure:
a check on the configuration is not a check on the behaviour.

This runs the real `embedding.fold_codes` on the real cached mhash codes, at the
real block configuration, and reports how the codes land. It takes seconds and
it is the difference between a five-hour sweep that measures something and one
that measures the old artifact again.

PASS CRITERION — REVISED after the first version failed on real data
--------------------------------------------------------------------
The first criterion ("largest class under 5%") assumed word frequencies are
UNIFORM. They are Zipfian: a single very common word can hold several percent
of all word tokens, and its class holds at least that much however finely the
rest is folded. The criterion was tested against uniform synthetic codes, which
is exactly why it passed in simulation and failed on real data.

The revised criterion is derived rather than guessed. Let f be the share of word
tokens in the single largest class at the FULL inventory — measured, not
assumed. A correct fold into k classes should put roughly

        expected(k) = f + (1 - f) / k

in its largest class: the dominant word, plus its fair share of everything else.
The fold PASSES when the observed share is within one percentage point of that.

This is falsifiable. If f turns out small, a large folded share is a real defect
and the check fails. The broken clamp misses by 70+ points and fails clearly.

    python scripts/audits/check_fold.py
    python scripts/audits/check_fold.py --k 64 512 4096
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('MORPHLM_NO_BUILD', '1')   # never trigger a cache build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, nargs='+', default=[64, 512, 4096])
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()

    import numpy as np
    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.stream import build_stream
    from morphlm.features.blocks import build_blocks
    from morphlm.modeling.embedding import fold_codes

    # load the real cached mhash codes once, at the full inventory
    base = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', f'data.bpe_path={args.bpe_path}',
        'frozen.facts=mhash', 'frozen.trainable=true'])
    pipe = build_pipeline(base, verbose=False)
    meta = build_stream(pipe, base, split=args.split, verbose=False)
    raw = np.asarray(meta['codes'][0]).astype(np.int64)
    full_block = build_blocks(base)[0]
    none_code = full_block.n_classes - 1
    words = raw[raw != none_code]

    print()
    print('=' * 78)
    print(f'FOLD CHECK — real cached mhash codes, real torch fold_codes')
    print('=' * 78)
    print(f'  {len(raw):,} {args.split} tokens, {len(words):,} carry a word code')
    print(f'  full inventory: {full_block.n_classes:,} classes')
    print()
    # f: the irreducible concentration from word frequency, MEASURED
    full_new = fold_codes(torch.from_numpy(words), full_block).numpy()
    f = np.bincount(full_new).max() / len(full_new)
    print(f'  full-inventory largest class f = {100*f:.2f}% of word tokens')
    print('  (the Zipf floor: the most frequent word, which no fold can split)')
    print()
    print(f'  {"k":>6}  {"used":>6}  {"observed":>9}  {"expected":>9}  '
          f'{"off":>6}  {"old clamp":>10}  verdict')

    ok = True
    for k in args.k:
        cfg = apply_overrides(base, [f'frozen.root_classes={k}'])
        block = build_blocks(cfg)[0]
        t = torch.from_numpy(words)
        new = fold_codes(t, block).numpy()
        old = t.clamp(0, block.n_classes - 1).numpy()

        obs = np.bincount(new).max() / len(new)
        exp = f + (1 - f) / k
        old_share = np.bincount(old).max() / len(old)
        used = len(np.unique(new))
        good = abs(obs - exp) < 0.01
        ok &= good
        print(f'  {k:6,}  {used:6,}  {100*obs:8.2f}%  {100*exp:8.2f}%  '
              f'{100*(obs-exp):+5.2f}  {100*old_share:9.1f}%  '
              f'{"PASS" if good else "FAIL"}')

    # SELF-DISCRIMINATION. A check that cannot fail is not evidence
    # (a check must be able to fail). Apply the SAME criterion to the known-broken
    # clamp: it must FAIL there. If the clamp would pass, the criterion cannot
    # tell a working fold from the bug it exists to catch, and a PASS above
    # means nothing.
    print()
    discriminates = True
    for k in args.k:
        cfg = apply_overrides(base, [f'frozen.root_classes={k}'])
        block = build_blocks(cfg)[0]
        old = torch.from_numpy(words).clamp(0, block.n_classes - 1).numpy()
        obs = np.bincount(old).max() / len(old)
        exp = f + (1 - f) / k
        if abs(obs - exp) < 0.01:
            discriminates = False
    print('  criterion vs the known-broken clamp: '
          + ('REJECTS it at every k — the check discriminates'
             if discriminates else
             'ACCEPTS it — the check CANNOT tell the bug from a fix'))
    ok &= discriminates

    # the sentinel must survive as its own class, not fold into class 0
    cfg = apply_overrides(base, [f'frozen.root_classes={args.k[0]}'])
    block = build_blocks(cfg)[0]
    sent = fold_codes(torch.tensor([none_code]), block).item()
    sent_ok = sent == block.n_classes - 1
    ok &= sent_ok
    print()
    print(f'  none sentinel {none_code} -> class {sent} '
          f'(expected {block.n_classes - 1})  {"PASS" if sent_ok else "FAIL"}')

    print()
    print('FOLD VERIFIED ON REAL DATA — safe to run the sweep' if ok else
          'FOLD CHECK FAILED — do NOT run the sweep')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
