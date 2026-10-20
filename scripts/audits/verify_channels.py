#!/usr/bin/env python3
"""
verify_channels.py — assert that a surface-channel ablation did what it says.

WHY THIS EXISTS
---------------
docs/record/bugs.md records the standing risk: alignment logic is the recurring
failure mode in this codebase, three of the fixed bugs were caught by numbers
that were *impossible* rather than merely surprising, and C-01/C-02 change what
the preprocessor emits — which is upstream of alignment.

The register's instruction is explicit: assert on the resulting token and code
distributions, not just on a passing test.

C-01 and C-02 work by DISABLING surface channels. A silently ineffective flag
would produce a run that looks fine, reports a plausible BPB, and measures
nothing. Nothing else in the codebase would catch it — the model trains happily
on whatever stream it is given.

This checks, on a few hundred lines and before any GPU time, that each requested
channel is actually off and every other channel is actually still on.

    python scripts/audits/verify_channels.py --pipeline det -o surface.emit_inflection=false
    python scripts/audits/verify_channels.py --pipeline det -o surface.emit_digits=false \\
                                                     -o surface.emit_punctuation=false
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: category -> the surface flag that controls it
CHANNEL_OF = {
    'punct': 'emit_punctuation',
    'digit': 'emit_digits',
    'infl': 'emit_inflection',
    'space': 'emit_whitespace',
    'case': 'emit_case',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='det', choices=['det', 'morph', 'bpe'])
    ap.add_argument('-o', '--override', action='append', default=[],
                    metavar='k=v')
    ap.add_argument('--lines', type=int, default=300)
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('verify_channels', args)

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from datasets import load_dataset

    cfg = apply_overrides(ExperimentConfig(),
                          [f'data.pipeline={args.pipeline}'] + args.override)
    cfg.validate()

    print(f'cell: {cfg.label()}')
    print(f'surface flags: '
          + ', '.join(f'{k}={v}' for k, v in sorted(
              cfg.surface.__dict__.items())))
    print(f'cache spec: {cfg.surface_spec()!r}')
    print()

    pipe = build_pipeline(cfg, verbose=False)
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)

    cats = Counter()
    n_tok = 0
    processed = 0
    for row in ds:
        if processed >= args.lines:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        for t in pipe.encode(text):
            cats[t.get('category')] += 1
            n_tok += 1
        processed += 1

    print(f'{processed} lines, {n_tok:,} tokens')
    print()
    print(f'  {"category":12} {"tokens":>10} {"share":>8}  channel')
    for c, n in cats.most_common():
        ch = CHANNEL_OF.get(c, '')
        print(f'  {str(c):12} {n:10,} {100.0*n/max(n_tok,1):7.2f}%  {ch}')

    print()
    print('  ASSERTIONS')
    ok = True
    for cat, flag in sorted(CHANNEL_OF.items()):
        want_on = getattr(cfg.surface, flag)
        seen = cats.get(cat, 0)
        if want_on and seen == 0:
            print(f'    FAIL  {flag}=True but ZERO {cat!r} tokens emitted')
            ok = False
        elif not want_on and seen > 0:
            print(f'    FAIL  {flag}=False but {seen:,} {cat!r} tokens emitted')
            ok = False
        else:
            state = 'on, present' if want_on else 'off, absent'
            print(f'    ok    {flag:18} {state}')

    # a disabled channel must also change the cache key, or the run would
    # silently reuse the stream it was meant to differ from
    from morphlm.config import ExperimentConfig as EC
    base_spec = EC(data=cfg.data).surface_spec()
    if cfg.surface_spec() == base_spec and any(
            not getattr(cfg.surface, f) for f in CHANNEL_OF.values()):
        print('    FAIL  a channel is disabled but the cache spec is unchanged '
              '— this run would reuse the default stream')
        ok = False
    else:
        print(f'    ok    cache spec distinguishes this cell '
              f'({cfg.surface_spec()!r})')

    print()
    rec.add('summary', {'label': cfg.label(), 'surface_spec': cfg.surface_spec(),
                        'lines': processed, 'tokens': n_tok, 'passed': ok,
                        'surface_flags': dict(cfg.surface.__dict__)})
    rec.add('category_counts', {str(k): v for k, v in cats.items()})
    rec.write()
    print('CHANNELS VERIFIED — safe to train' if ok
          else 'CHANNEL VERIFICATION FAILED — do NOT train')
    sys.exit(0 if ok else 1)



if __name__ == '__main__':
    main()
