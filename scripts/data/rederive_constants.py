#!/usr/bin/env python3
"""
rederive_constants.py — derive every constant null again, as the records say it
was derived (G-73, journey §33.56).

A constant null (`derive_lexical_codes.py --transform const`: `mconst`, its
widths, the zero slots) is one class for every word. Derived before G-73, a word
unseen in train got the class `none` at validation — a mark the null met only on
non-words in training — and runs of `mconst` now refuse such codes. This replays
every constant derivation in the derivation records
(results/audits/derive_lexical_codes_*.json) with its recorded arguments, in the
order each was first made, so a zero slot derived from the null follows it.

Each replay reports how many tokens it changed in the file it replaced: train
must change by 0 tokens — the null's training codes are unchanged, every train
word being in its lexicon — and validation by the unseen words, now given the
constant. A train change stops the replay.

    python scripts/data/rederive_constants.py --dry-run     # the plan
    python scripts/data/rederive_constants.py               # replay (resumable)

An output whose files already declare the rule is skipped. Nothing else is
touched; check afterwards with scripts/audits/codes_audit.py.
"""
import os
import re
import sys
import glob
import json
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def iso_epoch(s):
    import datetime
    try:
        return datetime.datetime.fromisoformat(str(s).replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError):
        return None


def stems(a):
    """The ids stems of each split a recorded derivation wrote."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem
    corpus = a.get('corpus') or 'wikitext103'
    o = ['data.pipeline=bpe', f'data.bpe_path={a["bpe_path"]}', f'data.corpus={corpus}']
    if corpus != 'wikitext103':
        o += [f'data.corpus_fraction={a.get("corpus_fraction") or 1.0}',
              f'data.corpus_holdout={a.get("corpus_holdout") or 2000}']
    spec = apply_overrides(ExperimentConfig(), o).surface_spec()
    return {s: cache_stem('bpe', s, None, spec) for s in (a.get('splits') or ['train', 'validation', 'test'])}


def done(a, out):
    """Every file this derivation wrote declares the constant rule."""
    cache = a.get('cache_dir') or 'data/cache'
    cache = cache if os.path.isabs(cache) else os.path.join(ROOT, cache)
    seen = 0
    for split, stem in stems(a).items():
        p = os.path.join(cache, f'{stem}_{out}.codes.json')
        if not os.path.exists(p):
            continue
        seen += 1
        if (json.load(open(p)).get('rules') or {}).get(out) != 'every word':
            return False
    return seen > 0


def command(a, out):
    cmd = [sys.executable, 'scripts/data/derive_lexical_codes.py', '--fact', a['fact'], '--out-fact', out,
           '--transform', 'const', '--bpe-path', a['bpe_path'], '--verify']
    if a.get('keep'):
        cmd += ['--keep', a['keep']]
    if a.get('cache_dir'):
        cmd += ['--cache-dir', a['cache_dir']]
    if a.get('corpus') and a['corpus'] != 'wikitext103':
        cmd += ['--corpus', a['corpus'], '--corpus-fraction', str(a.get('corpus_fraction') or 1.0),
                '--corpus-holdout', str(a.get('corpus_holdout') or 2000)]
    if a.get('lexicon_fraction') is not None:
        cmd += ['--lexicon-fraction', str(a['lexicon_fraction'])]
    if a.get('splits'):
        cmd += ['--splits'] + list(a['splits'])
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    plan = {}
    for p in sorted(glob.glob(os.path.join(a.audit_dir, 'derive_lexical_codes_*.json'))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        args, t = r.get('args') or {}, iso_epoch(r.get('timestamp_utc'))
        if args.get('transform') != 'const' or not args.get('bpe_path') or t is None:
            continue
        bpe = args['bpe_path'] if os.path.isabs(args['bpe_path']) else os.path.join(ROOT, args['bpe_path'])
        if not os.path.exists(bpe):
            continue                                        # a tokenizer not on this machine
        out = args.get('out_fact') or args['fact'] + 'lex'
        key = (args['bpe_path'], args.get('corpus') or 'wikitext103', args.get('corpus_fraction') or 1.0,
               args.get('corpus_holdout') or 2000, out, args.get('cache_dir') or 'data/cache')
        first = plan[key][0] if key in plan else t
        plan[key] = (min(first, t), args, out)             # the latest arguments, the first time
    order = sorted(plan.values(), key=lambda x: x[0])
    print(f'{len(order)} constant derivations in the records')
    todo = []
    for t, args, out in order:
        cond = (f'{os.path.basename(args["bpe_path"])}'
                + (f' {args["corpus"]} {args.get("corpus_fraction") or 1.0:g}' if (args.get('corpus') or 'wikitext103') != 'wikitext103' else ''))
        state = 'done' if done(args, out) else 'to replay'
        print(f'  {out:13} {cond:34} from {args["fact"]:7} keep {args.get("keep") or "all":7} {state}')
        if state != 'done':
            todo.append((args, out))
    if a.dry_run or not todo:
        print('\n  nothing to replay' if not todo else '\n  dry run — run without --dry-run to replay')
        return
    failed = []
    for args, out in todo:
        cmd = command(args, out)
        print(f'\n=== {time.strftime("%F %T")}  {out} — ' + ' '.join(cmd[1:]), flush=True)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if re.search(r'tokens -> |TRANSFORMED|not found|skipped', line):
                print(line)
        if r.returncode != 0:
            print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else '')
            failed.append(out)
            continue
        m = re.search(r'^\s+train: .*replaced a file: ([\d,]+) tokens changed', r.stdout, re.M)
        if m and int(m.group(1).replace(',', '')) != 0:
            sys.exit(f'  STOP: {out}\'s TRAIN codes changed ({m.group(1)} tokens) — the replay '
                     f'must leave them as they were; nothing further replayed')
    print(f'\n  replayed {len(todo) - len(failed)} of {len(todo)}' + (f'; failed: {failed}' if failed else ''))
    print('  check: python scripts/audits/codes_audit.py   (its CONSTANTS section must mark no word)')


if __name__ == '__main__':
    main()
