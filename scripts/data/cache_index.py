#!/usr/bin/env python3
"""
cache_index.py — what is in data/cache/, and what would a given cell reuse?

THE PROBLEM
-----------
Cache filenames are md5 stems: `det_train_None_67ae829830`. Which experiment
built that, and is it reusable? Until now, unanswerable. The directory holds
roughly 15 GB across a dozen variants with no index, so before a four-hour
rebuild there was no way to check whether an equivalent stream already existed.

The sidecar `.json` records token counts and byte counts but NOT the surface
spec or fact list that produced the stream, so it cannot identify itself.

THE TRICK
---------
`cache_stem()` is a pure deterministic function of
`(pipeline, split, max_articles, spec)`. So instead of asking each file what it
is, we ENUMERATE every configuration the project can express, hash each one, and
build a reverse map from stem to description. Existing caches are then labelled
without ever having recorded anything — no migration, no rebuild.

For the morpheme pipelines the surface space is all 32 combinations of the five
channels. For BPE it is the set of tokenizer files on disk. Codes files add the
fact subsets on top.

    python scripts/data/cache_index.py                  # label everything on disk
    python scripts/data/cache_index.py --plan --pipeline det \\
        -o surface.emit_whitespace=false           # would this cell hit or miss?
"""

import os
import sys
import json
import glob
import time
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: the five surface channels, in the order SurfaceConfig.spec() emits them
CHANNELS = [('emit_punctuation', 'p'), ('emit_digits', 'd'),
            ('emit_inflection', 'i'), ('emit_whitespace', 'w'),
            ('emit_case', 'c')]

SPLITS = ('train', 'validation', 'test')


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f}{unit}' if unit == 'B' else f'{n:.1f}{unit}'
        n /= 1024


def build_reverse_map(cfg_template, caps=(None,)):
    """
    stem -> human description, for every configuration the project can express.

    Enumerates rather than reads, because the stem is a deterministic hash of
    the config and the sidecars do not record their own spec.
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem, codes_spec, PIPELINE_CONTEXT_FACTS

    rev = {}

    # BPE: the stream identity is the tokenizer file.
    # Known stems are included unconditionally, because a cache can outlive the
    # tokenizer file that produced it and an unidentifiable cache is exactly
    # what this tool exists to prevent.
    KNOWN_TOK = ['bpe_tokenizer', 'bpe_bytelevel', 'bpe_17k',
                 'unigram_17k']
    tok_files = [os.path.basename(p)[:-5]
                 for p in sorted(glob.glob('data/*.json'))
                 if 'bpe' in os.path.basename(p)]
    for tok in sorted(set(tok_files) | set(KNOWN_TOK)):
        spec = '' if tok == 'bpe_tokenizer' else f'tok:{tok}'
        for split in SPLITS:
            for cap in caps:
                rev[cache_stem('bpe', split, cap, spec)] = \
                    ('bpe', split, cap, f'tokenizer={tok}', 'ids')
                for facts in _fact_subsets(PIPELINE_CONTEXT_FACTS.get('bpe', ())):
                    cs = codes_spec(facts, 'coarse')
                    full = f'{spec}|{cs}' if spec else cs
                    rev[cache_stem('bpe', split, cap, full)] = \
                        ('bpe', split, cap, f'tokenizer={tok}',
                         'codes: ' + '+'.join(facts))

    # morph / det: the stream identity is the surface-channel combination
    for pipe in ('morph', 'det'):
        for bits in itertools.product([True, False], repeat=len(CHANNELS)):
            spec_str = ''.join(ch if on else '-'
                               for (_, ch), on in zip(CHANNELS, bits))
            spec = '' if spec_str == '-----' else f'sfc:{spec_str}'
            pretty = spec_str if spec else 'legacy (no channels)'
            for split in SPLITS:
                for cap in caps:
                    rev[cache_stem(pipe, split, cap, spec)] = \
                        (pipe, split, cap, pretty, 'ids')
                    for facts in _fact_subsets(
                            PIPELINE_CONTEXT_FACTS.get(pipe, ())):
                        cs = codes_spec(facts, 'coarse')
                        full = f'{spec}|{cs}' if spec else cs
                        rev[cache_stem(pipe, split, cap, full)] = \
                            (pipe, split, cap, pretty,
                             'codes: ' + '+'.join(facts))
    return rev


def _fact_subsets(available):
    """
    EVERY subset of the available facts.

    Guessing which combinations were historically used does not work: the BPE
    inventory has grown from (dep, pos, ectx) to seven facts over the project,
    and a cache built at any point in between is a subset nobody would think to
    enumerate by hand. With at most seven facts the full power set is 128
    entries per pipeline, which costs nothing and cannot miss.
    """
    avail = tuple(sorted(available))
    out = []
    for r in range(1, len(avail) + 1):
        out.extend(itertools.combinations(avail, r))
    return out


def cmd_list(args):
    from morphlm.config import ExperimentConfig
    rev = build_reverse_map(ExperimentConfig(), caps=tuple(args.cap_list))

    files = sorted(glob.glob(os.path.join(args.cache_dir, '*.npy')))
    if not files:
        print(f'no caches under {args.cache_dir}/')
        return

    rows, unknown, total = [], 0, 0
    for f in files:
        base = os.path.basename(f)
        stem = base[:-4]
        kind = 'ids'
        if stem.endswith('.codes'):
            stem = stem[:-len('.codes')]
            kind = 'codes'
        size = os.path.getsize(f)
        total += size
        meta = {}
        side = f[:-4] + '.json'
        if os.path.exists(side):
            try:
                meta = json.load(open(side))
            except Exception:
                pass
        desc = rev.get(stem if kind == 'ids' else base[:-4].replace('.codes', ''))
        # codes files hash the FULL spec, so look them up by their own stem
        if kind == 'codes':
            desc = rev.get(base[:-len('.codes.npy')])
        if desc is None:
            unknown += 1
        rows.append({
            'file': base, 'kind': kind, 'size': size, 'desc': desc,
            'tokens': meta.get('num_tokens'),
            'facts': meta.get('fact_names'),
            'mtime': os.path.getmtime(f),
        })

    print()
    print('=' * 104)
    print(f'CACHE INDEX — {args.cache_dir}/')
    print('=' * 104)
    print(f'  {"pipeline":8} {"split":11} {"stream":22} {"contents":26} '
          f'{"tokens":>12} {"size":>8}  age')
    print('  ' + '-' * 100)
    for r in sorted(rows, key=lambda x: (x['desc'] or ('zzz',))[0:1] + (
            x['desc'][1:2] if x['desc'] else ('',)) + (x['file'],)):
        d = r['desc']
        if d:
            pipe, split, cap, stream, kind = d
            contents = kind if r['facts'] is None else 'codes: ' + '+'.join(
                r['facts'])
        else:
            pipe, split, stream, contents = '?', '?', 'UNIDENTIFIED', r['kind']
        tok = f'{r["tokens"]:,}' if r['tokens'] else '-'
        age = f'{(time.time()-r["mtime"])/86400:.0f}d'
        print(f'  {pipe:8} {split:11} {stream:22} {contents:26} '
              f'{tok:>12} {human(r["size"]):>8}  {age}')

    # ── redundancy: a codes file whose facts are a strict subset of another
    #    at the SAME token count can never be preferred by content-based
    #    reuse, because the loader takes the first file that contains what was
    #    asked for and a superset always qualifies. Those are dead weight.
    codes = [r for r in rows if r['kind'] == 'codes' and r['facts']
             and r['tokens']]
    redundant = []
    for a in codes:
        for b in codes:
            if a is b or a['tokens'] != b['tokens']:
                continue
            if set(a['facts']) < set(b['facts']):
                redundant.append((a, b))
                break
    if redundant:
        print()
        print('  REDUNDANT CODES (a strict subset of another file at the same')
        print('  token count, so content-based reuse will never choose it):')
        freed = 0
        for a, b in redundant:
            freed += a['size']
            print(f'    {a["file"]}')
            print(f'      {"+".join(a["facts"])}')
            print(f'      superseded by {b["file"]} '
                  f'({"+".join(b["facts"])})')
            print(f'      {human(a["size"])} reclaimable')
        print(f'    total reclaimable: {human(freed)} of {human(total)}')
        print('    (delete the .codes.npy and .codes.json together, or keep '
              'them — nothing breaks either way)')

    print()
    print(f'  {len(rows)} files, {human(total)} total'
          + (f', {unknown} unidentified' if unknown else ''))
    if unknown:
        print('  (unidentified = built by a config the current code cannot '
              'express, e.g. an article cap. Pass --caps to enumerate those.)')
    print()


def cmd_plan(args):
    """Would the given cell hit the cache, and if not, what will it cost?"""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import (cache_stem, codes_spec, facts_to_build)

    cfg = apply_overrides(ExperimentConfig(),
                          [f'data.pipeline={args.pipeline}'] + args.override)
    cfg.validate()
    sfc = cfg.surface_spec()
    build = list(facts_to_build(cfg))
    cs = codes_spec(build, cfg.frozen.dep_granularity)

    print()
    print(f'cell: {cfg.label()}')
    print(f'  surface spec : {sfc!r}')
    print(f'  facts to build: {build or "none"}')
    print()

    need_build = False
    for split in ('train', 'validation'):
        ids = cache_stem(cfg.data.pipeline, split, None, sfc)
        ip = os.path.join(args.cache_dir, ids + '.npy')
        hit = os.path.exists(ip)
        print(f'  {split:11} ids   {ids}  '
              f'{"HIT  " + human(os.path.getsize(ip)) if hit else "MISS"}')
        if not hit:
            need_build = True
        if build:
            full = f'{sfc}|{cs}' if sfc else cs
            cst = cache_stem(cfg.data.pipeline, split, None, full)
            cp = os.path.join(args.cache_dir, cst + '.codes.npy')
            chit = os.path.exists(cp)
            note = ''
            if not chit and hit:
                # Content-based reuse, with the SAME guard the loader uses:
                # codes are one value per token of a specific ids array, so a
                # matching fact list is not sufficient — the token count must
                # match too. Reuse is also only meaningful when the ids stream
                # already exists; otherwise there is nothing to align to.
                ids_tokens = None
                im = ip[:-4] + '.json'
                if os.path.exists(im):
                    try:
                        ids_tokens = json.load(open(im)).get('num_tokens')
                    except Exception:
                        pass
                for cand in sorted(glob.glob(os.path.join(
                        args.cache_dir,
                        f'{cfg.data.pipeline}_{split}_None_*.codes.json'))):
                    try:
                        cm = json.load(open(cand))
                    except Exception:
                        continue
                    if not set(build) <= set(cm.get('fact_names', [])):
                        continue
                    if cm.get('num_tokens') != ids_tokens:
                        continue
                    note = f'  -> REUSABLE via {os.path.basename(cand)}'
                    chit = True
                    break
            print(f'  {split:11} codes {cst}  '
                  f'{"HIT" if chit else "MISS"}{note}')
            if not chit:
                need_build = True

    print()
    if need_build:
        est = '~4h' if cfg.data.pipeline in ('morph', 'det') else '~4-5h'
        print(f'  VERDICT: a cache build is required ({est} for the train split).')
        print('  Nothing on disk encodes this stream. That is expected when the')
        print('  surface channels or the tokenizer change — a different stream')
        print('  is different data.')
    else:
        print('  VERDICT: fully cached. This cell will start training '
              'immediately.')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--plan', action='store_true',
                    help='check one cell instead of listing everything')
    ap.add_argument('--pipeline', default='det')
    ap.add_argument('-o', '--override', action='append', default=[],
                    metavar='k=v')
    ap.add_argument('--caps', default='',
                    help='comma-separated article caps to also enumerate, for '
                         'caches built with --max-articles (e.g. the 3,000-'
                         'article calibration run of journey section 13). '
                         'Example: --caps 3000,12000')
    args = ap.parse_args()
    args.cap_list = [None] + [int(c) for c in args.caps.split(',') if c.strip()]
    (cmd_plan if args.plan else cmd_list)(args)


if __name__ == '__main__':
    main()
