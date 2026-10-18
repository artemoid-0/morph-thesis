#!/usr/bin/env python3
"""
slim_codes.py — one fact of a stream's codes, in a file of its own, for a machine that holds only what
its runs read (journey §33.101, §33.112).

A cache build writes all its facts into one codes file, and a derivation writes the facts it kept beside
the one it made; either can be several hundred megabytes per fact. The second machine needs one or two
of them per stream. This writes `<ids stem>_slim_<fact>.codes.npy` (int16, shape (1, n)) and its
`.json` (fact_names, num_tokens, and where it came from) next to the source. Named for its ids stream,
it is a derived file: the loader reads it for that stream only (`stream_codes_files`, G-72). Before
writing, the row is read back and compared with the source, value for value; a wrong shape or a missing
fact stops the tool.

    python scripts/ops/slim_codes.py --ids-stem bpe_train_None_d0a4e3909d --fact ucaselex
    python scripts/ops/slim_codes.py --ids-stem bpe_train_None_562b1cc17e --fact mconst mfreq
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids-stem', required=True, help='the ids stream, e.g. bpe_train_None_d0a4e3909d')
    ap.add_argument('--fact', nargs='+', required=True)
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    a = ap.parse_args()
    import numpy as np
    from morphlm.data.stream import stream_codes_files
    stem = a.ids_stem
    pipeline, split = stem.split('_')[0], stem.split('_')[1]
    n = int(np.load(os.path.join(a.cache_dir, stem + '.npy'), mmap_mode='r').shape[0])
    files = stream_codes_files(a.cache_dir, pipeline, split, None, stem, n)
    for fact in a.fact:
        src = next(((j, npy, m) for j, npy, m in files
                    if fact in m.get('fact_names', []) and '_slim_' not in os.path.basename(npy)), None)
        if src is None:
            sys.exit(f'  {stem}: no codes file holds {fact}')
        j, npy, meta = src
        codes = np.load(npy, mmap_mode='r')
        if codes.ndim != 2 or codes.shape[1] != n:
            sys.exit(f'  {os.path.basename(npy)}: shape {codes.shape}, not (facts, {n:,})')
        row = np.asarray(codes[meta['fact_names'].index(fact)], dtype=np.int16)[None, :]
        out = os.path.join(a.cache_dir, f'{stem}_slim_{fact}.codes')
        np.save(out + '.npy', row)
        back = np.load(out + '.npy', mmap_mode='r')
        if back.shape != (1, n) or back.dtype != np.int16 or not np.array_equal(back[0], codes[meta['fact_names'].index(fact)]):
            sys.exit(f'  {out}.npy does not read back as the source row')
        keep = {k: v for k, v in meta.items() if k not in ('fact_names', 'num_tokens')}
        json.dump(dict(keep, fact_names=[fact], num_tokens=n, slim_of=os.path.basename(npy)),
                  open(out + '.json', 'w'), indent=2)
        print(f'  {os.path.basename(out)}.npy: {fact} from {os.path.basename(npy)}, {n:,} tokens, '
              f'{os.path.getsize(out + ".npy") >> 20} MB, read back equal')


if __name__ == '__main__':
    main()
