#!/usr/bin/env python3
"""
stream_roundtrip.py — does every tokenizer's cached stream decode to the same
text, byte for byte? (journey §33.65)

Bits per byte compares tokenizers fairly only if each scores the same text: a
stream that silently dropped or altered characters would have less to predict
over the same byte count, and look better than it is. BPE that keeps endings
apart scores below plain BPE at the same size (journey §33.65); before that is
believed, its stream must be the same text as the others'.

For each tokenizer with a cache of the condition, this decodes the cached ids of
a split, compares the decoded text's UTF-8 length with the bytes the cache
recorded when it was built, and hashes it; the condition passes when every
tokenizer's text has the recorded length and all hashes agree. A few seconds for
validation; the train split (`--splits train`) takes minutes per tokenizer.

    python scripts/audits/stream_roundtrip.py
    python scripts/audits/stream_roundtrip.py --tokenizers bpe_uk_17k bpeend_uk_17k --splits validation train

Writes results/audits/stream_roundtrip_<UTC timestamp>.json; exits 1 if any
tokenizer's text differs.
"""
import os
import sys
import json
import hashlib
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TOKENIZERS = ['bpe_uk_4k', 'bpe_uk_8k', 'bpe_uk_17k', 'bpe_uk_32k', 'unigram_uk_17k', 'bpeend_uk_17k']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default='ukwiki')
    ap.add_argument('--corpus-fraction', type=float, default=0.05)
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--tokenizers', nargs='+', default=TOKENIZERS)
    ap.add_argument('--splits', nargs='+', default=['validation'])
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    a = ap.parse_args()
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem
    from morphlm.data.tokenization import load_tokenizer
    from morphlm.record import AuditRecord
    rec = AuditRecord('stream_roundtrip', a, out_dir=a.audit_dir)
    ok = True
    for split in a.splits:
        rows, hashes = {}, set()
        print(f'\n  {a.corpus} at {a.corpus_fraction:g}, {split}')
        for t in a.tokenizers:
            path = os.path.join(ROOT, 'data', f'{t}.json')
            over = ['data.pipeline=bpe', f'data.bpe_path=data/{t}.json', f'data.corpus={a.corpus}',
                    f'data.corpus_fraction={a.corpus_fraction}', f'data.corpus_holdout={a.corpus_holdout}']
            stem = cache_stem('bpe', split, None, apply_overrides(ExperimentConfig(), over).surface_spec())
            meta_path = os.path.join(a.cache_dir, stem + '.json')
            if not (os.path.exists(path) and os.path.exists(meta_path)):
                print(f'    {t:18} no tokenizer or no cache — skipped')
                continue
            meta = json.load(open(meta_path))
            from morphlm.data.stream import require_unpacked
            require_unpacked(a.cache_dir, stem)         # a packed stream is unpacked first (cache_pack.py)
            ids = np.load(os.path.join(a.cache_dir, stem + '.npy'), mmap_mode='r')
            tok = load_tokenizer(path, vocab_only=True)        # decoding needs only the vocabulary
            text = tok.decode([int(i) for i in ids])
            raw = text.encode('utf-8')
            digest = hashlib.sha256(raw).hexdigest()
            same_len = len(raw) == meta['num_bytes']
            ok &= same_len
            hashes.add(digest)
            rows[t] = {'tokens': int(ids.shape[0]), 'recorded_bytes': meta['num_bytes'],
                       'decoded_bytes': len(raw), 'sha256': digest}
            print(f'    {t:18} {ids.shape[0]:>11,} tokens  recorded {meta["num_bytes"]:>12,} bytes  '
                  f'decoded {len(raw):>12,}  {"OK " if same_len else "DIFFERENT"}  {digest[:16]}')
        agree = len(hashes) <= 1
        ok &= agree
        print(f'    every tokenizer decodes to {"the same text" if agree else "DIFFERENT texts"} '
              f'({len(rows)} tokenizers)')
        rec.add(split, {'tokenizers': rows, 'same_text': agree})
    rec.write()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
