#!/usr/bin/env python3
"""
derive_causal_pos.py — `poscausal`: part of speech from a strictly causal tagger, every word
tagged from its LEFT context only (journey §33.98; DIRECTIONS, the queue's item 2).

WHY. `pos` is read from spaCy's tagger over the whole paragraph, so a word's tag can depend on the
words after it — the next word above all: "to" is a particle before a verb and a preposition
before a noun, a capitalised noun a proper noun before another name. Shifted to the previous word,
the feature then carries part of the word being predicted (§33.3: 7.27% of tags in WikiText-103's
train text change when the right context is removed). In Ukrainian that leak sat at prepositions,
and a control that read only them from their form removed it (`ucaseprep`, §33.74). In English it
is spread over thousands of words, mostly nouns, proper nouns and adjectives (§33.98), so no word
list removes it. This removes it at every word: the tag a causal tagger would give.

HOW. A causal tagger is spaCy's TAGGER reading the word's left context, its tags mapped to the
coarse classes by the tag map. The tagger sees four tokens on each side, so the window of a word's
last WINDOW + 1 = 5 tokens gives exactly the tag its whole prefix gives (100.000% of 9,992 training
words; §33.98), and the check below repeats that on every run. `pos` as the pipeline builds it also
passes the tags through rules that read the dependency parse ("which" as a determiner or a pronoun,
"have" as an auxiliary or a verb: 0.68% of its codes). A parse needs the whole sentence, and a parser
given a prefix that ends in "the" has nothing to attach it to — it calls it a pronoun 4,498 times in
4,499 (§33.98, as §30.6b found for the parser) — so a causal tagger cannot make those refinements.
They are information from the right, and `poscausal` goes without them.

The full pipeline, parser included, is run on the paragraphs only to prove the alignment: in full
context it must reproduce the cached `pos` codes exactly.
Each training paragraph is encoded exactly as the stream builder encodes it — the same text, the
same BPE tokenizer, the same spaCy tokens — and each BPE token takes the tag of the word holding its
first annotated character (`_first_annotated`, as the pipeline does). The model's causal shift is
applied at training time, as for `pos`.

TWO CHECKS THAT CAN FAIL, both before anything is written:
  * the re-encoded ids must equal the cached ids of the stream, token for token;
  * the same paragraphs tagged in full context must reproduce the cached `pos` codes exactly —
    which shows the alignment, the tokenization and the class order are the pipeline's.

Writes <ids_stem>_poscausal.codes.npy / .json with the rows ['pos', 'poscausal'] (a derived file,
read only by the stream it was derived for), and a record to results/audits/.

    python scripts/data/derive_causal_pos.py --limit-rows 3000          # a check on the first rows
    python scripts/data/derive_causal_pos.py --workers 8                 # every split
"""
import os
import sys
import json
import time
import argparse
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

W = {}      # per-worker state: the tokenizer, spaCy, the class index


def _init(bpe_path, window, check_rows=0):
    import spacy
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.features.blocks import POS_CLASSES
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', f'data.bpe_path={bpe_path}'])
    W['tok'] = build_pipeline(cfg, verbose=False).tokenizer
    # the causal tagger: the tagger and the tag map, no parse
    W['nlp'] = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
    # the pipeline's own part of speech (tagger, parser, parse-reading rules), for the check that
    # proves the alignment; the entity recognizer and the lemmatizer cannot change it
    W['full'] = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    W['classes'] = {c: i for i, c in enumerate(POS_CLASSES)}
    W['window'] = window


def _code(tag):
    c = W['classes']
    return c.get(tag, len(c) - 1)


def _row(text, check):
    """ids, the pipeline's full-context codes (when `check`) and the causal codes of one paragraph,
    as the stream encodes it."""
    from spacy.tokens import Doc
    from morphlm.pipelines.bpe import _first_annotated
    nlp, k = W['nlp'], W['window']
    enc = W['tok'].encode(text)
    doc = W['full'](text) if check else nlp.make_doc(text)
    words = [t.text for t in doc]
    spaces = [bool(t.whitespace_) for t in doc]
    wins = [Doc(nlp.vocab, words=words[max(0, i - k):i + 1], spaces=spaces[max(0, i - k):i + 1])
            for i in range(len(doc))]
    causal = [w[-1].pos_ for w in nlp.pipe(wins, batch_size=512)]
    n = len(text)
    full_c = [None] * n
    caus_c = [None] * n
    for t, cz in zip(doc, causal):
        for c in range(t.idx, min(t.idx + len(t.text), n)):
            full_c[c] = t.pos_ if check else None
            caus_c[c] = cz
    offsets = enc.offsets
    full, caus = [], []
    for idx in range(len(enc.ids)):
        s, e = offsets[idx]
        full.append(_code(_first_annotated(full_c, s, e)) if check else -1)
        caus.append(_code(_first_annotated(caus_c, s, e)))
    return enc.ids, full, caus


def _chunk(task):
    texts, check = task
    ids, full, caus = [], [], []
    for t in texts:
        a, b, c = _row(t, check)
        ids.extend(a); full.extend(b); caus.extend(c)
    return (np.asarray(ids, dtype=np.int32), np.asarray(full, dtype=np.int16), np.asarray(caus, dtype=np.int16))


def _prefix_check(texts, window, limit=2000):
    """The window's tag against the whole prefix's, on the first `limit` alphabetic words."""
    from spacy.tokens import Doc
    nlp = W['nlp']
    same = total = 0
    for text in texts:
        doc = nlp(text)
        words = [t.text for t in doc]
        spaces = [bool(t.whitespace_) for t in doc]
        for i, t in enumerate(doc):
            if not t.is_alpha:
                continue
            pre = nlp(Doc(nlp.vocab, words=words[:i + 1], spaces=spaces[:i + 1]))[-1].pos_
            win = nlp(Doc(nlp.vocab, words=words[max(0, i - window):i + 1],
                          spaces=spaces[max(0, i - window):i + 1]))[-1].pos_
            same += pre == win
            total += 1
            if total >= limit:
                return same, total
    return same, total


def rows_of(cfg, split, limit):
    """The paragraphs exactly as the stream builder takes them (morphlm/data/stream.py)."""
    from morphlm.data.stream import load_corpus
    out = []
    for row in load_corpus(cfg, split):
        text = row['text'].strip()
        if text:
            text += row.get('sep', '')
        if not text or text.startswith(' ='):
            continue
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--splits', nargs='+', default=['validation', 'test', 'train'])
    ap.add_argument('--window', type=int, default=4, help='left tokens each word is tagged with (the tagger sees 4)')
    ap.add_argument('--check-rows', type=int, default=20000,
                    help='paragraphs of each split on which the full pipeline must reproduce the cached `pos`')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--rows-per-task', type=int, default=400)
    ap.add_argument('--limit-rows', type=int, default=0, help='a check on the first rows only; writes nothing')
    ap.add_argument('--cache-dir', default='data/cache')
    args = ap.parse_args()
    os.chdir(ROOT)

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem, stream_codes_files
    from morphlm.record import AuditRecord
    rec = AuditRecord('derive_causal_pos', args)
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', f'data.bpe_path={args.bpe_path}'])
    _init(args.bpe_path, args.window)

    for split in args.splits:
        t0 = time.time()
        ids_stem = cache_stem('bpe', split, None, cfg.surface_spec())
        ids_path = os.path.join(args.cache_dir, ids_stem + '.npy')
        if not os.path.exists(ids_path):
            raise SystemExit(f'{split}: no cached ids {ids_stem}')
        cached_ids = np.load(ids_path, mmap_mode='r')
        src = None
        for cj, cn, cm in stream_codes_files(args.cache_dir, 'bpe', split, None, ids_stem, int(cached_ids.shape[0])):
            if 'pos' in cm.get('fact_names', []) and not cj.endswith('_poscausal.codes.json'):
                src = (cn, cm)
                break
        if src is None:
            raise SystemExit(f'{split}: no codes file holding `pos` for {ids_stem}')
        cached_pos = np.load(src[0], mmap_mode='r')[src[1]['fact_names'].index('pos')]

        texts = rows_of(cfg, split, args.limit_rows)
        same, total = _prefix_check(texts, args.window)
        print(f'  {split}: {len(texts):,} paragraphs; window of {args.window} against the whole prefix: '
              f'{same:,} of {total:,} words the same')
        if same != total:
            raise SystemExit('the window does not reproduce the prefix tagging — widen --window')

        tasks = [(texts[i:i + args.rows_per_task], i < args.check_rows)
                 for i in range(0, len(texts), args.rows_per_task)]
        if args.workers > 1:
            with Pool(args.workers, initializer=_init, initargs=(args.bpe_path, args.window)) as pool:
                parts = pool.map(_chunk, tasks, chunksize=1)
        else:
            parts = [_chunk(t) for t in tasks]
        ids = np.concatenate([p[0] for p in parts])
        full = np.concatenate([p[1] for p in parts])
        caus = np.concatenate([p[2] for p in parts])
        n = len(ids)
        # CHECK 1: the same token stream
        ids_ok = bool(np.array_equal(ids, np.asarray(cached_ids[:n]))) and (bool(args.limit_rows) or n == len(cached_ids))
        # CHECK 2: on the checked paragraphs, the pipeline's full-context codes are the cached `pos`
        checked = full >= 0
        pos_ok = bool(checked.any()) and bool(np.array_equal(full[checked], np.asarray(cached_pos[:n])[checked]))
        agree = float((caus == np.asarray(cached_pos[:n])).mean())
        print(f'  {split}: {n:,} tokens ({len(cached_ids):,} cached); ids identical: {ids_ok}; '
              f'the pipeline reproduces the cached `pos` on {int(checked.sum()):,} tokens: {pos_ok}; '
              f'causal = cached `pos` at {100 * agree:.2f}% of tokens; {time.time() - t0:.0f}s')
        rec.add(split, {'paragraphs': len(texts), 'tokens': n, 'cached_tokens': int(len(cached_ids)),
                        'window_vs_prefix': [same, total], 'ids_identical': ids_ok,
                        'pos_identical': pos_ok, 'pos_checked_tokens': int(checked.sum()), 'causal_equals_pos': agree,
                        'source': os.path.basename(src[0]), 'seconds': round(time.time() - t0)})
        if not (ids_ok and pos_ok):
            rec.write()
            raise SystemExit(f'{split}: a check failed — nothing is written')
        if args.limit_rows:
            continue
        out_stem = ids_stem + '_poscausal'
        out_npy = os.path.join(args.cache_dir, out_stem + '.codes.npy')
        out_json = os.path.join(args.cache_dir, out_stem + '.codes.json')
        if os.path.exists(out_npy):
            raise SystemExit(f'{out_stem} exists — a derived file is never overwritten; move it aside to rebuild')
        arr = np.lib.format.open_memmap(out_npy, mode='w+', dtype=np.int16, shape=(2, n))
        arr[0] = np.asarray(cached_pos[:n])
        arr[1] = caus
        arr.flush()
        json.dump({'fact_names': ['pos', 'poscausal'], 'num_tokens': int(n),
                   'dep_granularity': src[1].get('dep_granularity', 'coarse'),
                   'derived_by': 'scripts/data/derive_causal_pos.py', 'window': args.window},
                  open(out_json, 'w'), indent=2)
        print(f'    wrote {out_stem}.codes.npy')
    rec.write()


if __name__ == '__main__':
    main()
