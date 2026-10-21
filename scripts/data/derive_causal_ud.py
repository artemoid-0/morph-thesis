#!/usr/bin/env python3
"""
derive_causal_ud.py — Universal Dependencies features from a strictly causal tagger, every word
tagged from its LEFT context only (journey §33.125; DIRECTIONS, the queue's item 2): for Ukrainian
gender and number, `ugendercausal` and `unumbercausal`.

WHY. `ugender` and `unumber` are read from spaCy's morphologizer over the whole paragraph, so a
word's value can depend on the words after it: an adjective's gender on the noun it agrees with,
which follows it. Shifted to the previous word, the feature then carries part of the word being
predicted. The leak audit of 21 October (`morph_causality_check.py --by-type`, 10,000 training
sentences) puts it at 3.2% of words for gender and 1.4% for number, spread over adjectives, proper
nouns, nouns and determiners and over hundreds of word types (1,635 and 554), so no word list
removes it, as the prepositions did for case (`ucaseprep`). As `derive_causal_pos.py` does for
English part of speech, this removes it at every word: the value a causal tagger would give.

HOW. As derive_causal_pos.py: each word is tagged in a window of its last WINDOW + 1 tokens. The value
is read as the stream builder reads it (morphlm/pipelines/bpe.py: `tok.morph`, alphabetic words only;
a value outside the block's inventory, or two values, is 'none'), and each BPE token takes the value
of the word holding its first annotated character (`_first_annotated`). The windows are tagged
without the parser, which never changes a morphological feature (22,208 of 22,208 words, journey
§33.125), and the window check below holds them against the whole prefix under the builder's full
pipeline. The model's causal shift is applied at training time. The paragraphs come from the cached
stream itself, decoded: every stream decodes to its corpus text byte for byte (G-75), so this runs
where the corpus is not on disk. The values belong to words, so one tagging serves every tokenizer
given: each is aligned, checked and written for its own stream.

THREE CHECKS THAT CAN FAIL, all before anything is written:
  * a window of WINDOW tokens without the parser tags each word as its whole prefix does under the
    full pipeline (the first 2,000 words);
  * the decoded paragraphs re-encode to each stream's cached ids, token for token;
  * on the first --check-rows paragraphs, tagged in full context, the codes equal the cached ones.

Writes <ids_stem>_udcausal.codes.npy / .json with the source rows and the causal rows (ugender,
unumber, ugendercausal, unumbercausal) — a derived file, read only by the stream it was derived
for — and a record to results/audits/.

    python scripts/data/derive_causal_ud.py --limit-rows 2000                        # the checks only
    python scripts/data/derive_causal_ud.py --workers 18      # BPE 17k and endings apart, every split
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

FACTS = {'ugender': 'ugendercausal', 'unumber': 'unumbercausal'}    # source fact -> causal fact
W = {}      # per-worker state: the tokenizer, spaCy, the blocks


def _init(bpe_paths, model, window):
    import spacy
    from morphlm.data.tokenization import load_tokenizer
    from morphlm.registry import _load, FACT_BLOCKS
    _load()
    W['toks'] = [load_tokenizer(p) for p in bpe_paths]       # each encodes as its stream does
    # the stream builder's pipeline for Ukrainian (morphlm/pipelines/bpe.py): NER and lemmatizer off
    W['full'] = spacy.load(model, disable=['ner', 'lemmatizer'])
    W['full'].max_length = max(W['full'].max_length, 4_000_000)
    W['fast'] = spacy.load(model, disable=['ner', 'lemmatizer', 'parser'])     # for the windows
    W['blocks'] = [FACT_BLOCKS[f](None) for f in FACTS]      # ugender, unumber: 8 classes each
    W['window'] = window


def _value(tok, feature):
    """A word's value as the pipeline stores it: its morphology by name, alphabetic words only."""
    return (tok.morph.to_dict() or {}).get(feature) if tok.is_alpha else None


def _code(blk, v):
    return blk.classes.index(v) if v in blk.values else len(blk.classes) - 1


def _row(text, check):
    """For each tokenizer: the ids, the full-context codes (when `check`; else -1) and the causal
    codes of one paragraph, per fact, as its stream encodes it."""
    from spacy.tokens import Doc
    from morphlm.pipelines.bpe import _first_annotated
    fast, k, blocks = W['fast'], W['window'], W['blocks']
    doc = W['full'](text) if check else fast.make_doc(text)
    words = [t.text for t in doc]
    spaces = [bool(t.whitespace_) for t in doc]
    wins = [Doc(fast.vocab, words=words[max(0, i - k):i + 1], spaces=spaces[max(0, i - k):i + 1])
            for i in range(len(doc))]
    lasts = [w[-1] for w in fast.pipe(wins, batch_size=512)]
    n = len(text)
    maps = []
    for blk in blocks:
        full_c = [None] * n
        caus_c = [None] * n
        for t, last in zip(doc, lasts):
            v_full = _value(t, blk.feature) if check else None
            v_caus = _value(last, blk.feature)
            for c in range(t.idx, min(t.idx + len(t.text), n)):
                full_c[c] = v_full
                caus_c[c] = v_caus
        maps.append((blk, full_c, caus_c))
    out = []
    for tok in W['toks']:
        enc = tok.encode(text)
        offsets = enc.offsets
        full = [[_code(blk, _first_annotated(fc, s, e)) if check else -1 for s, e in offsets] for blk, fc, _ in maps]
        caus = [[_code(blk, _first_annotated(cc, s, e)) for s, e in offsets] for blk, _, cc in maps]
        out.append((list(enc.ids), full, caus))
    return out


def _chunk(task):
    texts, check = task
    per = [([], [[] for _ in FACTS], [[] for _ in FACTS]) for _ in W['toks']]
    for t in texts:
        for (ids, full, caus), (a, b, c) in zip(per, _row(t, check)):
            ids.extend(a)
            for j in range(len(FACTS)):
                full[j].extend(b[j])
                caus[j].extend(c[j])
    return [(np.asarray(ids, dtype=np.int32), np.asarray(full, dtype=np.int16).reshape(len(FACTS), -1),
             np.asarray(caus, dtype=np.int16).reshape(len(FACTS), -1)) for ids, full, caus in per]


def _prefix_check(texts, window, limit=2000):
    """The fast window's values against the whole prefix's under the full pipeline, on the first
    `limit` alphabetic words."""
    from spacy.tokens import Doc
    full, fast = W['full'], W['fast']
    feats = [b.feature for b in W['blocks']]
    same = total = 0
    for text in texts:
        doc = full(text)
        words = [t.text for t in doc]
        spaces = [bool(t.whitespace_) for t in doc]
        for i, t in enumerate(doc):
            if not t.is_alpha:
                continue
            pre = full(Doc(full.vocab, words=words[:i + 1], spaces=spaces[:i + 1]))[-1]
            win = fast(Doc(fast.vocab, words=words[max(0, i - window):i + 1],
                           spaces=spaces[max(0, i - window):i + 1]))[-1]
            same += all(_value(pre, f) == _value(win, f) for f in feats)
            total += 1
            if total >= limit:
                return same, total
    return same, total


def paragraphs_of(ids, tok, limit):
    """The stream's paragraphs, decoded: each ends in the newline the stream builder appended.
    With `limit`, only the stream's start is decoded, and only whole paragraphs are kept."""
    if limit:
        ids = ids[:min(len(ids), limit * 400)]
    text = tok.decode([int(i) for i in ids])
    parts = text.split('\n')
    out = [p + '\n' for p in parts[:-1]]
    if parts[-1] and not limit:
        out.append(parts[-1])           # a last paragraph without its newline: the check will say
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bpe-paths', nargs='+', default=['data/bpe_uk_17k.json', 'data/bpeend_uk_17k.json'],
                    help='the tokenizers whose streams get the codes; the paragraphs are decoded from the first')
    ap.add_argument('--corpus', default='ukwiki')
    ap.add_argument('--corpus-fraction', type=float, default=0.05)
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--spacy-model', default='uk_core_news_sm')
    ap.add_argument('--splits', nargs='+', default=['validation', 'train'])
    ap.add_argument('--window', type=int, default=4, help='left tokens each word is tagged with')
    ap.add_argument('--check-rows', type=int, default=20000,
                    help='paragraphs of each split on which the full context must reproduce the cached codes')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--rows-per-task', type=int, default=400)
    ap.add_argument('--limit-rows', type=int, default=0, help='the checks on the first rows only; writes nothing')
    ap.add_argument('--cache-dir', default='data/cache')
    args = ap.parse_args()
    os.chdir(ROOT)

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem, require_unpacked, stream_codes_files
    from morphlm.record import AuditRecord
    rec = AuditRecord('derive_causal_ud', args)
    _init(args.bpe_paths, args.spacy_model, args.window)
    sources = list(FACTS)

    for split in args.splits:
        t0 = time.time()
        streams = []        # per tokenizer: (bpe path, ids stem, cached ids, cached codes per fact, source)
        for bpe in args.bpe_paths:
            cfg = apply_overrides(ExperimentConfig(), [
                'data.pipeline=bpe', f'data.bpe_path={bpe}', f'data.corpus={args.corpus}',
                f'data.corpus_fraction={args.corpus_fraction}', f'data.corpus_holdout={args.corpus_holdout}',
                f'data.spacy_model={args.spacy_model}'])
            ids_stem = cache_stem('bpe', split, None, cfg.surface_spec())
            ids_path = os.path.join(args.cache_dir, ids_stem + '.npy')
            require_unpacked(args.cache_dir, ids_stem)      # a packed stream is unpacked first (cache_pack.py)
            if not os.path.exists(ids_path):
                raise SystemExit(f'{split}: no cached ids {ids_stem} ({bpe})')
            cached_ids = np.load(ids_path, mmap_mode='r')
            src = None
            for cj, cn, cm in stream_codes_files(args.cache_dir, 'bpe', split, None, ids_stem, int(cached_ids.shape[0])):
                if all(f in cm.get('fact_names', []) for f in sources) and not cj.endswith('_udcausal.codes.json'):
                    src = (cn, cm)
                    break
            if src is None:
                raise SystemExit(f'{split}: no codes file of {ids_stem} holding {sources}')
            held = np.load(src[0], mmap_mode='r')
            streams.append((bpe, ids_stem, cached_ids, [held[src[1]['fact_names'].index(f)] for f in sources], src))

        texts = paragraphs_of(streams[0][2], W['toks'][0], args.limit_rows)
        same, total = _prefix_check(texts, args.window)
        print(f'  {split}: {len(texts):,} paragraphs; a window of {args.window} without the parser against the '
              f'whole prefix with it: {same:,} of {total:,} words the same', flush=True)
        if same != total:
            rec.add(split, {'window_vs_prefix': [same, total]})
            rec.write()
            raise SystemExit('the window does not reproduce the prefix tagging — widen --window')

        tasks = [(texts[i:i + args.rows_per_task], i < args.check_rows)
                 for i in range(0, len(texts), args.rows_per_task)]
        if args.workers > 1:
            with Pool(args.workers, initializer=_init, initargs=(args.bpe_paths, args.spacy_model, args.window)) as pool:
                parts = pool.map(_chunk, tasks, chunksize=1)
        else:
            parts = [_chunk(t) for t in tasks]

        results, ok = [], True
        for k, (bpe, ids_stem, cached_ids, cached, src) in enumerate(streams):
            ids = np.concatenate([p[k][0] for p in parts])
            full = np.concatenate([p[k][1] for p in parts], axis=1)
            caus = np.concatenate([p[k][2] for p in parts], axis=1)
            n = len(ids)
            # CHECK: the same token stream
            ids_ok = bool(np.array_equal(ids, np.asarray(cached_ids[:n]))) and (bool(args.limit_rows) or n == len(cached_ids))
            # CHECK: on the checked paragraphs the full context gives the cached codes, for every fact
            checked = full[0] >= 0
            codes_ok = {f: bool(checked.any()) and bool(np.array_equal(full[j][checked], np.asarray(cached[j][:n])[checked]))
                        for j, f in enumerate(sources)}
            agree = {f: float((caus[j] == np.asarray(cached[j][:n])).mean()) for j, f in enumerate(sources)}
            print(f'  {split}, {os.path.basename(bpe)} ({ids_stem}): {n:,} tokens ({len(cached_ids):,} cached); '
                  f'ids identical: {ids_ok}; the full context reproduces the cached codes on {int(checked.sum()):,} '
                  f'tokens: {codes_ok}; causal = cached at ' + ', '.join(f'{f} {100 * a:.2f}%' for f, a in agree.items())
                  + f' of tokens; {time.time() - t0:.0f}s', flush=True)
            rec.add(f'{split}:{os.path.basename(bpe)}', {
                'stream': ids_stem, 'paragraphs': len(texts), 'tokens': n, 'cached_tokens': int(len(cached_ids)),
                'window_vs_prefix': [same, total], 'ids_identical': ids_ok, 'codes_identical': codes_ok,
                'checked_tokens': int(checked.sum()), 'causal_equals_cached': agree,
                'source': os.path.basename(src[0]), 'seconds': round(time.time() - t0)})
            ok = ok and ids_ok and all(codes_ok.values())
            results.append((ids_stem, n, caus, cached, src))
        if not ok:
            rec.write()
            raise SystemExit(f'{split}: a check failed — nothing is written')
        if args.limit_rows:
            continue
        for ids_stem, n, caus, cached, src in results:
            out_stem = ids_stem + '_udcausal'
            out_npy = os.path.join(args.cache_dir, out_stem + '.codes.npy')
            out_json = os.path.join(args.cache_dir, out_stem + '.codes.json')
            if os.path.exists(out_npy):
                raise SystemExit(f'{out_stem} exists — a derived file is never overwritten; move it aside to rebuild')
            names = sources + [FACTS[f] for f in sources]
            arr = np.lib.format.open_memmap(out_npy, mode='w+', dtype=np.int16, shape=(len(names), n))
            for j in range(len(sources)):
                arr[j] = np.asarray(cached[j][:n])
                arr[len(sources) + j] = caus[j]
            arr.flush()
            json.dump({'fact_names': names, 'num_tokens': int(n),
                       'dep_granularity': src[1].get('dep_granularity', 'coarse'),
                       'derived_by': 'scripts/data/derive_causal_ud.py', 'window': args.window},
                      open(out_json, 'w'), indent=2)
            print(f'    wrote {out_stem}.codes.npy', flush=True)
    rec.write()


if __name__ == '__main__':
    main()
