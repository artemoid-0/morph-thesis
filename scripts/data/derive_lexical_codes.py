#!/usr/bin/env python3
"""
derive_lexical_codes.py — a strictly causal version of a contextual feature.

`mfeat` is read from spaCy's tagger, whose window extends about four tokens to
the right, so a word's bundle can depend on the words after it ("reading" is
VBG in "reading books", NN in "reading is fun"). Shifted to the previous word,
it may still carry a little of the word being predicted.

This script gives every word type its MOST FREQUENT bundle in the training
corpus — a lexicon, looked up from the word's own form, like `mstruct` from
MorphoLEX — and writes that as a new fact (`mfeatlex`). After the causal shift
it depends only on words already written.

It reads the cached ids and the cached contextual codes (no spaCy), writes:
  data/cache/lexicon_<fact>_<tok>[_<corpus><fraction>].npz (+ .json)
                                                   the lexicon, reused on new text;
                                                   named per corpus and size outside
                                                   WikiText (G-66)
  <ids_stem>_<out_fact>.codes.npy / .json          source facts + the new row
                                                   (named per feature: deriving
                                                   another never overwrites this)
so any request including the new fact resolves by content-based reuse.

Controls derived the same way: `--transform` (the constant null, frequency bands),
`--shuffle-types` (classes shuffled among words of similar frequency), and
`--recode-pos` (G-38: the value in context everywhere but at the word types whose
majority part of speech is given — prepositions — which get their form's value).

    python scripts/data/derive_lexical_codes.py --bpe-path data/bpe_17k.json --verify
    python scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaseprep --recode-pos ADP --keep ucase \
        --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_17k.json --verify
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CHUNK = 8_000_000

#: Words whose majority bundle is known in advance. If the key-to-code mapping
#: were misaligned anywhere, these would come back wrong — the check that can
#: fail on the known-broken case (a check must be able to fail).
KNOWN = {
    'mfeat': {
        ' the': 'Definite=Def|PronType=Art',
        ' these': 'Number=Plur|PronType=Dem',
        ' they': 'Case=Nom|Number=Plur|Person=3|PronType=Prs',
        ' were': 'Mood=Ind|Tense=Past|VerbForm=Fin',
        ' their': 'Number=Plur|Person=3|Poss=Yes|PronType=Prs',
        ' was': 'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin',
    },
    # universal POS as spaCy assigns them in context (checked with the model)
    'pos': {
        ' the': 'DET', ' these': 'DET', ' they': 'PRON',
        ' were': 'AUX', ' their': 'PRON', ' was': 'AUX',
    },
    # Ukrainian case: prepositions that govern one case, and nominative
    # pronouns (checked with uk_core_news_sm, 29 Sep 2026)
    'ucase': {
        ' до': 'Gen', ' для': 'Gen', ' від': 'Gen', ' без': 'Gen',
        ' через': 'Acc', ' про': 'Acc',
        ' він': 'Nom', ' вони': 'Nom', ' ми': 'Nom',
    },
    # Ukrainian gender and number: personal pronouns and nouns of one gender (their majority
    # values over the corpus; the derivation checks them, 21 October 2026)
    'ugender': {' він': 'Masc', ' вона': 'Fem', ' воно': 'Neut', ' місто': 'Neut', ' країна': 'Fem'},
    'unumber': {' він': 'Sing', ' вона': 'Sing', ' вони': 'Plur', ' ми': 'Plur', ' люди': 'Plur'},
}
MIN_KNOWN_PRESENT = 3


def cut_points(ids, tables, chunk):
    """Chunk boundaries that never split a word."""
    alpha_lut, opens_lut = tables
    frag_lut = getattr(tables, 'frag', None)
    n = len(ids)
    cuts = [0]
    while cuts[-1] + chunk < n:
        t = cuts[-1] + chunk
        while t < n:
            i = int(ids[t])
            if frag_lut is not None and frag_lut[i]:
                t += 1                  # a fragment's class is its predecessor's
                continue
            prev_alpha = alpha_lut[int(ids[t - 1])]
            continuation = alpha_lut[i] and prev_alpha and not opens_lut[i]
            if not continuation:
                break
            t += 1
        cuts.append(t)
    cuts.append(n)
    return cuts


def find_source(cache_dir, split, ids_stem, fact, n_tokens, out_fact=None):
    """The fullest codes file of the stream `ids_stem` holding `fact` — never the
    file this derivation will replace (a rerun would otherwise read its own
    previous output), and never a file derived for another stream (G-72: a
    corpus's validation split is the same tokens at every fraction, and a
    derived row is the lexicon of the fraction it was derived for)."""
    from morphlm.data.stream import stream_codes_files
    best = None
    for cand, npy, cm in stream_codes_files(cache_dir, 'bpe', split, None, ids_stem, n_tokens):
        if out_fact and cand.endswith(f'_{out_fact}.codes.json'):
            continue
        if fact not in cm.get('fact_names', []):
            continue
        if best is None or len(cm['fact_names']) > len(best[1]['fact_names']):
            best = (npy, cm, cand)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--corpus', default='wikitext103', help='wikitext103 | ukwiki — part of the cache name')
    ap.add_argument('--corpus-fraction', type=float, default=1.0)
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--fact', default='mfeat')
    ap.add_argument('--out-fact', default=None, help='default: <fact>lex')
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--lexicon-fraction', type=float, default=None,
                    help="build the lexicon from this corpus fraction's train split instead "
                         "of the target's own. Sizes are nested (one per-article hash), so a "
                         "larger fraction's lexicon covers every word of a smaller one — a "
                         "little-data run then gets the same context-free knowledge as a "
                         "full-data run, not a lexicon starved by its own small corpus")
    ap.add_argument('--shuffle-types', type=int, default=None, metavar='SEED',
                    help='CONTROL: permute the lexicon\'s codes among word types '
                         'of similar frequency (see lexical.shuffle_within_frequency)')
    ap.add_argument('--shuffle-bin', type=int, default=64)
    ap.add_argument('--transform', choices=['const', 'freq', 'flat'], default=None,
                    help='CONTROL: const = one class for every word, seen in train or not (the null; G-73); '
                         'freq = the word\'s frequency band (lexical.frequency_bands); '
                         'flat = one class at EVERY position, words and non-words alike')
    ap.add_argument('--recode-pos', default=None, metavar='TAGS',
                    help='CONTROL for the tagger\'s right-context leak (G-38): the word types whose '
                         'majority part of speech in train is one of TAGS (comma-separated, e.g. ADP) '
                         'get the lexicon\'s value at every occurrence — what the form alone says; every '
                         'other word keeps its value in context. Reads `pos` from the same codes file; '
                         'needs its own --out-fact (ucaseprep)')
    ap.add_argument('--keep', default='all',
                    help='source facts to carry into the derived file, comma-separated '
                         '(default all). Fewer rows, less disk; a run can only reuse '
                         'the file if every fact it requests is in it')
    args = ap.parse_args()
    out_fact = args.out_fact or args.fact + 'lex'

    from morphlm.record import AuditRecord
    from morphlm.data.stream import cache_stem, _atomic_save_npy, _atomic_write_json
    from morphlm.features.lexical import (tokenizer_tables, word_spans,
                                          build_lexicon, lookup, lexicon_path, lexicon_condition,
                                          save_lexicon, shuffle_within_frequency,
                                          class_shares, frequency_bands)
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    rec = AuditRecord('derive_lexical_codes', args)

    blk = build_blocks(apply_overrides(ExperimentConfig(),
                       ['data.pipeline=bpe', f'frozen.facts={args.fact}']))[0]
    none_code = blk.n_classes - 1
    tables = tokenizer_tables(args.bpe_path)
    stem = os.path.basename(args.bpe_path).replace('.json', '')
    # the cache name exactly as the training run computes it (config.surface_spec)
    def spec_for(frac):
        return apply_overrides(ExperimentConfig(), [
            'data.pipeline=bpe', f'data.bpe_path={args.bpe_path}', f'data.corpus={args.corpus}',
            f'data.corpus_fraction={frac}',
            f'data.corpus_holdout={args.corpus_holdout}']).surface_spec()
    lex_spec = None if args.lexicon_fraction is None else spec_for(args.lexicon_fraction)
    spec = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', f'data.bpe_path={args.bpe_path}', f'data.corpus={args.corpus}',
        f'data.corpus_fraction={args.corpus_fraction}',
        f'data.corpus_holdout={args.corpus_holdout}']).surface_spec()

    def load_split(split, sp=None):
        ids_stem = cache_stem('bpe', split, None, sp or spec)
        ids_path = os.path.join(args.cache_dir, ids_stem + '.npy')
        from morphlm.data.stream import require_unpacked
        require_unpacked(args.cache_dir, ids_stem)      # never a packed stream taken for a missing one
        if not os.path.exists(ids_path):
            return None
        ids = np.load(ids_path, mmap_mode='r')
        src = find_source(args.cache_dir, split, ids_stem, args.fact, int(ids.shape[0]),
                          out_fact=args.out_fact)
        return ids_stem, ids, src

    # ── 1. lexicon from TRAIN ────────────────────────────────────────────────
    tr = load_split('train', lex_spec)
    if lex_spec is not None:
        print(f'  lexicon from corpus fraction {args.lexicon_fraction:g}; codes written for {args.corpus_fraction:g}')
    if tr is None or tr[2] is None:
        raise SystemExit(f'train ids or a codes file holding {args.fact!r} not found')
    ids_stem, ids, (src_npy, src_meta, _) = tr
    row = src_meta['fact_names'].index(args.fact)
    codes = np.load(src_npy, mmap_mode='r')[row]
    print(f'  train: {len(ids):,} tokens; source {os.path.basename(src_npy)} row {row}')
    recode = None
    if args.recode_pos:
        from morphlm.features.blocks import POS_CLASSES
        tags = [t.strip() for t in args.recode_pos.split(',') if t.strip()]
        unknown = [t for t in tags if t not in POS_CLASSES]
        if unknown or not tags:
            raise SystemExit(f'--recode-pos: {unknown or tags} not among {POS_CLASSES}')
        if 'pos' not in src_meta['fact_names']:
            raise SystemExit(f'--recode-pos reads `pos` from the source file, which holds {src_meta["fact_names"]}')
        recode = {'tags': tags, 'ids': np.array([POS_CLASSES.index(t) for t in tags]),
                  'none': len(POS_CLASSES) - 1}
        pos_codes = np.load(src_npy, mmap_mode='r')[src_meta['fact_names'].index('pos')]

    keys_all, codes_all, pos_all = [], [], []
    cuts = cut_points(ids, tables, CHUNK)
    for a, b in zip(cuts[:-1], cuts[1:]):
        chunk = np.asarray(ids[a:b], dtype=np.int64)
        wid, key, starts = word_spans(chunk, tables)
        keys_all.append(key[starts])
        codes_all.append(np.asarray(codes[a:b])[starts])
        if recode is not None:
            pos_all.append(np.asarray(pos_codes[a:b])[starts])
    keys_all = np.concatenate(keys_all)
    codes_all = np.concatenate(codes_all).astype(np.int64)
    print(f'  {len(keys_all):,} word occurrences in {len(cuts) - 1} chunks')

    lex = build_lexicon(keys_all, codes_all)
    agree = float((lookup(keys_all, lex, none_code) == codes_all).mean())
    print(f'  lexicon: {len(lex[0]):,} word types; each word\'s majority value matches its in-context value for {100 * agree:.2f}% of occurrences')
    if recode is not None:
        # the word types recoded: those whose MAJORITY part of speech is a recoded tag — a
        # property of the form, so whether a word is recoded never depends on its context
        pos_all = np.concatenate(pos_all).astype(np.int64)
        poslex = build_lexicon(keys_all, pos_all)
        recode['lex'] = poslex
        is_rec = np.isin(lookup(keys_all, poslex, recode['none']), recode['ids'])
        tagged = np.isin(pos_all, recode['ids'])
        changed = is_rec & (lookup(keys_all, lex, none_code) != codes_all)
        recode['stats'] = {
            'types_recoded': int(np.isin(poslex[1], recode['ids']).sum()),
            'occurrences_recoded': float(is_rec.mean()),
            # how well the form's majority tag matches the tag in context, both ways
            'recoded_tagged_so_in_context': float(tagged[is_rec].mean()) if is_rec.any() else None,
            'tagged_so_and_recoded': float(is_rec[tagged].mean()) if tagged.any() else None,
            'occurrences_changed': float(changed.mean()),
            'recoded_occurrences_changed': float(changed[is_rec].mean()) if is_rec.any() else None}
        st = recode['stats']
        print(f'  RECODED ({",".join(recode["tags"])} by majority tag): {st["types_recoded"]:,} word types, '
              f'{100 * st["occurrences_recoded"]:.2f}% of word occurrences; their value changes at '
              f'{100 * st["recoded_occurrences_changed"]:.1f}% of them ({100 * st["occurrences_changed"]:.2f}% of all); '
              f'{100 * st["recoded_tagged_so_in_context"]:.1f}% of recoded occurrences are tagged so in context, '
              f'{100 * st["tagged_so_and_recoded"]:.1f}% of occurrences tagged so are recoded')
        rec.add('recode', {'tags': recode['tags'], **st})

    controls = [x for x in (args.transform, args.shuffle_types is not None or None, recode) if x]
    if len(controls) > 1:
        raise SystemExit('--transform, --shuffle-types and --recode-pos are separate controls; pick one')
    if controls and not args.out_fact:
        raise SystemExit('a control needs its own --out-fact (mconst, mfreq, mstructshuf, posshuf, ucaseprep)')
    if controls:
        out_blk = build_blocks(apply_overrides(ExperimentConfig(),
                               ['data.pipeline=bpe', f'frozen.facts={out_fact}']))[0]
        if out_blk.n_classes != blk.n_classes:
            raise SystemExit(f'{out_fact} has {out_blk.n_classes} classes, {args.fact} '
                             f'{blk.n_classes}: codes would not line up')
    transformed = args.transform is not None
    if transformed:
        if args.transform in ('const', 'flat'):
            new = np.zeros(len(lex[1]), dtype=np.int64)
        else:
            new = frequency_bands(lex[2], out_blk.n_classes - 1)
        shares = class_shares(new, lex[2], out_blk.n_classes)
        print(f'  TRANSFORMED ({args.transform}): token shares by class ' +
              ' '.join(f'{out_blk.classes[k]}={100 * v:.1f}%' for k, v in enumerate(shares) if v > 0))
        rec.add('transform', {'kind': args.transform,
                              'shares': {out_blk.classes[k]: float(v) for k, v in enumerate(shares)}})
        lex = (lex[0], new.astype(np.int16), lex[2])
    shuffled = args.shuffle_types is not None
    if shuffled:
        new = shuffle_within_frequency(lex[1], lex[2], args.shuffle_types, args.shuffle_bin)
        moved = new != lex[1]
        changed_types = float(moved.mean())
        changed_tokens = float(lex[2][moved].sum() / lex[2].sum())
        before = class_shares(lex[1], lex[2], blk.n_classes)
        after = class_shares(new, lex[2], blk.n_classes)
        tvd = 0.5 * float(np.abs(before - after).sum())
        # the informative part: tokens whose original class is not the most
        # common one (in English the most frequent words are nearly all
        # 'simple', and a fair shuffle leaves them so)
        top = int(np.argmax(before))
        informative = lex[1] != top
        changed_informative = float(lex[2][moved & informative].sum() /
                                    max(lex[2][informative].sum(), 1))
        print(f'  SHUFFLED (seed {args.shuffle_types}, bins of {args.shuffle_bin} types): '
              f'class changed for {100 * changed_types:.1f}% of types, '
              f'{100 * changed_tokens:.1f}% of tokens; token-level class shares '
              f'moved by {100 * tvd:.2f}% (total variation); '
              f'{100 * changed_informative:.1f}% of tokens outside the majority '
              f'class \'{blk.classes[top]}\' changed class')
        rec.add('shuffle', {'seed': args.shuffle_types, 'bin': args.shuffle_bin,
                            'changed_types': changed_types,
                            'changed_tokens': changed_tokens, 'tvd': tvd,
                            'changed_informative': changed_informative})
        lex = (lex[0], new.astype(np.int16), lex[2])
    # a shuffled lexicon is its own fact; an unshuffled one is named by its source
    # named for the corpus and size it was built from (G-66); WikiText keeps its names
    lpath = lexicon_path(args.bpe_path, out_fact if (shuffled or transformed) else args.fact,
                         args.cache_dir, condition=lexicon_condition(
                             args.corpus, args.lexicon_fraction if args.lexicon_fraction is not None
                             else args.corpus_fraction))
    meta = {'fact': args.fact, 'out_fact': out_fact, 'word_types': int(len(lex[0])),
            'occurrences': int(len(keys_all)), 'agreement_with_context': agree,
            'source': os.path.basename(src_npy), 'bpe': stem}
    if recode is None:
        save_lexicon(lpath, lex, meta)
        print(f'  wrote {lpath}')
    else:
        # the same majority lexicon the word-form variant saved under its source fact:
        # not written again — a recoded row is read from its codes file only
        print(f'  lexicon not written ({os.path.basename(lpath)} is the word-form variant\'s)')
    rec.add('lexicon', meta)

    if args.verify and shuffled:
        # known words cannot be checked — their codes are random by design.
        # Check instead that the control is what it claims to be.
        if tvd > 0.05:
            raise SystemExit(f'token-level class shares moved by {tvd:.2%} — the shuffle '
                             f'did not preserve the class profile; no codes written')
        if changed_tokens < 0.15:
            raise SystemExit(f'only {changed_tokens:.1%} of tokens changed class — '
                             f'frequency bins are too homogeneous for a real control; '
                             f'no codes written')
    if args.verify and transformed:
        # the check is that the control is what it claims to be
        if args.transform in ('const', 'flat') and len(set(lex[1].tolist())) != 1:
            raise SystemExit('const transform left more than one class')
        if args.transform == 'freq':
            used = [v for v in shares[:-1] if v > 0]
            if len(used) < out_blk.n_classes - 2 or max(used) > 3.0 / (out_blk.n_classes - 1):
                raise SystemExit(f'frequency bands badly unbalanced: {shares}')
    if args.verify and not (shuffled or transformed):
        from morphlm.data.tokenization import load_tokenizer
        tok = load_tokenizer(args.bpe_path)          # encodes as the stream does
        bad, absent = [], []
        known = KNOWN.get(args.fact)
        if known is None:
            raise SystemExit(f'no known words defined for {args.fact!r}; add them to KNOWN')
        for text, want in known.items():
            wids = np.array(tok.encode(text).ids, dtype=np.int64)
            _, k, st = word_spans(wids, tables)
            key = k[st][:1]
            li = np.searchsorted(lex[0], key)
            present = bool(len(key)) and li[0] < len(lex[0]) and lex[0][li[0]] == key[0]
            if not present:
                # ABSENT is not MISALIGNED: a word missing from the corpus says
                # nothing about the mapping. Only a present word mapped to the
                # wrong bundle is evidence of a broken key-to-code alignment.
                absent.append(text)
                print(f'    {text!r:10} -> not in the lexicon (absent from the corpus)')
                continue
            got = blk.classes[int(lex[1][li[0]])]
            ok = got == want
            print(f'    {text!r:10} -> {got:48} {"OK" if ok else "EXPECTED " + want}')
            if not ok:
                bad.append(text)
        rec.add('known_words', {'mismatched': bad, 'absent': absent})
        if recode is not None:
            # the recoded set is what it claims: these words are prepositions by their
            # majority tag, and these are not (checked with uk_core_news_sm)
            from morphlm.features.blocks import POS_CLASSES
            want_in = {' до', ' для', ' від', ' без', ' через', ' про', ' на', ' в', ' з'}
            want_out = {' він', ' вони', ' ми', ' було', ' року'}
            wrong = []
            for text in sorted(want_in | want_out):
                wids = np.array(tok.encode(text).ids, dtype=np.int64)
                _, k, st = word_spans(wids, tables)
                key = k[st][:1]
                li = np.searchsorted(recode['lex'][0], key)
                if not (len(key) and li[0] < len(recode['lex'][0]) and recode['lex'][0][li[0]] == key[0]):
                    print(f'    {text!r:10} -> not in the lexicon (absent from the corpus)')
                    continue
                tag = POS_CLASSES[int(recode['lex'][1][li[0]])]
                ok = (tag in recode['tags']) == (text in want_in)
                print(f'    {text!r:10} -> majority tag {tag:6} {"recoded" if tag in recode["tags"] else "kept":8} '
                      f'{"OK" if ok else "EXPECTED " + ("recoded" if text in want_in else "kept")}')
                if not ok:
                    wrong.append(text)
            rec.add('recode_known_words', {'wrong': wrong})
            if wrong:
                raise SystemExit(f'recoded set wrong for {wrong} — no codes written')
        if agree < 0.5:
            raise SystemExit(f'agreement {agree:.2%} is implausibly low — '
                             f'the key-to-code mapping is suspect')
        if bad:
            raise SystemExit(f'known words mapped wrongly: {bad} — '
                             f'lexicon misaligned; no codes written')
        if len(known) - len(absent) < MIN_KNOWN_PRESENT:
            raise SystemExit(f'only {len(known) - len(absent)} known words present — '
                             f'too few to vouch for the mapping; no codes written')

    # ── 2. codes for every split ─────────────────────────────────────────────
    for split in args.splits:
        # the train split loaded for the lexicon is the target's own only when no
        # other fraction supplied the lexicon
        got = tr if (split == 'train' and lex_spec is None) else load_split(split)
        if got is None:
            print(f'  {split}: no cached ids — skipped')
            continue
        s_stem, s_ids, s_src = got
        if s_src is None:
            print(f'  {split}: no codes file holding {args.fact!r} — skipped')
            continue
        s_npy, s_meta, _ = s_src
        src_codes = np.load(s_npy, mmap_mode='r')
        n = int(s_ids.shape[0])
        new_row = np.empty(n, dtype=np.int16)
        oov = words = n_rec = n_changed = 0
        s_cuts = cut_points(s_ids, tables, CHUNK)
        for a, b in zip(s_cuts[:-1], s_cuts[1:]):
            chunk = np.asarray(s_ids[a:b], dtype=np.int64)
            wid, key, starts = word_spans(chunk, tables)
            if args.transform == 'const':
                # ONE CLASS FOR EVERY WORD, seen in train or not (G-73): a lexicon
                # lookup gave unseen validation words `none`, the mark the null met
                # only on non-words in training. Train codes are unchanged — every
                # train word is in the lexicon
                c = np.where(wid >= 0, 0, none_code).astype(np.int64)
            elif recode is not None:
                # the value in context, except at the recoded word types, which get
                # their form's majority value; a word unseen in train is never
                # recoded (it has no majority tag) and keeps its value in context
                ctx = np.asarray(src_codes[s_meta['fact_names'].index(args.fact)][a:b], dtype=np.int64)
                is_rec = (wid >= 0) & np.isin(lookup(key, recode['lex'], recode['none']), recode['ids'])
                c = np.where(is_rec, lookup(key, lex, none_code), ctx)
                n_rec += int(is_rec.sum())
                n_changed += int((c != ctx).sum())
                if bool((c[~is_rec] != ctx[~is_rec]).any()):
                    raise SystemExit(f'{split}: a word not recoded lost its value in context')
            else:
                c = lookup(key, lex, none_code)
                c[wid < 0] = none_code
            if args.transform == 'flat':
                c[:] = 0                  # non-word positions too: truly constant
            new_row[a:b] = c
            ws = key[starts]
            words += len(ws)
            li = np.searchsorted(lex[0], ws)
            li = np.minimum(li, len(lex[0]) - 1)
            oov += int((lex[0][li] != ws).sum())
        if recode is not None and args.verify and not (0 < n_changed < 0.1 * n):
            raise SystemExit(f'{split}: {n_changed:,} of {n:,} tokens changed — not what a recode of '
                             f'{recode["tags"]} does; no codes written')
        names = list(s_meta['fact_names'])
        if out_fact in names:
            names.remove(out_fact)
        if args.keep != 'all':
            keep = [f.strip() for f in args.keep.split(',') if f.strip()]
            missing = [f for f in keep if f not in names]
            if missing:
                raise SystemExit(f'--keep asks for {missing}, not in the source file')
            names = keep
        out = np.lib.format.open_memmap(
            os.path.join(args.cache_dir, s_stem + f'_{out_fact}.codes.tmp.npy'),
            mode='w+', dtype=np.int16, shape=(len(names) + 1, n))
        for r, nm in enumerate(names):
            out[r] = src_codes[s_meta['fact_names'].index(nm)]
        out[-1] = new_row
        out.flush()
        del out
        final = os.path.join(args.cache_dir, s_stem + f'_{out_fact}.codes.npy')
        # what replacing an existing file changes, row by row: a re-derivation
        # shows here that it touched only what it meant to (G-73: the null's
        # train codes must not change)
        changed = None
        if os.path.exists(final) and os.path.exists(final[:-4] + '.json'):
            try:
                om = json.load(open(final[:-4] + '.json'))
                if om.get('num_tokens') == n and out_fact in om.get('fact_names', []):
                    old_rows = np.load(final, mmap_mode='r')
                    changed = int((np.asarray(old_rows[om['fact_names'].index(out_fact)]) != new_row).sum())
            except (OSError, ValueError):
                changed = None
        os.replace(os.path.join(args.cache_dir, s_stem + f'_{out_fact}.codes.tmp.npy'), final)
        # each row's rule, carried from the source for the rows it copied: what a
        # run of a constant fact checks before it reads the codes (stream.py)
        rules = {f: r for f, r in (s_meta.get('rules') or {}).items() if f in names}
        if args.transform in ('const', 'flat'):
            rules[out_fact] = 'every word' if args.transform == 'const' else 'every position'
        _atomic_write_json(final[:-4] + '.json',
                           {'fact_names': names + [out_fact],
                            'dep_granularity': s_meta.get('dep_granularity', 'coarse'),
                            'num_tokens': n,
                            **({'rules': rules} if rules else {}),
                            **({'recoded': {out_fact: {'from': args.fact, 'pos': recode['tags'],
                                                       'by': 'majority tag of the word type in train'}}}
                               if recode is not None else {})})
        if recode is not None:
            print(f'  {split}: {n_rec:,} tokens in recoded words, {n_changed:,} changed from the value in context '
                  f'({100 * n_changed / max(n, 1):.2f}% of tokens)')
            rec.add(f'recoded:{split}', {'tokens_recoded': n_rec, 'tokens_changed': n_changed})
        rate = oov / max(words, 1)
        print(f'  {split}: {n:,} tokens -> {os.path.basename(final)} '
              f'({len(names) + 1} facts); words unseen in train {100 * rate:.2f}%'
              + (f'; replaced a file: {changed:,} tokens changed ({100 * changed / max(n, 1):.2f}%)'
                 if changed is not None else ''))
        rec.add(f'split:{split}', {'tokens': n, 'words': words, 'unseen_rate': rate,
                                   'replaced_changed': changed})

    rec.note('Unseen words get the none class, never their contextual code, '
             'which would readmit the right-context information — except under '
             '--transform const, where every word gets the one class (G-73).')
    rec.write()


if __name__ == '__main__':
    main()
