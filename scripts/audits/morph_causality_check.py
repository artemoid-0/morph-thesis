#!/usr/bin/env python3
"""
morph_causality_check.py — how much do tagger-derived features depend on the
words to their RIGHT?

`mstruct` is looked up from the word's letters in MorphoLEX: no context at all,
so shifted to the previous word it is strictly causal. `pos`, `minfl`, `magr`
and `mfeat` come from spaCy's TAGGER, whose convolutional window reaches about
four tokens to the right. "reading" is VBG in "reading books" and NN in
"reading is fun" — the tag of a word can depend on the word being predicted
next, and shifting it back one word does not remove that.

For each word, this compares its features tagged in the FULL sentence against
its features tagged from the prefix ENDING at that word — left context only,
exactly what a causal tagger would see. Whatever differs is information from
the right. Unlike the dependency parser (dep_causality_check.py, whose prefix
test failed because a parser must build a tree over any fragment), a tagger
has no global constraint, so the prefix IS the causal setting and this
measures the quantity directly.

Harness check built in: a sentence tagged as its own prefix must match
exactly, or the run stops.

    python scripts/audits/morph_causality_check.py --sentences 3000
    python scripts/audits/morph_causality_check.py --text some_file.txt      # local text
    python scripts/audits/morph_causality_check.py --split train --sentences 10000 --by-type

--by-type also counts, for part of speech, where its right-context dependence sits: per word
type, and per class of the type's majority part of speech in the sample — which words a control
must read from their form to remove it (journey §33.98; the list of 15 October, item 2).
--by-type-attrs counts other features the same way, e.g. Ukrainian gender and number. Where the
corpus itself is not on the machine, --stream reads the text from a cached token stream: every
stream decodes to its corpus text byte for byte (G-75), and it is the text a cell trains on:

    python scripts/audits/morph_causality_check.py --spacy-model uk_core_news_sm \
        --stream data/cache/bpe_train_None_c360a47f1f.npy --tokenizer data/bpe_uk_17k.json \
        --sentences 10000 --full-pipeline --by-type --by-type-attrs pos Gender Number
"""

import os
import sys
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ATTRS = ('tag', 'pos', 'mfeat', 'magr', 'minfl')


UD_ATTRS = ('pos', 'mfeat', 'Case', 'Gender', 'Number', 'Person', 'Tense',
            'Aspect', 'Animacy')


def features(tok, ud=False):
    if ud:
        # language-neutral: each Universal Dependencies feature by name, the
        # values the Ukrainian blocks (ucase, ugender, ...) are built from
        out = {'pos': tok.pos_, 'mfeat': str(tok.morph) or None}
        for a in UD_ATTRS[2:]:
            out[a] = ','.join(tok.morph.get(a)) or None
        return out
    from morphlm.features.inflection import inflection_of, agreement_of
    return {'tag': tok.tag_, 'pos': tok.pos_,
            'mfeat': str(tok.morph) or None,
            'magr': agreement_of(tok), 'minfl': inflection_of(tok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sentences', type=int, default=3000)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--max-words', type=int, default=40)
    ap.add_argument('--text', help='read sentences from a local file instead')
    ap.add_argument('--stream', help='read them from a cached token stream instead (an unpacked '
                    'data/cache/<stem>.npy), decoded with --tokenizer')
    ap.add_argument('--tokenizer', help='the tokenizer file that encoded --stream')
    ap.add_argument('--stream-tokens', type=int, default=2_500_000,
                    help="how many of the stream's first tokens to decode (whole paragraphs only)")
    ap.add_argument('--spacy-model', default='en_core_web_sm')
    ap.add_argument('--corpus', default='wikitext103', help='wikitext103 | ukwiki')
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--full-pipeline', action='store_true',
                    help='keep the parser: the pipeline sets part of speech with rules that read the parse')
    ap.add_argument('--by-type', action='store_true',
                    help='record part of speech changes per word type and per majority class')
    ap.add_argument('--by-type-attrs', nargs='+', default=['pos'], metavar='FEATURE',
                    help='with --by-type, the features counted per word type and class: pos (the default) '
                         'and, with --ud, Universal Dependencies names such as Gender and Number')
    ap.add_argument('--ud', action='store_true',
                    help='compare Universal Dependencies features by name '
                         '(default for any non-English model)')
    args = ap.parse_args()

    import spacy
    from morphlm.record import AuditRecord
    rec = AuditRecord('morph_causality_check', args)
    # the tagger and the rule table are all these features need; the parser
    # and NER only cost time and cannot change a tag
    ud = args.ud or not args.spacy_model.startswith('en_')
    global ATTRS
    if ud:
        ATTRS = UD_ATTRS
    BY = list(args.by_type_attrs) if args.by_type else []
    unknown = [a for a in BY if a not in ATTRS]
    if unknown:
        ap.error(f'--by-type-attrs: {unknown} are not among the features compared: {list(ATTRS)}')
    # without --full-pipeline only the tagger: the parser and NER cost time, and the tagger's
    # tags never depend on them — but the PIPELINE's part of speech does: rules after the parser
    # set "which" as DET or PRON, "have" as AUX or VERB, from the parse (§33.98)
    nlp = spacy.load(args.spacy_model, disable=(['ner', 'lemmatizer'] if args.full_pipeline
                                                else ['parser', 'ner', 'lemmatizer']))
    if not args.full_pipeline:
        nlp.add_pipe('sentencizer')

    if args.text:
        lines = [l.strip() for l in open(args.text) if l.strip()]
    elif args.stream:
        import numpy as np
        from tokenizers import Tokenizer
        if not args.tokenizer:
            ap.error('--stream needs --tokenizer')
        ids = np.load(args.stream, mmap_mode='r')[:args.stream_tokens]
        text = Tokenizer.from_file(args.tokenizer).decode([int(i) for i in ids])
        lines = [l.strip() for l in text[:text.rfind('\n') + 1].split('\n') if l.strip()]
    else:
        from morphlm.config import ExperimentConfig, apply_overrides
        from morphlm.data.stream import load_corpus
        ds = load_corpus(apply_overrides(ExperimentConfig(), [
            f'data.corpus={args.corpus}', f'data.corpus_holdout={args.corpus_holdout}']), args.split)
        lines = (r['text'].strip() for r in ds)

    sents = []
    for t in lines:
        if not t or t.startswith('='):
            continue
        for s in nlp(t).sents:
            words = [w for w in s if not w.is_space]
            if 3 <= len(words) <= args.max_words:
                sents.append(s.text)
            if len(sents) >= args.sentences:
                break
        if len(sents) >= args.sentences:
            break
    print(f'  {len(sents)} sentences')

    agree = Counter()
    total = 0
    by_dist = defaultdict(Counter)
    n_dist = Counter()
    changes = Counter()
    pos_seen = defaultdict(Counter)       # each word type's part of speech in full context: its class
    n_by = {a: Counter() for a in BY}
    changed_by = {a: Counter() for a in BY}
    pairs_by = {a: defaultdict(Counter) for a in BY}
    skipped = 0
    harness_ok = harness_n = 0
    for s in sents:
        full = nlp(s)
        toks = [t for t in full if not t.is_space]
        # HARNESS CHECK: the prefix ending at the final token IS the sentence,
        # so its features must match exactly. If they do not, the comparison
        # itself is broken (tokenisation, alignment) and nothing below means
        # anything.
        last = toks[-1]
        whole = [x for x in nlp(full[:last.i + 1].text) if not x.is_space]
        harness_n += 1
        harness_ok += (len(whole) == len(toks) and
                       features(whole[-1], ud) == features(last, ud))
        for i, t in enumerate(toks):
            if not t.is_alpha:
                continue
            prefix = nlp(full[:t.i + 1].text)
            pt = [x for x in prefix if not x.is_space]
            if len(pt) != i + 1 or pt[-1].text != t.text:
                skipped += 1                 # the prefix re-tokenised; do not guess
                continue
            f_full, f_left = features(t, ud), features(pt[-1], ud)
            total += 1
            d = min(len(toks) - 1 - i, 5)
            n_dist[d] += 1
            for a in ATTRS:
                same = f_full[a] == f_left[a]
                agree[a] += same
                by_dist[d][a] += same
            if f_full['mfeat'] != f_left['mfeat']:
                changes[(t.lower_, f_left['mfeat'] or '-', f_full['mfeat'] or '-')] += 1
            if args.by_type:
                w = t.lower_
                pos_seen[w][f_full['pos']] += 1
                for a in BY:
                    n_by[a][w] += 1
                    if f_full[a] != f_left[a]:
                        changed_by[a][w] += 1
                        pairs_by[a][w][f'{f_left[a]}->{f_full[a]}'] += 1

    print()
    print('=' * 84)
    print('TAGGER CAUSALITY — each word tagged in the full sentence vs from its prefix')
    print('=' * 84)
    print(f'  {total:,} alphabetic words ({skipped} skipped: the prefix re-tokenised)')
    print(f'  {"feature":8} {"agree":>8}   = share of words whose value uses NO right context')
    for a in ATTRS:
        print(f'  {a:8} {100 * agree[a] / max(total, 1):7.2f}%')
    print(f'  harness check (a sentence tagged as its own prefix): '
          f'{harness_ok}/{harness_n} identical')
    if harness_ok != harness_n:
        raise SystemExit('harness check failed: prefix and full tagging differ '
                         'even when the prefix is the whole sentence')
    print()
    print('  by number of tokens AFTER the word')
    print(f'  {"after":>6} {"n":>7} ' + ' '.join(f'{a:>7}' for a in ATTRS))
    for d in sorted(n_dist):
        lab = f'{d}' if d < 5 else '5+'
        print(f'  {lab:>6} {n_dist[d]:7,} ' + ' '.join(
            f'{100 * by_dist[d][a] / n_dist[d]:6.1f}%' for a in ATTRS))
    print()
    print('  most frequent mfeat changes (left-context -> full sentence):')
    for (w, a, b), n in changes.most_common(8):
        print(f'    {w:12} {a:40} -> {b:40} {n:4}')
    print()
    print('  READING THIS')
    print('    The disagreement rate is an UPPER BOUND on the share of shifted')
    print('    feature values that could carry information about the next word.')
    print('    It is not the VALUE of that information: most changes are')
    print('    disambiguations the model can make itself from the word it sees.')

    rec.add('summary', {'words': total, 'skipped': skipped,
                        'harness': [harness_ok, harness_n],
                        'agreement': {a: agree[a] / max(total, 1) for a in ATTRS}})
    rec.add('by_words_after', {str(d): {a: by_dist[d][a] / n_dist[d] for a in ATTRS}
                               for d in sorted(n_dist)})
    rec.add('mfeat_changes', {f'{w}: {a} -> {b}': n for (w, a, b), n in changes.most_common(40)})
    if args.by_type:
        major = {w: c.most_common(1)[0][0] for w, c in pos_seen.items()}
        for a in BY:
            n_w, ch_w, pairs = n_by[a], changed_by[a], pairs_by[a]
            total_changed = sum(ch_w.values())
            by_class_n, by_class_changed = Counter(), Counter()
            for w in n_w:
                by_class_n[major[w]] += n_w[w]
                by_class_changed[major[w]] += ch_w[w]
            name = 'PART OF SPEECH' if a == 'pos' else a.upper()
            print()
            print(f'  {name}, by where it changes: {total_changed:,} changed of {total:,} words '
                  f'({100 * total_changed / max(total, 1):.2f}%)')
            print(f'  {"majority class":14} {"words":>9} {"changed":>8} {"rate":>7} {"share of all changes":>21}')
            for c, n in by_class_changed.most_common():
                print(f'  {c:14} {by_class_n[c]:9,} {n:8,} {100 * n / by_class_n[c]:6.1f}% '
                      f'{100 * n / max(total_changed, 1):20.1f}%')
            top = sorted(ch_w, key=lambda w: -ch_w[w])
            cum = 0
            print(f'  {"word type":14} {"seen":>7} {"changed":>8} {"rate":>7} {"cumulative":>11}   most common changes')
            for w in top[:40]:
                cum += ch_w[w]
                print(f'  {w:14} {n_w[w]:7,} {ch_w[w]:8,} {100 * ch_w[w] / n_w[w]:6.1f}% '
                      f'{100 * cum / max(total_changed, 1):10.1f}%   ' + ', '.join(f'{k} {v}' for k, v in pairs[w].most_common(3)))
            # the part-of-speech keys keep their names of 15 October
            key = 'pos' if a == 'pos' else a.lower()
            rec.add(f'{key}_by_class', {c: {'words': by_class_n[c], 'changed': by_class_changed[c]} for c in by_class_n})
            rec.add(f'{key}_by_type', {w: {'seen': n_w[w], 'changed': ch_w[w], 'majority': major[w],
                                          'changes': dict(pairs[w].most_common(5))}
                                      for w in top})
            rec.add(f'{key}_changed_total', total_changed)
    rec.write()


if __name__ == '__main__':
    main()
