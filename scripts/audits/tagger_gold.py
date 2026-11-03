#!/usr/bin/env python3
"""
tagger_gold.py — the tagger's case against gold labels, where context decides (the text's list of
30 October, item 8; journey §33.153).

WHY
---
The contextual case feature (`ucase`) is uk_core_news_sm's reading of each word in its sentence. Its
published case F1, 0.963, is on Ukr-Synth, whose labels are themselves automatic, and over all words. The
strictly causal feature (`ucaselex`) gives each word form its majority case in the training corpus; the
two differ on about a tenth of word occurrences. Whether the tagger reads context where context decides
bounds what the thesis can say: an accurate tagger means the context is read and the model does not need
it; a poor one bounds the answer by the tagger.

HOW
---
UD Ukrainian-IU's gold test and dev sentences (`external/ud_ukrainian_iu/`, CC BY-NC-SA 4.0), each tagged by
uk_core_news_sm over the gold tokens, so no alignment is needed. For every word with a gold case:
  * the tagger's case in context;
  * the form's majority case: the training corpus's lexicon for `ucaselex` (BPE 17k at 0.2), looked up by
    the word's key as the streams compute it (`word_spans`), the form preceded by a space as in running
    text, and without one if that is not found;
  * the dictionary's readings (pymorphy3): the form is case-ambiguous when its readings carry two cases
    or more.
Reported: each reading's accuracy over all words; by part of speech; on the dictionary's ambiguous forms;
on the words whose gold case differs from their form's majority case (context decides); and where the
tagger and the majority case disagree, which is right. Plus the tagger's case on words gold gives none.

    python scripts/audits/tagger_gold.py
"""
import os
import sys
import glob
import argparse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

CASES = ('Nom', 'Gen', 'Dat', 'Acc', 'Ins', 'Loc', 'Voc')          # the ucase block's classes, in order
PYMORPHY = {'nomn': 'Nom', 'gent': 'Gen', 'gen2': 'Gen', 'datv': 'Dat', 'accs': 'Acc', 'ablt': 'Ins',
            'loct': 'Loc', 'loc2': 'Loc', 'voct': 'Voc'}
GROUPS = {'NOUN': 'nouns and pronouns', 'PROPN': 'nouns and pronouns', 'PRON': 'nouns and pronouns',
          'ADJ': 'adjectives, determiners, numerals', 'DET': 'adjectives, determiners, numerals',
          'NUM': 'adjectives, determiners, numerals', 'ADP': 'prepositions'}


def sentences(path):
    """[(forms, spaces, upos, cases)] of a CoNLL-U file; multiword ranges and empty nodes skipped."""
    out, cur = [], []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line:
            if cur:
                out.append(cur)
            cur = []
            continue
        if line.startswith('#'):
            continue
        cols = line.split('\t')
        if '-' in cols[0] or '.' in cols[0]:
            continue
        feats = dict(f.split('=', 1) for f in cols[5].split('|') if '=' in f)
        cur.append((cols[1], 'SpaceAfter=No' not in cols[9], cols[3], feats.get('Case')))
    if cur:
        out.append(cur)
    return out


class Lexicon:
    """The form's majority case in the training corpus, as `ucaselex` reads it."""

    def __init__(self, bpe_path, lexicon):
        import numpy as np
        from morphlm.data.tokenization import load_tokenizer
        from morphlm.features.lexical import tokenizer_tables
        self.np = np
        self.tok = load_tokenizer(bpe_path)
        self.tables = tokenizer_tables(bpe_path)
        z = np.load(lexicon)
        self.keys, self.codes = z['keys'], z['codes']
        self.cache = {}

    def case(self, form):
        if form in self.cache:
            return self.cache[form]
        from morphlm.features.lexical import word_spans
        got = None
        for text in (' ' + form, form):
            ids = self.tok.encode(text).ids
            _, key, starts = word_spans(ids, self.tables)
            if starts.sum() != 1:
                continue
            k = key[self.np.flatnonzero(starts)[0]]
            i = int(self.np.searchsorted(self.keys, k))
            if i < len(self.keys) and self.keys[i] == k:
                c = int(self.codes[i])
                got = CASES[c] if c < len(CASES) else 'none'
                break
        self.cache[form] = got
        return got


def share(n, d):
    return {'n': d, 'right': n, 'share': (n / d) if d else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--treebank', default=os.path.join(ROOT, 'external', 'ud_ukrainian_iu'))
    ap.add_argument('--bpe-path', default=os.path.join(ROOT, 'data', 'bpe_uk_17k.json'))
    ap.add_argument('--lexicon', default=os.path.join(ROOT, 'data', 'cache', 'lexicon_ucase_bpe_uk_17k.npz'))
    a = ap.parse_args()
    import spacy
    import pymorphy3
    from spacy.tokens import Doc
    from morphlm.record import AuditRecord
    rec = AuditRecord('tagger_gold', a, out_dir=os.path.join(ROOT, 'results', 'audits'))
    nlp = spacy.load('uk_core_news_sm')
    morph = pymorphy3.MorphAnalyzer(lang='uk')
    lex = Lexicon(a.bpe_path, a.lexicon)
    rec.add('tagger', {'model': nlp.meta.get('name'), 'version': nlp.meta.get('version'),
                       'sources': [s.get('name') for s in nlp.meta.get('sources', [])]})
    for split in ('test', 'dev'):
        path = glob.glob(os.path.join(a.treebank, '*', f'uk_iu-ud-{split}.conllu'))
        if len(path) != 1:
            raise SystemExit(f'expected one {split} file under {a.treebank}, found {path}')
        sents = sentences(path[0])
        docs = nlp.pipe(Doc(nlp.vocab, words=[t[0] for t in s], spaces=[t[1] for t in s]) for s in sents)
        c = defaultdict(Counter)
        for s, doc in zip(sents, docs):
            for (form, _, upos, gold), tok in zip(s, doc):
                pred = (tok.morph.get('Case') or [None])[0]
                if gold is None:
                    c['no gold case']['all'] += 1
                    c['no gold case']['tagger gives one'] += pred is not None
                    continue
                maj = lex.case(form)
                cases = {PYMORPHY[p.tag.case] for p in morph.parse(form.lower()) if p.tag.case in PYMORPHY}
                keys = ['all', GROUPS.get(upos, 'other'),
                        'dictionary: ambiguous' if len(cases) > 1 else
                        'dictionary: one case' if cases else 'dictionary: unknown']
                if maj is not None and maj != gold:
                    keys.append('context decides (gold ≠ majority)')
                if maj is not None and pred is not None and maj != pred:
                    keys.append('tagger ≠ majority')
                for k in keys:
                    c[k]['n'] += 1
                    c[k]['tagger'] += pred == gold
                    c[k]['majority'] += maj == gold
                    c[k]['majority known'] += maj is not None
        out = {k: {'n': v['n'], 'tagger': share(v['tagger'], v['n']), 'majority': share(v['majority'], v['n']),
                   'majority_known': v['majority known']} for k, v in c.items() if k != 'no gold case'}
        out['no gold case'] = {'n': c['no gold case']['all'], 'tagger_gives_one': c['no gold case']['tagger gives one']}
        rec.add(split, out)
        print(f'\n  UD Ukrainian-IU {split}: {len(sents):,} sentences')
        print(f'  {"words with a gold case":42} {"n":>7} {"tagger":>8} {"majority":>9} {"known":>7}')
        for k, v in out.items():
            if k == 'no gold case':
                continue
            print(f'  {k:42} {v["n"]:7,} {100 * v["tagger"]["share"]:7.1f}% {100 * v["majority"]["share"]:8.1f}% '
                  f'{100 * v["majority_known"] / v["n"]:6.1f}%')
        ng = out['no gold case']
        print(f'  words gold gives no case: {ng["n"]:,}; the tagger gives one to {ng["tagger_gives_one"]:,} '
              f'({100 * ng["tagger_gives_one"] / max(ng["n"], 1):.1f}%)')
    rec.write()


if __name__ == '__main__':
    main()
