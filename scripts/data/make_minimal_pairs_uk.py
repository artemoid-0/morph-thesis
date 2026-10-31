#!/usr/bin/env python3
"""
make_minimal_pairs_uk.py — Ukrainian minimal pairs beyond subject–verb agreement (journey §33.147).

WHY
---
MultiBLiMP's Ukrainian pairs test one phenomenon, a verb agreeing with its subject. The tokenizer's
grammatical gain is claimed from them alone. These pairs test two more, the ones an inflectional
ending carries inside a noun phrase and after a preposition:
  AN-C, AN-#, AN-G   an adjective agreeing with the noun right after it, in case, number or gender
                     (у великому місті → у великим місті);
  PP-C               a noun right after a preposition, in a case the preposition governs
                     (біля будинку → біля будинком).

HOW
---
From the corpus's TEST split — held out from training and never used to choose a checkpoint — each
paragraph is parsed by the corpus's tagger (spaCy), and each sentence of 6 to 40 tokens is searched
for an adjective whose head is the noun directly after it, both in the same case and number (and
gender, in the singular), and for one of the prepositions in GOVERNS directly before a noun in a
case it governs. pymorphy3 then inflects ONE word to a form that changes one feature, and the pair is kept
only if the changed form is unambiguously wrong there, by the dictionary, with no context:
  agreement   no reading of the new adjective form agrees with any reading of the noun's form;
  government  no reading of the new noun form is in a case the preposition governs.
Both words must be known to the dictionary and wholly Ukrainian, with no apostrophe; capitals are kept.

Each pair is one row of a MultiBLiMP-style table (`sen`, `wrong_sen`, `phenomenon`, then what was
changed), so `scripts/eval/multiblimp_eval.py --data` scores it as it scores MultiBLiMP. A sample for
a native speaker's check is drawn by `--sample`, stratified by phenomenon, with a fixed seed.

    python scripts/data/make_minimal_pairs_uk.py --out data/minimal_pairs/uk_np_pp_v1.tsv
    python scripts/data/make_minimal_pairs_uk.py --out data/minimal_pairs/uk_np_pp_v1.tsv --sample 100
"""
import os
import re
import sys
import json
import random
import argparse
import datetime
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

#: spaCy's UD values → pymorphy3's grammemes
CASE = {'Nom': 'nomn', 'Gen': 'gent', 'Dat': 'datv', 'Acc': 'accs', 'Ins': 'ablt', 'Loc': 'loct'}
NUMBER = {'Sing': 'sing', 'Plur': 'plur'}
GENDER = {'Masc': 'masc', 'Fem': 'femn', 'Neut': 'neut'}
CASES = tuple(CASE.values())

#: prepositions and every case each governs in standard Ukrainian, kept to those whose government is
#: not in doubt; a preposition that also takes a case in idioms (в/у with the genitive, по, о) is left out
GOVERNS = {
    'до': {'gent'}, 'від': {'gent'}, 'для': {'gent'}, 'без': {'gent'}, 'після': {'gent'}, 'біля': {'gent'},
    'серед': {'gent'}, 'крім': {'gent'}, 'окрім': {'gent'}, 'навколо': {'gent'}, 'протягом': {'gent'},
    'замість': {'gent'}, 'проти': {'gent'}, 'через': {'accs'}, 'про': {'accs'}, 'крізь': {'accs'},
    'над': {'ablt'}, 'перед': {'ablt', 'accs'}, 'між': {'ablt', 'gent'}, 'при': {'loct'},
    'завдяки': {'datv'}, 'всупереч': {'datv'}, 'на': {'accs', 'loct'}, 'за': {'accs', 'ablt', 'gent'},
    'під': {'accs', 'ablt'}, 'з': {'gent', 'ablt', 'accs'}, 'із': {'gent', 'ablt', 'accs'},
    'зі': {'gent', 'ablt', 'accs'},
}
PHENOMENA = {'AN-C': 'adjective–noun agreement in case', 'AN-#': 'adjective–noun agreement in number',
             'AN-G': 'adjective–noun agreement in gender', 'PP-C': 'preposition–noun government'}
WORD = re.compile(r'[а-щьюяєіїґА-ЩЬЮЯЄІЇҐ]+')


def morph_of(tok, key):
    v = tok.morph.get(key)
    return v[0] if v else None


def num(tag):
    """A reading's number: pymorphy3 names the plural only; a singular reading carries its gender."""
    return 'plur' if tag.number == 'plur' else 'sing'


def cap_like(new, old):
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


class Inflector:
    """pymorphy3's readings and forms, with the checks that make a change unambiguous."""

    def __init__(self):
        import pymorphy3
        self.m = pymorphy3.MorphAnalyzer(lang='uk')
        self._parse = {}

    def parse(self, w):
        k = w.lower()
        if k not in self._parse:
            self._parse[k] = self.m.parse(k)
        return self._parse[k]

    def known(self, w):
        return self.m.word_is_known(w.lower())

    def readings(self, w, pos):
        return [p for p in self.parse(w) if p.tag.POS == pos]

    @staticmethod
    def agrees(a, n):
        """An adjective reading agrees with a noun reading: case, number, and gender in the singular."""
        if a.tag.case != n.tag.case or num(a.tag) != num(n.tag):
            return False
        return num(a.tag) == 'plur' or a.tag.gender == n.tag.gender

    def adjective_change(self, adj, noun, kind, rng):
        """A form of the adjective that changes `kind` and agrees with no reading of the noun, or None."""
        case, number = CASE.get(morph_of(noun, 'Case')), NUMBER.get(morph_of(noun, 'Number'))
        gender = GENDER.get(morph_of(noun, 'Gender'))
        if not case or not number or (number == 'sing' and not gender):
            return None
        mine = [p for p in self.readings(adj.text, 'ADJF')
                if p.tag.case == case and num(p.tag) == number and (number == 'plur' or p.tag.gender == gender)]
        nouns = self.readings(noun.text, 'NOUN')
        if not mine or not nouns:
            return None
        if kind == 'AN-C':
            targets = [({c}, f'{case}→{c}') for c in CASES if c != case]
        elif kind == 'AN-#':
            other = 'plur' if number == 'sing' else 'sing'
            if other == 'sing':
                targets = [({g}, f'plur→sing,{g}') for g in ('masc', 'femn', 'neut')]
            else:
                targets = [({other}, 'sing→plur')]
        else:
            if number != 'sing':
                return None
            targets = [({g}, f'{gender}→{g}') for g in ('masc', 'femn', 'neut') if g != gender]
        rng.shuffle(targets)
        for want, change in targets:
            f = mine[0].inflect(want)
            if f is None or f.word == adj.text.lower() or f.tag.case != (case if kind != 'AN-C' else next(iter(want))):
                continue
            if any(self.agrees(r, n) for r in self.readings(f.word, 'ADJF') for n in nouns):
                continue                    # some reading still agrees: not unambiguous
            if not self.readings(f.word, 'ADJF'):
                continue
            return f.word, change
        return None

    def noun_change(self, prep, noun, rng):
        """A form of the noun in a case the preposition never governs, by every reading, or None."""
        gov = GOVERNS.get(prep.text.lower())
        case, number = CASE.get(morph_of(noun, 'Case')), NUMBER.get(morph_of(noun, 'Number'))
        if not gov or case not in gov or not number:
            return None
        mine = [p for p in self.readings(noun.text, 'NOUN') if p.tag.case == case and num(p.tag) == number]
        if not mine:
            return None
        targets = [c for c in CASES if c not in gov]
        rng.shuffle(targets)
        for c in targets:
            f = mine[0].inflect({c})
            if f is None or f.word == noun.text.lower() or f.tag.case != c or num(f.tag) != number:
                continue
            news = self.readings(f.word, 'NOUN')
            if not news or any(r.tag.case in gov for r in news):
                continue
            return f.word, f'{case}→{c}'
        return None


def write_table(path, rows, cols):
    """Tab-separated, unquoted, as MultiBLiMP's table is read (csv.QUOTE_NONE): no field holds a tab."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')


def pairs_from(doc, inf, rng, doc_id):
    out = []
    for sent in doc.sents:
        toks = list(sent)
        if not 6 <= len(toks) <= 40 or '\n' in sent.text:
            continue
        base = sent.start_char
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            if not (WORD.fullmatch(a.text) and WORD.fullmatch(b.text)):
                continue
            if not (inf.known(a.text) and inf.known(b.text)):
                continue
            found = []
            if a.pos_ == 'ADJ' and b.pos_ == 'NOUN' and a.dep_ == 'amod' and a.head.i == b.i:
                for kind in ('AN-C', 'AN-#', 'AN-G'):
                    ch = inf.adjective_change(a, b, kind, rng)
                    if ch:
                        found.append((kind, a, ch, b.text))
            elif a.pos_ == 'ADP' and b.pos_ == 'NOUN' and b.head.i != a.i and a.head.i == b.i:
                ch = inf.noun_change(a, b, rng)
                if ch:
                    found.append(('PP-C', b, ch, a.text))
            for kind, tok, (form, change), other in found:
                s, e = tok.idx - base, tok.idx - base + len(tok.text)
                wrong = sent.text[:s] + cap_like(form, tok.text) + sent.text[e:]
                out.append({'sen': sent.text, 'wrong_sen': wrong, 'phenomenon': kind, 'doc': doc_id,
                            'word': tok.text, 'wrong_word': cap_like(form, tok.text), 'change': change,
                            'other': other})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='the pairs table (.tsv); a sidecar .json describes it')
    ap.add_argument('--split', default='test')
    ap.add_argument('--per-phenomenon', type=int, default=1000, help='pairs kept per phenomenon, one per sentence')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample', type=int, default=0, help='also write a stratified sample of this many for a check')
    a = ap.parse_args()

    import spacy
    import pymorphy3
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import load_corpus
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.corpus=ukwiki', 'data.corpus_fraction=1.0',
                                               'data.corpus_holdout=2000', 'data.spacy_model=uk_core_news_sm'])
    nlp = spacy.load('uk_core_news_sm', disable=['ner'])
    inf, rng = Inflector(), random.Random(a.seed)
    rows, paragraphs = [], 0
    # paragraphs, as the streams see them (each starts after a newline)
    texts = [(f'{i}.{j}', t.strip()) for i, r in enumerate(load_corpus(cfg, a.split))
             for j, t in enumerate(r['text'].split('\n')) if t.strip()]
    for (doc_id, text), doc in zip(texts, nlp.pipe((t for _, t in texts), batch_size=32)):
        paragraphs += 1
        rows += pairs_from(doc, inf, rng, doc_id)
    found = Counter(r['phenomenon'] for r in rows)
    # one pair per sentence and phenomenon, then a fixed-seed draw of each phenomenon
    by = defaultdict(dict)
    for r in rows:
        by[r['phenomenon']].setdefault(r['sen'], r)
    kept = []
    for k in PHENOMENA:
        pool = list(by[k].values())
        rng.shuffle(pool)
        kept += pool[:a.per_phenomenon]
    for n, r in enumerate(kept):
        r['item'] = n
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    cols = ['item', 'sen', 'wrong_sen', 'phenomenon', 'word', 'wrong_word', 'change', 'other', 'doc']
    kept = [r for r in kept if not any(ch in str(r[c]) for c in cols for ch in '\t\n')]
    write_table(a.out, kept, cols)
    meta = {'script': 'scripts/data/make_minimal_pairs_uk.py', 'args': vars(a), 'split': a.split,
            'corpus': 'ukwiki', 'holdout': 2000, 'paragraphs': paragraphs, 'phenomena': PHENOMENA,
            'found': dict(found), 'kept': dict(Counter(r['phenomenon'] for r in kept)),
            'governs': {k: sorted(v) for k, v in GOVERNS.items()},
            'spacy': spacy.__version__, 'spacy_model': nlp.meta.get('name') + ' ' + nlp.meta.get('version', ''),
            'pymorphy3': pymorphy3.__version__,
            'made_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    if a.sample:
        srng = random.Random(a.seed + 1)
        per = a.sample // len(PHENOMENA)
        sample = []
        for k in PHENOMENA:
            pool = [r for r in kept if r['phenomenon'] == k]
            sample += srng.sample(pool, min(per, len(pool)))
        srng.shuffle(sample)
        spath = a.out[:-4] + f'_sample{a.sample}.tsv'
        write_table(spath, sample, cols)
        meta['sample'] = {'path': os.path.relpath(spath, ROOT), 'items': [r['item'] for r in sample], 'seed': a.seed + 1}
    json.dump(meta, open(a.out[:-4] + '.json', 'w'), indent=1, ensure_ascii=False)
    print(f'{paragraphs:,} paragraphs of the {a.split} split; found {dict(found)}; kept {meta["kept"]} -> {a.out}')
    if a.sample:
        print(f'  a sample of {len(meta["sample"]["items"])} for the check -> {meta["sample"]["path"]}')


if __name__ == '__main__':
    main()
