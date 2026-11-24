#!/usr/bin/env python3
"""
tokenizer_known_words.py — the tokenizer's gain on words the morphological analyzer does not know (the
review's remark on text rich in new words; journey §33.161).

WHY
---
The endings rule cuts a word by pymorphy3's readings. For a word outside its dictionary those are the
analyzer's guesses (`FakeDictionary`), so the rule still cuts new words — кринжов|ий, тіктокерк|и — or
leaves them whole when the guesses disagree. Whether the gain holds on such words says what to expect of
text rich in neologisms, slang or surzhyk, which the work did not study.

HOW
---
The 5M model's full-training nulls (151M tokens, 50k; BPE 17k and endings apart, seeds 0–3, the thesis
machine) are scored again token by token, exactly as training evaluated them (morphlm.runs.rescore; each
figure must equal its record within --tol on validation, evaluate_split's on test). Each target is put in
its word as item 3 does (`loss_by_kind.classify`: a word the rule cuts, a word it does not, not a word).
Each word is then read from the text as the rule reads it (`endings.ANY_WORD`: letters with internal
apostrophes, so that м'ясо is one word), and marked known or unknown by pymorphy3's `word_is_known`; an
unknown word is Cyrillic or not. The
classes, the same words under both tokenizers as the text is the same:
  known, cut · known, not cut · unknown, cut · unknown Cyrillic, not cut · unknown, other script · not a word
and the first two and the next three together, known and unknown. For each: its share of the text's words and bytes; the gain (BPE 17k's loss minus endings apart's) as its
part of the whole in bits per byte; and the gain per byte of the class's own text, which says whether a
byte of unknown words gains as much as a byte of known ones. Paired by seed.

    python scripts/analysis/tokenizer_known_words.py
"""
import os
import re
import sys
import json
import glob
import bisect
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MORPHLM_NO_BUILD', '1')
from morphlm import runs as RUNS  # noqa: E402
from loss_by_kind import classify, checkpoint, KINDS, WORD_CLASS  # noqa: E402   (beside this script)

N = 'bpe_mconst-LEARN-CAUSAL_'
TOKENIZERS = ('bpe_uk_17k', 'bpeend_uk_17k')
CLASSES = ('known, cut', 'known, not cut', 'unknown, cut', 'unknown Cyrillic, not cut', 'unknown, other script',
           'not a word')
CYRILLIC = re.compile(r'[Ѐ-ӿ]')


class Words:
    """The text's words as the rule reads them, each known or not to pymorphy3."""

    def __init__(self, text, morph, rule):
        self.starts, self.ends, self.cls = [], [], []
        self.text_counts = {c: [0, 0] for c in CLASSES[:-1]}      # words, bytes
        cache = {}
        from morphlm.data.endings import ANY_WORD, APOSTROPHES
        for m in ANY_WORD.finditer(text):
            w = m.group(0)
            if w not in cache:
                lw = w.lower()
                for ap in APOSTROPHES:              # as the rule reads a form: one apostrophe for all
                    lw = lw.replace(ap, "'")
                known = morph.word_is_known(lw)
                cut = bool(rule.cut(w))
                cyr = bool(CYRILLIC.search(w))
                cache[w] = (CLASSES[0] if cut else CLASSES[1]) if known else (
                    CLASSES[2] if cut else CLASSES[3] if cyr else CLASSES[4])
            c = cache[w]
            self.starts.append(m.start())
            self.ends.append(m.end())
            self.cls.append(c)
            self.text_counts[c][0] += 1
            self.text_counts[c][1] += len(w.encode('utf-8'))

    def at(self, char):
        """The class of the word holding this character, or None."""
        i = bisect.bisect_right(self.starts, char) - 1
        return self.cls[i] if i >= 0 and char < self.ends[i] else None


def target_classes(ds, bpe_path, kind, morph, rule):
    """The class of every scored target, and the text's counts by class."""
    from morphlm.features.lexical import tokenizer_tables, word_spans, _bytelevel_decoder
    from morphlm.data.tokenization import load_tokenizer
    ids = np.asarray(ds.ids, dtype=np.int64)
    tok = load_tokenizer(bpe_path, vocab_only=True)
    dec = _bytelevel_decoder()
    raw = [bytes(dec[ch] for ch in (tok.id_to_token(i) or '') if ch in dec) for i in range(tok.get_vocab_size())]
    off = np.zeros(len(ids) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(raw[i]) for i in ids])
    text = b''.join(raw[i] for i in ids).decode('utf-8')
    char_byte = np.zeros(len(text) + 1, dtype=np.int64)
    char_byte[1:] = np.cumsum([len(ch.encode('utf-8')) for ch in text])
    words = Words(text, morph, rule)
    wid, _, starts = word_spans(ids, tokenizer_tables(bpe_path))
    # each word of the models' reading, by its first token: the text's word holding its first letter
    n_w = int(wid.max()) + 1
    first = np.flatnonzero(starts)
    wcls = []
    for p in first:
        b = int(off[p]) + len(raw[ids[p]]) - len(raw[ids[p]].lstrip(b' '))
        c = int(np.searchsorted(char_byte, b, side='right')) - 1
        wcls.append(words.at(c))
    if len(wcls) != n_w:
        raise RuntimeError('words and their first tokens disagree in number')
    n_t = len(kind)
    p = np.arange(1, n_t + 1)
    w = wid[p]
    word_class = np.array([CLASSES.index('not a word') if k == 'non-word' else -1 for k in (KINDS[i] for i in kind)])
    out = np.full(n_t, CLASSES.index('not a word'), dtype=np.int64)
    named = np.array([CLASSES.index(c) if c is not None else CLASSES.index('not a word') for c in wcls])
    inword = (w >= 0) & (word_class < 0)
    out[inword] = named[w[inword]]
    # the rule's cut is the stream's own (classify): a word the class calls cut must be one the stream cuts
    cut_kind = np.array([WORD_CLASS[k] == 'cut' for k in KINDS])[kind]
    unknown_cut = CLASSES.index('unknown, cut')
    known_cut = CLASSES.index('known, cut')
    agree = float(np.mean(cut_kind[inword] == np.isin(out[inword], [known_cut, unknown_cut]))) if inword.any() else 1.0
    tb = sum(len(raw[i]) for i in ids[1:n_t + 1])
    return out, words.text_counts, len(words.starts), agree, nbytes_by_class(ids, raw, out, n_t), tb


def nbytes_by_class(ids, raw, cls, n_t):
    """The bytes of the scored targets in each class: a class's own text, as the models predict it."""
    nb = np.array([len(raw[i]) for i in ids[1:n_t + 1]], dtype=np.int64)
    return {c: int(nb[cls == i].sum()) for i, c in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol', type=float, default=1e-5)
    ap.add_argument('--splits', nargs='+', default=['validation', 'test'])
    a = ap.parse_args()
    import pymorphy3
    from morphlm.data.endings import EndingRule
    from morphlm.record import AuditRecord
    rec = AuditRecord('tokenizer_known_words', a, out_dir=RUNS.AUDITS)
    R = RUNS.load(machine='thesis', keep=lambda r: 'seed' in r and 'bits_per_byte' in r)
    test_bpb = {r['run_id']: r['bpb'] for f in glob.glob(os.path.join(RUNS.AUDITS, 'evaluate_split_*.json'))
                for r in json.load(open(f)).get('results', {}).get('rows', []) if r.get('split') == 'test'}
    rule, morph = EndingRule(), pymorphy3.MorphAnalyzer(lang='uk')
    report = {}
    for split in a.splits:
        bits, info = {}, {}
        for t in TOKENIZERS:
            label = f'{N}{t}_ukwiki0.2'
            cls = None
            for s in range(4):
                r = R[(label, s)]
                cfg, pipe, ds, nll, _, _ = RUNS.rescore(*checkpoint(r), split=split)
                nll = nll.astype(np.float64)
                scored = ds.num_bytes * len(nll) / max(ds.num_tokens, 1)
                bpb = float(nll.sum() / np.log(2) / scored)
                want = r['bits_per_byte'] if split == 'validation' else test_bpb.get(r['run_id'])
                if want is None or abs(bpb - want) > a.tol:
                    raise SystemExit(f'{r["run_id"]} {split}: re-scored {bpb:.6f} against recorded {want} — nothing reported')
                if cls is None:
                    kind, _ = classify(ds, os.path.join(ROOT, 'data', f'{t}.json'), rule)
                    cls, counts, n_words, agree, cbytes, tbytes = target_classes(
                        ds, os.path.join(ROOT, 'data', f'{t}.json'), kind, morph, rule)
                    info[t] = {'text_words': n_words, 'text_counts': counts, 'cut_agreement': agree,
                               'target_bytes': cbytes, 'target_bytes_all': tbytes}
                    print(f'  {split} {t}: {n_words:,} words; the classes\' cut agrees with the stream\'s for '
                          f'{100 * agree:.2f}% of word targets', flush=True)
                bits[(t, s)] = {c: float(nll[cls == i].sum() / np.log(2)) for i, c in enumerate(CLASSES)}
                print(f'  {split} {t} seed {s}: {bpb:.4f}', flush=True)
        ref = info['bpe_uk_17k']
        total_bytes = ref['target_bytes_all']
        out = {'text': {'words': ref['text_words'], 'by_class': ref['text_counts']},
               'cut_agreement': {t: info[t]['cut_agreement'] for t in TOKENIZERS}, 'by_class': {}}
        tw = sum(v[0] for v in ref['text_counts'].values())
        tb = sum(v[1] for v in ref['text_counts'].values())
        print(f'\n  {split}: the gain by class, bits per byte of the whole text and of the class\'s own bytes, paired by seed')
        for c in CLASSES:
            g = [bits[('bpe_uk_17k', s)][c] - bits[('bpeend_uk_17k', s)][c] for s in range(4)]
            own = ref['target_bytes'][c]
            whole = RUNS.summary([x / total_bytes for x in g])
            per = RUNS.summary([x / own for x in g]) if own else None
            words_c, bytes_c = ref['text_counts'].get(c, [None, None])
            out['by_class'][c] = {'gain_whole': whole, 'gain_per_own_byte': per, 'own_target_bytes': own,
                                  'text_words': words_c, 'text_bytes': bytes_c}
            share = (f'{100 * words_c / tw:5.1f}% of words, {100 * bytes_c / tb:5.1f}% of word bytes'
                     if words_c is not None else 'punctuation, digits, line breaks')
            print(f'    {c:26} {share:44} whole {whole["mean"]:+.4f} ± {whole["se"]:.4f} ({whole["positive"]}/4)'
                  + (f'   per own byte {per["mean"]:+.4f} ± {per["se"]:.4f}' if per else ''))
        # the two groups: a word the dictionary knows, and one it does not
        for name, members in (('known', CLASSES[:2]), ('unknown', CLASSES[2:5])):
            g = [sum(bits[('bpe_uk_17k', s)][c] - bits[('bpeend_uk_17k', s)][c] for c in members) for s in range(4)]
            own = sum(ref['target_bytes'][c] for c in members)
            words_g = sum(ref['text_counts'][c][0] for c in members)
            bytes_g = sum(ref['text_counts'][c][1] for c in members)
            out['by_class'][name] = {'members': list(members), 'gain_whole': RUNS.summary([x / total_bytes for x in g]),
                                     'gain_per_own_byte': RUNS.summary([x / own for x in g]), 'own_target_bytes': own,
                                     'text_words': words_g, 'text_bytes': bytes_g}
            per = out['by_class'][name]['gain_per_own_byte']
            print(f'    {name:26} {100 * words_g / tw:5.1f}% of words, {100 * bytes_g / tb:5.1f}% of word bytes'
                  f'{"":9} per own byte {per["mean"]:+.4f} ± {per["se"]:.4f} ({per["positive"]}/4)')
        report[split] = out
    rec.add('report', report)
    rec.write()


if __name__ == '__main__':
    main()
