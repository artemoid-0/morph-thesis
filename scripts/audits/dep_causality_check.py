#!/usr/bin/env python3
"""
dep_causality_check.py — how much does a `dep` label depend on LATER words?

THE QUESTION
------------
`lookahead_check` cleared `dep` of the WITHIN-word leak: its gain sits on
word-initial targets (+0.1487) and it slightly hurts continuations (−0.0593),
the signature of a genuine word-level context feature rather than an oracle.

But it looks only inside words. `dep` comes from spaCy, whose encoder reads the
whole sentence in both directions, so a label can still encode information about
words that have not been produced yet. Until that is measured, `dep`'s 0.0412
BPB — the largest surviving effect in the project — is an upper bound.

WHY NOT JUST BUILD A CAUSAL `dep`
---------------------------------
Because it costs about thirty hours of preprocessing. Re-parsing every prefix of
every sentence is O(words x sentence length): roughly a billion word-parses over
the training corpus. That is worth paying only if the leak is large, and this
script finds out for a few minutes of compute.

WHAT IS MEASURED
----------------
For each word, the label from the FULL sentence parse against the label from a
parse of the sentence PREFIX ending at that word — which is all a causal feature
could ever see. Agreement bounds the leak: if a label is the same either way,
using the full-sentence one leaks nothing, whatever the parser saw.

Agreement is reported on the COARSE labels the feature actually uses (eight
classes), not the fine ones. A prefix parse may well disagree about `nsubj` vs
`csubj` while agreeing that the word is a subject, and only the coarse
distinction reaches the model.

    python scripts/audits/dep_causality_check.py --sentences 3000
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sentences', type=int, default=3000)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--max-words', type=int, default=40,
                    help='skip longer sentences; prefix parsing is quadratic')
    args = ap.parse_args()

    import spacy
    from datasets import load_dataset
    from morphlm.features.blocks import DEP_COARSE_MAP
    from morphlm.record import AuditRecord

    rec = AuditRecord('dep_causality_check', args)

    def coarse(lab):
        return DEP_COARSE_MAP.get(lab, 'other')

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)

    sents = []
    for row in ds:
        t = row['text'].strip()
        if not t or t.startswith(' ='):
            continue
        for s in nlp(t).sents:
            w = [tok for tok in s if not tok.is_space]
            if 3 <= len(w) <= args.max_words:
                sents.append([tok.text for tok in w])
            if len(sents) >= args.sentences:
                break
        if len(sents) >= args.sentences:
            break

    print(f'  {len(sents)} sentences, '
          f'{sum(len(s) for s in sents):,} words')

    agree_fine = agree_coarse = total = 0
    by_pos = {}
    confus = Counter()

    for words in sents:
        full = nlp(' '.join(words))
        full_toks = [t for t in full if not t.is_space]
        if len(full_toks) != len(words):
            continue                       # retokenised; skip rather than guess
        prefix_labels = []
        for i in range(len(words)):
            pre = nlp(' '.join(words[:i + 1]))
            pt = [t for t in pre if not t.is_space]
            prefix_labels.append(pt[-1].dep_ if len(pt) == i + 1 else None)

        for i, (ft, pl) in enumerate(zip(full_toks, prefix_labels)):
            if pl is None:
                continue
            total += 1
            agree_fine += (ft.dep_ == pl)
            cf, cp = coarse(ft.dep_), coarse(pl)
            agree_coarse += (cf == cp)
            # position from the END: the last word of a sentence has no future
            from_end = len(words) - 1 - i
            b = min(from_end, 5)
            d = by_pos.setdefault(b, [0, 0])
            d[0] += 1
            d[1] += (cf == cp)
            if cf != cp:
                confus[(cf, cp)] += 1

    print()
    print('=' * 72)
    print('DEP CAUSALITY — full-sentence label vs prefix-parse label')
    print('=' * 72)
    print(f'  {total:,} words compared')
    print(f'    fine labels agree   {100.0*agree_fine/max(total,1):6.2f}%')
    print(f'    COARSE labels agree {100.0*agree_coarse/max(total,1):6.2f}%  '
          f'<- the feature uses these')
    print()
    print('  agreement by distance from the end of the sentence')
    print('  (a word at the end has no future left to depend on)')
    print(f'    {"words after it":>16} {"n":>8} {"coarse agree":>13}')
    for b in sorted(by_pos):
        n, a = by_pos[b]
        lab = f'{b}' if b < 5 else '5+'
        print(f'    {lab:>16} {n:8,} {100.0*a/max(n,1):12.2f}%')
    if confus:
        print()
        print('  most common disagreements (full -> prefix)')
        for (cf, cp), n in confus.most_common(6):
            print(f'    {cf:>10} -> {cp:<10} {n:6,}')

    share = 1.0 - agree_coarse / max(total, 1)
    print()
    print('  READING THIS')
    print(f'    {100*share:.2f}% of coarse labels change when the parser is')
    print('    denied the rest of the sentence. A label that does NOT change')
    print('    leaks nothing, whatever the parser saw, so this bounds the')
    print('    cross-sentence lookahead in `dep`.')
    print('    Low  -> dep is effectively causal; its 0.0412 BPB stands and a')
    print('            30-hour prefix-parsed rebuild is not worth running.')
    print('    High -> dep is an oracle like the word-form features, and the')
    print('            largest surviving effect in the project is not safe.')

    rec.add('summary', {'words': total,
                        'fine_agreement': agree_fine / max(total, 1),
                        'coarse_agreement': agree_coarse / max(total, 1),
                        'coarse_disagreement': share})
    rec.add('by_distance_from_end',
            {str(b): {'n': v[0], 'agree': v[1] / max(v[0], 1)}
             for b, v in sorted(by_pos.items())})
    rec.add('confusions', {f'{a}->{b}': n for (a, b), n in confus.most_common(20)})
    rec.note('Agreement bounds the leak but does not measure its VALUE: a '
             'label that changes may or may not have been carrying useful '
             'information about the upcoming word.')
    rec.write()


if __name__ == '__main__':
    main()
