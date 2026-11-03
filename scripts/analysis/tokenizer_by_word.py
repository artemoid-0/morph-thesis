#!/usr/bin/env python3
"""
tokenizer_by_word.py — where the tokenizer's gain sits, by word, and how much of it is the text's sample
(the text's list of 30 October, item 3 and the bootstrap over articles; journey §33.156).

WHY
---
The text places case's gain by target, not the tokenizer's. If the endings-apart null's lead over BPE 17k's
sits in the words the endings rule cuts, the model predicts endings better; if in the other words and
around them, keeping endings apart improves how context is represented. And the lead is 0.0039 smaller on
the test articles than on validation (§33.152): a bootstrap over articles says whether two samples of
Wikipedia differ by that much.

HOW
---
Each of the 5M model's full-training nulls (151M tokens, 50k; both tokenizers, four seeds, the thesis
machine) is scored again token by token on the validation and the test streams, exactly as training
evaluated (morphlm.runs.rescore; on validation each model's figure must equal its record's within --tol,
on test the figure evaluate_split recorded). The two streams of one split hold the same text, so the same
words and paragraphs. Every target is put in its word by the words the models read (`word_spans`) and the
endings rule's cuts in the stream's text (`loss_by_kind.classify`): a word the rule cuts, a word it does
not, or not a word (punctuation, digits, line breaks). Words are also banded by their count in the
training corpus, from each tokenizer's training lexicon. Each paragraph's loss is summed, in bits, per
class and band; the gain is BPE 17k's minus endings apart's, over the text's bytes, paired by seed.

The paragraphs are mapped back to their articles: the split's articles, read in the corpus's order with
its hash rule, each hold their non-empty paragraphs, and the count must equal the stream's paragraphs or
the tool stops. The bootstrap resamples articles (--draws) for the four seeds' mean gain.

    python scripts/analysis/tokenizer_by_word.py
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MORPHLM_NO_BUILD', '1')
from morphlm import runs as RUNS  # noqa: E402
from loss_by_kind import classify, checkpoint, KINDS, WORD_CLASS  # noqa: E402   (beside this script)

N = 'bpe_mconst-LEARN-CAUSAL_'
TOKENIZERS = {'bpe_uk_17k': 'lexicon_ucase_bpe_uk_17k.npz', 'bpeend_uk_17k': 'lexicon_ucase_bpeend_uk_17k_ukwiki0.2.npz'}
CLASSES = ('cut', 'uncut', 'non-word')
BANDS = (('unseen', 0, 0), ('1-99', 1, 99), ('100-9,999', 100, 9999), ('10,000+', 10000, None))


def articles_of(split, holdout):
    """The number of non-empty paragraphs of each article of `split`, in the corpus's order."""
    from datasets import load_dataset
    from morphlm.data.corpora import CORPORA, assigned_split
    spec = CORPORA['ukwiki']
    ds = load_dataset(spec.dataset, spec.config, split='train')
    out = []
    for doc in ds:
        key = doc[spec.key_field] if spec.key_field else doc[spec.text_field]
        if assigned_split(key, holdout) != split:
            continue
        out.append(sum(1 for p in doc[spec.text_field].split('\n') if p.strip()))
    return np.asarray(out, dtype=np.int64)


def per_paragraph(ds, nll, kind, bpe_path, lexicon, nl):
    """Per paragraph: bits by class, bits by (class, band), and bytes, over the scored targets."""
    from morphlm.features.lexical import tokenizer_tables, word_spans, lookup, _bytelevel_decoder
    from morphlm.data.tokenization import load_tokenizer
    ids = np.asarray(ds.ids, dtype=np.int64)
    tok = load_tokenizer(bpe_path, vocab_only=True)
    dec = _bytelevel_decoder()
    nbytes = np.array([len(bytes(dec[ch] for ch in (tok.id_to_token(i) or '') if ch in dec))
                       for i in range(tok.get_vocab_size())], dtype=np.int64)
    para = np.concatenate([[0], np.cumsum(ids == nl)[:-1]])          # a newline belongs to the paragraph it ends
    _, key, _ = word_spans(ids, tokenizer_tables(bpe_path))
    z = np.load(lexicon)
    count = lookup(key, (z['keys'], z['counts'].astype(np.int64)), 0).astype(np.int64)
    count = np.where(key == 0, -1, count)                             # not a word
    n_t = len(nll)
    p = np.arange(1, n_t + 1)
    bits = nll / np.log(2)
    cls = np.array([CLASSES.index(WORD_CLASS[k]) for k in KINDS])[kind]
    band = np.full(n_t, -1, dtype=np.int64)
    c = count[p]
    for b, (_, lo, hi) in enumerate(BANDS):
        band[(c >= lo) & ((c <= hi) if hi is not None else True) & (cls != CLASSES.index('non-word'))] = b
    n_par = int(para[p].max()) + 1
    out = {'bytes': np.bincount(para[p], weights=nbytes[ids[p]], minlength=n_par),
           'total': np.bincount(para[p], weights=bits, minlength=n_par)}
    for ci, cn in enumerate(CLASSES):
        m = cls == ci
        out[cn] = np.bincount(para[p][m], weights=bits[m], minlength=n_par)
        for bi, (bn, _, _) in enumerate(BANDS):
            mb = m & (band == bi)
            out[f'{cn} {bn}'] = np.bincount(para[p][mb], weights=bits[mb], minlength=n_par)
    out['last_full'] = int(para[n_t]) - 1 if n_t < len(ids) else n_par - 1   # the last paragraph scored whole
    out['n_paragraphs'] = int((ids == nl).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol', type=float, default=1e-5)
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--holdout', type=int, default=2000)
    a = ap.parse_args()
    from morphlm.data.endings import EndingRule
    from morphlm.record import AuditRecord
    rec = AuditRecord('tokenizer_by_word', a, out_dir=RUNS.AUDITS)
    R = RUNS.load(machine='thesis', keep=lambda r: 'seed' in r and 'bits_per_byte' in r)
    test_bpb = {r['run_id']: r['bpb'] for f in glob.glob(os.path.join(RUNS.AUDITS, 'evaluate_split_*.json'))
                for r in json.load(open(f)).get('results', {}).get('rows', []) if r.get('split') == 'test'}
    rule = EndingRule()
    rng = np.random.default_rng(0)
    report = {}
    for split in ('validation', 'test'):
        sums = {}
        for t, lexname in TOKENIZERS.items():
            label = f'{N}{t}_ukwiki0.2'
            kind, nl = None, None
            for s in range(4):
                r = R[(label, s)]
                cfg, pipe, ds, nll, _, _ = RUNS.rescore(*checkpoint(r), split=split)
                nll = nll.astype(np.float64)
                scored = ds.num_bytes * len(nll) / max(ds.num_tokens, 1)
                bpb = float(nll.sum() / np.log(2) / scored)
                want = r['bits_per_byte'] if split == 'validation' else test_bpb.get(r['run_id'])
                if want is None or abs(bpb - want) > a.tol:
                    raise SystemExit(f'{r["run_id"]} {split}: re-scored {bpb:.6f} against recorded {want} — nothing reported')
                if kind is None:
                    nl = pipe.encode_ids('\n')[0]
                    kind, counts = classify(ds, os.path.join(ROOT, 'data', f'{t}.json'), rule)
                sums[(t, s)] = per_paragraph(ds, nll, kind, os.path.join(ROOT, 'data', f'{t}.json'),
                                             os.path.join(ROOT, 'data', 'cache', lexname), nl)
                print(f'  {split} {t} seed {s}: {bpb:.4f}', flush=True)
        n_par = {sums[k]['n_paragraphs'] for k in sums}
        if len(n_par) != 1:
            raise SystemExit(f'{split}: the streams hold {n_par} paragraphs — not the same text')
        arts = articles_of(split, a.holdout)
        if int(arts.sum()) != n_par.pop():
            raise SystemExit(f'{split}: the articles hold {int(arts.sum())} paragraphs, the stream another number')
        article = np.repeat(np.arange(len(arts)), arts)
        keep = min(min(v['last_full'] for v in sums.values()) + 1, len(article))
        b0 = sums[('bpe_uk_17k', 0)]['bytes'][:keep]
        keys = ['total'] + list(CLASSES) + [f'{c} {b}' for c in CLASSES[:2] for b, _, _ in BANDS]
        per_seed = {k: [] for k in keys}
        per_article = []
        for s in range(4):
            B, E = sums[('bpe_uk_17k', s)], sums[('bpeend_uk_17k', s)]
            for k in keys:
                per_seed[k].append(float((B[k][:keep] - E[k][:keep]).sum() / b0.sum()))
            per_article.append(np.bincount(article[:keep], weights=B['total'][:keep] - E['total'][:keep]))
        art_bytes = np.bincount(article[:keep], weights=b0)
        g_art = np.mean(per_article, axis=0)                               # the four seeds' mean, per article
        draws = rng.integers(0, len(g_art), size=(a.draws, len(g_art)))
        boot = g_art[draws].sum(1) / art_bytes[draws].sum(1)
        out = {'paragraphs': keep, 'articles': int(article[keep - 1]) + 1, 'bytes': float(b0.sum()),
               'by': {k: RUNS.summary(v) for k, v in per_seed.items()},
               'bootstrap': {'mean': float(g_art.sum() / art_bytes.sum()), 'se': float(boot.std(ddof=1)),
                             'low': float(np.quantile(boot, 0.025)), 'high': float(np.quantile(boot, 0.975)),
                             'draws': a.draws}}
        report[split] = out
        tot = out['by']['total']['mean']
        print(f'\n  {split}: {keep:,} paragraphs of {out["articles"]:,} articles, {out["bytes"]:,.0f} bytes')
        for k in keys:
            sm = out['by'][k]
            print(f'    {k:20} {sm["mean"]:+.4f} ± {sm["se"]:.4f} ({100 * sm["mean"] / tot:5.1f}% of the gain, {sm["positive"]}/{sm["n"]})')
        bs = out['bootstrap']
        print(f'    over articles: {bs["mean"]:+.4f}, bootstrap se {bs["se"]:.4f}, 95% {bs["low"]:+.4f} to {bs["high"]:+.4f}')
    v, t = report['validation']['bootstrap'], report['test']['bootstrap']
    d = v['mean'] - t['mean']
    se = (v['se'] ** 2 + t['se'] ** 2) ** 0.5
    report['validation_minus_test'] = {'difference': d, 'se': se, 'z': d / se if se else None}
    print(f'\n  validation minus test: {d:+.4f}, se {se:.4f} over articles (z = {d / se:.2f})')
    rec.add('report', report)
    rec.write()


if __name__ == '__main__':
    main()
