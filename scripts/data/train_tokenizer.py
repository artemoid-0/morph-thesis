#!/usr/bin/env python3
"""
train_tokenizer.py — a tokenizer for any corpus, size and algorithm, and what it
does to words (journey §33.53; morphlm/data/tokenization.py).

TRAIN — on the corpus's train split at a given fraction; the standard name
(`bpe_17k`, `bpe_uk_8k`, `unigram_uk_17k`), a full sidecar (word rule, recipe,
the file's sha256). An existing tokenizer is never overwritten: its file name is
in every cache key built from it, and Unigram training is not deterministic
(the same pieces, scores differing by up to 0.02 between two trainings), so a
retrained file under an old name would silently change that stream.

    python scripts/data/train_tokenizer.py --corpus ukwiki --corpus-fraction 0.2 --vocab 8000
    python scripts/data/train_tokenizer.py --corpus ukwiki --corpus-fraction 0.2 --vocab 17000 --algorithm unigram
    python scripts/data/train_tokenizer.py --corpus ukwiki --corpus-fraction 0.2 --vocab 17000 --algorithm bpe-endings

`bpe-endings`: BPE that never joins a stem to its ending — trained on text marked
at every cut of morphlm/data/endings.py, by the rule of the corpus's language
(Ukrainian `uk_endings_v1`, English `en_endings_v1`), and opened (load_tokenizer)
so that the text it encodes is marked the same way; named bpeend_uk_<k>k, and
bpeend_<k>k on WikiText-103.

    python scripts/data/train_tokenizer.py --corpus wikitext103 --vocab 17000 --algorithm bpe-endings

STATS — on held-out text (the corpus's validation split), with spaCy's words:
characters per token, tokens per word, words split, and ending exposure (the
stem|ending boundary, all inflected words and the case-bearing ones). One
record per call in results/audits/. For Ukrainian and English, exposure is also
measured at the cut of the language's ending rule (cut|), the rule the
bpe-endings tokenizer applies.

    python scripts/data/train_tokenizer.py --stats data/bpe_uk_8k.json data/bpe_uk_17k.json --corpus ukwiki
"""
import os
import sys
import json
import hashlib
import argparse
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def corpus_cfg(a):
    from morphlm.config import ExperimentConfig, apply_overrides
    return apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', f'data.corpus={a.corpus}', f'data.corpus_fraction={a.corpus_fraction}',
        f'data.corpus_holdout={a.corpus_holdout}'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--corpus-fraction', type=float, default=1.0)
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--vocab', type=int, default=17000)
    ap.add_argument('--algorithm', choices=['bpe', 'unigram', 'bpe-endings'], default='bpe')
    ap.add_argument('--out', default=None, help='default: data/<standard name>.json')
    ap.add_argument('--lines', type=int, default=0, help='train on at most this many rows (0 = all)')
    ap.add_argument('--stats', nargs='+', default=None, metavar='TOKENIZER',
                    help='measure these tokenizers instead of training one')
    ap.add_argument('--paragraphs', type=int, default=3000,
                    help='held-out paragraphs the statistics read')
    ap.add_argument('--spacy-model', default=None, help="default: the corpus's tagger")
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    a = ap.parse_args()

    from morphlm.data.corpora import get_corpus
    from morphlm.data import tokenization as T
    spec = get_corpus(a.corpus)
    cfg = corpus_cfg(a)

    if a.stats is None:
        out = a.out or os.path.join('data', T.standard_name(a.corpus, a.algorithm, a.vocab) + '.json')
        if os.path.exists(out):
            sys.exit(f'{out} exists — a tokenizer is never overwritten (its name is in every cache key '
                     f'built from it). Measure it with --stats, or give another --out.')
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        print(f'training {a.algorithm} at {a.vocab:,} on {a.corpus} train, fraction {a.corpus_fraction:g} -> {out}')
        T.train_tokenizer(cfg, a.vocab, out, algorithm=a.algorithm, max_lines=a.lines, sidecar='full')
        meta_path = out[:-5] + '.meta.json'
        meta = json.load(open(meta_path))
        meta['sha256'] = sha256(out)
        json.dump(meta, open(meta_path, 'w'), indent=2)
        print(f'  {meta["training_texts"]:,} training rows; sha256 {meta["sha256"][:16]}…  -> {meta_path}')
        print(f'  measure it: python scripts/data/train_tokenizer.py --stats {out} --corpus {a.corpus}')
        return

    import spacy
    model = a.spacy_model or spec.spacy_model
    nlp = spacy.load(model, disable=['parser', 'ner'])
    from morphlm.data.stream import load_corpus
    texts = []
    for row in load_corpus(cfg, 'validation'):
        t = row['text'].strip()
        if t:
            texts.append(t)
        if len(texts) >= a.paragraphs:
            break
    print(f'{len(texts):,} held-out paragraphs of {a.corpus} ({sum(map(len, texts)):,} characters), words by {model}')
    rows = []
    for path in a.stats:
        s = T.tokenizer_stats(path, texts, nlp)
        s['sha256'] = sha256(path)
        rows.append(s)
    print(f'\n  {"tokenizer":24} {"vocab":>7} {"chars/tok":>9} {"tok/word":>8} {"split":>7} '
          f'{"ending|":>8} {"own":>6} {"case ending|":>13} {"own":>6}')
    for s in rows:
        print(f'  {s["tokenizer"][:24]:24} {s["vocab_size"]:>7,} {s["chars_per_token"]:9.2f} {s["tokens_per_word"]:8.2f} '
              f'{100 * s["words_split"]:6.1f}% {100 * s["ending_boundary_all"]:7.1f}% {100 * s["ending_own_token_all"]:5.1f}% '
              f'{100 * s["ending_boundary_case"]:12.1f}% {100 * s["ending_own_token_case"]:5.1f}%')
    print(f'\n  ending| — a token boundary exactly between stem and ending, among {rows[0]["inflected_all"]:,} inflected words '
          f'({rows[0]["inflected_case"]:,} nouns, proper nouns, adjectives); own — the ending is a token by itself')
    if 'cut_words_all' in rows[0]:
        print(f'\n  {"tokenizer":24} {"cut|":>8} {"own":>6} {"case cut|":>10} {"own":>6}')
        for s in rows:
            print(f'  {s["tokenizer"][:24]:24} {100 * s["cut_boundary_all"]:7.1f}% {100 * s["cut_own_token_all"]:5.1f}% '
                  f'{100 * s["cut_boundary_case"]:9.1f}% {100 * s["cut_own_token_case"]:5.1f}%')
        print(f'\n  cut| — a token boundary at the ending rule\'s cut ({rows[0]["cut_rule"]}, fingerprint '
              f'{rows[0]["cut_fingerprint"]}), among {rows[0]["cut_words_all"]:,} words it cuts '
              f'({rows[0]["cut_words_case"]:,} nouns, proper nouns, adjectives)')
    os.makedirs(a.audit_dir, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '+0000'
    path = os.path.join(a.audit_dir, f'tokenizer_stats_{a.corpus}_{ts}.json')
    json.dump({'corpus': a.corpus, 'paragraphs': len(texts), 'spacy_model': model,
               'spacy_version': spacy.__version__, 'tokenizers': rows}, open(path, 'w'), indent=1)
    print(f'  -> {path}')


if __name__ == '__main__':
    main()
