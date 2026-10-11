#!/usr/bin/env python3
"""
corpus_info.py — look at a dataset before it becomes a corpus (journey §33.53).
Run it on a machine that can reach Hugging Face.


A corpus is added as a spec in `morphlm/data/corpora.py`. This shows what the
spec needs: the dataset's splits and sizes (from its metadata, nothing
downloaded), its fields, a few documents (streamed, not downloaded), and how
long its documents and paragraphs are. For a corpus already registered it also
estimates, per train fraction, the tokens and the tagging time.

    python scripts/data/corpus_info.py --dataset wikimedia/wikipedia --config 20231101.uk
    python scripts/data/corpus_info.py --corpus ukwiki --sample 200
"""
import os
import sys
import argparse
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: Measured rates, for estimates only: Ukrainian Wikipedia at fraction 0.2 —
#: 151.1M tokens tagged in 5 h 22 min; ~3.6 characters per token (17k BPE).
TAG_TOKENS_PER_SECOND = 151.1e6 / (5 * 3600 + 22 * 60)
CHARS_PER_TOKEN = {'uk': 3.6, 'en': 4.4}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', help='a Hugging Face dataset path')
    ap.add_argument('--config', default=None)
    ap.add_argument('--corpus', help='a registered corpus (morphlm/data/corpora.py)')
    ap.add_argument('--sample', type=int, default=100, help='documents to stream for statistics')
    ap.add_argument('--text-field', default=None)
    a = ap.parse_args()

    spec = None
    if a.corpus:
        from morphlm.data.corpora import get_corpus
        spec = get_corpus(a.corpus)
        a.dataset, a.config = spec.dataset, spec.config
        print(f'corpus {spec.name}: {spec.note}')
        print(f'  spec: language {spec.language}, tagger {spec.spacy_model}, split {spec.split_rule}, '
              f'served as {spec.serve}, word rule {spec.word_rule}, tokenizer code {spec.code!r}')
    if not a.dataset:
        sys.exit('give --dataset (and --config) or --corpus')

    import datasets
    print(f'\ndataset {a.dataset}' + (f' [{a.config}]' if a.config else ''))
    splits = {}
    try:
        info = datasets.load_dataset_builder(a.dataset, a.config).info
        for name, s in (info.splits or {}).items():
            splits[name] = s.num_examples
            print(f'  split {name:12} {s.num_examples:>12,} documents  {s.num_bytes / 1e9:7.2f} GB')
        if info.features:
            print(f'  fields: {", ".join(info.features)}')
    except Exception as e:                    # metadata missing: streaming still works
        print(f'  (no metadata: {type(e).__name__}: {e})')

    split = 'train' if not splits or 'train' in splits else next(iter(splits))
    try:
        ds = datasets.load_dataset(a.dataset, a.config, split=split, streaming=True)
    except TypeError:                         # the offline stand-in has no streaming
        ds = datasets.load_dataset(a.dataset, a.config, split=split)
    field = a.text_field or (spec.text_field if spec else 'text')
    lengths, paras, shown = [], [], 0
    for i, doc in enumerate(ds):
        if i >= a.sample:
            break
        if i == 0:
            print(f'  first document\'s fields: {", ".join(doc)}')
        text = doc.get(field, '')
        lengths.append(len(text))
        paras.append(len([p for p in text.split('\n') if p.strip()]))
        if shown < 2:
            print(f'  sample {shown}: {text[:240]!r}')
            shown += 1
    if lengths:
        print(f'  {len(lengths)} documents streamed: {st.mean(lengths):,.0f} characters each '
              f'(median {st.median(lengths):,.0f}), {st.mean(paras):.1f} non-empty paragraphs each')
    if spec and lengths and split in splits:
        cpt = CHARS_PER_TOKEN.get(spec.language, 4.0)
        n = splits[split]
        hold = 2000
        train_docs = n * (hold - 2) / hold if spec.split_rule == 'hash' else n
        print(f'\n  estimates (~{cpt} characters per token; tagging at {TAG_TOKENS_PER_SECOND:,.0f} tokens/s):')
        for frac in (1.0, 0.5, 0.2, 0.05, 0.01):
            tokens = train_docs * frac * st.mean(lengths) / cpt
            print(f'    fraction {frac:<5g} ~{tokens / 1e6:8.1f}M train tokens, tagging ~{tokens / TAG_TOKENS_PER_SECOND / 3600:5.1f} h')
    if not spec:
        print('\n  a spec skeleton for morphlm/data/corpora.py:')
        print(f"    'NAME': CorpusSpec(name='NAME', language='??', dataset={a.dataset!r}, config={a.config!r},\n"
              f"        split_rule={'native' if 'validation' in splits else 'hash'!r}, serve='paragraphs',\n"
              f"        spacy_model='??_core_news_sm', word_rule='??', code='??', text_field={field!r},\n"
              f"        key_field=None, note='...'),")


if __name__ == '__main__':
    main()
