#!/usr/bin/env python3
"""
multilingual_endings.py — how widely used multilingual tokenizers cut Ukrainian words: do they keep the
inflectional ending apart? (the text's analyses of 30 October; journey §33.154)

WHY
---
The thesis's tokenizer keeps every inflectional ending its rule finds apart from the stem. Whether the
tokenizers of widely used models already do — by chance, or because a large vocabulary learns it — says
whether the finding is news for them. Unigram 17k, trained here, puts a token boundary at the rule's cut
for 43% of the words the rule cuts; plain BPE 17k for 8.8%.

HOW
---
On the same held-out text as the project's tokenizer statistics (the first 3,000 paragraphs of the
Ukrainian corpus's validation split, words by uk_core_news_sm), `morphlm.data.tokenization.tokenizer_stats`
measures each tokenizer: characters per token, tokens per word, words split, a token boundary at the
stem|ending boundary (the word's common prefix with its lemma) and at the endings rule's cut, and the
ending a token of its own. The tokenizers:
  * the project's BPE 17k, endings apart and Unigram 17k, for reference;
  * XLM-R's (SentencePiece Unigram, 250k), mBERT's (WordPiece, 119k) and Qwen2.5's (byte-level BPE,
    152k), from their `tokenizer.json` at a recorded commit (`external/tokenizers/`), special tokens off;
  * GPT-4o's `o200k_base` (byte-level BPE, 200k), through tiktoken in an environment of its own
    (`--tiktoken-python`): the spans come back from it as character offsets.
Each file's SHA-256 is recorded.

    python scripts/audits/multilingual_endings.py --tiktoken-python scratch/env_tiktoken/bin/python
"""
import os
import sys
import glob
import json
import hashlib
import argparse
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

EXTERNAL = os.path.join(ROOT, 'external', 'tokenizers')
HF = {'XLM-R (Unigram, 250k)': 'FacebookAI_xlm-roberta-base', 'mBERT (WordPiece, 119k)': 'google-bert_bert-base-multilingual-cased',
      'Qwen2.5 (byte-level BPE, 152k)': 'Qwen_Qwen2.5-7B'}
OURS = {'BPE 17k': 'bpe_uk_17k.json', 'endings apart': 'bpeend_uk_17k.json', 'Unigram 17k': 'unigram_uk_17k.json'}

TIKTOKEN = r'''
import sys, json
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import tiktoken_ext.openai_public as op
path = sys.argv[1]
op.load_tiktoken_bpe = lambda url, expected_hash=None: load_tiktoken_bpe(path, expected_hash)
enc = tiktoken.Encoding(**op.o200k_base())
out = []
for line in open(sys.argv[2], encoding='utf-8'):
    text = json.loads(line)
    ids = enc.encode(text, disallowed_special=())
    dec, starts = enc.decode_with_offsets(ids)
    assert dec == text
    ends = starts[1:] + [len(text)]
    out.append({'ids': ids, 'offsets': list(zip(starts, ends))})
json.dump({'vocab': enc.n_vocab, 'rows': out}, open(sys.argv[3], 'w'))
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Encoded:
    def __init__(self, ids, offsets):
        self.ids, self.offsets = ids, [tuple(o) for o in offsets]


class HFTokenizer:
    """A tokenizer.json, with no special tokens added."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)

    def encode(self, text):
        e = self.tok.encode(text, add_special_tokens=False)
        return Encoded(e.ids, e.offsets)

    def get_vocab_size(self):
        return self.tok.get_vocab_size()


class Precomputed:
    """Spans computed elsewhere (tiktoken's environment), looked up by text."""

    def __init__(self, texts, rows, vocab):
        self.by = {t: r for t, r in zip(texts, rows)}
        self.vocab = vocab

    def encode(self, text):
        r = self.by[text]
        return Encoded(r['ids'], r['offsets'])

    def get_vocab_size(self):
        return self.vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paragraphs', type=int, default=3000)
    ap.add_argument('--tiktoken-python', default=None, help="a Python with tiktoken; GPT-4o's tokenizer is skipped without it")
    a = ap.parse_args()
    import spacy
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import load_corpus
    from morphlm.data import tokenization as T
    from morphlm.record import AuditRecord
    rec = AuditRecord('multilingual_endings', a, out_dir=os.path.join(ROOT, 'results', 'audits'))
    cfg = apply_overrides(ExperimentConfig(), ['data.corpus=ukwiki', 'data.spacy_model=uk_core_news_sm'])
    texts = []
    for row in load_corpus(cfg, 'validation'):
        t = row['text'].strip()
        if t:
            texts.append(t)
        if len(texts) >= a.paragraphs:
            break
    nlp = spacy.load('uk_core_news_sm', disable=['parser', 'ner'])
    print(f'{len(texts):,} held-out paragraphs ({sum(map(len, texts)):,} characters), words by uk_core_news_sm')
    todo = [(name, os.path.join(ROOT, 'data', f), None) for name, f in OURS.items()]
    for name, d in HF.items():
        found = glob.glob(os.path.join(EXTERNAL, d, '*', 'tokenizer.json'))
        if len(found) != 1:
            raise SystemExit(f'{name}: expected one tokenizer.json under {d}, found {found}')
        todo.append((name, found[0], HFTokenizer(found[0])))
    if a.tiktoken_python:
        found = [p for p in glob.glob(os.path.join(EXTERNAL, 'openai_o200k_base', '*')) if os.path.isfile(p)]
        if len(found) != 1:
            raise SystemExit(f'GPT-4o: expected one o200k_base file, found {found}')
        with tempfile.TemporaryDirectory() as tmp:
            src, inp, out = (os.path.join(tmp, n) for n in ('tk.py', 'in.jsonl', 'out.json'))
            open(src, 'w').write(TIKTOKEN)
            with open(inp, 'w', encoding='utf-8') as f:
                for t in texts:
                    f.write(json.dumps(t, ensure_ascii=False) + '\n')
            subprocess.run([a.tiktoken_python, '-I', src, found[0], inp, out], check=True)
            got = json.load(open(out))
        todo.append(('GPT-4o (byte-level BPE, 200k)', found[0], Precomputed(texts, got['rows'], got['vocab'])))
    rows = []
    for name, path, tok in todo:
        s = T.tokenizer_stats(path, texts, nlp, tok=tok)
        s.update(name=name, path=os.path.relpath(path, ROOT), sha256=sha256(path))
        rows.append(s)
        print(f'  {name:32} vocab {s["vocab_size"]:>7,}  {s["chars_per_token"]:5.2f} chars/token  {s["tokens_per_word"]:4.2f} tokens/word  '
              f'split {100 * s["words_split"]:5.1f}%  ending| {100 * s["ending_boundary_all"]:5.1f}%  '
              f'cut| {100 * s["cut_boundary_all"]:5.1f}%  own {100 * s["cut_own_token_all"]:5.1f}%  '
              f'(case-bearing cut| {100 * s["cut_boundary_case"]:5.1f}%)', flush=True)
    print(f'\n  cut| — a token boundary at the endings rule\'s cut, among {rows[0]["cut_words_all"]:,} words it cuts; '
          f'ending| — at the stem|ending boundary, among {rows[0]["inflected_all"]:,} inflected words; own — the ending a token of its own')
    rec.add('paragraphs', len(texts))
    rec.add('tokenizers', rows)
    rec.write()


if __name__ == '__main__':
    main()
