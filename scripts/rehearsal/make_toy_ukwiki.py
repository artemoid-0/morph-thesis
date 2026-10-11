#!/usr/bin/env python3
"""
make_toy_ukwiki.py — toy Ukrainian Wikipedia for a rehearsal, from real sentences.

The sandbox cannot reach Hugging Face, so the Ukrainian corpus is stood in for by
the Universal Dependencies treebank UD_Ukrainian-IU (GitHub, CC BY-NC-SA 4.0,
used locally and never redistributed): its sentences, grouped into documents by
the treebank's `# newdoc` markers and into paragraphs by `# newpar`, are written
as one article per line — {"title", "text"}, paragraphs joined by newlines,
exactly the shape of `wikimedia/wikipedia` rows. The offline `datasets` stand-in
serves the file when REHEARSAL_UKWIKI points at it.

A rehearsal tests the MACHINERY, never an effect: about 120k words.

    python scripts/rehearsal/make_toy_ukwiki.py /tmp/rehearsal
    python scripts/rehearsal/make_toy_ukwiki.py /tmp/rehearsal --conllu uk_iu-ud-train.conllu
"""
import os
import sys
import json
import argparse
import urllib.request

URL = ('https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-IU/'
       'master/uk_iu-ud-train.conllu')


def documents(lines):
    """Yield (doc_id, [paragraph, ...]) from CoNLL-U comment lines."""
    doc, paras, cur = None, [], []
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('# newdoc'):
            if cur:
                paras.append(' '.join(cur))
                cur = []
            if doc is not None and paras:
                yield doc, paras
            doc = line.split('=', 1)[-1].strip() or f'doc{len(paras)}'
            paras = []
        elif line.startswith('# newpar'):
            if cur:
                paras.append(' '.join(cur))
                cur = []
        elif line.startswith('# text ='):
            cur.append(line.split('=', 1)[1].strip())
    if cur:
        paras.append(' '.join(cur))
    if doc is not None and paras:
        yield doc, paras


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('work', nargs='?', default='/tmp/rehearsal')
    ap.add_argument('--conllu', default=None, help='a local copy of the treebank (default: download)')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.conllu:
        lines = open(a.conllu, encoding='utf-8').readlines()
    else:
        with urllib.request.urlopen(URL, timeout=120) as r:
            lines = r.read().decode('utf-8').splitlines()
    out = os.path.join(a.work, 'ukwiki_articles.jsonl')
    n_docs = n_words = 0
    with open(out, 'w', encoding='utf-8') as f:
        for i, (doc, paras) in enumerate(documents(lines)):
            # one "article" per treebank document; the title is what the corpus
            # loader hashes to assign a split, so it must be unique and stable
            text = '\n'.join(paras)
            f.write(json.dumps({'title': f'{doc} #{i}', 'text': text}, ensure_ascii=False) + '\n')
            n_docs += 1
            n_words += len(text.split())
    print(f'toy ukwiki: {n_docs} articles, ~{n_words:,} words -> {out}')


if __name__ == '__main__':
    main()
