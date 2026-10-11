#!/usr/bin/env python3
"""
prepare_ubertext_fiction.py — UberText 2.0's fiction subcorpus as documents of
paragraphs: the second Ukrainian corpus (DIRECTIONS D9; journey §33.85).

THE SOURCE. `external/ubertext/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2`
(the cleansed export, 433,084,760 bytes; lang.org.ua, Chaplynskyi 2023; novels,
prose and some poetry from two public libraries; no licence stated on the page —
used here for research only, never redistributed: `external/` is gitignored and
never packed). Downloaded 11 October 2026.

THE FORMAT, as found. Texts are separated by exactly three blank lines (25,564
runs of three; no other run but single blank lines). A text opens with its title
and one blank line. Most texts hold one paragraph per line; some are hard-wrapped
at a fixed width (the first: ~80 characters), where a line ends mid-sentence.

WHAT THIS WRITES. `external/ubertext/fiction.jsonl`, one document per line:
  title  the text's first line, when a blank line follows it ('' otherwise)
  key    title + ' | ' + the first 12 hex digits of the body's SHA-1 — unique per
         text, what the corpus's hash split reads (corpora.py, `ukfiction`)
  text   the body's paragraphs joined by '\\n'
A wrapped text — at least 20 lines, 90% of them at most 82 characters long and
half of them 55–82 — is unwrapped: lines join with a space until one ends a
paragraph (shorter than three quarters of the text's 90th-percentile line length
and ending in . ! ? … » " ” : ; or ), or followed by a line opening with a dash,
or by a blank line). Any other text keeps its lines as paragraphs — verse among
them — and drops blank lines. Nothing else is changed: no character is added,
removed or replaced inside a line.

A record goes to results/audits/ubertext_fiction_<UTC timestamp>.json.

    python scripts/data/prepare_ubertext_fiction.py
"""
import os
import sys
import bz2
import json
import hashlib
import datetime
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'external', 'ubertext', 'ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2')
OUT = os.path.join(ROOT, 'external', 'ubertext', 'fiction.jsonl')
ENDS = tuple('.!?…»"”:;)')
DASHES = ('—', '–', '-')


def documents(path):
    """Texts as lists of lines, split at runs of three or more blank lines."""
    doc, blanks = [], 0
    with bz2.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                blanks += 1
                continue
            if blanks >= 3 and doc:
                yield doc
                doc = []
            elif blanks and doc:
                doc.extend([''] * blanks)
            blanks = 0
            doc.append(line)
    if doc:
        yield doc


def wrapped(lines):
    body = [l for l in lines if l]
    if len(body) < 20:
        return False
    n = sorted(len(l) for l in body)
    p90 = n[int(0.9 * (len(n) - 1))]
    mid = sum(1 for x in n if 55 <= x <= 82) / len(n)
    return p90 <= 82 and mid >= 0.5


def paragraphs(lines):
    if not wrapped(lines):
        return [l for l in lines if l], False
    n = sorted(len(l) for l in lines if l)
    width = n[int(0.9 * (len(n) - 1))]
    out, cur = [], []
    for i, line in enumerate(lines):
        if not line:
            if cur:
                out.append(' '.join(cur)); cur = []
            continue
        cur.append(line.strip())
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        ends = (len(line) < 0.75 * width and line.rstrip().endswith(ENDS)) or not nxt \
            or nxt.lstrip().startswith(DASHES)
        if ends:
            out.append(' '.join(cur)); cur = []
    if cur:
        out.append(' '.join(cur))
    return out, True


def main():
    if not os.path.exists(SRC):
        sys.exit(f'{SRC} not found')
    if os.path.exists(OUT):
        sys.exit(f'{OUT} exists — a derived corpus is never overwritten; move it aside to rebuild')
    stats = {'documents': 0, 'titled': 0, 'wrapped': 0, 'paragraphs': 0, 'characters': 0, 'keys_repeated': 0}
    keys, sizes = set(), []
    tmp = OUT + '.part'
    with open(tmp, 'w', encoding='utf-8') as out:
        for lines in documents(SRC):
            title = lines[0] if len(lines) > 1 and lines[1] == '' else ''
            body = lines[2:] if title else lines
            paras, was_wrapped = paragraphs(body)
            if not paras:
                continue
            text = '\n'.join(paras)
            key = f'{title} | {hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]}'
            stats['keys_repeated'] += key in keys
            keys.add(key)
            out.write(json.dumps({'title': title, 'key': key, 'text': text}, ensure_ascii=False) + '\n')
            stats['documents'] += 1
            stats['titled'] += bool(title)
            stats['wrapped'] += was_wrapped
            stats['paragraphs'] += len(paras)
            stats['characters'] += len(text)
            sizes.append(len(text))
    os.replace(tmp, OUT)
    stats['bytes_written'] = os.path.getsize(OUT)
    stats['document_characters'] = {'median': st.median(sizes), 'max': max(sizes), 'min': min(sizes)}
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '+0000'
    rec = os.path.join(ROOT, 'results', 'audits', f'ubertext_fiction_{ts}.json')
    json.dump({'script': 'scripts/data/prepare_ubertext_fiction.py', 'source': os.path.relpath(SRC, ROOT),
               'source_bytes': os.path.getsize(SRC), 'output': os.path.relpath(OUT, ROOT),
               'timestamp_utc': ts, 'results': stats}, open(rec, 'w'), indent=1, ensure_ascii=False)
    print(f'  [record] {rec}')


if __name__ == '__main__':
    main()
