#!/usr/bin/env python3
"""
check_docs.py — keep the documents honest: links, names, citations, kinds, numbers.

Checks every Markdown file the repository would hold — tracked, or new and not
ignored — except generated or data folders, and fails (exit 1) on:

  links      a relative link whose file, or whose #anchor in a Markdown file,
             does not exist
  names      a backticked file name (`x.md`, `scripts/a/b.py`, or a path inside
             morphlm/) that the repository does not hold; records may name files
             that existed then. A retired document name (research_journey.md, ...)
             is allowed only in a sentence that says it is a former name
  journey    a "journey §X" citation naming a section the journey does not have
  kinds      a document under docs/ or scripts/ without its **Kind:** line
  numbers    a "x.xxxx ± y.yyyy" figure on a summary page (README.md,
             docs/README.md, and every file given with --also) that does not
             appear verbatim in its home: docs/FINDINGS.md, docs/record/results.md
             or the journey

`--also PATH ...` adds Markdown files, or the Markdown files of folders, that the
repository does not hold (local notes): their links, names and numbers are checked the same way, and their
names may point at each other.

    python scripts/ops/check_docs.py
    python scripts/ops/check_docs.py --also notes
"""
import os
import re
import sys
import glob
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP = ('results/', 'results_invalid/', 'data/', 'exports/', 'paper/', 'packs/', 'logs/', 'external/', 'scratch/')
GENERATED = ('data/', 'results/', 'logs/', 'packs/', 'exports/', 'paper/', 'external/', 'scratch/', 'manuscript/')
RETIRED = ('research_journey.md', 'results_reference.md', 'ukrainian_plan.md', 'PAPERWORK_GUIDE.md')
HISTORICAL = re.compile(r'formerly|until 1 october|this was|keep their numbers from|archived|was one file', re.I)
SUMMARY = ('README.md', 'docs/README.md')
HOMES = ['docs/FINDINGS.md', 'docs/record/results.md'] + sorted(glob.glob(os.path.join(ROOT, 'docs/record/journey/*.md')))


def slug(h):
    return re.sub(r'[^\w\- ]', '', h.strip().lower(), flags=re.U).replace(' ', '-')


def anchors(path):
    out = set()
    for line in open(path, encoding='utf-8'):
        m = re.match(r'^#{1,6} (.+?)\s*$', line)
        if m:
            out.add(slug(m.group(1)))
    return out


def committable():
    """Every file the repository would hold: tracked, or new and not ignored."""
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and os.path.exists(os.path.join(ROOT, f))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--also', nargs='*', default=[], metavar='PATH',
                    help='folders or Markdown files the repository does not hold, checked as well (as summary pages)')
    args = ap.parse_args()
    os.chdir(ROOT)
    held = committable()
    extra = sorted({p for d in args.also
                    for p in ([d] if d.endswith('.md') else glob.glob(os.path.join(d, '**', '*.md'), recursive=True))})
    docs = sorted(p for p in held if p.endswith('.md') and not p.startswith(SKIP)) + extra
    every = {os.path.basename(p) for p in held + extra}
    if extra:       # local notes may name any file on disk, held by the repository or not
        every |= {os.path.basename(p) for p in glob.glob('**/*', recursive=True) if os.path.isfile(p)}
    every |= {os.path.basename(p) for p in glob.glob('data/**/*', recursive=True) if os.path.isfile(p)}
    sections = set()
    for f in glob.glob('docs/record/journey/*.md'):
        for line in open(f, encoding='utf-8'):
            m = re.match(r'^#{2,4} (\d+(?:\.\d+)*)\.?\s', line)
            if m:
                sections.add(m.group(1))
    home_text = ''.join(open(h if os.path.isabs(h) else os.path.join(ROOT, h), encoding='utf-8').read() for h in HOMES)
    errors = {'links': [], 'names': [], 'journey': [], 'kinds': [], 'numbers': []}
    n_links = 0
    for p in docs:
        text = open(p, encoding='utf-8').read()
        here = os.path.dirname(p)
        lines = text.splitlines()
        is_record = '**Kind:** record' in '\n'.join(lines[:15])
        in_code = False
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue
            for m in re.finditer(r'\[[^\]]*\]\(([^)\s]+)\)', line):
                target = m.group(1)
                if re.match(r'^[a-z]+:', target):
                    continue
                n_links += 1
                path, _, anchor = target.partition('#')
                dest = os.path.normpath(os.path.join(here, path)) if path else p
                if not os.path.exists(dest):
                    errors['links'].append(f'{p}:{i}  {target}')
                elif anchor and dest.endswith('.md') and anchor not in anchors(dest):
                    errors['links'].append(f'{p}:{i}  {target}  (no such anchor)')
            for m in re.finditer(r'`([\w./<>-]+\.(?:md|py|sh))`', line):
                name = m.group(1)
                if '<' in name or name.startswith(GENERATED):
                    continue
                base = os.path.basename(name)
                if base in RETIRED:
                    if not HISTORICAL.search(line):
                        errors['names'].append(f'{p}:{i}  `{name}` is retired')
                elif is_record:
                    continue                 # a record may name files that existed then
                elif '/' in name:
                    if not any(os.path.exists(c) for c in (name, os.path.join(here, name), os.path.join('morphlm', name))):
                        errors['names'].append(f'{p}:{i}  `{name}` does not exist')
                elif base not in every:
                    errors['names'].append(f'{p}:{i}  `{name}` exists nowhere')
            for m in re.finditer(r'journey\s+((?:§\d+(?:\.\d+)*\s*(?:[–,-]|and)?\s*)+)', line):
                for sec in re.findall(r'§(\d+(?:\.\d+)*)', m.group(1)):
                    if sec not in sections:
                        errors['journey'].append(f'{p}:{i}  journey §{sec}')
        if p.startswith(('docs/', 'scripts/')) and '**Kind:**' not in '\n'.join(lines[:15]):
            errors['kinds'].append(p)
        if (p.startswith(SUMMARY) or p in extra) and p not in HOMES:
            for m in re.finditer(r'[+−-]?\d\.\d{4} ± \d\.\d{4}', text):
                if m.group(0) not in home_text and m.group(0).lstrip('+−-') not in home_text:
                    errors['numbers'].append(f'{p}  {m.group(0)} appears in no home')
    bad = sum(len(v) for v in errors.values())
    print(f'  {len(docs)} documents, {n_links} links, {len(sections)} journey sections')
    for k, v in errors.items():
        print(f'  {k:8} {"OK" if not v else f"{len(v)} problem(s)"}')
        for e in v[:40]:
            print(f'      {e}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
