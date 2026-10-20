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
  parts      a journey part's "Covers:" range, or a range in the journey index's
             part table, that is not the part's first and last numbered heading
  sections   in docs/record/results.md, a bare §0 or §3–§5 citation that is not a
             heading there, or a bare §N.M (N ≥ 6) that is not a journey section;
             a bare §1, §2 or §6–§9 (sections moved out on 1 October) passes only
             on a line that says so ("moved", "→", "never in this file", ...)
  records    a record name (<stem>_YYYYMMDDTHHMMSS, or "…THHMMSS" after one) that
             no file under results/audits/ holds; in a record, a 12-hex run id
             found neither in results/, results_invalid/ nor as a twin under
             results/audits/
  tally      docs/predictions.md's tally that its table's verdicts, counted by
             their first words and the key under the tally, do not reproduce
  scripts    a script (.py, .sh) under scripts/ named in neither scripts/README.md
             nor scripts/chains/README.md

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
JOURNEY_DIR = 'docs/record/journey'
RESULTS = 'docs/record/results.md'
PREDICTIONS = 'docs/predictions.md'
SCRIPT_INDEXES = ('scripts/README.md', 'scripts/chains/README.md')
SCRIPT_EXEMPT = ('scripts/rehearsal/offline_datasets/datasets/__init__.py',)
HEADING = re.compile(r'^#{2,4} (\d+(?:\.\d+)*)\.?\s')
SECTION_RANGE = re.compile(r'§(\d+(?:\.\d+)*)(?:\s*[–-]\s*§?(\d+(?:\.\d+)*))?')
JOURNEY_CITE = re.compile(r'journey\s+((?:§\d+(?:\.\d+)*\s*(?:[–,-]|and)?\s*)+)')
MOVED_SAID = re.compile(r'→|moved|never in this file|formerly|until 1 october', re.I)
RECORD_NAME = re.compile(r'\b([a-z][a-z0-9_]*?)_(\d{8})T(\d{6})\b')
RECORD_SHORT = re.compile(r'…T(\d{6})\b')
RUN_ID = re.compile(r'(?<![0-9A-Za-z])(?=[0-9]*[a-f])[0-9a-f]{12}(?![0-9A-Za-z])')
VERDICTS = ('right', 'half right', 'wrong', 'never tested', 'withdrawn', 'pending')


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


def numbered_headings(path):
    return [m.group(1) for m in map(HEADING.match, open(path, encoding='utf-8')) if m]


def check_parts(errors):
    """Each part's "Covers:" range and the index's part table against the part's own headings."""
    def compare(where, part, text):
        r = SECTION_RANGE.search(text)
        if not r or not os.path.exists(part):
            return
        a, b = r.group(1), r.group(2) or r.group(1)
        heads = numbered_headings(part)
        first = next((h for h in heads if h.count('.') == a.count('.')), None)
        last = next((h for h in reversed(heads) if h.count('.') == b.count('.')), None)
        if not r.group(2) and re.match(r'\s*onwards?\b', text[r.end():]):
            last = b = '…'                                  # the open part: "§X onward"
        if (first, last) != (a, b):
            errors['parts'].append(f'{where}  says §{a}–§{b}; {os.path.basename(part)} runs §{first}–§{last}')
    for f in sorted(glob.glob(os.path.join(JOURNEY_DIR, '*.md'))):
        if os.path.basename(f) != 'README.md':
            m = re.search(r'\*\*Covers:\*\*\s*([^\[\n]*)', ''.join(open(f, encoding='utf-8').readlines()[:12]))
            if m:
                compare(f'{f} (Covers)', f, m.group(1))
    index = os.path.join(JOURNEY_DIR, 'README.md')
    for i, line in enumerate(open(index, encoding='utf-8') if os.path.exists(index) else [], 1):
        m = re.match(r'^\|[^|]*\|\s*\[[^\]]*\]\(([^)#]+\.md)\)\s*\|([^|]*)\|', line)
        if m:
            compare(f'{index}:{i}', os.path.join(JOURNEY_DIR, m.group(1)), m.group(2))


def check_results_sections(errors, sections):
    """A bare § in the results record: its own heading (§0, §3–§5) or a journey section (§6.x on)."""
    heads = set(numbered_headings(RESULTS))
    for i, line in enumerate(open(RESULTS, encoding='utf-8'), 1):
        cited = [m.span() for m in JOURNEY_CITE.finditer(line)]
        for m in SECTION_RANGE.finditer(line):
            if any(a <= m.start() < b for a, b in cited):
                continue
            for sec in filter(None, m.groups()):
                top = int(sec.split('.')[0])
                if top in (0, 3, 4, 5):
                    ok = sec in heads
                elif top >= 10 or (top >= 6 and '.' in sec):
                    ok = sec in sections
                else:
                    ok = False                       # §1, §2, §6–§9 moved out on 1 October
                if not ok and not MOVED_SAID.search(line):
                    errors['sections'].append(f'{RESULTS}:{i}  §{sec}')


def known_records_and_runs():
    """Record names under results/audits/ (any depth), and every run id kept anywhere."""
    names, runs = set(), set()
    for p in glob.glob('results/audits/**/*', recursive=True):
        base = os.path.basename(p)
        m = RECORD_NAME.match(base)
        if m:
            names.add(m.group(0))
        if re.fullmatch(r'[0-9a-f]{12}(\.json)?', base):
            runs.add(base[:12])                      # a twin's record or model folder
    for p in glob.glob('results/*') + glob.glob('results_invalid/*/*'):
        if re.fullmatch(r'[0-9a-f]{12}', os.path.basename(p)):
            runs.add(os.path.basename(p))
    return names, runs


def check_records(errors, path, lines, is_record, names, runs):
    for i, line in enumerate(lines, 1):
        last = None
        for m in re.finditer(f'{RECORD_NAME.pattern}|{RECORD_SHORT.pattern}', line):
            if m.group(1):
                last, name = (m.group(1), m.group(2)), m.group(0)
            elif last:
                name = f'{last[0]}_{last[1]}T{m.group(4)}'
            else:
                continue
            if name not in names:
                errors['records'].append(f'{path}:{i}  {name}: no such record under results/audits/')
        if is_record:
            for rid in RUN_ID.findall(line):
                if rid not in runs:
                    errors['records'].append(f'{path}:{i}  run {rid}: not in results/, results_invalid/ or the twins')


def check_tally(errors):
    """The tally above the predictions table against the table's own verdicts."""
    if not os.path.exists(PREDICTIONS):
        return
    text = open(PREDICTIONS, encoding='utf-8').read()
    m = re.search(r'\*\*Tally:\*\*([^\n]*?)— of (\d+)', text)
    if not m:
        errors['tally'].append(f'{PREDICTIONS}  no "**Tally:** … — of N" line')
        return
    stated = dict.fromkeys(VERDICTS, 0)
    for n, k in re.findall(r'(\d+) (half right|right|wrong|never tested|withdrawn|pending)', re.sub(r'\([^)]*\)', '', m.group(1))):
        stated[k] = int(n)
    key = {}
    for kind in ('half right', 'right'):                     # the key under the tally
        k = re.search(rf'[Cc]ounted {kind}: ((?:P\d+(?:, )?)+)', text)
        key.update(dict.fromkeys(re.findall(r'P\d+', k.group(1)) if k else [], kind))
    counted, rows = dict.fromkeys(VERDICTS, 0), 0
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split(' | ')]
        if len(cells) != 4 or not re.fullmatch(r'P\d+', cells[0]):
            continue
        out = re.sub(r'\*\([^)]*\)\*', '', cells[2]).strip().lower()
        if cells[0] in key:
            kind = key[cells[0]]
        elif out.startswith(('half right', 'partly right')):
            kind = 'half right'
        elif re.match(r'right(?! (?:on|by|in) )\b', out):
            kind = 'right'
        elif re.match(r'(?:badly |narrowly )?wrong\b', out):
            kind = 'wrong'
        elif out.startswith('not tested'):
            kind = 'never tested'
        elif out.startswith(('withdrawn', 'pending')):
            kind = out.split()[0]
        else:
            errors['tally'].append(f'{PREDICTIONS}  {cells[0]}: "{cells[2][:50]}" — put it in the key under the tally')
            continue
        counted[kind] += 1
        rows += 1
    if counted != stated or rows != int(m.group(2)):
        errors['tally'].append(f'{PREDICTIONS}  the tally says {stated}, of {m.group(2)}; '
                               f'the table gives {counted}, of {rows}')


def check_scripts_index(errors, held):
    index = ''.join(open(p, encoding='utf-8').read() for p in SCRIPT_INDEXES if os.path.exists(p))
    for f in held:
        if f.startswith('scripts/') and f.endswith(('.py', '.sh')) and f not in SCRIPT_EXEMPT:
            if not re.search(r'(?<![\w.-])' + re.escape(os.path.basename(f)) + r'(?![\w-])', index):
                errors['scripts'].append(f'{f}  is in neither script index')


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
    errors = {'links': [], 'names': [], 'journey': [], 'kinds': [], 'numbers': [],
              'parts': [], 'sections': [], 'records': [], 'tally': [], 'scripts': []}
    record_names, run_ids = known_records_and_runs()
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
        check_records(errors, p, lines, is_record, record_names, run_ids)
    check_parts(errors)
    check_results_sections(errors, sections)
    check_tally(errors)
    check_scripts_index(errors, held)
    bad = sum(len(v) for v in errors.values())
    print(f'  {len(docs)} documents, {n_links} links, {len(sections)} journey sections')
    for k, v in errors.items():
        print(f'  {k:8} {"OK" if not v else f"{len(v)} problem(s)"}')
        for e in v[:40]:
            print(f'      {e}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
