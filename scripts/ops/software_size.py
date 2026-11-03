#!/usr/bin/env python3
"""
software_size.py — the software's size, from what git holds at HEAD: modules, lines of code and tests
(the text's request of 30 October, for its section on the software).

Counted over HEAD's tree, never the working tree, so an uncommitted change, a scratch file or a local
copy cannot enter the count:
  * Python modules and shell scripts, by directory: the package (`morphlm/`), the scripts
    (`scripts/`, the chains among them) and the tests (`tests/`);
  * their lines: every line; the non-blank ones; and the lines of code — non-blank, not a comment,
    and not inside a docstring (module, class or function), found with Python's own parser;
  * the tests: the files `tests/test_*.py` and the `test_` functions they define, each of which the
    file runs when it is run (README, "tests/").

A check that could fail: every Python file counted must parse, or the tool reports it and writes
nothing.

    python scripts/ops/software_size.py
"""
import io
import os
import ast
import sys
import argparse
import tokenize
import subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PARTS = ('morphlm', 'scripts', 'tests')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def tracked(suffix):
    """The files of HEAD's tree with this suffix, in the three parts."""
    return [p for p in git('ls-tree', '-r', '--name-only', 'HEAD').splitlines()
            if p.endswith(suffix) and p.split('/')[0] in PARTS]


def docstring_lines(tree):
    """The line numbers inside docstrings: a module's, each class's and each function's."""
    lines = set()
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                                                                       ast.AsyncFunctionDef))]:
        body = getattr(node, 'body', None)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def python_counts(text):
    tree = ast.parse(text)
    doc = docstring_lines(tree)
    comment_only = set()
    code_tokens = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            comment_only.add(tok.start[0])
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                              tokenize.ENDMARKER, tokenize.ENCODING):
            code_tokens.update(range(tok.start[0], tok.end[0] + 1))
    lines = text.splitlines()
    nonblank = [i for i, l in enumerate(lines, 1) if l.strip()]
    code = [i for i in nonblank if i in code_tokens and i not in doc]
    tests = sum(1 for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_'))
    return {'lines': len(lines), 'nonblank': len(nonblank), 'code': len(code),
            'comment_only': len(comment_only - code_tokens), 'docstring': len(doc), 'test_functions': tests}


def shell_counts(text):
    lines = text.splitlines()
    nonblank = [l for l in lines if l.strip()]
    code = [l for l in nonblank if not l.lstrip().startswith('#')]
    return {'lines': len(lines), 'nonblank': len(nonblank), 'code': len(code)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('software_size', argparse.Namespace(), out_dir=os.path.join(ROOT, 'results', 'audits'))
    head = git('rev-parse', '--short', 'HEAD').strip()
    by = defaultdict(lambda: defaultdict(int))
    bad = []
    for path in tracked('.py'):
        part = 'scripts/chains' if path.startswith('scripts/chains/') else path.split('/')[0]
        try:
            c = python_counts(git('show', f'HEAD:{path}'))
        except (SyntaxError, tokenize.TokenError) as e:
            bad.append((path, str(e)))
            continue
        by[part]['python_files'] += 1
        for k, v in c.items():
            by[part][f'python_{k}'] += v
        if path.startswith('tests/test_') and c['test_functions']:
            by[part]['test_files'] += 1
    for path in tracked('.sh'):
        part = 'scripts/chains' if path.startswith('scripts/chains/') else path.split('/')[0]
        c = shell_counts(git('show', f'HEAD:{path}'))
        by[part]['shell_files'] += 1
        for k, v in c.items():
            by[part][f'shell_{k}'] += v
    if bad:
        raise SystemExit(f'{len(bad)} file(s) do not parse — nothing written: {bad[:3]}')
    total = defaultdict(int)
    for part in by.values():
        for k, v in part.items():
            total[k] += v
    rows = {p: dict(v) for p, v in sorted(by.items())}
    print(f'  the software at {head} (HEAD's tree):')
    print(f'  {"part":16} {"py files":>9} {"py lines":>9} {"py code":>9} {"sh files":>9} {"sh code":>8} {"tests":>6}')
    for p, v in list(rows.items()) + [('all', dict(total))]:
        print(f'  {p:16} {v.get("python_files", 0):9,} {v.get("python_lines", 0):9,} {v.get("python_code", 0):9,} '
              f'{v.get("shell_files", 0):9,} {v.get("shell_code", 0):8,} {v.get("python_test_functions", 0):6,}')
    print(f'  {total["test_files"]} test files define {by["tests"]["python_test_functions"]} test functions')
    rec.add('head', head)
    rec.add('parts', rows)
    rec.add('total', dict(total))
    rec.add('tests', {'files': by['tests']['test_files'], 'functions': by['tests']['python_test_functions']})
    rec.write()


if __name__ == '__main__':
    main()
