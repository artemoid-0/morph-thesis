"""
The operations scripts a chain leans on (scripts/ops/), each on a copy or a fake cache of its own.

- refresh_exports.sh: every step runs, a failed step or a file left stale is reported, and
  exports/REFRESHED.txt is written only when the refresh is whole (on 12 and 14 October a step
  crashed, "refreshed" was printed all the same, and exports/audits.md stayed stale).
- cells_ready.py: a cell of two facts, given as the chains give it (`--facts pos,dep`, the CLI's
  syntax), is ready when its streams hold both facts; a packed stream is reported as packed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── refresh_exports.sh ────────────────────────────────────────────────────────

STEP_STUB = '''import os, sys
name = os.path.splitext(os.path.basename(__file__))[0]
out = sys.argv[sys.argv.index('--out') + 1]
if os.environ.get('FAIL_STEP') == name:
    sys.exit(1)
if os.environ.get('SKIP_WRITE') != name:
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, name + '.out'), 'w').write('drawn\\n')
'''


def _refresh_tree(d):
    """A project holding refresh_exports.sh, a stand-in for each of its steps, and an exports/
    as an earlier refresh left it. Returns (the files the steps draw, the files on NOT_DRAWN)."""
    import re
    src = open(os.path.join(ROOT, 'scripts', 'ops', 'refresh_exports.sh')).read()
    # NOT_DRAWN is tested with an entry of the test's own, whatever the script's list holds
    m = re.search(r'^NOT_DRAWN=\((.*?)^\)', src, re.M | re.S)
    assert m, 'refresh_exports.sh: its NOT_DRAWN=( … ) block not found'
    src = src[:m.end(1)] + '    exports/figures/kept_by_hand.png\n' + src[m.end(1):]
    os.makedirs(os.path.join(d, 'scripts', 'ops'))
    open(os.path.join(d, 'scripts', 'ops', 'refresh_exports.sh'), 'w').write(src)
    steps = re.findall(r'^\s*"(scripts/\S+\.py) --out (\S+?)/?"', src, re.M)
    assert len(steps) >= 4, f'refresh_exports.sh: its STEPS ("script --out dir", one a line) not found: {steps}'
    drawn = []
    for script, out in steps:
        os.makedirs(os.path.join(d, os.path.dirname(script)), exist_ok=True)
        open(os.path.join(d, script), 'w').write(STEP_STUB)
        drawn.append(os.path.join(out, os.path.splitext(os.path.basename(script))[0] + '.out'))
    block = re.search(r'^NOT_DRAWN=\((.*?)^\)', src, re.M | re.S).group(1)
    not_drawn = [ln.split()[0] for ln in block.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    assert 'exports/figures/kept_by_hand.png' in not_drawn, not_drawn
    for f in drawn + not_drawn + ['exports/REFRESHED.txt']:
        p = os.path.join(d, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write('old\n')
    return drawn, not_drawn


def _age(d, days=3):
    import time
    past = time.time() - days * 86400
    for dirpath, _, files in os.walk(os.path.join(d, 'exports')):
        for f in files:
            os.utime(os.path.join(dirpath, f), (past, past))


def _refresh(d, **env):
    import subprocess
    e = dict(os.environ, PY=sys.executable)
    e.pop('FAIL_STEP', None)
    e.pop('SKIP_WRITE', None)
    e.update(env)
    p = subprocess.run(['bash', os.path.join(d, 'scripts', 'ops', 'refresh_exports.sh')],
                       capture_output=True, text=True, env=e, cwd=d)
    return p.returncode, p.stdout + p.stderr, open(os.path.join(d, 'exports', 'REFRESHED.txt')).read()


def test_refresh_exports_writes_its_mark_only_when_whole():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        drawn, not_drawn = _refresh_tree(d)

        # every step succeeds and every file is rewritten, but for those no step draws
        _age(d)
        rc, out, mark = _refresh(d)
        assert rc == 0 and 'exports/ refreshed' in mark and 'NOT refreshed' not in out, (rc, out)
        assert not any(n in out for n in not_drawn), out
        assert all(open(os.path.join(d, f)).read() == 'drawn\n' for f in drawn)

        # a step that fails: the steps after it still run, and the mark keeps the last full refresh
        open(os.path.join(d, 'exports', 'REFRESHED.txt'), 'w').write('old\n')
        _age(d)
        failing = os.path.splitext(os.path.basename(drawn[1]))[0]
        rc, out, mark = _refresh(d, FAIL_STEP=failing)
        assert rc == 1 and 'exports/ NOT refreshed' in out and mark == 'old\n', (rc, out, mark)
        assert f'failed:        scripts/analysis/{failing}.py' in out, out
        assert f'not rewritten: {drawn[1]}' in out, out
        assert sum('not rewritten' in ln for ln in out.splitlines()) == 1, out    # the others were redrawn

        # a step that succeeds without drawing its file (its runs missing): stale, so not refreshed
        _age(d)
        skipped = os.path.splitext(os.path.basename(drawn[-1]))[0]
        rc, out, mark = _refresh(d, SKIP_WRITE=skipped)
        assert rc == 1 and 'failed:' not in out and f'not rewritten: {drawn[-1]}' in out and mark == 'old\n', out

        # a file no step draws and NOT_DRAWN does not name
        _age(d)
        open(os.path.join(d, 'exports', 'figures', 'forgotten.png'), 'w').write('old\n')
        _age(d)
        rc, out, mark = _refresh(d)
        assert rc == 1 and 'not rewritten: exports/figures/forgotten.png' in out and mark == 'old\n', out
        assert os.path.exists(os.path.join(d, 'logs', '.refresh_exports.start'))
    print('  refresh_exports.sh: REFRESHED.txt only when every step ran and every file was redrawn OK')


# ── cells_ready.py ────────────────────────────────────────────────────────────

def _fake_stream(d, facts_held, cfg):
    """A stream's train and validation ids, and one built codes file per split holding facts_held
    (validation's may differ: facts_held is {split: [fact, ...]})."""
    import json
    import numpy as np
    from morphlm.data.stream import cache_stem, codes_spec
    sfc = cfg.surface_spec()
    stems = {}
    for split, n in (('train', 2048), ('validation', 256)):
        ids = cache_stem('bpe', split, None, sfc)
        stems[split] = ids
        np.save(os.path.join(d, ids + '.npy'), np.arange(n, dtype=np.int32))
        json.dump({'num_tokens': n, 'num_bytes': n, 'num_chars': n, 'num_articles': 1},
                  open(os.path.join(d, ids + '.json'), 'w'))
        held = facts_held[split]
        build = cache_stem('bpe', split, None, f'{sfc}|{codes_spec(held, "coarse")}')
        np.save(os.path.join(d, build + '.codes.npy'), np.zeros((len(held), n), dtype=np.int16))
        json.dump({'fact_names': held, 'dep_granularity': 'coarse', 'num_tokens': n},
                  open(os.path.join(d, build + '.codes.json'), 'w'))
    return stems


def _cells_ready(d, facts):
    import subprocess
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'ops', 'cells_ready.py'),
                        '--facts', facts, '--cache-dir', d,
                        '-o', 'data.bpe_path=data/bpe_17k.json', '-o', 'train.seed=0'],
                       capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout + p.stderr


def test_cells_ready_reads_two_facts_as_the_cli_does():
    import tempfile
    from morphlm.config import ExperimentConfig, apply_overrides
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json',
                                               'frozen.facts=pos,dep'])
    assert cfg.frozen.facts == ('pos', 'dep')
    with tempfile.TemporaryDirectory() as d:
        stems = _fake_stream(d, {'train': ['dep', 'pos'], 'validation': ['dep', 'pos']}, cfg)
        rc, out = _cells_ready(d, 'pos,dep')
        assert rc == 0, out                      # before the fix: "codes of pos,dep" missing, forever
        rc, out = _cells_ready(d, 'pos')
        assert rc == 0, out
    with tempfile.TemporaryDirectory() as d:
        stems = _fake_stream(d, {'train': ['dep', 'pos'], 'validation': ['pos']}, cfg)
        rc, out = _cells_ready(d, 'pos,dep')
        assert rc == 1 and f'validation codes of dep for {stems["validation"]}' in out, out
        assert 'codes of pos' not in out and 'train codes' not in out, out
        # a packed stream is named as packed, with the command that unpacks it
        os.rename(os.path.join(d, stems['train'] + '.npy'), os.path.join(d, stems['train'] + '.npy.zst'))
        rc, out = _cells_ready(d, 'pos,dep')
        assert rc == 1 and f'train stream {stems["train"]}, packed on this machine' in out, out
        assert 'cache_pack.py unpack' in out, out
    print('  cells_ready.py: a two-fact cell (--facts pos,dep) is ready when both are held OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_ops: PASS')
