"""
The chain library (scripts/chains/lib/chain_v1.sh), driven as a rehearsal drives a chain: a small
chain that sources it runs in a project of its own under a temporary folder, with the stub python
of scripts/rehearsal/stub first on PATH (every call recorded, none run) and the rehearsal hooks
(FREE_GB, GPU_FREE_GB, MINUTES, GPU_WAIT_SECONDS, PLAN, END_BY, STUB_MAP, FAIL_ON). Nothing
reaches the project's results/ or logs/.

A seed's runs that are finished are not run again; a failed twin check skips the later seeds; a
piece of one seed runs once; END_BY leaves sets for a later start; too little disk stops the chain;
and the guards, the log and PLAN's account behave as the chains' own did.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import subprocess
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = 'chainlib_rehearsal_2026-01-01'

CHAIN = r'''#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh
PIECES=${PIECES-twins}
C="-o frozen.trainable=true"
null_a() { echo "--facts mconst -o data.bpe_path=data/tok_a.json $C -o train.seed=$1"; }
case_a() { echo "--facts ucase -o data.bpe_path=data/tok_a.json $C -o train.seed=$1"; }
fic()    { echo "--facts mconst -o data.bpe_path=data/$1.json $C -o train.seed=$2"; }
minutes() {
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in twins|one|pair|near) echo 5 ;; esac
}
run_piece() {
    case "$1" in
        twins) seedpiece "twins: the null and case" twins "0 1 2 3" twin:1.000000 null_a case_a ;;
        one)   seedpiece "one: a piece of one seed" one "2" twin:1.000000 null_a ;;
        pair)  seedpiece "pair: two tokenizers' nulls" pair "0 1" pair "fic tok_x" "fic tok_y" ;;
        near)  seedpiece "near: against this machine's run" near "0 1" near:1.200000 case_a ;;
        nominutes) seedpiece "nominutes: a kind minutes() does not know" nominutes "0" none null_a ;;
    esac
}
chain_main "a rehearsal of the chain library"
'''


def _args(cell, seed):
    """A cell's run arguments, as the chain above prints them."""
    c = '-o frozen.trainable=true'
    if cell == 'null_a':
        return f'--facts mconst -o data.bpe_path=data/tok_a.json {c} -o train.seed={seed}'
    if cell == 'case_a':
        return f'--facts ucase -o data.bpe_path=data/tok_a.json {c} -o train.seed={seed}'
    tok = cell.split()[1]                      # "fic tok_x"
    return f'--facts mconst -o data.bpe_path=data/{tok}.json {c} -o train.seed={seed}'


def _project(d):
    """A project holding the library, the guard, the stub python and the chain above."""
    for rel in ('scripts/chains/lib/chain_v1.sh', 'scripts/chains/guard.sh', 'scripts/rehearsal/stub/python'):
        os.makedirs(os.path.join(d, os.path.dirname(rel)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(d, rel))
    chain = os.path.join(d, 'scripts', 'chains', NAME + '.sh')
    open(chain, 'w').write(CHAIN)
    return chain


# every knob and hook the library or the stub reads: none is inherited from the caller's shell
KNOBS = ('PIECES', 'END_BY', 'PLAN', 'MIN_FREE_GB', 'MIN_GPU_GB', 'TWIN_TOL', 'TWIN_WHAT', 'SANITY_TOL',
         'WAIT_RUNS', 'TABLES', 'RUN', 'LEFT_ONLY', 'MINUTES', 'FREE_GB', 'GPU_FREE_GB', 'GPU_WAIT_SECONDS',
         'STUB_LOG', 'STUB_MAP', 'FAIL_ON', 'STUB_STDOUT_ON', 'STUB_STDOUT')


def rehearse(pieces, bpb=None, fail_on=(), hold_lock=False, **env):
    """Run the chain once in a fresh project. bpb maps (cell, seed) to the BPB its finished run
    reports; fail_on lists the stub's FAIL_ON patterns. Returns (exit code, output, the RUN calls,
    the project's log text or None)."""
    with tempfile.TemporaryDirectory() as d:
        chain = _project(d)
        calls, stub_map = os.path.join(d, 'calls.txt'), os.path.join(d, 'map.tsv')
        open(calls, 'w').close()
        with open(stub_map, 'w') as f:
            for (cell, seed), value in (bpb or {}).items():
                f.write(f'--bpb {_args(cell, seed)}\t{value}\n')
        e = {k: v for k, v in os.environ.items() if k not in KNOBS}
        e.update(PATH=os.path.join(d, 'scripts', 'rehearsal', 'stub') + os.pathsep + e.get('PATH', ''),
                 STUB_LOG=calls, STUB_MAP=stub_map, FAIL_ON='|'.join(fail_on), PIECES=pieces,
                 WAIT_RUNS='0', FREE_GB='50', GPU_FREE_GB='20', GPU_WAIT_SECONDS='1')
        e.update({k: str(v) for k, v in env.items()})
        resolved = subprocess.run(['bash', '-c', 'command -v python'], env=e, capture_output=True, text=True)
        assert resolved.stdout.strip() == os.path.join(d, 'scripts', 'rehearsal', 'stub', 'python'), resolved
        lock = None
        if hold_lock:
            import fcntl
            os.makedirs(os.path.join(d, 'logs'), exist_ok=True)
            lock = open(os.path.join(d, 'logs', f'.{NAME}.lock'), 'w')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            p = subprocess.run(['bash', chain], env=e, capture_output=True, text=True, timeout=120)
        finally:
            if lock:
                lock.close()
        runs = [ln.strip() for ln in open(calls) if '-m morphlm.cli run' in ln]
        log = os.path.join(d, 'logs', NAME + '.log')
        return p.returncode, p.stdout + p.stderr, runs, (open(log).read() if os.path.exists(log) else None)


def _ran(runs, cell, seed):
    return sum(ln.endswith(_args(cell, seed)) for ln in runs)


PENDING = ('--done',)            # every run unfinished: the stub fails each `run_ids.py --done`
TWIN_OK = {('null_a', 0): '1.000400'}


def test_every_seed_runs_after_a_passing_twin_check():
    rc, out, runs, log = rehearse('twins', bpb=TWIN_OK, fail_on=PENDING)
    assert rc == 0 and len(runs) == 8, (len(runs), out)
    assert 'twin check passed — seeds 1 2 3' in out and 'CHAIN COMPLETE' in out, out
    assert "1.000400 against the other machine's 1.000000 — within 0.01, passes" in out, out
    assert all(_ran(runs, c, s) == 1 for c in ('null_a', 'case_a') for s in range(4)), runs
    assert runs[0] == '-m morphlm.cli run --pipeline bpe ' + _args('null_a', 0), runs[0]   # python's arguments
    assert log is not None and 'CHAIN COMPLETE' in log, 'outside PLAN the chain writes its log in logs/'
    print('  after a passing twin check every seed runs, and the log is written OK')


def test_finished_runs_are_not_run_again():
    pending = [f'--done {_args("null_a", 1)}', f'--done {_args("case_a", 1)}', f'--done {_args("case_a", 3)}']
    rc, out, runs, _ = rehearse('twins', bpb=TWIN_OK, fail_on=pending)
    assert rc == 0 and len(runs) == 3, (runs, out)
    assert _ran(runs, 'null_a', 1) == _ran(runs, 'case_a', 1) == _ran(runs, 'case_a', 3) == 1, runs
    rc, out, runs, _ = rehearse('twins', bpb=TWIN_OK)                   # everything finished
    assert rc == 0 and runs == [] and 'twin check passed' in out and 'CHAIN COMPLETE' in out, out
    print('  a seed\'s finished runs are skipped, its unfinished ones run OK')


def test_a_failed_twin_check_skips_the_later_seeds():
    rc, out, runs, _ = rehearse('twins', bpb={('null_a', 0): '1.050000'}, fail_on=PENDING)
    assert len(runs) == 2 and _ran(runs, 'null_a', 0) == _ran(runs, 'case_a', 0) == 1, (runs, out)
    assert 'NOT within 0.01, fails' in out and 'TWIN CHECK FAILED — its other seeds are skipped' in out, out
    rc, out, runs, _ = rehearse('twins', fail_on=PENDING)                # the first run reports nothing
    assert len(runs) == 2 and 'no finished run to compare' in out and 'TWIN CHECK FAILED' in out, out
    print('  a failed twin check skips the later seeds OK')


def test_a_piece_of_one_seed_runs_once():
    rc, out, runs, _ = rehearse('one', bpb={('null_a', 2): '0.999000'}, fail_on=PENDING)
    assert rc == 0 and runs == [r for r in runs if r.endswith(_args('null_a', 2))] and len(runs) == 1, (runs, out)
    assert 'twin check passed — seeds (none left)' in out, out
    rc, out, runs, _ = rehearse('one', fail_on=PENDING, PLAN=1, MINUTES=5)
    assert runs == [] and 'PLAN: 1 run(s), ~5 min of GPU' in out, out
    print('  a piece of one seed runs once (the first runners went over its seed a second time) OK')


def test_end_by_leaves_sets_for_a_later_start():
    soon = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 30 * 60))
    rc, out, runs, _ = rehearse('twins', fail_on=PENDING, MINUTES=60, END_BY=soon)
    assert rc == 0 and runs == [], (runs, out)
    assert '[left for the next chain] twins, seed 0' in out and 'and seeds 1 2 3 with it' in out, out
    assert '8 run(s) left for the next start' in out, out
    later = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 25 * 60))
    rc, out, runs, _ = rehearse('twins', fail_on=PENDING, MINUTES=10, END_BY=later, PLAN=1)
    assert runs == [] and 'PLAN: 2 run(s), ~20 min of GPU' in out and '6 run(s) left for a later start' in out, out
    rc, out, runs, _ = rehearse('twins', fail_on=PENDING, END_BY='not a date')
    assert rc == 1 and 'END_BY not understood' in out and runs == [], out
    print('  END_BY leaves the sets that would end after it for a later start OK')


def test_too_little_disk_stops_the_chain():
    rc, out, runs, _ = rehearse('twins near', bpb=TWIN_OK, fail_on=PENDING, FREE_GB=1)
    assert rc == 0 and runs == [], (runs, out)
    assert 'DISK: 1 GB free, below 5 GB — no further run starts' in out, out
    assert 'CHAIN STOPPED — too little disk' in out and 'FAILED' not in out, out
    print('  too little disk stops the chain, and no check reads the runs it did not make OK')


def test_a_busy_gpu_waits_until_end_by():
    soon = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 3))
    rc, out, runs, _ = rehearse('twins', fail_on=PENDING, GPU_FREE_GB=2, MINUTES=0, END_BY=soon)
    assert rc == 0 and runs == [] and '[gpu] 2 GB free, below 6 GB — waiting' in out, (runs, out)
    assert 'CHAIN STOPPED — the GPU stayed busy until END_BY' in out, out
    print('  a busy GPU is waited for, and the chain stops at END_BY OK')


def test_the_other_checks_and_guards():
    # a pair of cells checked against each other, and a cell against this machine's run
    pair_ok = {('fic tok_x', 0): '1.100000', ('fic tok_y', 0): '1.130000'}
    rc, out, runs, _ = rehearse('pair', bpb=pair_ok, fail_on=PENDING)
    assert len(runs) == 4 and 'sanity passed — seeds 1' in out, (runs, out)
    rc, out, runs, _ = rehearse('pair', bpb={('fic tok_x', 0): '1.10', ('fic tok_y', 0): '1.30'}, fail_on=PENDING)
    assert len(runs) == 2 and 'SANITY FAILED — its other seeds are skipped' in out, (runs, out)
    rc, out, runs, _ = rehearse('near', bpb={('case_a', 0): '1.210000'}, fail_on=PENDING)
    assert len(runs) == 2 and "against this machine's 1.200000 — within 0.05" in out, (runs, out)
    # caches not ready: the piece is skipped
    rc, out, runs, _ = rehearse('twins', bpb=TWIN_OK, fail_on=PENDING + ('cells_ready.py',))
    assert runs == [] and '[not ready]' in out and 'seed 0 is skipped (above) — its other seeds too' in out, out
    # a kind minutes() does not know is skipped, not run without a time
    rc, out, runs, _ = rehearse('nominutes', fail_on=PENDING)
    assert runs == [] and 'minutes() gives no time for nominutes' in out, out
    # PLAN runs nothing, writes no log, and gives its account
    rc, out, runs, log = rehearse('twins', bpb=TWIN_OK, fail_on=PENDING, PLAN=1, MINUTES=5)
    assert rc == 0 and runs == [] and log is None and 'PLAN: 8 run(s), ~40 min of GPU' in out, out
    # a second copy of a running chain does not start
    rc, out, runs, _ = rehearse('twins', bpb=TWIN_OK, fail_on=PENDING, hold_lock=True)
    assert rc == 1 and 'is already running — this copy does not start' in out and runs == [], out
    print('  pair and near checks, readiness, PLAN and the one-copy guard OK')


def test_the_project_is_untouched():
    for f in (NAME + '.log', f'.{NAME}.lock'):
        assert not os.path.exists(os.path.join(ROOT, 'logs', f)), f'logs/{f} was written by a rehearsal'
    assert not os.path.exists(os.path.join(ROOT, 'results', NAME)), 'results/ was written by a rehearsal'
    print('  the project\'s results/ and logs/ untouched OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_chainlib: PASS')
