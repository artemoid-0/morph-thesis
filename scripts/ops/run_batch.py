#!/usr/bin/env python3
"""
run_batch.py — run a block of experiments unattended, safely.

WHY THIS EXISTS
---------------
Two ways of working, with different needs:

  * During the day, small blocks — run a few cells, look at the results, decide
    the next step.
  * Overnight, one long block — nobody watching, so a failure at hour one must
    not waste the other seven, and a bad config must be caught BEFORE the GPU
    is committed.

It also fixes a methodological habit. Replication seeds were being run one at a
time with a look in between. Seed 1 never depended on seed 0's result, but
seeing seed 0 alone repeatedly produced a claim that seed 1 then overturned.
Running the seeds of a cell together in one block means the claim can only be
formed once the replicates exist.

WHAT IT DOES
------------
PRE-FLIGHT, before any training, for every job:
  * parses the command with the CLI's OWN parser (`morphlm.cli.build_parser`),
    so a job validates exactly as it will run
  * builds and validates the config — a typo in `-o frozen.causl_identity`
    fails here, not six hours in
  * computes the run_id and label
  * flags jobs whose result already exists (they are skipped — see RESUME)
  * flags two jobs in the batch that would write the same run_id
  * flags a missing token cache, which would silently add a multi-hour build
The batch refuses to start if any job fails pre-flight.

RUN: jobs execute in order, each with its output teed to
`results/batch_logs/<batch>/<NN>_<run_id>.log`. A job that fails is recorded and
the batch CONTINUES to the next (`--stop-on-fail` to override).

RESUME: a job whose `results/<run_id>/result.json` already exists is skipped, so
re-running an interrupted batch picks up where it stopped. `--force` re-runs
everything.

BATCH FILE: one command per line, written as you would after
`python -m morphlm.cli`. Blank lines and `#` comments are ignored. Lines starting
with `!` are run as plain shell commands (e.g. an analysis script) and are not
pre-flighted.

    python scripts/ops/run_batch.py batches/overnight.txt --dry-run   # check only
    python scripts/ops/run_batch.py batches/overnight.txt
"""

import os
import sys
import json
import time
import shlex
import argparse
import subprocess
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_batch(path):
    jobs = []
    for n, raw in enumerate(open(path), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('!'):
            jobs.append({'kind': 'shell', 'line': n, 'cmd': line[1:].strip()})
        else:
            jobs.append({'kind': 'cli', 'line': n, 'cmd': line})
    return jobs


def preflight(jobs, results_dir, cache_dir, force):
    """Validate every job without training. Returns (ok, annotated jobs)."""
    from morphlm.cli import build_parser, _base_config
    parser = build_parser()
    seen = {}
    ok = True
    for j in jobs:
        if j['kind'] == 'shell':
            j['status'] = 'shell'
            continue
        try:
            argv = shlex.split(j['cmd'])
            args = parser.parse_args(argv)
        except SystemExit:
            j['status'] = 'BAD ARGS'
            ok = False
            continue
        except Exception as e:
            j['status'] = f'BAD ARGS: {e}'
            ok = False
            continue
        if args.command != 'run':
            j['status'] = f'{args.command} (not a training run)'
            continue
        try:
            cfg = _base_config(args)
        except Exception as e:
            j['status'] = f'BAD CONFIG: {e}'
            ok = False
            continue
        rid = cfg.run_id()
        j.update(run_id=rid, label=cfg.label(), seed=cfg.train.seed)

        if rid in seen:
            j['status'] = f'DUPLICATE of line {seen[rid]}'
            ok = False
            continue
        seen[rid] = j['line']

        if not force and os.path.exists(
                os.path.join(results_dir, rid, 'result.json')):
            j['status'] = 'exists — skip'
            j['skip'] = True
            continue

        # a missing ids cache means a silent multi-hour build inside the batch
        try:
            from morphlm.data.stream import cache_stem
            spec = cfg.surface_spec()
            stem = cache_stem(cfg.data.pipeline, 'train', None, spec)
            cached = os.path.exists(os.path.join(cache_dir, stem + '.npy'))
        except Exception:
            cached = None
        j['cached'] = cached
        j['status'] = 'ready' if cached else 'ready — CACHE MISS (+hours)'
    return ok, jobs


def show(jobs):
    print()
    print(f'  {"#":>3} {"line":>4}  {"status":30} {"seed":>4}  label')
    print('  ' + '-' * 96)
    for i, j in enumerate(jobs, 1):
        lab = j.get('label') or j['cmd'][:48]
        seed = j.get('seed', '')
        print(f'  {i:3} {j["line"]:4}  {j["status"][:30]:30} {str(seed):>4}  {lab}')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--dry-run', action='store_true',
                    help='pre-flight only; train nothing')
    ap.add_argument('--force', action='store_true',
                    help='re-run jobs whose result already exists')
    ap.add_argument('--stop-on-fail', action='store_true')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--cache-dir', default='data/cache')
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.batch))[0]
    jobs = parse_batch(args.batch)
    print(f'\nbatch {name}: {len(jobs)} job(s)')

    ok, jobs = preflight(jobs, args.results_dir, args.cache_dir, args.force)
    show(jobs)
    todo = [j for j in jobs if not j.get('skip')]
    misses = sum(1 for j in todo if j.get('cached') is False)
    print(f'  {len(todo)} to run, {len(jobs) - len(todo)} skipped'
          + (f', {misses} CACHE MISS' if misses else ''))
    if not ok:
        print('\n  PRE-FLIGHT FAILED — fix the lines above; nothing was run.')
        sys.exit(1)
    if args.dry_run:
        print('\n  dry run — pre-flight passed, nothing was trained.')
        return

    logdir = os.path.join(args.results_dir, 'batch_logs', name)
    os.makedirs(logdir, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    record = []
    for i, j in enumerate(todo, 1):
        tag = j.get('run_id') or f'shell{i:02d}'
        log = os.path.join(logdir, f'{i:02d}_{tag}.log')
        if j['kind'] == 'cli':
            cmd = [sys.executable, '-m', 'morphlm.cli'] + shlex.split(j['cmd'])
        else:
            cmd = ['bash', '-c', j['cmd']]
        print(f'\n[{i}/{len(todo)}] {j.get("label") or j["cmd"][:60]}')
        print(f'  log: {log}')
        t0 = time.time()
        with open(log, 'w') as fh:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for out in proc.stdout:
                fh.write(out)
                fh.flush()
                if 'step ' in out and '|' in out:
                    print('   ' + out.rstrip()[:110], end='\r')
            proc.wait()
        dt = time.time() - t0
        status = 'ok' if proc.returncode == 0 else f'FAILED ({proc.returncode})'
        print(f'\n  {status} in {dt/60:.1f} min')
        record.append({'line': j['line'], 'cmd': j['cmd'],
                       'run_id': j.get('run_id'), 'label': j.get('label'),
                       'status': status, 'minutes': round(dt / 60, 1),
                       'log': log})
        if proc.returncode != 0 and args.stop_on_fail:
            print('  --stop-on-fail: stopping.')
            break

    summary = os.path.join(logdir, 'summary.json')
    json.dump({'batch': name, 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(
                   timespec='seconds'),
               'jobs': record}, open(summary, 'w'), indent=2)
    print('\n' + '=' * 70)
    print(f'BATCH {name} — done')
    for r in record:
        print(f'  {r["status"]:14} {r["minutes"]:6.1f} min  '
              f'{r["run_id"] or "-":14} {r["label"] or r["cmd"][:40]}')
    failed = [r for r in record if r['status'] != 'ok']
    print(f'\n  {len(record) - len(failed)} ok, {len(failed)} failed. '
          f'Summary: {summary}')


if __name__ == '__main__':
    main()
