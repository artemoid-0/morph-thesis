#!/usr/bin/env python3
"""
work_size.py — the work's size, from the run records: how many models were trained, the hours each
GPU spent training them, and how many tokens they were trained on (the text's request of 24 October;
journey §33.138).

Every training run leaves a record: the thesis machine's and the second machine's new runs in
results/<run id>/result.json, the second machine's repeats of the thesis machine's runs (twins) in
results/audits/device_twins_*/<run id>.json. A record names its GPU (environment.gpu), its wall time
and the steps it completed (timing); its tokens are the steps times the batch and the window, 32 × 256
in every run (tests/test_labels.py checks both in the records). Toy rehearsals and smoke tests run in
copies of the code and leave no record here; probes and other audits are not training runs.

The 25 records of September that predate the environment block are the thesis machine's: the second
machine first trained on 16 October, and every record of it names its GPU. A check that could fail:
every other record must name one of the two GPUs, with a positive wall time and step count, or it is
reported and the tool stops before writing anything.

    python scripts/ops/work_size.py
"""
import os
import sys
import glob
import json
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BATCH, WINDOW = 32, 256           # every run (tests/test_labels.py, FIXED)


THESIS, SECOND = 'thesis machine (RTX 4060 Laptop)', 'second machine (RTX 5090 Laptop)'


def machine_of(rec):
    """(machine, how it is known). A record of September without an environment block names no GPU;
    the second machine first trained on 16 October and every record of it names its GPU, so such a
    record is the thesis machine's."""
    env = rec.get('environment')
    gpu = (env or {}).get('gpu') or ''
    if '5090' in gpu:
        return SECOND, 'named'
    if '4060' in gpu:
        return THESIS, 'named'
    if not env:
        return THESIS, 'no environment recorded (September)'
    return f'unknown GPU ({gpu or "not recorded"})', 'named'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.results, '*', 'result.json')))
    twins = sorted(glob.glob(os.path.join(a.results, 'audits', 'device_twins_*', '*.json')))
    by = defaultdict(lambda: {'runs': 0, 'seconds': 0.0, 'steps': 0})
    bad, seen = [], set()
    for kind, group in (('run', paths), ('twin', twins)):
        for p in group:
            rec = json.load(open(p))
            rid = rec.get('run_id') or os.path.basename(os.path.dirname(p))
            t = rec.get('timing') or {}
            sec, steps = t.get('wall_seconds'), t.get('steps_completed')
            m, how = machine_of(rec)
            if m.startswith('unknown'):
                bad.append(f'{kind} {rid}: {m}')
                continue
            if not sec or sec <= 0 or not steps or steps <= 0:
                bad.append(f'{kind} {rid}: wall {sec}, steps {steps}')
                continue
            key = (kind, rid, m)
            if key in seen:
                continue
            seen.add(key)
            b = by[(m, kind if how == 'named' else 'run, ' + how)]
            b['runs'] += 1
            b['seconds'] += sec
            b['steps'] += steps
    if bad:
        raise SystemExit('records without a wall time or steps — nothing written:\n  ' + '\n  '.join(bad))
    rows, total = [], {'runs': 0, 'hours': 0.0, 'tokens': 0}
    for (m, kind), b in sorted(by.items()):
        tokens = b['steps'] * BATCH * WINDOW
        rows.append({'machine': m, 'records': {'twin': 'twins', 'run': 'runs'}.get(kind, kind), 'models': b['runs'],
                     'gpu_hours': round(b['seconds'] / 3600, 1), 'steps': b['steps'], 'tokens': tokens})
        total['runs'] += b['runs']; total['hours'] += b['seconds'] / 3600; total['tokens'] += tokens
    per_machine = defaultdict(lambda: {'models': 0, 'gpu_hours': 0.0, 'tokens': 0})
    for r in rows:
        pm = per_machine[r['machine']]
        pm['models'] += r['models']; pm['gpu_hours'] += r['gpu_hours']; pm['tokens'] += r['tokens']
    print(f'  {"machine":34} {"records":42} {"models":>7} {"GPU-hours":>10} {"tokens":>18}')
    for r in rows:
        print(f'  {r["machine"]:34} {r["records"]:42} {r["models"]:7d} {r["gpu_hours"]:10.1f} {r["tokens"]:18,d}')
    for m, pm in sorted(per_machine.items()):
        print(f'  {m:34} {"all":42} {pm["models"]:7d} {pm["gpu_hours"]:10.1f} {pm["tokens"]:18,d}')
    print(f'  {"in all":34} {"":42} {total["runs"]:7d} {total["hours"]:10.1f} {total["tokens"]:18,d}')
    from morphlm.record import AuditRecord
    rec = AuditRecord('work_size', a, out_dir=os.path.join(a.results, 'audits'))
    rec.add('batch_window', [BATCH, WINDOW])
    rec.add('rows', rows)
    rec.add('per_machine', {m: {k: (round(v, 1) if k == 'gpu_hours' else v) for k, v in pm.items()}
                            for m, pm in per_machine.items()})
    rec.add('total', {'models': total['runs'], 'gpu_hours': round(total['hours'], 1), 'tokens': total['tokens']})
    rec.write()


if __name__ == '__main__':
    main()
