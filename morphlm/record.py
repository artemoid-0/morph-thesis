"""
record.py — make every analysis script write its numbers to a file.

Training runs write `result.json` with full provenance; the analysis scripts
once printed a table and exited, so their numbers lived only in a terminal's
scrollback. Several are quoted in the thesis, and transcription is a risk of its
own: one figure (the alpha-coverage row) survived for weeks because nobody
could re-derive it without re-running the script.

Records land in `results/audits/<script>_<timestamp>.json` and carry the same
environment block as a training run, so an audit is as reproducible as a cell.

    from morphlm.record import AuditRecord
    rec = AuditRecord('coverage_audit', args)
    rec.add('bpe', {...})
    rec.write()
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

AUDIT_DIR = os.path.join('results', 'audits')


class AuditRecord:
    """
    Accumulates an analysis script's findings and writes them once.

    Deliberately permissive about structure: each script knows the right shape
    for its own results, and forcing a schema would either distort them or stop
    scripts adopting this at all. What IS enforced is the envelope — script
    name, arguments, environment, timestamp — so any record can be traced back
    to the invocation that produced it.
    """

    def __init__(self, script, args=None, out_dir=AUDIT_DIR):
        self.script = script
        self.out_dir = out_dir
        self.data = {
            'script': script,
            'args': ({k: v for k, v in vars(args).items()}
                     if args is not None else {}),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(
                timespec='seconds'),
            'results': {},
        }
        try:
            from .provenance import environment
            # the tagger, where the script was given one (G-74)
            self.data['environment'] = environment(getattr(args, 'spacy_model', None))
        except Exception as e:
            self.data['environment_error'] = f'{type(e).__name__}: {e}'

    def add(self, key, value):
        """Attach a named block of findings."""
        self.data['results'][key] = value
        return self

    def note(self, text):
        """A caveat that belongs with the numbers rather than in prose."""
        self.data.setdefault('notes', []).append(text)
        return self

    def write(self, path=None, verbose=True):
        if path is None:
            os.makedirs(self.out_dir, exist_ok=True)
            stamp = self.data['timestamp_utc'].replace(':', '').replace('-', '')
            path = os.path.join(self.out_dir, f'{self.script}_{stamp}.json')
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2, default=_safe)
        if verbose:
            print(f'\n  [record] {path}')
        return path


def _safe(o):
    """Numpy scalars and arrays are common here; make them JSON-able."""
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)
