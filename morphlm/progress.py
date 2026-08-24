"""
progress.py — elapsed time and ETA for the long-running loops.

Two of this project's loops run for hours (the spaCy + MorphoLex cache pass and
training), and both previously printed progress without any indication of how
much was left. That makes it impossible to tell a slow job from a hung one, and
it was the reason an earlier cache build was interrupted by hand.

The ETA is a simple overall average rather than a windowed rate: throughput here
is dominated by per-item work that does not drift much, and an overall average
is far more stable early on, when the estimate is actually needed.
"""

from __future__ import annotations

import time


def fmt_duration(seconds) -> str:
    """Compact human duration: 45s, 4m12s, 1h23m."""
    if seconds is None:
        return '?'
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f'{m}m{s:02d}s'
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f'{h}h{m:02d}m'


class Progress:
    """
    Elapsed / ETA tracker.

        p = Progress(total=1_800_000)
        ...
        print(p.status(done))       # '18% | 4m12s elapsed | eta 18m40s'

    `total` may be None (unknown), in which case only elapsed is reported.
    """

    def __init__(self, total=None):
        self.total = total if (total and total > 0) else None
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def eta(self, done) -> float:
        """Seconds remaining, or None when it cannot be estimated yet."""
        if not self.total or done <= 0:
            return None
        frac = min(done / self.total, 1.0)
        if frac <= 0:
            return None
        return self.elapsed * (1.0 - frac) / frac

    def status(self, done) -> str:
        parts = []
        if self.total:
            parts.append(f'{100.0 * min(done / self.total, 1.0):.0f}%')
        parts.append(f'{fmt_duration(self.elapsed)} elapsed')
        eta = self.eta(done)
        if eta is not None:
            parts.append(f'eta {fmt_duration(eta)}')
        return ' | '.join(parts)

    def rate(self, done) -> float:
        """Items per second so far."""
        e = self.elapsed
        return done / e if e > 0 else 0.0
