"""
registry.py — name -> class lookup.

Adding a new pipeline or a new frozen fact is one decorator, not a chain of
edits across the codebase. Registration happens on import of the implementing
module; `_load()` triggers those imports lazily so there is no import cycle
between config, registry and the modules that register into it.
"""

from __future__ import annotations

PIPELINES = {}     # name -> factory callable(cfg) -> pipeline
FACT_BLOCKS = {}   # name -> FactBlock subclass

_loaded = False


def _load():
    """Import the modules that register themselves. Idempotent."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from .features import blocks as _blocks     # noqa: F401
    from .pipelines import registry as _pipes   # noqa: F401


def register_pipeline(name):
    def deco(fn):
        PIPELINES[name] = fn
        return fn
    return deco


def register_fact(name):
    def deco(cls):
        cls.name = name
        FACT_BLOCKS[name] = cls
        return cls
    return deco


# ── queries used by config validation and the cache key ───────────────────────

def context_facts(facts):
    """
    Subset of `facts` whose value varies per OCCURRENCE, not per token id.
    These are the ones a static lookup table provably cannot represent, and
    the only ones that change the cached token stream.
    """
    _load()
    return tuple(f for f in facts
                 if f in FACT_BLOCKS and FACT_BLOCKS[f].source == 'context')


def static_facts(facts):
    """Subset of `facts` derivable from the vocabulary alone (function of id)."""
    _load()
    return tuple(f for f in facts
                 if f in FACT_BLOCKS and FACT_BLOCKS[f].source == 'vocab')


def get_fact(name):
    _load()
    if name not in FACT_BLOCKS:
        raise KeyError(f'unknown fact {name!r}; choose from {sorted(FACT_BLOCKS)}')
    return FACT_BLOCKS[name]


def get_pipeline(name):
    _load()
    if name not in PIPELINES:
        raise KeyError(f'unknown pipeline {name!r}; choose from {sorted(PIPELINES)}')
    return PIPELINES[name]


def all_facts():
    _load()
    return dict(FACT_BLOCKS)
