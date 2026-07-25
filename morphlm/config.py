"""
config.py — the single source of truth for every knob.

Design rules (learned the hard way):
  * ONE default per knob. The old codebase had max_steps defaulting to 2000,
    40000 and 50000 in three different places, which is how a learning-rate
    schedule confound stayed invisible for a whole calibration run.
  * max_steps and lr_horizon are SEPARATE. In the old code a single field did
    both jobs, so two cells run to different lengths had different learning
    rates at the same step and were not comparable. Now the cosine schedule is
    pinned by lr_horizon while max_steps only decides when to stop.
  * Every config is hashable. run_id = hash(config), so results never collide
    and every number is traceable to what produced it.
  * data_key() hashes ONLY the fields that affect the token stream, so changing
    lr or seed never invalidates a multi-hour spaCy cache.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Optional, Tuple


# ── leaf configs ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataConfig:
    """What text goes in, and how it is chopped."""
    pipeline: str = 'bpe'                     # bpe | morph | det
    seq_len: int = 256
    max_articles_train: Optional[int] = None  # None = full corpus
    max_articles_val: Optional[int] = None
    cache_dir: str = 'data/cache'
    # asset paths
    morpholex_path: str = 'data/MorphoLEX_en.xlsx'
    vocab_csv: str = 'data/our_final_vocab.csv'
    bpe_path: str = 'data/bpe_tokenizer.json'
    spacy_model: str = 'en_core_web_sm'


@dataclass(frozen=True)
class FrozenConfig:
    """
    Which facts get frozen into the embedding.

    facts = () is the baseline control (T = 0, a plain trained embedding).

    Fact names are resolved through registry.FACT_BLOCKS. Each block declares
    its own cardinality and whether it is sourced from the vocabulary (static,
    a function of token id) or from per-position annotation (context-dependent,
    a fact a static lookup table provably cannot represent).

    Blocks only exist if requested, so the dead-dimension bug of the previous
    design (12 of 23 columns constant) cannot recur by construction.
    """
    facts: Tuple[str, ...] = ()
    dep_granularity: str = 'coarse'     # coarse | fine  (one-line switch)
    amplify: int = 1                    # Lever 2: tile the frozen block

    def spec(self) -> str:
        """
        Canonical string for the DATA side of the feature request.
        Only context facts affect the cached token stream; static facts are
        derived from the vocabulary at model-build time and cost no cache.
        Empty string when no context facts are needed -> cache key reduces to
        the historical format and existing caches stay valid.
        """
        from .registry import context_facts
        ctx = context_facts(self.facts)
        if not ctx:
            return ''
        parts = []
        for name in ctx:
            if name == 'dep':
                parts.append(f'dep:{self.dep_granularity}')
            else:
                parts.append(name)
        return '+'.join(parts)


@dataclass(frozen=True)
class ModelConfig:
    """The fixed apparatus. Identical across every cell except vocab size."""
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    tie_output: str = 'none'            # none | frozen (option C hook)

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    warmup: int = 100
    max_steps: int = 50000      # hard ceiling — when to STOP
    lr_horizon: int = 50000     # cosine denominator — PIN THIS ACROSS CELLS
    eval_every: int = 500
    patience: Optional[int] = 6
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    frozen: FrozenConfig = field(default_factory=FrozenConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ── identity ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def run_id(self) -> str:
        """Short stable hash of the WHOLE config. Results land in results/<run_id>/."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def label(self) -> str:
        """Human-readable cell name, e.g. 'morph_dep' or 'bpe_baseline'."""
        facts = '+'.join(self.frozen.facts) if self.frozen.facts else 'baseline'
        return f'{self.data.pipeline}_{facts}'

    def describe(self) -> str:
        return f'{self.label()} [seed {self.train.seed}] ({self.run_id()})'

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Fail loudly and early on configurations that cannot be compared."""
        from .registry import _load, PIPELINES, FACT_BLOCKS
        _load()   # populate the registries before reading them

        if self.data.pipeline not in PIPELINES:
            raise ValueError(f'unknown pipeline {self.data.pipeline!r}; '
                             f'choose from {sorted(PIPELINES)}')
        for f in self.frozen.facts:
            if f not in FACT_BLOCKS:
                raise ValueError(f'unknown fact {f!r}; '
                                 f'choose from {sorted(FACT_BLOCKS)}')
        if len(set(self.frozen.facts)) != len(self.frozen.facts):
            raise ValueError(f'duplicate facts in {self.frozen.facts}')
        if self.model.d_model % self.model.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.train.lr_horizon < self.train.max_steps:
            raise ValueError(
                f'lr_horizon ({self.train.lr_horizon}) < max_steps '
                f'({self.train.max_steps}): the cosine schedule would run past '
                f'its horizon. Set lr_horizon >= max_steps.')
        if self.frozen.amplify < 1:
            raise ValueError('amplify must be >= 1')
        # static facts need a vocabulary; BPE has none
        from .registry import static_facts
        if self.data.pipeline == 'bpe' and static_facts(self.frozen.facts):
            raise ValueError(
                'BPE has no morpheme vocabulary, so static facts '
                f'{static_facts(self.frozen.facts)} cannot be built for it. '
                'Context facts (dep/pos/ent) DO work for BPE via character '
                'offset alignment.')


# ── override plumbing ─────────────────────────────────────────────────────────

def _coerce(value: str, target_type):
    """Turn a CLI string into the type the dataclass field expects."""
    origin = getattr(target_type, '__origin__', None)
    # Optional[X] -> X
    if origin is not None and type(None) in getattr(target_type, '__args__', ()):
        if value.lower() in ('none', 'null', ''):
            return None
        inner = [a for a in target_type.__args__ if a is not type(None)][0]
        return _coerce(value, inner)
    if origin is tuple or target_type is tuple or origin is Tuple:
        if value.strip() == '':
            return ()
        return tuple(v.strip() for v in value.split(',') if v.strip())
    if target_type is bool:
        return value.lower() in ('1', 'true', 'yes', 'on')
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def apply_overrides(cfg: ExperimentConfig, overrides) -> ExperimentConfig:
    """
    Apply 'section.field=value' strings to a config, returning a NEW config.

        apply_overrides(cfg, ['train.seed=1', 'frozen.facts=dep',
                              'data.pipeline=morph'])

    Dotted paths keep the CLI honest: there is exactly one place each knob
    lives, and the override names it explicitly.
    """
    import dataclasses
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'override {item!r} must be section.field=value')
        path, raw = item.split('=', 1)
        parts = path.split('.')
        if len(parts) != 2:
            raise ValueError(
                f'override {item!r}: expected section.field, e.g. train.seed=1')
        section, key = parts
        if not hasattr(cfg, section):
            raise ValueError(f'unknown config section {section!r}; '
                             f'choose from data/frozen/model/train')
        sub = getattr(cfg, section)
        if not is_dataclass(sub):
            raise ValueError(f'{section!r} is not a config section')
        names = {f.name: f.type for f in fields(sub)}
        if key not in names:
            raise ValueError(f'unknown field {key!r} in section {section!r}; '
                             f'choose from {sorted(names)}')
        typ = names[key]
        # dataclass field types may arrive as strings under `from __future__`
        if isinstance(typ, str):
            typ = {'int': int, 'float': float, 'bool': bool, 'str': str}.get(
                typ.replace('Optional[', '').replace(']', ''), str)
            if 'Tuple' in names[key] or 'tuple' in names[key]:
                typ = tuple
            if 'Optional' in names[key]:
                if raw.lower() in ('none', 'null', ''):
                    typ = None
        value = None if typ is None else _coerce(raw, typ)
        sub = dataclasses.replace(sub, **{key: value})
        cfg = dataclasses.replace(cfg, **{section: sub})
    return cfg
