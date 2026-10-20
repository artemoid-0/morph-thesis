"""
config.py — the single source of truth for every knob.

Design rules (journey §16):
  * ONE default per knob: max_steps once had three, which hid a learning-rate
    confound for a whole calibration run.
  * max_steps and lr_horizon are SEPARATE: the cosine schedule is pinned by
    lr_horizon, max_steps only decides when to stop (bugs.md #4).
  * Every config is hashable. run_id = hash(config), so results never collide
    and every number is traceable to what produced it.
  * The cache keys (`surface_spec`, `FrozenConfig.spec`) hold ONLY what changes
    the token stream, so lr or seed never invalidates a multi-hour spaCy cache.
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
    corpus: str = 'wikitext103'           # wikitext103 | ukwiki (docs/reference/ukrainian_pipeline.md)
    corpus_fraction: float = 1.0          # share of a non-default corpus's TRAIN articles
    corpus_holdout: int = 2000            # 1 in N articles to validation, 1 in N to test


@dataclass(frozen=True)
class FrozenConfig:
    """
    Which facts get frozen into the embedding.

    facts = () is the baseline control (T = 0, a plain trained embedding).
    Fact names resolve through registry.FACT_BLOCKS; each block declares its
    cardinality and whether it is static (a function of the token id) or
    context-sourced (per position, which a static table cannot represent). A
    block exists only if requested, so a dead dimension cannot recur (bugs.md #3).

    trainable   False: a fixed identity table, the GUARANTEED signal — the thesis
                condition. True: an nn.Embedding of the same shape, the factored
                LM baseline (Alexandrescu & Kirchhoff, 2006). Same T dims, same
                codes: only fixed-or-negotiable differs (embedding.py).
    scale       the frozen factor's magnitude: a one-hot entered the residual
                stream ~4.6x louder than the trained slice, confounding
                trainability with scale; ~0.22 matches them (bugs.md #13, §21.3).
    shuffle_codes  the ABLATION CONTROL for context facts: codes permuted within
                each window — same distribution, width and parameter count, the
                alignment to the tokens gone. Separates "the linguistic content
                helps" from "any fixed structured input helps" (journey §18).
    """
    facts: Tuple[str, ...] = ()
    dep_granularity: str = 'coarse'     # coarse | fine  (one-line switch)
    amplify: int = 1                    # Lever 2: tile the frozen block
    shuffle_codes: bool = False         # ABLATION CONTROL — see above
    trainable: bool = False             # False = frozen (thesis condition)
                                        # True  = learned (2006 FNLM baseline)
    scale: float = 1.0                  # magnitude of the frozen factor
    root_width: int = 16                # dims for the ~15k-class root factor
    root_classes: int = 0               # 0 = full inventory; else coarsen to N
    causal_identity: bool = False       # see embedding.causal_shift
    struct_keep: str = 'all'            # all | morphology | names (G-25)
    inject: str = 'concat'              # concat: facts take T of d_model dims
                                        # add: facts ADDED to a full-width token
                                        # embedding — no carved slot (§33.13)
    feature_off_step: Optional[int] = None  # the feature switch: from step K on
                                        # every feature contributes zeros (the
                                        # slot becomes a zero slot); 0 = off from
                                        # the start. None = no switch (journey §33.33)
    feature_on_step: Optional[int] = None   # the reverse: features contribute
                                        # zeros until step K, then switch on
                                        # (journey §33.34); not with feature_off_step
    feature_dropout_until: Optional[int] = None  # weaning: each token's features
                                        # dropped with probability rising linearly
                                        # to 1 at step K (journey §33.47)
    role: str = 'input'                 # input | target: the fact as an auxiliary
                                        # TARGET, predicted from the final hidden
                                        # state; nothing injected (journey §33.47)
    aux_weight: float = 0.1             # the auxiliary loss's weight (role target)

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
class SurfaceConfig:
    """
    How the morpheme pipelines encode content that is not morphological: one
    flag per information channel the audits measured (journey §17, "The six
    channels"; bugs.md #9), so each one's cost can be ABLATED. Defaults are
    all-on, i.e. information-preserving; all-off is the legacy stream.

      emit_punctuation / emit_digits   dropped by the old loop: 5.32% of corpus
          bytes BPE encoded and morph did not, an ASYMMETRIC bias in Q1
      emit_inflection   MorphoLex stores lemmas, so `cats` collided with `cat`
          (0.0440 BPB); explicit inflection tokens remove the collision
      emit_case         a <cap>/<upper> marker before a word, rather than
          capitalised variants doubling the vocabulary
      emit_whitespace   18.8% of bytes, dropped by BOTH pipelines: on, both
          BPBs are true compression rates. Morph pays a token per gap where
          ByteLevel BPE folds the space into the next token, so off is the
          cleaner segmentation-only comparison
    """
    emit_punctuation: bool = True
    emit_digits: bool = True
    emit_inflection: bool = True
    emit_whitespace: bool = True
    emit_case: bool = True

    def spec(self) -> str:
        """
        Canonical string for the cache key. MUST vary with every field: these
        options change the token stream, and a stale cache hit here would mean
        training on silently wrong data.
        """
        bits = [
            'p' if self.emit_punctuation else '-',
            'd' if self.emit_digits else '-',
            'i' if self.emit_inflection else '-',
            'w' if self.emit_whitespace else '-',
            'c' if self.emit_case else '-',
        ]
        s = ''.join(bits)
        # all-off reproduces the legacy stream, so it must map to the
        # historical cache filename and keep those caches valid
        return '' if s == '-----' else f'sfc:{s}'


@dataclass(frozen=True)
class ModelConfig:
    """The fixed apparatus. Identical across every cell except vocab size."""
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    tie_output: str = 'none'            # none | frozen (option C hook)
    position: str = 'learned'           # learned | rope: rotary positions in
                                        # attention, none added at the input (§33.47)
    norm: str = 'layernorm'             # layernorm | rmsnorm (no mean subtraction) (§33.54)
    mlp: str = 'gelu'                   # gelu | swiglu (gated, same weight count) (§33.54)

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
    eval_buckets: bool = True           # per-frequency breakdown at the end


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    frozen: FrozenConfig = field(default_factory=FrozenConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ── identity ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """
        The config a recorded dict describes — a run's config.json, a checkpoint's
        'config' — with all five sections. A field the dict lacks was added after
        it was written and takes its default, which is what the run did before the
        field existed. The one exception is `surface`: a dict without it predates
        SurfaceConfig, and its stream was the legacy one, every channel off (the
        all-off spec is the historical cache key). An unknown field raises.
        """
        surface = d.get('surface')
        if surface is None:
            surface = {f.name: False for f in fields(SurfaceConfig)}
        frozen = dict(d.get('frozen') or {})
        if 'facts' in frozen:
            frozen['facts'] = tuple(frozen['facts'])
        return cls(data=DataConfig(**(d.get('data') or {})),
                   surface=SurfaceConfig(**surface),
                   frozen=FrozenConfig(**frozen),
                   model=ModelConfig(**(d.get('model') or {})),
                   train=TrainConfig(**(d.get('train') or {})))

    def run_id(self) -> str:
        """Short stable hash of the WHOLE config. Results land in results/<run_id>/."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def surface_spec(self) -> str:
        """
        Surface spec RELEVANT TO THIS PIPELINE.

        The SurfaceConfig flags drive the morpheme encoder, so they mean nothing
        for BPE — its punctuation and whitespace behaviour is fixed by the
        pre-tokenizer baked into its tokenizer file. For BPE the stream instead
        depends on WHICH tokenizer file is loaded, so that identity goes in the
        key. Retraining BPE with a ByteLevel pre-tokenizer to a new filename
        therefore invalidates its cache automatically, which is the behaviour we
        want: the old Whitespace stream and the new ByteLevel stream are
        different data.
        """
        import os as _os
        if self.data.pipeline == 'bpe':
            stem = _os.path.basename(self.data.bpe_path)
            stem = stem[:-5] if stem.endswith('.json') else stem
            base = '' if stem == 'bpe_tokenizer' else f'tok:{stem}'
            # a non-default corpus is different data under the same tokenizer
            # name; defaults add nothing, so every English key is unchanged
            corpus = getattr(self.data, 'corpus', 'wikitext103')
            if corpus != 'wikitext103':
                frac = getattr(self.data, 'corpus_fraction', 1.0)
                hold = getattr(self.data, 'corpus_holdout', 2000)
                # ':nl' — paragraphs end in a newline (G-64); streams built
                # before it glued them together and must not be reused
                base = (base + '|' if base else '') + f'corpus:{corpus}:{frac:g}:{hold}:nl'
            return base
        return self.surface.spec()

    def label(self) -> str:
        """Human-readable cell name, e.g. 'morph_dep' or 'bpe_baseline'."""
        facts = '+'.join(self.frozen.facts) if self.frozen.facts else 'baseline'
        if self.frozen.facts and self.frozen.shuffle_codes:
            facts += '-SHUF'
        if self.frozen.facts and self.frozen.trainable:
            facts += '-LEARN'
        if self.frozen.facts and self.frozen.root_classes:
            facts += f'-k{self.frozen.root_classes}'
        if self.frozen.facts and self.frozen.causal_identity:
            facts += '-CAUSAL'
        if self.frozen.facts and getattr(self.frozen, 'inject', 'concat') == 'add':
            facts += '-ADD'
        if self.frozen.facts and getattr(self.frozen, 'feature_off_step', None) is not None:
            facts += f'-OFF{self.frozen.feature_off_step}'
        if self.frozen.facts and getattr(self.frozen, 'feature_on_step', None) is not None:
            facts += f'-ON{self.frozen.feature_on_step}'
        if self.frozen.facts and getattr(self.frozen, 'feature_dropout_until', None) is not None:
            facts += f'-DROP{self.frozen.feature_dropout_until}'
        if self.frozen.facts and getattr(self.frozen, 'role', 'input') == 'target':
            facts += f'-TGT{self.frozen.aux_weight:g}'
        # tiling widens the slot: an amplified run labelled like a plain one
        # would join its cell as more seeds (journey §33.48)
        if self.frozen.facts and self.frozen.amplify != 1:
            facts += f'-x{self.frozen.amplify}'
        if self.frozen.facts and self.frozen.struct_keep != 'all':
            facts += f'-S{self.frozen.struct_keep}'
        elif self.frozen.facts and self.frozen.scale != 1.0:
            facts += f'-S{self.frozen.scale:g}'
        sfc = self.surface_spec()
        tag = ''
        if sfc.startswith('sfc:'):
            tag = '_S' + sfc[4:]
        elif sfc.startswith('tok:'):
            tag = '_' + sfc[4:]

        # NON-DEFAULT TRAINING HYPERPARAMETERS MUST APPEAR IN THE LABEL: runs are
        # grouped into cells by label, so an unlabelled lr sweep would be read
        # as seeds of one cell and fabricate a noise floor. Defaults add
        # nothing, so existing labels are untouched.
        tag = tag.split('|')[0]              # the corpus part is added cleanly below
        corpus = getattr(self.data, 'corpus', 'wikitext103')
        if corpus != 'wikitext103':
            frac = getattr(self.data, 'corpus_fraction', 1.0)
            tag += f'_{corpus}' + ('' if frac == 1.0 else f'{frac:g}')
        d = TrainConfig()
        if self.train.lr != d.lr:
            tag += f'_lr{self.train.lr:g}'
        if self.train.max_steps != d.max_steps:
            tag += f'_st{self.train.max_steps // 1000}k'
        # The model's size too: a wider model labelled like the base one would
        # be read as more seeds of the base cell (journey §33.39).
        dm = ModelConfig()
        if self.model.d_model != dm.d_model:
            tag += f'_w{self.model.d_model}'
        if self.model.n_layers != dm.n_layers:
            tag += f'_L{self.model.n_layers}'
        if self.model.n_heads != dm.n_heads:
            tag += f'_h{self.model.n_heads}'
        if getattr(self.model, 'position', 'learned') != dm.position:
            tag += f'_{self.model.position}'
        if getattr(self.model, 'norm', 'layernorm') != dm.norm:
            tag += f'_{self.model.norm}'
        if getattr(self.model, 'mlp', 'gelu') != dm.mlp:
            tag += f'_{self.model.mlp}'
        # dropout's masks are drawn on the GPU, differently on each GPU model; runs without it
        # tell the GPUs' rounding from their random draws (journey §33.105)
        if self.model.dropout != dm.dropout:
            tag += f'_do{self.model.dropout:g}'
        return f'{self.data.pipeline}_{facts}{tag}'

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
        from .data.corpora import get_corpus
        corpus = getattr(self.data, 'corpus', 'wikitext103')
        get_corpus(corpus)                                          # unknown names fail here
        if self.data.pipeline != 'bpe' and corpus != 'wikitext103':
            # the morpheme segmenters are English (MorphoLEX), and their cache
            # key carries no corpus: the stream would share the English files
            raise ValueError(f'data.pipeline={self.data.pipeline!r} reads English only; '
                             f'data.corpus={corpus!r} needs data.pipeline=bpe')
        for section, field, allowed in (
                ('model', 'position', ('learned', 'rope')), ('model', 'norm', ('layernorm', 'rmsnorm')),
                ('model', 'mlp', ('gelu', 'swiglu')),
                ('frozen', 'role', ('input', 'target')), ('frozen', 'inject', ('concat', 'add')),
                ('frozen', 'struct_keep', ('all', 'morphology', 'names')),      # embedding.keep_struct
                ('frozen', 'dep_granularity', ('coarse', 'fine'))):             # blocks.DependencyBlock
            value = getattr(getattr(self, section), field)
            if value not in allowed:
                raise ValueError(f'{section}.{field} must be one of {allowed}, not {value!r}')
        if self.frozen.struct_keep != 'all' and 'mstruct' not in self.frozen.facts:
            # keep_struct acts on the mstruct block alone: without it the cell is
            # the plain one under another label
            raise ValueError(f'frozen.struct_keep={self.frozen.struct_keep!r} needs the mstruct fact')
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

_TRUE = ('1', 'true', 'yes', 'on')
_FALSE = ('0', 'false', 'no', 'off')


def _coerce(value: str, target_type):
    """Turn a CLI string into the type the dataclass field expects (apply_overrides
    resolves the field's type first: tuple, bool, int, float or str)."""
    if target_type is tuple:
        return tuple(v.strip() for v in value.split(',') if v.strip())
    if target_type is bool:
        if value.lower() in _TRUE:
            return True
        if value.lower() in _FALSE:
            return False
        raise ValueError(f'{value!r} is not a boolean; use one of {_TRUE + _FALSE}')
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
                             f'choose from data/surface/frozen/model/train')
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
