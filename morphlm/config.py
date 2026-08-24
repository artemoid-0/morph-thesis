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

    trainable selects between the two conditions the thesis compares:

        False  factor table is a fixed identity matrix; the linguistic signal
               is GUARANTEED and no gradient can reshape it. Thesis condition.
        True   factor table is an nn.Embedding of identical shape; the model
               may reshape the factor space. This is Factored Neural Language
               Models (Alexandrescu & Kirchhoff, 2006), the direct ancestor of
               this architecture and the baseline a reviewer will ask for.

    Both occupy the same T dimensions and index the same codes, so the only
    variable is whether the signal is fixed or negotiable.

    scale exists because that claim was NOT initially true. A frozen one-hot
    has L2 norm 1.0, while the trained 120-dim slice initialises at std 0.02
    and so has norm ~0.22, and the positional embedding ~0.23. The frozen
    factor therefore entered the residual stream ~4.6x louder than the entire
    trained embedding, and pre-norm LayerNorm then squashed everything else. A
    LEARNED table initialises at std 0.02 and is balanced by construction, and
    can additionally adjust its own magnitude during training.

    So a naive frozen-vs-learned comparison confounds TRAINABILITY with SCALE.
    Setting scale to ~0.22 matches the frozen factor to the trained slice and
    isolates trainability properly. Run both before drawing a conclusion.

    shuffle_codes is the ABLATION CONTROL for context facts. It permutes the
    per-position codes within each window, preserving their exact distribution
    while destroying their alignment to the tokens. The frozen block keeps the
    same width and the model keeps the same parameter count, so the ONLY thing
    removed is the linguistic content.

        dep beats baseline           -> could be information, could be an
                                        artifact of having 8 frozen dims
        shuffled dep beats baseline  -> it was the artifact
        shuffled dep ~= baseline     -> the gain really is the linguistics

    Without this control a positive Q2 result cannot distinguish 'guaranteed
    linguistic facts help' from 'any fixed structured input helps'.
    """
    facts: Tuple[str, ...] = ()
    dep_granularity: str = 'coarse'     # coarse | fine  (one-line switch)
    amplify: int = 1                    # Lever 2: tile the frozen block
    shuffle_codes: bool = False         # ABLATION CONTROL — see below
    trainable: bool = False             # False = frozen (thesis condition)
                                        # True  = learned (2006 FNLM baseline)
    scale: float = 1.0                  # magnitude of the frozen factor
    root_width: int = 16                # dims for the ~15k-class root factor

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

    def morph_facts(self):
        """Requested facts that need the morphological analyser."""
        return tuple(f for f in self.facts if f in ('mroot', 'mstruct', 'mrole'))


@dataclass(frozen=True)
class SurfaceConfig:
    """
    How the pipeline encodes content that is not morphological.

    Every field here corresponds to a measured information channel from the
    audits (journey section 33). They are flags rather than hardcoded fixes so
    each channel's contribution can be ABLATED and reported separately — the
    audits told us the channels exist; ablation tells us what each one costs.

    Defaults are all-on, i.e. information-preserving.

      emit_punctuation / emit_digits
          The old encode loop began `if not spacy_tok.is_alpha: continue`, so
          the morpheme pipelines emitted NOTHING for punctuation, digits or
          symbols. Measured at 5.32% of corpus bytes that BPE encodes and morph
          did not, while BPB divided both by the same denominator. This was the
          single largest bias and it was ASYMMETRIC between the two arms of Q1.

      emit_inflection
          MorphoLex stores lemmas, so `cat` and `cats` both mapped to [cat] —
          a genuine collision. Measured cost 0.0440 BPB. Emitting explicit
          inflectional morphemes removes the collision and is linguistically
          more correct: section 5 already classifies inflection as morphology.

      emit_case
          _make_token lowercased every morpheme, so 'The' and 'the' produced
          the same token. BPE preserves case, making this another ASYMMETRIC
          loss. Restored with a marker token before capitalised words rather
          than by doubling the vocabulary with capitalised variants.

      emit_whitespace
          Whitespace is 18.8% of corpus bytes and BOTH pipelines dropped it, so
          it never biased Q1 — but it means neither BPB was a real compression
          rate or comparable to any published figure. With this on (and BPE
          retrained with a ByteLevel pre-tokenizer) both become genuinely
          lossless.

          NOTE the encoding schemes differ in EFFICIENCY: morph emits a
          separate boundary token per word, while ByteLevel BPE folds the space
          into the following token. Both encode the same information, so BPB is
          fair, but morph pays more tokens for it. Run with emit_whitespace off
          for the cleanest segmentation-only comparison, and on for the
          literature-comparable compression rate. Two questions, one flag.
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
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
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
            return '' if stem == 'bpe_tokenizer' else f'tok:{stem}'
        return self.surface.spec()

    def label(self) -> str:
        """Human-readable cell name, e.g. 'morph_dep' or 'bpe_baseline'."""
        facts = '+'.join(self.frozen.facts) if self.frozen.facts else 'baseline'
        if self.frozen.facts and self.frozen.shuffle_codes:
            facts += '-SHUF'
        if self.frozen.facts and self.frozen.trainable:
            facts += '-LEARN'
        elif self.frozen.facts and self.frozen.scale != 1.0:
            facts += f'-S{self.frozen.scale:g}'
        sfc = self.surface_spec()
        tag = ''
        if sfc.startswith('sfc:'):
            tag = '_S' + sfc[4:]
        elif sfc.startswith('tok:'):
            tag = '_' + sfc[4:]
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
