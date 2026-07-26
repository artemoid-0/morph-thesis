"""
registry.py — pipeline factories.

One decorator per pipeline. Each factory takes the ExperimentConfig and returns
a ready pipeline, so callers never assemble constructor arguments by hand and
there is exactly one place that knows how a pipeline is built.
"""

from __future__ import annotations

from ..registry import register_pipeline, context_facts


def _needs_context(cfg) -> bool:
    return bool(context_facts(cfg.frozen.facts))


@register_pipeline('bpe')
def _build_bpe(cfg, verbose=True):
    from .bpe import BPEPipeline
    return BPEPipeline(tokenizer_path=cfg.data.bpe_path,
                       spacy_model=cfg.data.spacy_model,
                       want_context=_needs_context(cfg),
                       verbose=verbose)


@register_pipeline('morph')
def _build_morph(cfg, verbose=True):
    from .morpheme import MorphemePipeline
    from ..vocab.build import load_vocab
    vocab_df = load_vocab(cfg.data.vocab_csv)
    return MorphemePipeline(vocab_df, cfg.data.morpholex_path,
                            spacy_model=cfg.data.spacy_model,
                            surface=cfg.surface, verbose=verbose)


@register_pipeline('det')
def _build_det(cfg, verbose=True):
    from .deterministic import DeterministicPipeline
    from ..vocab.build import load_vocab
    vocab_df = load_vocab(cfg.data.vocab_csv)
    return DeterministicPipeline(vocab_df, cfg.data.morpholex_path,
                                 spacy_model=cfg.data.spacy_model,
                                 surface=cfg.surface, verbose=verbose)


def build_pipeline(cfg, verbose=True):
    """Instantiate the pipeline named by cfg.data.pipeline."""
    from ..registry import get_pipeline
    return get_pipeline(cfg.data.pipeline)(cfg, verbose=verbose)


def pipeline_vocab(cfg):
    """
    Return (vocab_df_or_None, vocab_size_or_None) for the configured pipeline.
    BPE has an integer vocabulary and no DataFrame; the morpheme pipelines have
    a DataFrame the static fact blocks are built from.
    """
    if cfg.data.pipeline == 'bpe':
        from .bpe import BPEPipeline
        bpe = BPEPipeline(tokenizer_path=cfg.data.bpe_path, verbose=False)
        return None, bpe.vocab_size_actual()
    from ..vocab.build import load_vocab
    return load_vocab(cfg.data.vocab_csv), None
