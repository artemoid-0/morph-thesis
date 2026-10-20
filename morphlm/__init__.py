"""
morphlm — Context-Aware Morphological Embeddings for Language Models.

Layout (grouped by rate of change):
    config.py      dataclass tree — one default per knob
    registry.py    name -> class lookup
    vocab/         STABLE   MorphoLex parsing, entity extraction, vocab build
    pipelines/     STABLE   bpe / morpheme / deterministic segmenters
    features/      VOLATILE context annotation + frozen fact blocks
    data/          cached id+code streams, windowed dataset
    modeling/      hybrid embedding + transformer
    training/      loop and metrics
    eval/          probing, per-frequency-bucket analysis
"""
__version__ = "0.2.0"
