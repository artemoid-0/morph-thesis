"""
Checkpoints: a model saved by a run reloads as the SAME cell — config, label,
run id and weights.

`load_checkpoint` rebuilt the config from four of its five sections, dropping
`surface`, so a det surface ablation came back labelled as the all-on cell and
was rebuilt over the all-on stream. It now reads every section
(`ExperimentConfig.from_dict`).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: tiny cells, each differing from the defaults in a section a run can change
CELLS = [
    ['data.pipeline=det', 'surface.emit_whitespace=false', 'frozen.facts=dep'],
    ['data.pipeline=det', 'surface.emit_punctuation=false', 'surface.emit_digits=false'],
    ['data.pipeline=morph', 'surface.emit_case=false', 'frozen.facts=pos', 'frozen.trainable=true'],
    ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json', 'frozen.facts=mconst', 'frozen.trainable=true',
     'frozen.causal_identity=true', 'model.position=rope', 'model.norm=rmsnorm', 'model.mlp=swiglu'],
    ['data.pipeline=bpe', 'data.bpe_path=data/bpe_uk_17k.json', 'data.corpus=ukwiki', 'data.corpus_fraction=0.05',
     'data.spacy_model=uk_core_news_sm', 'frozen.facts=ucase', 'frozen.trainable=true', 'frozen.inject=add'],
    ['data.pipeline=bpe', 'frozen.facts=pos', 'frozen.trainable=true', 'frozen.role=target', 'train.seed=3'],
]
TINY = ['model.d_model=32', 'model.n_layers=1', 'model.n_heads=2', 'data.seq_len=16']


def test_checkpoint_round_trip_keeps_the_cell():
    """Saved and reloaded on CPU, every cell keeps its config, label, run id and weights."""
    import tempfile
    import torch
    import morphlm.pipelines.registry as reg
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.transformer import build_model
    from morphlm.training.metrics import save_checkpoint, load_checkpoint
    real = reg.pipeline_vocab
    reg.pipeline_vocab = lambda cfg: (None, 60)       # a tiny vocabulary for every pipeline
    cwd = os.getcwd()
    os.chdir(ROOT)                                    # data/bpe_17k.json for the causal cell
    try:
        with tempfile.TemporaryDirectory() as d:
            for over in CELLS:
                cfg = apply_overrides(ExperimentConfig(), over + TINY)
                cfg.validate()
                torch.manual_seed(0)
                model, _ = build_model(cfg, verbose=False)
                save_checkpoint(cfg, model, results_dir=d)
                back, cfg2 = load_checkpoint(cfg.run_id(), results_dir=d)
                assert cfg2 == cfg, (over, cfg2)
                assert cfg2.label() == cfg.label() and cfg2.run_id() == cfg.run_id(), over
                a, b = model.state_dict(), back.state_dict()
                assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), over
    finally:
        reg.pipeline_vocab = real
        os.chdir(cwd)
    print(f'  {len(CELLS)} cells saved and reloaded with config, label, run id and weights OK')


def test_recorded_surface_ablations_reload_as_themselves():
    """
    The det surface ablations that kept their weights reload under their own
    label — before the fix, as det_baseline_Spdiwc. Skipped where the records
    or their checkpoints are absent.
    """
    import json, glob
    from morphlm.training.metrics import load_checkpoint
    found = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'results', '*', 'config.json'))):
        d = json.load(open(p))
        run = os.path.dirname(p)
        if d.get('surface') and not all(d['surface'].values()) \
                and os.path.exists(os.path.join(run, 'model.pt')) and os.path.exists(os.path.join(run, 'result.json')):
            found.append((os.path.basename(run), json.load(open(os.path.join(run, 'result.json')))['label']))
    if not found:
        print('  (skipped: no checkpointed surface ablation here)')
        return
    cwd = os.getcwd()
    os.chdir(ROOT)                                    # the vocabulary the det model is built from
    try:
        for rid, label in found:
            _, cfg = load_checkpoint(rid, results_dir=os.path.join(ROOT, 'results'))
            assert cfg.label() == label, (rid, label, cfg.label())
    finally:
        os.chdir(cwd)
    print(f'  {len(found)} checkpointed surface ablations reload under their own labels OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_checkpoint: PASS')
