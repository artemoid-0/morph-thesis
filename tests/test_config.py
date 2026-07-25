"""Config invariants: one default per knob, and the lr_horizon guard."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphlm.config import ExperimentConfig, apply_overrides
import dataclasses


def test_lr_horizon_guard():
    cfg = ExperimentConfig()
    cfg.validate()
    bad = apply_overrides(cfg, ['train.max_steps=60000'])
    try:
        bad.validate()
    except ValueError as e:
        assert 'lr_horizon' in str(e)
        print('  lr_horizon < max_steps is refused OK')
        return
    raise AssertionError('expected lr_horizon guard to fire')


def test_run_id_includes_seed():
    a = ExperimentConfig()
    b = apply_overrides(a, ['train.seed=1'])
    assert a.run_id() != b.run_id(), 'seed must change run identity'
    print('  seed changes run_id (no silent overwrite) OK')


def test_overrides():
    cfg = ExperimentConfig()
    cfg = apply_overrides(cfg, ['data.pipeline=morph', 'train.seed=3',
                                'train.lr=0.001', 'frozen.facts=cat,dep'])
    assert cfg.data.pipeline == 'morph'
    assert cfg.train.seed == 3
    assert abs(cfg.train.lr - 0.001) < 1e-12
    assert cfg.frozen.facts == ('cat', 'dep')
    print('  dotted overrides coerce types OK')


def test_bpe_rejects_static_facts():
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=bpe', 'frozen.facts=cat'])
    try:
        cfg.validate()
    except ValueError as e:
        assert 'BPE' in str(e)
        print('  BPE + static facts refused OK')
        return
    raise AssertionError('expected BPE static-fact guard to fire')


def test_bpe_allows_context_facts():
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=bpe', 'frozen.facts=dep'])
    cfg.validate()
    print('  BPE + context facts allowed OK (the 2x3 matrix)')


if __name__ == '__main__':
    test_lr_horizon_guard()
    test_run_id_includes_seed()
    test_overrides()
    test_bpe_rejects_static_facts()
    test_bpe_allows_context_facts()
    print('test_config: PASS')
