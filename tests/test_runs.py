"""
The run records, read one way (morphlm/runs.py): loading, the machine a run ran on,
pairing by seed within one machine (FINDINGS, method point 11), curves along
training, interpolation at a level, and the mean ± standard error over seeds.

Every test builds its own records in a temporary directory, with known answers.
"""
import os
import sys
import json
import math
import argparse
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphlm import runs  # noqa: E402

GPU = {'thesis': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'second': 'NVIDIA GeForce RTX 5090 Laptop GPU',
       'old': None, 'other': 'NVIDIA A100'}


def record(label, seed, end, where='thesis', run_id=None, best=None, steps=10000, slope=0.5, **extra):
    """A run record whose BPB falls in a straight line from end + slope to `end` over `steps`."""
    log = [{'step': s, 'bits_per_byte': end + slope * (1 - s / steps)} for s in range(0, steps + 1, 1000)]
    r = {'label': label, 'seed': seed, 'run_id': run_id or f'{label[-6:]}{seed}{where[:2]}',
         'bits_per_byte': end, 'best_step': best or steps, 'max_steps': steps, 'log': log}
    if GPU[where] is not None:
        r['environment'] = {'gpu': GPU[where]}
    r.update(extra)
    return r


def write(root, rec, twin=None):
    """A record as results/<id>/result.json, or as a twin copy results/audits/<twin>/<id>.json."""
    if twin:
        d = os.path.join(root, 'audits', twin)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, rec['run_id'] + '.json')
    else:
        d = os.path.join(root, rec['run_id'])
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'result.json')
    json.dump(rec, open(p, 'w'))
    return p


def test_machine():
    """device_twins.py's rule: the 5090 is the second machine; the 4060, or no GPU named (records
    from before they named one), the thesis machine; any other GPU, neither."""
    assert runs.machine({'environment': {'gpu': GPU['second']}}) == 'second'
    assert runs.machine({'environment': {'gpu': GPU['thesis']}}) == 'thesis'
    assert runs.machine({'environment': {'gpu': None}}) == 'thesis'
    assert runs.machine({'environment': None}) == 'thesis'
    assert runs.machine({}) == 'thesis'
    assert runs.machine({'environment': {'gpu': GPU['other']}}) is None
    print('  machine: 5090 second, 4060 or none thesis, others neither OK')


def test_load_sorted_first_wins_and_twins():
    with tempfile.TemporaryDirectory() as d:
        write(d, record('cellA', 0, 1.10, run_id='bbb000000000'))
        write(d, record('cellA', 0, 1.20, run_id='aaa000000000'))       # sorts first: wins
        write(d, record('cellA', 1, 1.11, run_id='ccc000000000'))
        write(d, record('cellA', 1, 1.31, 'second', run_id='ccc000000000'), twin='device_twins_1')
        write(d, record('cellA', 2, 1.12, 'second', run_id='ddd000000000'), twin='device_calibration_1')
        os.makedirs(os.path.join(d, 'broken'))
        open(os.path.join(d, 'broken', 'result.json'), 'w').write('{not json')
        os.makedirs(os.path.join(d, 'notarun'))
        json.dump({'something': 1}, open(os.path.join(d, 'notarun', 'result.json'), 'w'))
        R = runs.load(d)
        assert set(R) == {('cellA', 0), ('cellA', 1)}, set(R)
        assert R[('cellA', 0)]['run_id'] == 'aaa000000000'
        assert R[('cellA', 0)]['_path'] == os.path.join(d, 'aaa000000000', 'result.json')
        # twins fill only keys no result.json holds
        T = runs.load(d, twins=True)
        assert T[('cellA', 1)]['bits_per_byte'] == 1.11 and T[('cellA', 2)]['bits_per_byte'] == 1.12
        # one machine's records, wherever they are
        S = runs.load(d, machine='second')
        assert set(S) == {('cellA', 1), ('cellA', 2)} and S[('cellA', 1)]['bits_per_byte'] == 1.31
        assert set(runs.load(d, machine='thesis')) == {('cellA', 0), ('cellA', 1)}
        # models beside their records; a twin's in device_twins_models/<id>
        assert runs.model_dir(R[('cellA', 0)]) == os.path.join(d, 'aaa000000000')
        assert runs.model_dir(S[('cellA', 1)]) == os.path.join(d, 'audits', 'device_twins_models', 'ccc000000000')
        # a predicate drops records before they are keyed
        K = runs.load(d, keep=lambda r: r['bits_per_byte'] < 1.15)
        assert K[('cellA', 0)]['run_id'] == 'bbb000000000'
        # the first root given wins over a later one
        with tempfile.TemporaryDirectory() as e:
            write(e, record('cellA', 0, 1.50, run_id='000000000000'))
            assert runs.load([d, e])[('cellA', 0)]['bits_per_byte'] == 1.20
            assert runs.load([e, d])[('cellA', 0)]['bits_per_byte'] == 1.50
        # by id: a result.json comes before a twin copy of the same id
        ids = runs.by_id(d, twins=True)
        assert ids['ccc000000000']['bits_per_byte'] == 1.11 and ids['ddd000000000']['bits_per_byte'] == 1.12
        assert runs.by_id(d, machine='second')['ccc000000000']['bits_per_byte'] == 1.31
    print('  load: sorted, first wins, twins and one machine, models beside records OK')


def test_load_by_key():
    with tempfile.TemporaryDirectory() as d:
        write(d, record('cellA', 0, 1.10, run_id='aaa000000000'))
        write(d, record('cellA', 0, 1.30, 'second', run_id='aaa000000000'), twin='device_twins_1')
        write(d, record('cellA', 0, 1.31, 'second', run_id='aaa000000000'), twin='device_twins_2')
        write(d, record('cellA', 0, 1.50, 'other', run_id='fff000000000'))
        K = runs.load_by_key(d)
        assert set(K) == {('cellA', 0, 10000)}, set(K)
        by = K[('cellA', 0, 10000)]
        assert [r['bits_per_byte'] for _, r in by['thesis']] == [1.10]
        assert [r['bits_per_byte'] for _, r in by['second']] == [1.30, 1.31]      # sorted path order
        assert all(p == r['_path'] for m in by.values() for p, r in m)
    print('  load_by_key: per machine, sorted, an unknown GPU left out OK')


def test_pair_within_one_machine():
    R = {('A', 0): record('A', 0, 1.0), ('B', 0): record('B', 0, 0.9),
         ('A', 1): record('A', 1, 1.0), ('B', 1): record('B', 1, 0.9, 'second'),
         ('A', 2): record('A', 2, 1.0, 'old'), ('B', 2): record('B', 2, 0.9),
         ('A', 3): record('A', 3, 1.0, 'other'), ('B', 3): record('B', 3, 0.9, 'other'),
         ('A', 4): record('A', 4, 1.0), ('C', 4): record('C', 4, 0.9)}
    pairs, left = runs.pair(R, 'A', 'B')
    assert sorted(pairs) == [0, 2, 3], pairs.keys()          # a record naming no GPU is the thesis machine's
    assert left == {1: ('thesis', 'second')}, left
    assert pairs[0] == (R[('A', 0)], R[('B', 0)])
    assert set(runs.pair(R, 'A', 'B', same_machine=False)[0]) == {0, 1, 2, 3}
    pairs, left = runs.pair(R, 'A', 'B', 'C')
    assert not pairs and not left
    assert runs.left_out({}) == ''
    assert runs.left_out({1: ('thesis', 'second')}) == 'left out, runs on two machines: seed 1 (thesis / second)'
    print('  pair: within one machine, the others named OK')


def test_curves_and_gain():
    a0, b0 = record('A', 0, 1.00), record('B', 0, 0.90)
    a1, b1 = record('A', 1, 1.02, best=8000), record('B', 1, 0.94)
    a1['log'] = a1['log'][:-1] + [{'step': 10000, 'bits_per_byte': 9.0}]   # over-fits after its best step
    R = {('A', 0): a0, ('B', 0): b0, ('A', 1): a1, ('B', 1): b1, ('A', 2): record('A', 2, 1.0),
         ('B', 2): record('B', 2, 0.9, 'second')}
    assert 0 not in runs.curve(a0) and runs.curve(a0)[10000] == 1.00
    pts, seeds, left = runs.gain_curve(R, 'A', 'B')
    assert seeds == [0, 1] and left == {2: ('thesis', 'second')}
    assert [p[0] for p in pts] == list(range(1000, 8001, 1000)), [p[0] for p in pts]   # to the earlier best step
    step, a, g, per = pts[-1]
    va = [r['log'][8]['bits_per_byte'] for r in (a0, a1)]
    vb = [r['log'][8]['bits_per_byte'] for r in (b0, b1)]
    assert per == [(va[0], va[0] - vb[0]), (va[1], va[1] - vb[1])]
    assert a == sum(va) / 2 and abs(g - ((va[0] - vb[0]) + (va[1] - vb[1])) / 2) < 1e-15
    pts, _, _ = runs.gain_curve(R, 'A', 'B', to_best=False)
    assert pts[-1][0] == 10000 and pts[-1][3][1] == (9.0, 9.0 - 0.94)
    # a step one seed lacks is dropped
    b1['log'] = [e for e in b1['log'] if e['step'] != 3000]
    assert 3000 not in [p[0] for p in runs.gain_curve(R, 'A', 'B')[0]]
    print('  curves: step 0 left out, gains to the best step, steps every seed has OK')


def test_at():
    pts = [(1.5, 0.010), (1.3, 0.020), (1.1, 0.030, 'extra'), (1.1, 0.5), (1.0, 0.040)]
    assert abs(runs.at(pts, 1.4) - 0.015) < 1e-15
    assert runs.at(pts, 1.5) == 0.010 and runs.at(pts, 1.3) == 0.020
    assert abs(runs.at(pts, 1.05) - 0.5 + (0.5 - 0.040) * 0.5) < 1e-12   # (1.1, 1.1) is no segment
    assert runs.at(pts, 1.6) is None and runs.at(pts, 0.9) is None and runs.at(pts[:1], 1.5) is None
    # the first crossing of a curve that turns back
    assert abs(runs.at([(1.2, 0.0), (1.0, 2.0), (1.2, 4.0)], 1.1) - 1.0) < 1e-12
    print('  at: linear interpolation, the first crossing, None outside OK')


def test_summary():
    s = runs.summary([0.002, None, 0.004, -0.003])
    assert s['n'] == 3 and s['positive'] == 2 and s['values'] == [0.002, 0.004, -0.003]
    assert abs(s['mean'] - 0.001) < 1e-15
    sd = math.sqrt(sum((x - 0.001) ** 2 for x in s['values']) / 2)
    assert abs(s['se'] - sd / math.sqrt(3)) < 1e-15
    assert list(s) == ['mean', 'se', 'n', 'positive', 'values']          # the shape records hold
    assert runs.summary([None]) is None and runs.summary([]) is None
    assert runs.summary([0.5])['se'] is None
    assert runs.summary([0.5], se_one=0.0)['se'] == 0.0
    assert math.isnan(runs.summary([0.5], se_one=float('nan'))['se'])
    assert runs.agree([0.1, 0.2, -0.1, 0.0]) == 2 and runs.agree([-0.1, -0.2, 0.3]) == 2
    assert runs.fmt(None) == '—'
    assert runs.fmt(runs.summary([0.0012, 0.0010], se_one=0.0)) == '+0.0011 ± 0.0001 (2/2)'
    assert runs.fmt(runs.summary([0.0012])) == '+0.0012 (1/1)'
    assert runs.fmt(runs.summary([0.0012], se_one=0.0)) == '+0.0012 ± 0.0000 (1/1)'
    assert runs.fmt(runs.summary([0.0012], se_one=0.0), scale=100, nd=1) == '+0.1 ± 0.0 (1/1)'
    print('  summary: mean ± se, the conventions at one seed, agree, fmt OK')


def test_own_baseline():
    assert runs.own_baseline('bpe_mstruct-LEARN-CAUSAL_bpe_17k') == 'bpe_baseline_bpe_17k'
    assert runs.own_baseline('bpe_mstruct_bpe_17k_st10k') == 'bpe_baseline_bpe_17k_st10k'
    assert runs.own_baseline('det_dep_Spdiwc') == 'det_baseline_Spdiwc'
    assert runs.own_baseline('bpe_dep') == 'bpe_baseline'
    print('  own_baseline OK')


def test_select():
    good = [{'causal_identity': True, 'causal_semantics': 3}]
    with tempfile.TemporaryDirectory() as d:
        write(d, record('bpe_x-LEARN-CAUSAL_t', 1, 1.0, run_id='b00000000000', frozen_blocks=good))
        write(d, record('bpe_x-LEARN-CAUSAL_t', 0, 1.0, run_id='c00000000000', frozen_blocks=good))
        write(d, record('bpe_x-LEARN-CAUSAL_t', 2, 1.0, run_id='a00000000000',
                        frozen_blocks=[{'causal_identity': True, 'causal_semantics': 2}]))
        write(d, record('bpe_baseline_t', 0, 1.0, run_id='d00000000000'))
        write(d, record('bpe_x-LEARN-CAUSAL_t', 3, 1.0, 'second', run_id='e00000000000', frozen_blocks=good),
              twin='device_twins_1')
        write(d, record('bpe_x-LEARN-CAUSAL_t', 3, 1.0, 'second', run_id='e00000000000', frozen_blocks=good),
              twin='device_twins_2')
        got = runs.select(['*x-LEARN*'], d)
        assert [(r['seed'], r['run_id']) for r in got] == [(0, 'c00000000000'), (1, 'b00000000000')], got
        assert [r['run_id'] for r in runs.select(['bpe_baseline_*', '*x-*'], d)] == \
            ['d00000000000', 'c00000000000', 'b00000000000']
        second = runs.select(['*'], d, machine='second')
        assert [r['run_id'] for r in second] == ['e00000000000']          # an id taken once
        assert runs.model_dir(second[0]).endswith(os.path.join('audits', 'device_twins_models', 'e00000000000'))
    print('  select: valid runs by pattern, sorted, one machine, an id once OK')


def test_results_flag():
    ap = runs.add_results_flag(argparse.ArgumentParser(), root=True)
    assert runs.results_of(ap.parse_args([])) == [runs.RESULTS]
    assert runs.results_of(ap.parse_args(['--roots', 'a', 'b'])) == ['a', 'b']
    assert runs.results_of(ap.parse_args(['--results-dir', 'a'])) == ['a']
    assert runs.results_of(ap.parse_args(['--results', 'a'])) == ['a']
    assert runs.results_of(ap.parse_args(['--root', 'p'])) == [os.path.join('p', 'results')]
    assert runs.RESULTS == os.path.join(runs.ROOT, 'results') and os.path.isdir(os.path.join(runs.ROOT, 'morphlm'))
    print('  --results, with --roots, --results-dir and --root OK')


def test_fit_linear():
    """The one linear probe (morphlm/eval/probing.py) that probe.py and probe_words.py fit."""
    import torch
    from morphlm.eval.probing import fit_linear
    g = torch.Generator().manual_seed(1)
    X = torch.randn(400, 6, generator=g)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).long()
    p1 = fit_linear(X[:300], y[:300], 2, epochs=200)(X[300:])
    p2 = fit_linear(X[:300], y[:300], 2, epochs=200)(X[300:])
    assert torch.equal(p1, p2), 'the probe is seeded: two fits must agree'
    assert (p1 == y[300:]).float().mean().item() > 0.9
    print('  fit_linear: deterministic, reads a linear property OK')


if __name__ == '__main__':
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_runs: PASS')
