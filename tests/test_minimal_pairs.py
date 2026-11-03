"""
The Ukrainian minimal pairs beyond subject–verb agreement (scripts/data/make_minimal_pairs_uk.py,
journey §33.147, §33.150).

The check of v1 found two kinds of error, both the dictionary's: an adjective form that is a noun
too, and a noun whose form belongs to two lexemes. `--refine` drops both kinds; the refined table
must keep only v1's rows and none the check marked invalid, and the scorer must read both tables.
"""
import os
import sys
import csv
import importlib.util
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PAIRS = os.path.join(ROOT, 'data', 'minimal_pairs')


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(spec.origin))
    spec.loader.exec_module(m)
    return m


def _rows(name):
    return list(csv.DictReader(open(os.path.join(PAIRS, name), newline='', encoding='utf-8'), delimiter='\t',
                               quoting=csv.QUOTE_NONE))


def test_the_two_errors_are_seen():
    b = _load('make_minimal_pairs_uk', 'scripts', 'data', 'make_minimal_pairs_uk.py')
    inf = b.Inflector()
    assert inf.also_noun('черговим')                 # черговий, the person on duty
    assert not inf.also_noun('київських')
    assert inf.lexemes_differ('голів', 'gent', 'plur', 'loct')        # гол: голах; голова: головах
    assert not inf.lexemes_differ('будинку', 'gent', 'sing', 'ablt')


def test_changed_token_when_the_new_form_only_extends_the_old():
    b = _load('make_minimal_pairs_uk', 'scripts', 'data', 'make_minimal_pairs_uk.py')
    doc = [SimpleNamespace(idx=0, text='здані'), SimpleNamespace(idx=6, text='на'), SimpleNamespace(idx=9, text='брухт')]
    row = {'sen': 'здані на брухт', 'wrong_sen': 'здані на брухтом', 'word': 'брухт'}
    assert b.changed_token(doc, row) is doc[2]


def test_the_refined_table_is_v1_without_the_invalid_pairs():
    v1 = {r['item']: r for r in _rows('uk_np_pp_v1.tsv')}
    v2 = _rows('uk_np_pp_v2.tsv')
    assert v2 and all(v1.get(r['item']) == r for r in v2)
    check = _rows('uk_np_pp_v1_sample100_check.tsv')
    assert len(check) == 100 and {r['verdict'] for r in check} == {'valid', 'invalid'}
    invalid = {r['item'] for r in check if r['verdict'] == 'invalid'}
    assert invalid and not invalid & {r['item'] for r in v2}


def test_the_scorer_reads_both_tables():
    e = _load('multiblimp_eval', 'scripts', 'eval', 'multiblimp_eval.py')
    for name in ('uk_np_pp_v1.tsv', 'uk_np_pp_v2.tsv'):
        rows = e.load_pairs(os.path.join(PAIRS, name))
        assert {e.PHENOMENA[r['phenomenon']] for r in rows} == set(e.GROUPS['noun phrase']) | {'government'}


if __name__ == '__main__':
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_minimal_pairs: PASS')
