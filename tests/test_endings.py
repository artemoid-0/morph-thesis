"""
The ending rule, and the tokenizer that never joins a Ukrainian stem to its
ending (morphlm/data/endings.py, morphlm/data/tokenization.py; journey §33.61).

The rule must read the word alone, cut only between letters, and leave every
other character exactly as it was; a tokenizer trained on marked text must be
opened only through `load_tokenizer`, encode with the same marks (every cut a
token boundary), decode to the unmarked text, report offsets into the unmarked
text, keep a word one word, and refuse to run under a rule with another
fingerprint.
"""
import os
import re
import sys
import json
import glob
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RULE = None


def rule():
    global _RULE
    if _RULE is None:
        from morphlm.data.endings import EndingRule
        _RULE = EndingRule()
    return _RULE


TEXT = ('Компанія „Порше“ — відомий виробник автомобілів. Вона продала 300 машин у 2020 році, '
        'а студентам із книгами про об’єкти м’ясо… НАТО, Google і Шевченка знають! '
        'Учителька читала дітям важливих книжок; вони зібралися біля школи в Києві.')
TRAIN = [
    'Учителька читала книгами важливих новин студентам. Студенти читають книги й книжки.',
    'Вона прочитала книгу, а вони читали книгами. Важливий день і важлива подія для школи.',
    'Компанія продала автомобілі; автомобілів було багато, а компанії раділи. Об’єктами м’ясо.',
    'Учительки зібралися біля школи. Школа і школи, у школі, зі школою — нові учителі в Києві.',
] * 30


def test_rule_cuts_from_the_word_alone():
    r = rule()
    show = lambda w: (w[:r.cut(w)] + '|' + w[r.cut(w):]) if r.cut(w) else w
    want = {'книгами': 'книг|ами', 'книга': 'книг|а', 'важливих': 'важлив|их',
            'зібралися': 'зібра|лися', 'студентам': 'студент|ам', 'читаю': 'чита|ю',
            'зображення': 'зображенн|я', "об'єктами": "об'єкт|ами", 'об’єктами': 'об’єкт|ами',
            'Книгами': 'Книг|ами', 'Шевченка': 'Шевченк|а', 'синього': 'син|ього'}
    for w, s in want.items():
        assert show(w) == s, (w, show(w), s)
    for w in ('його', 'ішов', 'метро', 'НАТО', 'Google', 'роблячи', 'студент', 'машин',
              'кінь', 'м', 'книгами1', 'dog-книгами'):
        assert r.cut(w) == 0, (w, show(w))
    from morphlm.data.endings import EndingRule
    other = EndingRule()
    assert all(other.cut(w) == r.cut(w) for w in want), 'two rules disagree'
    assert other.fingerprint() == r.fingerprint()
    print('  ending rule: cuts, refusals, determinism OK')


def test_marking_changes_nothing_else():
    from morphlm.data.endings import mark_text, MARK
    marked, cuts = mark_text(TEXT, rule())
    assert marked.replace(MARK, '') == TEXT, 'marks must be the only change'
    assert len(cuts) == marked.count(MARK) > 5
    assert all(TEXT[p - 1].isalpha() and TEXT[p].isalpha() for p in cuts), 'a cut outside a word'
    assert [i - n for n, i in enumerate(m.start() for m in re.finditer(MARK, marked))] == cuts
    try:
        mark_text('a' + MARK, rule())
        raise AssertionError('text holding the mark must be refused')
    except ValueError:
        pass
    print('  marking: only marks added, only between letters OK')


def test_fingerprint_follows_the_rule():
    import morphlm.data.endings as E
    r = rule()
    fp = r.fingerprint()
    old = E.MIN_STEM
    try:
        E.MIN_STEM = 3
        r2 = E.EndingRule()
        assert r2.fingerprint() != fp, 'a changed constant must change the fingerprint'
    finally:
        E.MIN_STEM = old
    assert E.EndingRule().fingerprint() == fp
    print('  fingerprint follows the rule OK')


def _train(d, presplit=True, vocab=600):
    """A tiny tokenizer, trained as train_tokenizer trains them; its sidecar as written there."""
    from morphlm.pipelines.bpe import BPEPipeline
    from morphlm.data.endings import mark_text, RULE, MARK
    path = os.path.join(d, 'bpeend_t.json' if presplit else 'bpe_t.json')
    texts = [mark_text(t, rule())[0] for t in TRAIN] if presplit else list(TRAIN)
    BPEPipeline(tokenizer_path=None, vocab_size=vocab, verbose=False).train(
        iter(texts), save_path=path, presplit=rule() if presplit else None)
    meta = {'word_rule': 'decoded', 'corpus': 'ukwiki'}
    if presplit:
        meta.update({'presplit': RULE, 'presplit_fingerprint': rule().fingerprint(),
                     'presplit_mark': f'U+{ord(MARK):04X}'})
    json.dump(meta, open(path[:-5] + '.meta.json', 'w'))
    return path


def test_ending_tokenizer_encodes_every_cut():
    import numpy as np
    from morphlm.data.tokenization import load_tokenizer
    from morphlm.data.endings import EndingTokenizer, mark_text
    from morphlm.features.lexical import tokenizer_tables, word_spans
    with tempfile.TemporaryDirectory() as d:
        path, plain_path = _train(d), _train(d, presplit=False)
        tok = load_tokenizer(path)
        assert isinstance(tok, EndingTokenizer), 'a presplit tokenizer must come wrapped'
        assert not isinstance(load_tokenizer(plain_path), EndingTokenizer)
        assert not isinstance(load_tokenizer(path, vocab_only=True), EndingTokenizer)
        tables = tokenizer_tables(path)
        for text in (TEXT, TRAIN[0], 'Книгами! А студентам — так.'):
            enc = tok.encode(text)
            assert tok.decode(enc.ids) == text, 'lossless'
            marked, cuts = mark_text(text, rule())
            raw = tok.inner.encode(marked)
            assert enc.ids == raw.ids
            starts = {a for a, _ in enc.offsets}
            missed = [p for p in cuts if p not in starts]
            assert not missed, f'cuts not at a token boundary: {[text[p - 3:p + 3] for p in missed]}'
            for (a, b), (ra, rb) in zip(enc.offsets, raw.offsets):
                assert text[a:b] == marked[ra:rb], 'offsets must point into the unmarked text'
            # one word stays one word: the ending is a continuation, not a new word
            wid, _, _ = word_spans(np.array(enc.ids, dtype=np.int64), tables)
            words = [tok.decode([int(i) for i in np.array(enc.ids)[wid == w]]).strip()
                     for w in sorted(set(wid[wid >= 0].tolist()))]
            runs = re.findall(r'[^\W\d_]+', text)
            assert words == runs, (words, runs)
        plain = load_tokenizer(plain_path)
        marked, cuts = mark_text(TEXT, rule())
        hit = sum(p in {a for a, _ in plain.encode(TEXT).offsets} for p in cuts)
        print(f'  ending tokenizer: lossless, every cut a boundary ({len(cuts)} cuts; '
              f'plain BPE on the same text: {hit}), offsets, words OK')


def test_a_changed_rule_is_refused():
    from morphlm.data.tokenization import load_tokenizer
    with tempfile.TemporaryDirectory() as d:
        path = _train(d)
        meta_path = path[:-5] + '.meta.json'
        meta = json.load(open(meta_path))
        for key, bad in (('presplit_fingerprint', '0000000000000000'), ('presplit', 'uk_endings_v0')):
            json.dump({**meta, key: bad}, open(meta_path, 'w'))
            try:
                load_tokenizer(path)
                raise AssertionError(f'a tokenizer with another {key} must be refused')
            except RuntimeError:
                pass
            assert load_tokenizer(path, vocab_only=True).get_vocab_size() > 0, 'the vocabulary stays readable'
    print('  changed rule refused; vocabulary still readable OK')


def test_tokenizers_open_only_through_the_loader():
    """A presplit tokenizer opened with Tokenizer.from_file would encode unmarked text — silently."""
    allowed = os.path.join('morphlm', 'data', 'tokenization.py')
    bad = []
    for p in glob.glob(os.path.join(ROOT, 'morphlm', '**', '*.py'), recursive=True) + \
            glob.glob(os.path.join(ROOT, 'scripts', '**', '*.py'), recursive=True):
        rel = os.path.relpath(p, ROOT)
        if rel != allowed and 'Tokenizer.from_file(' in open(p, encoding='utf-8').read():
            bad.append(rel)
    assert not bad, f'open tokenizers with tokenization.load_tokenizer: {bad}'
    print('  every tokenizer opened through load_tokenizer OK')


def test_the_stream_pipeline_aligns_through_the_wrapper():
    """BPEPipeline encodes through the wrapper; with the tagger, every token aligns."""
    from morphlm.pipelines.bpe import BPEPipeline
    try:
        import spacy
        spacy.load('uk_core_news_sm')
    except Exception as e:                       # the project's machine has it; say so if not
        print(f'  [skipped: no uk_core_news_sm here: {type(e).__name__}]')
        return
    with tempfile.TemporaryDirectory() as d:
        path = _train(d)
        pipe = BPEPipeline(tokenizer_path=path, spacy_model='uk_core_news_sm', want_context=True, verbose=False)
        units = pipe.encode(TEXT)
        assert pipe.tokenizer.decode([u['token_id'] for u in units]) == TEXT
        offsets = pipe.tokenizer.encode(TEXT).offsets
        unaligned = [TEXT[a:b] for u, (a, b) in zip(units, offsets) if u['dep'] is None]
        assert all(not t.strip() for t in unaligned), f'tokens with text failed to align: {unaligned}'
    print(f'  pipeline: encodes through the wrapper; every token with text aligned '
          f'({len(unaligned)} lone spaces, as under plain BPE) OK')


def test_stats_measure_exposure_at_the_cut():
    from morphlm.data.tokenization import tokenizer_stats
    try:
        import spacy
        nlp = spacy.load('uk_core_news_sm', disable=['parser', 'ner'])
    except Exception as e:
        print(f'  [skipped: no uk_core_news_sm here: {type(e).__name__}]')
        return
    with tempfile.TemporaryDirectory() as d:
        s_end = tokenizer_stats(_train(d), [TEXT] + TRAIN[:4], nlp)
        s_bpe = tokenizer_stats(_train(d, presplit=False), [TEXT] + TRAIN[:4], nlp)
    assert s_end['cut_words_all'] > 10 and s_end['cut_boundary_all'] == 1.0, s_end
    assert s_bpe['cut_boundary_all'] < 1.0, s_bpe
    print(f'  stats: exposure at the cut {s_end["cut_boundary_all"]:.0%} (plain BPE {s_bpe["cut_boundary_all"]:.0%}) OK')



_EN = None


def en_rule():
    global _EN
    if _EN is None:
        from morphlm.data.endings import EnglishEndingRule
        _EN = EnglishEndingRule()
    return _EN


def test_english_rule_cuts_from_the_word_alone():
    """en_endings_v1: the cut after the common prefix with every reading's lemma, all readings agreeing."""
    r = en_rule()
    want = {'walks': 4, 'walked': 4, 'studies': 4, 'children': 5, 'boxes': 3, 'teachers': 7,
            'Walks': 4, 'cities': 3,
            # ambiguous or uninflected readings, names, abbreviations: never cut
            'leaves': 0, 'news': 0, 'running': 0, 'went': 0, 'Williams': 0, 'NATO': 0, 'walk': 0,
            'is': 0, "don't": 0, 'x2': 0}
    got = {w: r.cut(w) for w in want}
    assert got == want, {w: (got[w], want[w]) for w in want if got[w] != want[w]}
    print(f'  English rule: {sum(1 for v in want.values() if v)} cuts, {sum(1 for v in want.values() if not v)} words kept whole OK')


def test_english_tokenizer_round_trip_and_refusal():
    """An English bpe-endings tokenizer: every cut a token boundary, lossless, and refused under another rule."""
    import os
    import json
    import tempfile
    from morphlm.data.endings import mark_text, RULE_EN, MARK, EndingTokenizer
    from morphlm.data.tokenization import load_tokenizer
    from morphlm.pipelines.bpe import BPEPipeline
    texts = ['The teachers walked to the cities with their boxes .',
             'She walks and he studies ; the children played .',
             'Teachers study cities , and boxes hold the walks of children .'] * 20
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bpeend_en_t.json')
        BPEPipeline(tokenizer_path=None, vocab_size=400, verbose=False).train(
            iter([mark_text(t, en_rule())[0] for t in texts]), save_path=path, presplit=en_rule())
        meta = {'presplit': RULE_EN, 'presplit_fingerprint': en_rule().fingerprint(),
                'presplit_mark': f'U+{ord(MARK):04X}'}
        json.dump(meta, open(path[:-5] + '.meta.json', 'w'))
        tok = load_tokenizer(path)
        assert isinstance(tok, EndingTokenizer)
        text = 'The teachers walked to the cities .'
        enc = tok.encode(text)
        starts = {a for a, _ in enc.offsets}
        for p in mark_text(text, en_rule())[1]:
            assert p in starts, f'no token starts at the cut {p} in {text!r}'
        assert tok.decode(enc.ids) == text, (tok.decode(enc.ids), text)
        json.dump({**meta, 'presplit_fingerprint': '0000000000000000'}, open(path[:-5] + '.meta.json', 'w'))
        try:
            load_tokenizer(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError('an English tokenizer with another fingerprint must be refused')
    print('  English tokenizer: cuts are boundaries, lossless, another rule refused OK')


if __name__ == '__main__':
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_endings: PASS')
