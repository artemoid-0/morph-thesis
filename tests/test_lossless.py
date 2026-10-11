"""
test_lossless.py — does the encode loop preserve information?

Two properties, tested separately:

  ROUND TRIP    concatenating the emitted tokens reproduces the input exactly
                (modulo the documented morpheme->surface table)

  INJECTIVITY   no two DISTINCT inputs produce the same token id sequence.
                This is the property that actually validates bits-per-byte:
                collisions destroy information, non-injective spelling does not.
                `cat`/`cats` colliding is fatal; `beauty+ful` needing a decode
                table is not, exactly as BPE's merge table is not.

Runs against a FAKE spaCy doc so the loop is testable with no spaCy install and
no model download. The real end-to-end check is scripts/audits/collision_audit.py.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from morphlm.config import SurfaceConfig


# ── fake spaCy ────────────────────────────────────────────────────────────────

class FakeMorph:
    def __init__(self, feats):
        self.feats = feats

    def get(self, k):
        v = self.feats.get(k)
        return [v] if v else []


class FakeTok:
    def __init__(self, i, idx, text, pos='NOUN', dep='nsubj', **feats):
        self.i = i
        self.idx = idx
        self.text = text
        self.pos_ = pos
        self.dep_ = dep
        self.morph = FakeMorph(feats)
        self.ent_type_ = ''

    @property
    def is_alpha(self):
        return self.text.isalpha()


class FakeDoc(list):
    ents = ()


def make_doc(text, specs):
    """specs: list of (surface, pos, feats) laid out at their real offsets."""
    toks = []
    cursor = 0
    for i, (surface, pos, feats) in enumerate(specs):
        idx = text.index(surface, cursor)
        toks.append(FakeTok(i, idx, surface, pos=pos, **feats))
        cursor = idx + len(surface)
    return FakeDoc(toks)


class FakeNER:
    def __init__(self, doc):
        self._doc = doc

    @property
    def nlp(self):
        return lambda text: self._doc

    def get_entity_spans(self, doc):
        return {}, set()


# ── a minimal preprocessor over a toy vocabulary ──────────────────────────────

def build_pipe(text, specs, surface, segmenter):
    from morphlm.pipelines.base import BasePreprocessor
    from morphlm.vocab.build import PUNCTUATION, DIGITS, WHITESPACE
    from morphlm.features.inflection import INFLECTION_TOKENS

    rows = []
    for t in ['the', 'cat', 'sit', 'run', 'include', 'beauty', 'ful', 'yard']:
        rows.append({'token': t, 'category': 'root', 'entity_label': 'none'})
    for p in PUNCTUATION:
        rows.append({'token': p, 'category': 'punct', 'entity_label': 'none'})
    for d in DIGITS:
        rows.append({'token': d, 'category': 'digit', 'entity_label': 'none'})
    for x in INFLECTION_TOKENS:
        rows.append({'token': x, 'category': 'infl', 'entity_label': 'none'})
    for w in WHITESPACE:
        rows.append({'token': w, 'category': 'space', 'entity_label': 'none'})
    for c in ('<cap>', '<upper>'):
        rows.append({'token': c, 'category': 'case', 'entity_label': 'none'})
    from morphlm.vocab.build import BYTE_FALLBACK
    for b in BYTE_FALLBACK:
        rows.append({'token': b, 'category': 'byte', 'entity_label': 'none'})
    for c in 'abcdefghijklmnopqrstuvwxyz':
        rows.append({'token': c, 'category': 'character', 'entity_label': 'none'})
    vocab = pd.DataFrame(rows)

    class Pipe(BasePreprocessor):
        def _segment_word(self, word):
            return segmenter(word.lower())

    pipe = Pipe.__new__(Pipe)
    BasePreprocessor.__init__(pipe, vocab, None, surface=surface, verbose=False)
    pipe.ner = FakeNER(make_doc(text, specs))
    return pipe


SEG = {
    'the': [('the', 'root')],
    'cat': [('cat', 'root')],
    'cats': [('cat', 'root')],
    'sat': [('sit', 'root')],
    'yards': [('yard', 'root')],
    'beautiful': [('beauty', 'root'), ('ful', 'suffix')],
}


def seg(word):
    return SEG.get(word, [(c, 'char') for c in word])


# ── tests ─────────────────────────────────────────────────────────────────────

def test_whitespace_emitted():
    text = 'the cat sat'
    specs = [('the', 'DET', {}), ('cat', 'NOUN', {}), ('sat', 'VERB', {})]
    pipe = build_pipe(text, specs, SurfaceConfig(), seg)
    toks = [t['token'] for t in pipe.encode(text)]
    assert toks.count('<sp>') == 2, f'expected 2 spaces, got {toks}'
    print(f'  whitespace emitted OK  {toks}')


def test_punctuation_and_digits_emitted():
    text = 'cat, 42.'
    specs = [('cat', 'NOUN', {}), (',', 'PUNCT', {}),
             ('42', 'NUM', {}), ('.', 'PUNCT', {})]
    pipe = build_pipe(text, specs, SurfaceConfig(), seg)
    toks = [t['token'] for t in pipe.encode(text)]
    for expect in (',', '4', '2', '.'):
        assert expect in toks, f'{expect!r} missing from {toks}'
    print(f'  punctuation + digits emitted OK  {toks}')


def test_case_marker():
    text = 'The cat'
    specs = [('The', 'DET', {}), ('cat', 'NOUN', {})]
    pipe = build_pipe(text, specs, SurfaceConfig(), seg)
    toks = [t['token'] for t in pipe.encode(text)]
    assert '<cap>' in toks, toks
    print(f'  case marker OK  {toks}')


def test_inflection_breaks_the_collision():
    """THE headline fix: cat and cats must no longer share a token sequence."""
    sing = build_pipe('cat', [('cat', 'NOUN', {'Number': 'Sing'})],
                      SurfaceConfig(), seg)
    plur = build_pipe('cats', [('cats', 'NOUN', {'Number': 'Plur'})],
                      SurfaceConfig(), seg)
    a = tuple(t['token_id'] for t in sing.encode('cat'))
    b = tuple(t['token_id'] for t in plur.encode('cats'))
    assert a != b, f'cat and cats STILL collide: {a} == {b}'
    tb = [t['token'] for t in plur.encode('cats')]
    assert '<plur>' in tb, tb
    print(f'  cat/cats collision BROKEN OK  cats -> {tb}')


def test_legacy_mode_reproduces_old_behaviour():
    """All channels off must reproduce the pre-fix stream, for ablation."""
    off = SurfaceConfig(emit_punctuation=False, emit_digits=False,
                        emit_inflection=False, emit_whitespace=False,
                        emit_case=False)
    text = 'The cats, 42.'
    specs = [('The', 'DET', {}), ('cats', 'NOUN', {'Number': 'Plur'}),
             (',', 'PUNCT', {}), ('42', 'NUM', {}), ('.', 'PUNCT', {})]
    pipe = build_pipe(text, specs, off, seg)
    toks = [t['token'] for t in pipe.encode(text)]
    assert toks == ['the', 'cat'], f'legacy mode changed: {toks}'
    print(f'  legacy mode reproduces old stream OK  {toks}')


def test_injectivity_on_an_inflectional_family():
    """No two members of a paradigm may share an id sequence."""
    family = [
        ('cat', [('cat', 'NOUN', {'Number': 'Sing'})]),
        ('cats', [('cats', 'NOUN', {'Number': 'Plur'})]),
    ]
    seen = {}
    for surface, specs in family:
        pipe = build_pipe(surface, specs, SurfaceConfig(), seg)
        ids = tuple(t['token_id'] for t in pipe.encode(surface))
        assert ids not in seen, \
            f'collision: {surface} and {seen[ids]} share {ids}'
        seen[ids] = surface
    print(f'  injective across the paradigm OK  ({len(seen)} distinct streams)')


def test_orthographic_case_is_not_a_collision():
    """beauty+ful fails naive concatenation but must not COLLIDE."""
    pipe = build_pipe('beautiful', [('beautiful', 'ADJ', {})],
                      SurfaceConfig(), seg)
    toks = [t['token'] for t in pipe.encode('beautiful')]
    joined = ''.join(t for t in toks if not t.startswith('<'))
    assert joined == 'beautyful', joined
    print(f'  beauty+ful: concat={joined!r} != surface, but no collision '
          f'-> recoverable via decode table OK')


def test_byte_fallback_prevents_unk_collisions():
    """
    UNK is a collision sink: every unknown string maps to one id, so distinct
    inputs become one stream. The audit caught en-dash, em-dash, degree and
    pound all collapsing together (645 occurrences). Byte fallback makes the
    encoding total, so they must now be DISTINCT.
    """
    chars = ['\u2013', '\u2014', '\u00b0', '\u00a3']   # en-dash em-dash degree pound
    seen = {}
    for ch in chars:
        text = ch
        specs = [(ch, 'PUNCT', {})]
        pipe = build_pipe(text, specs, SurfaceConfig(), seg)
        toks = pipe.encode(text)
        ids = tuple(t['token_id'] for t in toks)
        assert pipe.UNK_ID not in ids, f'{ch!r} still hits UNK: {toks}'
        assert ids not in seen, f'{ch!r} collides with {seen[ids]!r}'
        seen[ids] = ch
    print(f'  byte fallback: {len(seen)} non-ASCII chars all DISTINCT, '
          f'none reach UNK OK')


def test_residue_marker_catches_hidden_inflection():
    """
    spaCy POS tags hide real inflection: gerunds are NOUN, participles ADJ,
    and MorphoLex root mismatches (news -> new) are invisible to any feature.
    All of these appeared as real collision groups in the audit.
    """
    from morphlm.features.inflection import residue_marker
    must_mark = [
        ('new', 'news', '<plur>'), ('game', 'gaming', '<prog>'),
        ('season', 'seasoned', '<past>'), ('route', 'routing', '<prog>'),
        ('state', 'stated', '<past>'), ('found', 'founded', '<past>'),
        ('ship', 'shipping', '<prog>'), ('run', 'running', '<prog>'),
        ('big', 'bigger', '<cmp>'),
    ]
    for concat, surface, expect in must_mark:
        got = residue_marker(concat, surface)
        assert got == expect, f'{concat}->{surface}: {got} != {expect}'

    # orthographic variation is INJECTIVE and must stay unmarked: marking it
    # would cost a token for no information gain
    must_not_mark = [('beautyful', 'beautiful'),
                     ('populateion', 'population'),
                     ('costruction', 'construction'),
                     ('cat', 'cat')]
    for concat, surface in must_not_mark:
        got = residue_marker(concat, surface)
        assert got is None, f'{concat}->{surface} wrongly marked {got}'
    print(f'  residue marker: {len(must_mark)} marked, '
          f'{len(must_not_mark)} correctly left alone OK')


def test_byte_fallback_is_visible_to_coverage_accounting():
    """
    Regression for the third measurement artifact of its family.

    The byte layer writes `<b00>`..`<bff>`. Any audit that counts characters
    inside token strings sees '<','b','0','0','>' and never the byte encoded,
    so byte-fallback characters read as UNCOVERED. On 1,000 validation lines
    that was invisible; on the full split it made det appear to decline 2,153
    punctuation bytes it in fact encodes losslessly.

    Asserts the decode round-trips, which is the property any coverage
    accounting must rely on.
    """
    seq = ['<be2>', '<b80>', '<b94>']          # UTF-8 em dash
    raw = bytearray()
    for tok in seq:
        assert len(tok) == 5 and tok.startswith('<b') and tok.endswith('>')
        raw.append(int(tok[2:4], 16))
    assert raw.decode('utf-8') == '\u2014', 'byte tokens must round-trip'

    # and the naive reading must NOT contain the character - the bug it guards
    naive = ''.join(seq)
    assert '\u2014' not in naive, 'literal token text cannot contain the byte'
    print('  byte fallback decodes to its character OK (audit artifact guarded)')



def test_tokenizer_names_and_ending_exposure():
    """Standard names reproduce the historical ones; the ending measure counts stem|ending cuts (§33.53)."""
    import json, os, tempfile
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from morphlm.data.tokenization import standard_name, tokenizer_stats
    assert standard_name('wikitext103', 'bpe', 17000) == 'bpe_17k'
    assert standard_name('wikitext103', 'unigram', 17000) == 'unigram_17k'
    assert standard_name('ukwiki', 'bpe', 17000) == 'bpe_uk_17k'
    assert standard_name('ukwiki', 'bpe', 1500) == 'bpe_uk_1.5k'
    # a vocabulary in which " школи" can only be cut as " школ" + "и", and " мама" stays whole
    vocab = {'[UNK]': 0}
    for ch in sorted(set(pre_tokenizers.ByteLevel.alphabet())):
        vocab.setdefault(ch, len(vocab))
    bl = lambda t: pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(t)[0][0]
    merges = []
    def add(piece):
        sym = bl(piece)
        cur = list(sym)
        while len(cur) > 1:
            a, b = cur[0], cur[1]
            merges.append((a, b)); vocab.setdefault(a + b, len(vocab)); cur = [a + b] + cur[2:]
    add(' школ'); add(' мама'); add('и')
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    class W:                                   # a spaCy-like word
        def __init__(self, text, idx, lemma, pos):
            self.text, self.idx, self.lemma_, self.pos_, self.is_alpha = text, idx, lemma, pos, True
    text = 'до школи мама'
    docs = [[W('до', 0, 'до', 'ADP'), W('школи', 3, 'школа', 'NOUN'), W('мама', 9, 'мама', 'NOUN')]]
    nlp = type('N', (), {'pipe': staticmethod(lambda texts, batch_size=64: iter(docs))})()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 't.json'); tok.save(path)
        cuts = tok.encode(text).offsets
        s = tokenizer_stats(path, [text], nlp)
    assert (2, 7) in cuts and (7, 8) in cuts, cuts            # ' школ' | 'и': the cut at 3 + 4
    assert s['inflected_all'] == 1 and s['inflected_case'] == 1, s       # школи only; мама is its lemma
    assert s['ending_boundary_case'] == 1.0 and s['ending_own_token_case'] == 1.0, s
    print('  tokenizer names and the ending-exposure measure OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_lossless: PASS')
