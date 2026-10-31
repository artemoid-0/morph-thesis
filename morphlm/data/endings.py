"""
endings.py — where a Ukrainian word's inflectional ending begins, read from the
word form alone, and a tokenizer that never joins a stem to its ending
(DIRECTIONS D7; journey §33.61).

THE RULE (`uk_endings_v1`). The stem is what a word form shares with its lemma's
stem; the ending is the rest. pymorphy3 (Ukrainian dictionary) gives every
reading of the lowercased form, with no context: part of speech and lemma. For
an inflecting part of speech the lemma's stem is the lemma minus its citation
ending (книга → книг, важливий → важлив, зібратися → зібра; a lemma ending in a
consonant is its own stem), and the cut falls after the longest common prefix
of the form and that stem: книгами → книг|ами, книга → книг|а, важливих →
важлив|их, зібралися → зібра|лися, студентам → студент|ам. Where the stem
alternates, the varying letters go with the ending (книзі → кни|зі): the ending
is the part of the word that varies across its paradigm.

A cut is made only if EVERY reading of the form gives the same cut — a reading
whose part of speech does not inflect, or that is indeclinable, gives none — so
an ambiguous form like `його` (pronoun, or a form of `йога`) stays whole; and
only if the stem keeps at least MIN_STEM letters and the ending has one to
MAX_END. Words with non-Ukrainian letters, and all-capital abbreviations, are
never cut.

WHY THE WORD ALONE. A tokenizer that consulted the sentence would make one
word's tokens depend on the words after it — the right-context leak of journey
§33.3, inside the tokenizer.

THE MARK. A cut is written into the text as MARK, a Unicode noncharacter that
never occurs in text; the tokenizer's pre-tokenizer splits on it and removes it
(`Split(MARK, 'removed')` before ByteLevel), so no token spans a cut and no
token contains the mark. Training text is marked, so BPE never learns a merge
across a cut; encoded text is marked too (`EndingTokenizer`), since merges
learned elsewhere — `книга` inside `книгарня` — would otherwise join a stem and
its ending again (a third of the endings, on the toy corpus: §33.61).

THE FINGERPRINT. A tokenizer's cuts depend on this code and on pymorphy3's
dictionary. Both, and the cuts of a fixed list of probe words, go into a
fingerprint stored with the tokenizer; `tokenization.load_tokenizer` refuses
to encode under another (G-73: a derived artifact must carry the rule that
made it).
"""

from __future__ import annotations

import re
import bisect
import hashlib
import json

RULE = 'uk_endings_v1'
MARK = '﷐'          # a noncharacter: never in text, valid in str and UTF-8

#: citation endings by pymorphy3 part of speech, longest first; a part of
#: speech not listed does not inflect for this rule (pronouns, gerunds, ...)
CITATION = {
    'NOUN': ('я', 'а', 'о', 'е'),
    'ADJF': ('ій', 'ий', 'їй'),
    'PRTF': ('ій', 'ий'),
    'NUMR': ('а', 'о', 'и'),
    'VERB': ('тися', 'тись', 'ти', 'ть'),
    'INFN': ('тися', 'тись', 'ти', 'ть'),
}
MIN_STEM = 2
MAX_END = 5
APOSTROPHES = '’ʼ`'
#: a word as the rule sees it: Ukrainian letters, apostrophes inside
UK_WORD = re.compile(r"[а-щьюяєіїґА-ЩЬЮЯЄІЇҐ]+(?:['’ʼ`][а-щьюяєіїґА-ЩЬЮЯЄІЇҐ]+)*")
#: any run of letters, for refusing words that mix scripts
ANY_WORD = re.compile(r"[^\W\d_]+(?:['’ʼ`][^\W\d_]+)*")
PROBES = ('книгами', 'книга', 'книзі', 'важливих', 'зібралися', 'студентам', 'читаю',
          'пишу', 'ішов', 'його', 'мати', 'коня', 'зображення', 'п\'ятьма', 'об\'єктами',
          'Шевченка', 'написана', 'метро', 'роблячи', 'НАТО', 'вчительки', 'синього',
          'людей', 'дітьми', 'одновірці', 'поїздці', 'батькова', 'імені', 'знищити', 'краю')


def common_prefix(a: str, b: str) -> int:
    """How many leading characters two strings share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class EndingRule:
    """The cut position of a word's ending (0: no cut), from the word alone."""

    def __init__(self):
        import pymorphy3
        self.morph = pymorphy3.MorphAnalyzer(lang='uk')
        self._cache = {}

    def _reading_cut(self, key, p):
        pos = p.tag.POS
        if pos not in CITATION or 'Fixd' in p.tag:
            return 0
        lemma = p.normal_form
        for a in APOSTROPHES:
            lemma = lemma.replace(a, "'")
        stem = lemma
        for e in CITATION[pos]:
            if lemma.endswith(e) and len(lemma) - len(e) >= MIN_STEM:
                stem = lemma[:-len(e)]
                break
        k = common_prefix(key, stem)
        if k < MIN_STEM or not (1 <= len(key) - k <= MAX_END):
            return 0
        if not (key[k - 1].isalpha() and key[k].isalpha()):
            return 0                       # never at an apostrophe
        return k

    def cut(self, word: str) -> int:
        """Where the ending begins in `word` (an index into it), or 0 for no cut."""
        if not UK_WORD.fullmatch(word):
            return 0
        if len(word) > 1 and word.isupper():
            return 0                       # abbreviations: НАТО, США
        key = word.lower()
        if len(key) != len(word):
            return 0
        for a in APOSTROPHES:
            key = key.replace(a, "'")
        k = self._cache.get(key)
        if k is None:
            cuts = {self._reading_cut(key, p) for p in self.morph.parse(key)}
            k = cuts.pop() if len(cuts) == 1 else 0
            self._cache[key] = k
        return k

    def fingerprint(self) -> str:
        """The rule's identity: this code's constants, pymorphy3 and its dictionary, probe cuts."""
        import pymorphy3
        meta = dict(self.morph.dictionary.meta)
        try:
            from pymorphy3_dicts_uk import version as dv
            dict_version = getattr(dv, '__version__', None) or getattr(dv, 'version', None)
        except Exception:                  # pragma: no cover — reported as missing
            dict_version = None
        blob = json.dumps({
            'rule': RULE, 'citation': CITATION, 'min_stem': MIN_STEM, 'max_end': MAX_END,
            'mark': ord(MARK), 'pymorphy3': pymorphy3.__version__,
            'dict': {k: meta.get(k) for k in ('compiled_at', 'source_lexemes_count', 'format_version')},
            'dict_version': str(dict_version),
            'probes': [self.cut(w) for w in PROBES],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def mark_text(text: str, rule: EndingRule):
    """
    The text with MARK inserted at every cut, and the cut positions in the
    ORIGINAL text, ascending. Only letter runs that are wholly Ukrainian words
    are cut; every other character is left exactly as it was.
    """
    if MARK in text:
        raise ValueError(f'text contains the cut mark U+{ord(MARK):04X}, which must never occur in text')
    out, cuts, last = [], [], 0
    for m in ANY_WORD.finditer(text):
        k = rule.cut(m.group(0))
        if k:
            p = m.start() + k
            out.append(text[last:p])
            out.append(MARK)
            cuts.append(p)
            last = p
    out.append(text[last:])
    return ''.join(out), cuts


class Encoding:
    """What `tokenizers.Encoding` offers our code: ids, tokens, offsets."""
    __slots__ = ('ids', 'tokens', 'offsets')

    def __init__(self, ids, tokens, offsets):
        self.ids, self.tokens, self.offsets = ids, tokens, offsets

    def __len__(self):
        return len(self.ids)


class EndingTokenizer:
    """
    A trained tokenizer whose input is marked at every cut before encoding.
    Offsets are mapped back to the unmarked text, so alignment with the tagger
    — by character offset — is unchanged. Decoding needs nothing: the marks
    were removed before any token was made.
    """

    def __init__(self, inner, rule: EndingRule):
        self.inner = inner
        self.rule = rule

    def encode(self, text: str) -> Encoding:
        marked, cuts = mark_text(text, self.rule)
        enc = self.inner.encode(marked)
        where = [p + i for i, p in enumerate(cuts)]            # the marks' indices in `marked`
        back = lambda o: o - bisect.bisect_left(where, o)
        return Encoding(list(enc.ids), list(enc.tokens), [(back(a), back(b)) for a, b in enc.offsets])

    def decode(self, ids, **kw):
        return self.inner.decode(ids, **kw)

    def get_vocab_size(self, *a, **kw):
        return self.inner.get_vocab_size(*a, **kw)

    def get_vocab(self, *a, **kw):
        return self.inner.get_vocab(*a, **kw)

    def id_to_token(self, i):
        return self.inner.id_to_token(i)

    def token_to_id(self, t):
        return self.inner.token_to_id(t)

    def save(self, *a, **kw):
        raise RuntimeError('a tokenizer is never overwritten (tokenization.py); save the inner one when training')


def pre_tokenizer():
    """Split on the mark (removing it), then ByteLevel — the tokenizer's own pre-tokenizer."""
    from tokenizers import pre_tokenizers
    return pre_tokenizers.Sequence([pre_tokenizers.Split(MARK, 'removed'),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False)])


# ─── English (DIRECTIONS, "The queue", 2; journey §33.85) ───────────────────

RULE_EN = 'en_endings_v1'
#: a word as the English rule sees it: Latin letters only
EN_WORD = re.compile(r'[A-Za-z]+')
#: the parts of speech spaCy's English rule lemmatizer has tables for
EN_POS = ('noun', 'verb', 'adj')
EN_PROBES = ('walks', 'walked', 'studies', 'children', 'leaves', 'news', 'running', 'went',
             'bigger', 'has', 'makes', 'cities', 'NATO', 'united', 'boxes', 'wolves', 'tried',
             'happiest', 'Williams', 'teachers', 'larger', 'women', 'species', 'is')


class EnglishEndingRule:
    """
    THE RULE (`en_endings_v1`), the English counterpart of `EndingRule`. spaCy's
    English rule lemmatizer (en_core_web_sm) ships three tables per part of
    speech — exceptions (`children` → child), the index of base forms, and
    suffix rules (`-ies` → `-y`, `-ed` → ``) — which together give every
    reading of a lowercased form with no context: a noun, verb or adjective
    reading is an exception's lemma, the form itself if it is a base form, or
    a suffix rule's result if that is a base form. The stem is what the form
    shares with the lemma (English citation forms carry no ending), and the cut
    falls after the longest common prefix: walks → walk|s, studies → stud|ies,
    children → child|ren, has → ha|s.

    As in Ukrainian, a cut is made only if EVERY reading gives the same cut —
    a base-form reading gives none — so `leaves` (leaf, leave), `news` and
    `running` (also a noun) stay whole; and only with a stem of MIN_STEM
    letters or more and an ending of one to MAX_END. A word not in the tables
    (most names) has no reading and is never cut; nor are all-capital words.
    """
    name = RULE_EN

    def __init__(self, model: str = 'en_core_web_sm'):
        import spacy
        nlp = spacy.load(model)
        look = nlp.get_pipe('lemmatizer').lookups
        self.model = model
        self.model_version = nlp.meta.get('version')
        self.spacy_version = spacy.__version__
        self.index = {p: set(look.get_table('lemma_index')[p]) for p in EN_POS}
        self.exc = {p: {k: tuple(v) for k, v in look.get_table('lemma_exc')[p].items()} for p in EN_POS}
        self.rules = {p: tuple(tuple(r) for r in look.get_table('lemma_rules')[p]) for p in EN_POS}
        self._cache = {}

    def readings(self, key: str) -> set:
        """Every lemma the tables give the lowercased form, with no context."""
        out = set()
        for p in EN_POS:
            out.update(self.exc[p].get(key, ()))
            if key in self.index[p]:
                out.add(key)
            for old, new in self.rules[p]:
                if key.endswith(old):
                    form = key[:len(key) - len(old)] + new
                    if form and form in self.index[p]:
                        out.add(form)
        return out

    def cut(self, word: str) -> int:
        """Where the ending begins in `word` (an index into it), or 0 for no cut."""
        if not EN_WORD.fullmatch(word):
            return 0
        if len(word) > 1 and word.isupper():
            return 0                       # abbreviations: NATO, USA
        key = word.lower()
        k = self._cache.get(key)
        if k is None:
            cuts = set()
            for lemma in self.readings(key):
                c = common_prefix(key, lemma) if lemma != key else 0
                cuts.add(c if c >= MIN_STEM and 1 <= len(key) - c <= MAX_END else 0)
            k = cuts.pop() if len(cuts) == 1 else 0
            self._cache[key] = k
        return k

    def fingerprint(self) -> str:
        """The rule's identity: this code's constants, spaCy, the model and its tables, probe cuts."""
        tables = json.dumps({'index': {p: sorted(self.index[p]) for p in EN_POS},
                             'exc': {p: sorted(self.exc[p].items()) for p in EN_POS},
                             'rules': {p: list(self.rules[p]) for p in EN_POS}}, sort_keys=True)
        blob = json.dumps({
            'rule': RULE_EN, 'pos': EN_POS, 'min_stem': MIN_STEM, 'max_end': MAX_END,
            'mark': ord(MARK), 'spacy': self.spacy_version, 'model': self.model,
            'model_version': self.model_version,
            'tables': hashlib.sha256(tables.encode('utf-8')).hexdigest(),
            'probes': [self.cut(w) for w in EN_PROBES],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


EndingRule.name = RULE


# ─── The cut moved off the boundary (journey §33.144) ───────────────────────

RULE_SHIFT = 'uk_endings_shift1_v1'


class ShiftedEndingRule(EndingRule):
    """
    THE CONTROL FOR WHERE THE CUT FALLS: the words `uk_endings_v1` cuts, each cut once, but one
    letter off the rule's boundary — later or earlier, by a stable hash of the form — so the
    number of cuts and the pieces' lengths stay about the same while no cut falls on the
    stem|ending boundary (книгами → книга|ми or кни|гами, never книг|ами). Where only one side
    keeps both pieces letters, that side; where neither, no cut. Read from the word alone, as
    the rule is.
    """
    name = RULE_SHIFT

    def cut(self, word: str) -> int:
        k = super().cut(word)
        if not k:
            return 0
        key = word.lower()
        for a in APOSTROPHES:
            key = key.replace(a, "'")
        later = hashlib.sha256(key.encode('utf-8')).digest()[0] & 1
        for j in ((k + 1, k - 1) if later else (k - 1, k + 1)):
            if 1 <= j < len(word) and word[j - 1].isalpha() and word[j].isalpha():
                return j
        return 0

    def fingerprint(self) -> str:
        """The shift's identity on top of the rule's: its name, and the probes' shifted cuts."""
        return hashlib.sha256((RULE_SHIFT + super().fingerprint()).encode('utf-8')).hexdigest()[:16]


#: every cut rule by the name a tokenizer's sidecar records (`presplit`)
RULES = {RULE: EndingRule, RULE_EN: EnglishEndingRule, RULE_SHIFT: ShiftedEndingRule}
#: the rule a `bpe-endings` tokenizer trained on a corpus of each language uses
RULE_OF_LANGUAGE = {'uk': RULE, 'en': RULE_EN}
