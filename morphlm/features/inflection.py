"""
inflection.py — explicit inflectional morphemes.

MorphoLex stores lemmas, so the morpheme pipelines mapped `cat` and `cats` to
the same tokens: a collision the audit measured at 0.0440 BPB, almost entirely
inflectional paradigms (journey §17, "Audit 2 — collisions"). Emitting the
inflection as its own token removes it, and is linguistically right: inflectional
morphemes are morphemes (journey §3). spaCy computes `token.morph` on every
parse anyway, so reading it costs nothing.

Some MorphoLex analyses already end in the inflection's surface suffix
(`happiest` -> happy + est); `already_marked` then skips the token, which would
only inflate morph's token count.
"""

from __future__ import annotations

#: Closed inventory. Order is fixed because these become vocabulary rows.
INFLECTION_TOKENS = (
    '<plur>',      # nouns: cats, cities
    '<3sg>',       # verbs: runs, includes
    '<past>',      # verbs: ran, included
    '<pastpart>',  # verbs: been, taken
    '<prog>',      # verbs: running, including
    '<cmp>',       # adjectives/adverbs: later, bigger
    '<sup>',       # adjectives/adverbs: latest, biggest
    '<poss>',      # possessive: 's
)

#: Surface suffixes that, if already present as the final morpheme, mean the
#: inflection is accounted for and the extra token would be redundant.
_EXPECTED_SUFFIX = {
    '<plur>': ('s', 'es'),
    '<3sg>': ('s', 'es'),
    '<past>': ('ed', 'd'),
    '<pastpart>': ('ed', 'en', 'd', 'n'),
    '<prog>': ('ing',),
    '<cmp>': ('er',),
    '<sup>': ('est',),
    '<poss>': ("'s", 's'),
}


def _feat(morph, key):
    """First value of a spaCy morphological feature, or None."""
    try:
        vals = morph.get(key)
    except Exception:
        return None
    return vals[0] if vals else None


def inflection_of(spacy_token):
    """
    Map a spaCy token to at most one inflection token, or None.

    Deliberately at most ONE: English inflection is almost entirely
    non-cumulative on a single word, and a single label keeps the token stream
    and the decode mapping simple. Possessive is checked first because it can
    co-occur with plurality ("teams'"), and the apostrophe is the part that
    would otherwise be unrecoverable.
    """
    morph = getattr(spacy_token, 'morph', None)
    if morph is None:
        return None
    pos = getattr(spacy_token, 'pos_', '')

    if _feat(morph, 'Poss') == 'Yes':
        return '<poss>'

    if pos in ('NOUN', 'PROPN'):
        if _feat(morph, 'Number') == 'Plur':
            return '<plur>'
        return None

    if pos in ('VERB', 'AUX'):
        verbform = _feat(morph, 'VerbForm')
        tense = _feat(morph, 'Tense')
        if verbform == 'Part':
            # spaCy marks -ing as VerbForm=Part with Tense=Pres
            return '<pastpart>' if tense == 'Past' else '<prog>'
        if verbform == 'Ger':
            return '<prog>'
        if tense == 'Past':
            return '<past>'
        if (tense == 'Pres'
                and _feat(morph, 'Person') == '3'
                and _feat(morph, 'Number') == 'Sing'):
            return '<3sg>'
        return None

    if pos in ('ADJ', 'ADV'):
        degree = _feat(morph, 'Degree')
        if degree == 'Cmp':
            return '<cmp>'
        if degree == 'Sup':
            return '<sup>'
        return None

    return None


def already_marked(infl, morphemes):
    """
    True if the morpheme sequence already spells the inflection, so emitting a
    token for it would be redundant.

        happiest -> [happy, est] + <sup>   -> redundant, skip <sup>
        cats     -> [cat]        + <plur>  -> needed
    """
    if not infl or not morphemes:
        return False
    last = str(morphemes[-1]).lower()
    return last in _EXPECTED_SUFFIX.get(infl, ())


#: Trailing differences that a single inflection marker can explain. Each is
#: compared with the whole trailing difference, so their order does not matter.
_RESIDUE_SUFFIXES = (
    ("'s", '<poss>'),
    ('ing', '<prog>'),
    ('est', '<sup>'),
    ('ed', '<past>'),
    ('es', '<plur>'),
    ('er', '<cmp>'),
    ('en', '<pastpart>'),
    ('d', '<past>'),
    ('n', '<pastpart>'),
    ('s', '<plur>'),
)


def residue_marker(concat, surface):
    """
    Explain a TRAILING difference between the morphemes and the surface — the
    inflection spaCy's tags hide: `gaming` (NOUN), `seasoned` (ADJ), `news`
    (root `new`), each of which collided with its base form (journey §17).
    Read from the surface string, so distinct surfaces get distinct markers.
    Internal orthographic variation (beauty+ful -> "beautiful") is left alone:
    it collides with nothing, and marking it would cost a token for no
    information.
    """
    if not concat or not surface or surface == concat:
        return None

    # straightforward addition: cat -> cats, new -> news, state -> stated
    # NOTE: no early return here. 'shipping' starts with 'ship' but its tail
    # ('ping') matches nothing, and it must still fall through to the
    # doubled-consonant branch below.
    if surface.startswith(concat):
        tail = surface[len(concat):]
        for suf, mark in _RESIDUE_SUFFIXES:
            if tail == suf:
                return mark

    # doubled final consonant: ship + ing -> shipping, run + ing -> running
    if len(concat) > 1 and surface.startswith(concat + concat[-1]):
        tail = surface[len(concat) + 1:]
        for suf, mark in _RESIDUE_SUFFIXES:
            if tail == suf:
                return mark

    # one dropped character, i.e. silent e: game + ing -> gaming
    if len(concat) > 2 and surface.startswith(concat[:-1]):
        tail = surface[len(concat) - 1:]
        for suf, mark in _RESIDUE_SUFFIXES:
            if tail == suf:
                return mark
    return None


#: Agreement classes for the `magr` context fact, from UD features Number and
#: Person. English marks Number on nouns, determiners, pronouns and finite verbs
#: and Person on pronouns and finite verbs; `you` carries Person=2 and no
#: Number. Order is fixed because these become embedding rows.
AGREEMENT_CLASSES = ('Sing', 'Plur', '1Sing', '1Plur', '2', '3Sing', '3Plur')


def agreement_of(spacy_token):
    """
    The agreement features (Number, Person) of a word, or None.

    This is deliberately NOT the tokenizer's inflection (`inflection_of`), which
    marks only inflectional morphemes on nouns, verbs and adjectives. Agreement
    is what the NEXT word must match, and English carries it on words that have
    no inflectional morpheme at all: `these` (Number=Plur), `they` (Person=3,
    Number=Plur), `you` (Person=2). A subword model sees those as single tokens
    whose number it must learn; this states it. Used only as a context fact —
    it never touches the morpheme tokenizer's stream, whose cache keys depend
    on `inflection_of` being unchanged.
    """
    morph = getattr(spacy_token, 'morph', None)
    if morph is None:
        return None
    num = _feat(morph, 'Number')
    per = _feat(morph, 'Person')
    if num is None and per is None:
        return None
    key = '2' if per == '2' else f"{per or ''}{num or ''}"
    return key if key in AGREEMENT_CLASSES else None
