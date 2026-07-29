"""
inflection.py — explicit inflectional morphemes.

THE PROBLEM THIS SOLVES
-----------------------
MorphoLex stores lemmas, so the morpheme pipelines mapped `cat` and `cats` to
the same token sequence [cat]. Two different texts, one stream: information
provably destroyed. `collision_audit.py` measured the cost at 0.0440 BPB, with
2,064 colliding groups covering 28.5% of word occurrences — almost all of them
inflectional paradigms:

    {include:40, included:88, includes:18, including:140}
    {year:120, years:108}
    {make:49, made:123, makes:8, making:23}

Emitting the inflection as its own token removes the collision. It is also
LINGUISTICALLY MORE CORRECT, not a workaround: journey section 5 already
classifies inflectional morphemes as morphemes. The pipeline never emitted them
only because its source database is derivational.

FREE RIDE
---------
spacy.load() enables the tagger and attribute_ruler by default, so
`token.morph` (Number=Plur, Tense=Past, ...) is already computed on every parse
the pipeline performs. Reading it costs nothing extra.

REDUNDANCY GUARD
----------------
Some MorphoLex analyses already carry the inflection as a suffix — journey
section 19 records `happiest -> happy + est`. Emitting <SUP> on top would waste
a token, which unfairly inflates morph's token count. `_already_marked` skips
the inflection token when the final morpheme is exactly the expected surface
suffix.
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


#: Trailing differences that a single inflection marker can explain.
#: Longest first so 'ing' is tried before 'g', 'est' before 'es'/'s'.
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
    Explain a TRAILING difference between the morphemes and the surface.

    Needed because spaCy's POS tags hide real inflection:

        gaming    tagged NOUN  -> no Tense feature at all
        seasoned  tagged ADJ   -> no Tense feature at all
        news      root 'new'   -> no feature could ever see this

    All three collided with their base form in the audit. Deriving the marker
    from the surface STRING instead of from features is deterministic, so
    distinct surfaces get distinct markers and injectivity is guaranteed.

    Only TRAILING differences are marked. Internal orthographic variation
    (beauty+ful -> "beautyful" vs "beautiful") is deliberately left alone: it
    does not collide with anything, so it needs no repair, and marking it would
    cost a token for no information gain. That distinction is the whole reason
    the collision audit supersedes the reconstruction audit.
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
