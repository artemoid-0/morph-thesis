"""
build.py — the morpheme pipelines' vocabulary, in twelve layers.

Layer 1  prefixes    MorphoLex, no frequency threshold
Layer 2  suffixes    MorphoLex, no frequency threshold
Layer 3  roots       MorphoLex, no frequency threshold
Layer 4  function    hardcoded closed class
Layer 5  entities    SPLIT threshold: multi-word (contains '_') freq>=3,
                     single-word freq>=10. Multi-word strings cannot be
                     common-word misclassifications, which is a structural
                     observation rather than a heuristic (journey section 6).
Layer 6  characters  a-z, never deduplicated, guarantees 100% coverage
Layers 7-12          punctuation, digits, inflection, whitespace, case, bytes:
                     the information-preserving layers (journey §17), below

Entity rows carry spaCy's `entity_label` where the entity source has it, so
the `etype` block has content; every other row gets 'none'.
"""

from __future__ import annotations

import pandas as pd

from .morpholex import (
    extract_morpheme_sets, FUNCTION_WORDS, CATEGORY_PRIORITY,
)
from ..features.inflection import INFLECTION_TOKENS

#: Layer 7 — ASCII punctuation and symbols, and Layer 8 — digits, one per
#: character so arbitrary numbers work: once dropped, 15,207 and 8,098 of
#: 422,168 validation bytes (results.md §4.3).
PUNCTUATION = tuple('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

DIGITS = tuple('0123456789')

#: Layer 12 — UTF-8 byte fallback: any character absent from every other layer
#: is emitted as its bytes, so no input reaches UNK, where en-dash, em-dash,
#: degree and pound once collided (bugs.md #10, journey §17).
BYTE_FALLBACK = tuple(f'<b{i:02x}>' for i in range(256))

#: Layer 11 — case markers. Restores capitalisation without doubling the
#: vocabulary with capitalised variants of every morpheme (~16k extra rows,
#: which would badly worsen the parameter asymmetry against BPE).
CASE_MARKERS = ('<cap>', '<upper>')

#: Layer 10 — whitespace, 18.8% of corpus bytes (journey §17). <sp> is a single
#: space; runs emit repeated tokens.
WHITESPACE = ('<sp>', '<nl>', '<tab>')


def build_vocabulary(morpholex_path, entity_csv_path=None,
                     entity_min_freq=10, entity_min_len=5,
                     multiword_min_freq=3, verbose=True):
    roots, prefixes, suffixes = extract_morpheme_sets(morpholex_path)
    CHARACTERS = set('abcdefghijklmnopqrstuvwxyz')

    entries = []
    for p in sorted(prefixes):
        entries.append({'token': p, 'category': 'prefix', 'layer': 1,
                        'entity_label': 'none'})
    for s in sorted(suffixes):
        entries.append({'token': s, 'category': 'suffix', 'layer': 2,
                        'entity_label': 'none'})
    for r in sorted(roots):
        if r not in FUNCTION_WORDS and r not in CHARACTERS:
            entries.append({'token': r, 'category': 'root', 'layer': 3,
                            'entity_label': 'none'})
    for fw in sorted(FUNCTION_WORDS):
        entries.append({'token': fw, 'category': 'function', 'layer': 4,
                        'entity_label': 'none'})

    if entity_csv_path:
        try:
            ents = pd.read_csv(entity_csv_path)
            has_label = 'entity_label' in ents.columns
            if verbose and not has_label:
                print('  NOTE: entity CSV has no entity_label column — the '
                      "'etype' fact will be unavailable. Rebuild entities to "
                      'capture labels.')
            ents['token_len'] = ents['entity'].str.len()
            ents = ents[ents['token_len'] >= entity_min_len]

            multi = ents[ents['entity'].str.contains('_')
                         & (ents['count'] >= multiword_min_freq)]
            single = ents[~ents['entity'].str.contains('_')
                          & (ents['count'] >= entity_min_freq)]

            for _, row in pd.concat([multi, single]).iterrows():
                entries.append({
                    'token': row['entity'],
                    'category': 'entity',
                    'layer': 5,
                    'entity_label': (row['entity_label'] if has_label
                                     else 'none'),
                })
        except FileNotFoundError:
            print(f'  Warning: entity file not found at {entity_csv_path}')

    for c in sorted(CHARACTERS):
        entries.append({'token': c, 'category': 'character', 'layer': 6,
                        'entity_label': 'none'})

    # ── information-preserving layers (journey section 17) ────────────────────
    for p in PUNCTUATION:
        entries.append({'token': p, 'category': 'punct', 'layer': 7,
                        'entity_label': 'none'})
    for d in DIGITS:
        entries.append({'token': d, 'category': 'digit', 'layer': 8,
                        'entity_label': 'none'})
    for t in INFLECTION_TOKENS:
        entries.append({'token': t, 'category': 'infl', 'layer': 9,
                        'entity_label': 'none'})
    for w in WHITESPACE:
        entries.append({'token': w, 'category': 'space', 'layer': 10,
                        'entity_label': 'none'})
    for c in CASE_MARKERS:
        entries.append({'token': c, 'category': 'case', 'layer': 11,
                        'entity_label': 'none'})
    for b in BYTE_FALLBACK:
        entries.append({'token': b, 'category': 'byte', 'layer': 12,
                        'entity_label': 'none'})

    PROTECTED = ('character', 'punct', 'digit', 'infl', 'space', 'case',
                 'byte')
    non_char = pd.DataFrame([e for e in entries
                             if e['category'] not in PROTECTED])
    char = pd.DataFrame([e for e in entries if e['category'] in PROTECTED])
    if not non_char.empty:
        non_char['_pri'] = non_char['category'].map(
            lambda c: CATEGORY_PRIORITY.get(c, 99))
        non_char = (non_char
                    .sort_values('_pri')
                    .drop_duplicates(subset=['token'], keep='first')
                    .drop(columns=['_pri']))
    vocab_df = pd.concat([non_char, char], ignore_index=True)

    if verbose:
        print(f'  vocabulary: {len(vocab_df):,} entries')
        print(vocab_df['category'].value_counts().to_string())
    return vocab_df


def load_vocab(path):
    """
    Load a vocabulary CSV.

    keep_default_na=False is REQUIRED, not cosmetic. Pandas' default NA list
    includes 'na', 'nan' and 'null' — all of which are legitimate MorphoLex
    morphemes ('nan' as in nanometer, 'null' as in nullify, 'na' as a suffix).
    Reading with defaults silently converts them to NaN, which then fails every
    dictionary lookup and sends those words to UNK. Row count is unaffected, so
    token ids do not shift and existing caches stay valid.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    if 'entity_label' not in df.columns:
        df['entity_label'] = 'none'
    else:
        df['entity_label'] = df['entity_label'].replace('', 'none')
    return df
