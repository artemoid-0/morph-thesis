"""
build.py — the six-layer vocabulary.

Layer 1  prefixes    MorphoLex, no frequency threshold
Layer 2  suffixes    MorphoLex, no frequency threshold
Layer 3  roots       MorphoLex, no frequency threshold
Layer 4  function    hardcoded closed class
Layer 5  entities    SPLIT threshold: multi-word (contains '_') freq>=3,
                     single-word freq>=10. Multi-word strings cannot be
                     common-word misclassifications, which is a structural
                     observation rather than a heuristic (journey section 23).
Layer 6  characters  a-z, never deduplicated, guarantees 100% coverage

Change from the previous version: `entity_label` is carried through to the
vocabulary CSV when the entity source provides it, so the 'etype' fact block
has real content. Non-entity rows get 'none'.
"""

from __future__ import annotations

import pandas as pd

from .morpholex import (
    extract_morpheme_sets, FUNCTION_WORDS, CATEGORY_PRIORITY,
)


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

    non_char = pd.DataFrame([e for e in entries if e['category'] != 'character'])
    char = pd.DataFrame([e for e in entries if e['category'] == 'character'])
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
    """Load a vocabulary CSV, tolerating the older label-free format."""
    df = pd.read_csv(path)
    if 'entity_label' not in df.columns:
        df['entity_label'] = 'none'
    return df


def has_entity_labels(vocab_df) -> bool:
    """True if any entity row carries a real (non-'none') label."""
    if 'entity_label' not in vocab_df.columns:
        return False
    ent = vocab_df[vocab_df['category'] == 'entity']
    if ent.empty:
        return False
    return bool((ent['entity_label'].fillna('none') != 'none').any())
