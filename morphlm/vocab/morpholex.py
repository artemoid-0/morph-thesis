"""
morpholex.py — MorphoLex parsing.

Ported verbatim from the previous common.py. This code is STABLE and validated
(the format-B fix in particular took real debugging: 4,941 of ~70k entries put
the prefix INSIDE the opening brace, and missing that silently dropped prefixes
from thousands of words). Do not "improve" it without a regression test.
"""

from __future__ import annotations

import os
import re
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


ARTICLES = {'the', 'a', 'an'}

KEEP_ENTITIES = {
    'ORG', 'PERSON', 'GPE', 'LOC', 'FAC',
    'NORP', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
}

FUNCTION_WORDS = {
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from',
    'by', 'about', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between',
    'under', 'over', 'out', 'off',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both',
    'either', 'neither', 'not', 'although', 'because',
    'since', 'unless', 'while', 'whereas', 'whether', 'if', 'though',
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'us', 'them', 'who', 'what', 'which',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'not', 'no', 'up', 'there', 'here', 'then', 'than',
    'too', 'also', 'just', 'only', 'more', 'most',
    'some', 'any', 'all', 'each', 'every', 'other', 'such', 'same',
}

CATEGORY_PRIORITY = {
    'entity': 0, 'function': 1, 'prefix': 2,
    'suffix': 3, 'root': 4, 'character': 5,
}


def parse_morpholex_segm(s):
    """
    Parse a MorphoLex segmentation string into (morphemes, roles).
    Handles both format A (<un<{(happy)}) and format B ({<re<(search)}>er>).
    """
    root_match = re.search(r'\(([a-zA-Z]+)\)', s)
    root = root_match.group(1) if root_match else None

    brace_open = s.index('{') if '{' in s else len(s)
    brace_close = s.rindex('}') + 1 if '}' in s else 0

    # format A: prefixes before the opening {
    format_a_prefixes = re.findall(r'<([a-zA-Z]+)<', s[:brace_open])

    # format B: prefixes inside the opening { before the root
    inner = s[brace_open:brace_close]
    paren_pos = inner.index('(') if '(' in inner else len(inner)
    format_b_prefixes = re.findall(r'<([a-zA-Z]+)<', inner[:paren_pos])

    prefixes = format_a_prefixes + format_b_prefixes

    inner_after_root = re.sub(r'^[^)]*\)', '', inner).rstrip('}')
    inner_suffixes = re.findall(r'>([a-zA-Z]+)>', inner_after_root)
    outer_suffixes = re.findall(r'>([a-zA-Z]+)>', s[brace_close:])

    suffixes = inner_suffixes + outer_suffixes
    morphemes = prefixes + ([root] if root else []) + suffixes
    roles = (
        ['prefix'] * len(prefixes)
        + (['root'] if root else [])
        + ['suffix'] * len(suffixes)
    )
    return morphemes, roles


def is_valid_morpheme(m):
    return bool(re.match(r'^[a-zA-Z]+$', m))


_SHEET_CACHE = {}


def _read_sheets(path, use_disk_cache=True, verbose=True):
    """
    Parse the MorphoLex workbook into one DataFrame.

    CACHED AT TWO LEVELS, because openpyxl parsing ~70k rows is slow enough to
    look like a hang, and every pipeline construction needs this data TWICE
    (build_morpholex_lookup + extract_morpheme_sets). Previously that meant two
    full parses on every single run.

      1. in-process dict  -> the second call in the same run is free
      2. on-disk pickle   -> every subsequent run is fast

    The disk cache is keyed on the workbook's (mtime, size), so editing the
    xlsx invalidates it automatically. Any failure to read or write the cache
    falls back to parsing — the cache can never be the reason something breaks.

    Callers immediately slice columns (which copies in pandas), so handing back
    the cached frame is safe.
    """
    key = os.path.abspath(path)
    if key in _SHEET_CACHE:
        return _SHEET_CACHE[key]

    stat = os.stat(path)
    stamp = f'{int(stat.st_mtime)}_{stat.st_size}'
    cache_path = os.path.join('data', 'cache', f'morpholex_sheets_{stamp}.pkl')

    if use_disk_cache and os.path.exists(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            _SHEET_CACHE[key] = df
            if verbose:
                print(f'  [morpholex cache hit] {len(df):,} rows')
            return df
        except Exception:
            pass  # corrupt or version-mismatched cache: reparse

    if verbose:
        print(f'  parsing {os.path.basename(path)} (slow, cached afterwards)...')

    xl = pd.ExcelFile(path)
    sheets = [s for s in xl.sheet_names
              if s != 'Presentation' and not s.startswith('All')]
    dfs = []
    for sheet in sheets:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
            if 'Word' in df.columns:
                dfs.append(df)
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True)

    if use_disk_cache:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_pickle(cache_path)
            if verbose:
                print(f'  [morpholex cached] {len(df):,} rows -> {cache_path}')
        except Exception as e:
            if verbose:
                print(f'  (could not write morpholex cache: {e})')

    _SHEET_CACHE[key] = df
    return df


def build_morpholex_lookup(path):
    """word -> {morphemes, roles, pos} for ~66k words, with normalized roots."""
    df = _read_sheets(path)
    df = df[['Word', 'POS', 'MorphoLexSegm']].dropna(
        subset=['Word', 'MorphoLexSegm'])
    df['Word'] = df['Word'].str.lower().str.strip()

    lookup = {}
    for _, row in df.iterrows():
        morphemes, roles = parse_morpholex_segm(row['MorphoLexSegm'])
        if morphemes and all(is_valid_morpheme(m) for m in morphemes):
            lookup[row['Word']] = {
                'morphemes': morphemes,
                'roles': roles,
                'pos': row['POS'],
            }
    return lookup


def extract_morpheme_sets(path):
    """
    Canonical prefix / suffix / root sets from MorphoLex.
    Source determines validity — no frequency threshold (journey section 12).
    """
    df = _read_sheets(path)
    df = df[['MorphoLexSegm']].dropna()

    roots, prefixes, suffixes = set(), set(), set()
    for _, row in df.iterrows():
        morphemes, roles = parse_morpholex_segm(row['MorphoLexSegm'])
        for m, r in zip(morphemes, roles):
            if not is_valid_morpheme(m):
                continue
            if r == 'root':
                roots.add(m)
            elif r == 'prefix':
                prefixes.add(m)
            elif r == 'suffix':
                suffixes.add(m)
    return roots, prefixes, suffixes


def compound_split(word, morpholex_lookup, function_words=None):
    """
    Split an OOV word into two known roots, longest-left-first.

    2-char right parts are accepted only if they are known compound particles
    (up/in/on/out/off/by). That exception is what makes 'startup' resolve as
    start+up rather than the star+tup false positive (journey section 25).
    """
    if function_words is None:
        function_words = set()

    n = len(word)
    for i in range(n - 2, 2, -1):
        left = word[:i]
        right = word[i:]

        if len(left) < 3:
            continue
        if len(right) < 2:
            continue
        if len(right) == 2 and right not in function_words:
            continue
        if left not in morpholex_lookup:
            continue

        if right in morpholex_lookup:
            le = morpholex_lookup[left]
            re_ = morpholex_lookup[right]
            result = list(zip(le['morphemes'], le['roles']))
            result += list(zip(re_['morphemes'], re_['roles']))
            return result
        elif right in function_words:
            le = morpholex_lookup[left]
            result = list(zip(le['morphemes'], le['roles']))
            result.append((right, 'root'))
            return result

    return None
