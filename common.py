"""
common.py

Shared utilities used by all three pipelines:
  - MorphoLex parser and lookup builder
  - spaCy NER wrapper
  - Vocabulary building (all 6 layers)
  - Base Preprocessor class defining the shared interface
"""

import re
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────

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


# ── MorphoLex parser ──────────────────────────────────────────────────────────

def parse_morpholex_segm(s):
    """
    Parse a MorphoLex segmentation string into (morphemes, roles).
    Handles both format A (<un<{(happy)}) and format B ({<re<(search)}>er>).
    """
    root_match = re.search(r'\(([a-zA-Z]+)\)', s)
    root = root_match.group(1) if root_match else None

    brace_open  = s.index('{') if '{' in s else len(s)
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
        ['prefix'] * len(prefixes) +
        (['root']  if root else []) +
        ['suffix'] * len(suffixes)
    )
    return morphemes, roles


def is_valid_morpheme(m):
    return bool(re.match(r'^[a-zA-Z]+$', m))


def build_morpholex_lookup(path):
    """
    Build word -> {morphemes, roles, pos} lookup from MorphoLex Excel file.
    Returns dict of 66k entries with normalized roots.
    """
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
    df = df[['Word', 'POS', 'MorphoLexSegm']].dropna(subset=['Word', 'MorphoLexSegm'])
    df['Word'] = df['Word'].str.lower().str.strip()

    lookup = {}
    for _, row in df.iterrows():
        morphemes, roles = parse_morpholex_segm(row['MorphoLexSegm'])
        if morphemes and all(is_valid_morpheme(m) for m in morphemes):
            lookup[row['Word']] = {
                'morphemes': morphemes,
                'roles':     roles,
                'pos':       row['POS'],
            }
    return lookup


def extract_morpheme_sets(path):
    """
    Extract the canonical prefix, suffix, and root sets from MorphoLex.
    These define what counts as a real morpheme — no frequency threshold.
    Returns: roots (set), prefixes (set), suffixes (set)
    """
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
    df = df[['MorphoLexSegm']].dropna()

    roots, prefixes, suffixes = set(), set(), set()
    for _, row in df.iterrows():
        morphemes, roles = parse_morpholex_segm(row['MorphoLexSegm'])
        for m, r in zip(morphemes, roles):
            if not is_valid_morpheme(m):
                continue
            if r == 'root':     roots.add(m)
            elif r == 'prefix': prefixes.add(m)
            elif r == 'suffix': suffixes.add(m)
    return roots, prefixes, suffixes


# ── Vocabulary builder ────────────────────────────────────────────────────────

def build_vocabulary(morpholex_path, entity_csv_path=None,
                     entity_min_freq=10, entity_min_len=5):
    """
    Build the full vocabulary with 6 layers.

    Layer 1 — prefixes     (from MorphoLex, no frequency threshold)
    Layer 2 — suffixes     (from MorphoLex, no frequency threshold)
    Layer 3 — roots        (from MorphoLex, no frequency threshold)
    Layer 4 — function     (hardcoded closed class)
    Layer 5 — entities     (frequency-filtered with split thresholds:
                            multi-word entities use freq>=3 since they
                            cannot be common word misclassifications;
                            single-word entities use entity_min_freq)
    Layer 6 — characters   (always present, never deduplicated)
    """
    roots, prefixes, suffixes = extract_morpheme_sets(morpholex_path)
    CHARACTERS = set('abcdefghijklmnopqrstuvwxyz')

    entries = []
    for p in sorted(prefixes):
        entries.append({'token': p, 'category': 'prefix', 'layer': 1})
    for s in sorted(suffixes):
        entries.append({'token': s, 'category': 'suffix', 'layer': 2})
    for r in sorted(roots):
        if r not in FUNCTION_WORDS and r not in CHARACTERS:
            entries.append({'token': r, 'category': 'root', 'layer': 3})
    for fw in sorted(FUNCTION_WORDS):
        entries.append({'token': fw, 'category': 'function', 'layer': 4})

    if entity_csv_path:
        try:
            entities = pd.read_csv(entity_csv_path)
            entities['token_len'] = entities['entity'].str.len()
            entities = entities[entities['token_len'] >= entity_min_len]

            # multi-word entities (containing _) cannot be common word
            # misclassifications — spaCy only produces them by recognising
            # genuine multi-word spans. Trust them at lower frequency.
            multi_word = entities[
                entities['entity'].str.contains('_') &
                (entities['count'] >= 3)
            ]

            # single-word entities need stricter filtering — common words
            # like 'light', 'world', 'child' leak in at low frequency
            single_word = entities[
                ~entities['entity'].str.contains('_') &
                (entities['count'] >= entity_min_freq)
            ]

            for _, row in pd.concat([multi_word, single_word]).iterrows():
                entries.append({
                    'token': row['entity'],
                    'category': 'entity',
                    'layer': 5,
                })
        except FileNotFoundError:
            print(f'  Warning: entity file not found at {entity_csv_path}')

    for c in sorted(CHARACTERS):
        entries.append({'token': c, 'category': 'character', 'layer': 6})

    non_char = pd.DataFrame([e for e in entries if e['category'] != 'character'])
    char     = pd.DataFrame([e for e in entries if e['category'] == 'character'])
    if not non_char.empty:
        non_char['_pri'] = non_char['category'].map(
            lambda c: CATEGORY_PRIORITY.get(c, 99)
        )
        non_char = (non_char
                    .sort_values('_pri')
                    .drop_duplicates(subset=['token'], keep='first')
                    .drop(columns=['_pri']))
    vocab_df = pd.concat([non_char, char], ignore_index=True)
    return vocab_df

def compound_split(word, morpholex_lookup, function_words=None):
    """
    Try to split an OOV word into two known roots.
    Longest-left-first. Both parts must be in MorphoLex, with an
    exception: 2-char right parts are accepted if they are known
    compound particles (function words like 'up', 'in', 'on', 'out').

    Examples:
      blockchain  → block + chain
      homepage    → home + page
      speedrun    → speed + run
      startup     → start + up   (up is a function word particle)
      holdup      → hold + up
    """
    if function_words is None:
        function_words = set()

    n = len(word)
    for i in range(n - 2, 2, -1):
        left  = word[:i]
        right = word[i:]

        if len(left) < 3:
            continue

        # right part: ≥3 chars, or exactly 2 chars if it's a known particle
        if len(right) < 2:
            continue
        if len(right) == 2 and right not in function_words:
            continue

        if left not in morpholex_lookup:
            continue

        if right in morpholex_lookup:
            left_entry  = morpholex_lookup[left]
            right_entry = morpholex_lookup[right]
            result = list(zip(left_entry['morphemes'], left_entry['roles']))
            result += list(zip(right_entry['morphemes'], right_entry['roles']))
            return result
        elif right in function_words:
            # particle like 'up', 'in', 'on' — not in MorphoLex as a root
            # but valid compound element; resolve_category will tag it correctly
            left_entry = morpholex_lookup[left]
            result = list(zip(left_entry['morphemes'], left_entry['roles']))
            result.append((right, 'root'))
            return result

    return None


# ── spaCy NER wrapper ─────────────────────────────────────────────────────────

class NERTagger:
    """Lazy-loading spaCy NER wrapper."""

    def __init__(self, model='en_core_web_sm'):
        self._model = model
        self._nlp   = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self._model)
        return self._nlp

    def get_entity_spans(self, doc):
        """
        Returns:
          entity_spans  : {start_token_idx: (surface_string, label)}
          entity_indices: set of all token indices belonging to any entity
        """
        entity_spans   = {}
        entity_indices = set()
        for ent in doc.ents:
            if ent.label_ not in KEEP_ENTITIES:
                continue
            tokens = [t for t in ent if t.text.lower() not in ARTICLES]
            if not tokens:
                continue
            surface = '_'.join(t.text.lower() for t in tokens)
            entity_spans[tokens[0].i] = (surface, ent.label_)
            for t in tokens:
                entity_indices.add(t.i)
        return entity_spans, entity_indices


# ── Base preprocessor ─────────────────────────────────────────────────────────

class BasePreprocessor:
    """
    Shared interface for all three pipelines.
    Subclasses implement _segment_word(word) -> [(morpheme, role), ...]
    """

    def __init__(self, vocab_df, morpholex_path, spacy_model='en_core_web_sm'):
        self.vocab      = vocab_df
        self.token2id   = {row['token']: idx
                           for idx, row in vocab_df.iterrows()}
        self.id2token   = {idx: row['token']
                           for idx, row in vocab_df.iterrows()}
        self.categories = {row['token']: row['category']
                           for _, row in vocab_df.iterrows()}

        # special token IDs beyond the vocabulary
        n = len(self.token2id)
        self.UNK_ID = n;     self.PAD_ID = n + 1
        self.BOS_ID = n + 2; self.EOS_ID = n + 3

        self.function_words = {t for t, c in self.categories.items()
                               if c == 'function'}
        self.entity_tokens  = {t for t, c in self.categories.items()
                               if c == 'entity'}

        self.ner = NERTagger(spacy_model)

        print(f'  Vocabulary: {self.vocab_size()} tokens '
              f'({len(self.token2id)} entries + 4 special)')

    def vocab_size(self):
        return len(self.token2id) + 4

    def _resolve_category(self, tok, override=None):
        if override:
            return override
        matches = self.vocab[self.vocab['token'] == tok]['category'].tolist()
        if not matches:
            return 'character'
        return min(matches, key=lambda c: CATEGORY_PRIORITY.get(c, 99))

    def _make_token(self, morpheme, role, word,
                    entity_label=None, category_override=None):
        tok      = morpheme.lower()
        token_id = self.token2id.get(tok, self.UNK_ID)
        # character fallback always wins regardless of vocabulary category
        category = 'character' if role == 'char' else \
                   self._resolve_category(tok, category_override)
        return {
            'token_id':     token_id,
            'token':        tok,
            'category':     category,
            'role':         role,
            'entity_label': entity_label,
            'word':         word,
        }

    def _segment_word(self, word):
        """
        Subclasses must implement this.
        Returns list of (morpheme_string, role) pairs.
        role: prefix / root / suffix / entity / char
        """
        raise NotImplementedError

    def encode(self, text):
        """
        Process a sentence -> list of token dicts.
        Handles NER entity spans, then delegates word segmentation
        to _segment_word which is pipeline-specific.
        """
        doc = self.ner.nlp(text)
        entity_spans, entity_indices = self.ner.get_entity_spans(doc)

        result = []
        i = 0
        while i < len(doc):
            spacy_tok = doc[i]

            if not spacy_tok.is_alpha:
                i += 1
                continue

            if spacy_tok.i in entity_spans:
                surface, label = entity_spans[spacy_tok.i]
                if surface in self.entity_tokens:
                    result.append(self._make_token(
                        surface, 'entity', surface,
                        entity_label=label,
                        category_override='entity',
                    ))
                else:
                    # unknown entity — honest character fallback
                    for w in surface.split('_'):
                        for ch in w:
                            result.append(self._make_token(
                                ch, 'char', w, entity_label=label
                            ))
                while i < len(doc) and doc[i].i in entity_indices:
                    i += 1
                continue

            if spacy_tok.i in entity_indices:
                i += 1
                continue

            word = spacy_tok.text
            for morpheme, role in self._segment_word(word):
                result.append(self._make_token(morpheme, role, word))
            i += 1

        return result

    def encode_ids(self, text):
        return [t['token_id'] for t in self.encode(text)]

    def pretty_print(self, text):
        units = self.encode(text)
        print(f'Input: {text}')
        print(f'  {"token":20} {"cat":10} {"role":8} {"entity":8} {"word"}')
        print(f'  {"-"*65}')
        for u in units:
            ent = u['entity_label'] or '-'
            print(f'  {u["token"]:20} {u["category"]:10} '
                  f'{u["role"]:8} {ent:8} {u["word"]}')
        print()