"""
entities.py — named entity extraction and the spaCy NER wrapper.

THE FIX IN THIS MODULE
----------------------
The old entity_vocab.csv had columns `entity,count` only — the spaCy label was
computed and then thrown away. That single omission is why the old 12-dim
entity block in the frozen embedding was constant across all 16,757 vocabulary
rows: there was no label to put in it.

extract_entities() now records the majority label per surface form, so the
vocabulary carries `entity_label` and the 'etype' fact block has something real
to encode. Backward compatible: a CSV without the column still loads, the label
comes back as 'none', and config validation will refuse to build 'etype' from
a vocabulary that lacks it rather than silently emitting a constant block.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from .morpholex import ARTICLES, KEEP_ENTITIES


class NERTagger:
    """
    Lazy-loading spaCy wrapper.

    NOTE ON COST: spacy.load() enables tagger, parser AND ner by default, so
    the morpheme pipelines have ALWAYS been paying for dependency parsing and
    discarding the result. Context facts (dep/pos) are therefore close to free
    for morph/det — only the BPE+context cell pays a genuinely new spaCy cost.
    """

    def __init__(self, model='en_core_web_sm'):
        self._model = model
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self._model)
        return self._nlp

    def get_entity_spans(self, doc):
        """
        Returns:
          entity_spans   : {start_token_idx: (surface_string, label)}
          entity_indices : set of token indices belonging to any kept entity
        """
        entity_spans = {}
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


def extract_entities(max_articles=6000, spacy_model='en_core_web_sm',
                     out_path='data/entity_vocab.csv', verbose=True):
    """
    Scan WikiText-103 and count named entities, KEEPING THE LABEL.

    Writes CSV with columns: entity, count, entity_label
    where entity_label is the majority spaCy label for that surface form.
    """
    from datasets import load_dataset

    tagger = NERTagger(spacy_model)
    counts = Counter()
    labels = defaultdict(Counter)

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    processed = 0
    for row in ds:
        if processed >= max_articles:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        doc = tagger.nlp(text)
        spans, _ = tagger.get_entity_spans(doc)
        for _, (surface, label) in spans.items():
            counts[surface] += 1
            labels[surface][label] += 1
        processed += 1
        if verbose and processed % 500 == 0:
            print(f'    {processed} articles, {len(counts):,} unique entities...')

    rows = []
    for surface, n in counts.items():
        majority = labels[surface].most_common(1)[0][0]
        rows.append({'entity': surface, 'count': n, 'entity_label': majority})

    df = pd.DataFrame(rows).sort_values('count', ascending=False)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f'  wrote {len(df):,} entities -> {out_path}')
        print(f'  label distribution:\n{df["entity_label"].value_counts()}')
    return df
