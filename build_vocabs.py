"""
build_vocabs.py

Run this once before using any pipeline.
Builds:
  - data/our_final_vocab.csv   (morpheme vocabulary, all 6 layers)
  - data/entity_vocab.csv      (named entity frequencies from dataset)
  - data/bpe_tokenizer.json    (trained BPE tokenizer)

Usage:
    python build_vocabs.py
"""

import re
import pandas as pd
from collections import Counter
from datasets import load_dataset

import spacy
from common import build_vocabulary, KEEP_ENTITIES

MORPHOLEX_PATH  = 'data/MorphoLEX_en.xlsx'
ENTITY_CSV      = 'data/entity_vocab.csv'
VOCAB_CSV       = 'data/our_final_vocab.csv'
BPE_PATH        = 'data/bpe_tokenizer.json'
ARTICLES        = {'the', 'a', 'an'}
N_ARTICLES      = 10000   # articles to process for entity counts and BPE
BPE_VOCAB_SIZE  = 10000


# ── Step 1: collect entity frequencies from WikiText-103 ──────────────────────

def build_entity_vocab(n_articles=N_ARTICLES):
    print(f'Collecting named entities from {n_articles} WikiText-103 articles...')
    nlp = spacy.load('en_core_web_sm')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')

    entity_counter = Counter()
    processed = 0

    for i, row in enumerate(dataset):
        if i >= n_articles:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ not in KEEP_ENTITIES:
                continue
            tokens = [t for t in ent if t.text.lower() not in ARTICLES]
            if not tokens:
                continue
            surface = '_'.join(t.text.lower() for t in tokens)
            entity_counter[surface] += 1
        processed += 1
        if processed % 1000 == 0:
            print(f'  {processed} articles, {len(entity_counter)} unique entities...')

    df = pd.DataFrame(entity_counter.most_common(), columns=['entity', 'count'])
    df.to_csv(ENTITY_CSV, index=False)
    print(f'  Saved {len(df)} entities to {ENTITY_CSV}')
    return df


# ── Step 2: build morpheme vocabulary ────────────────────────────────────────

def build_morpheme_vocab():
    print('Building morpheme vocabulary...')
    vocab_df = build_vocabulary(
        morpholex_path=MORPHOLEX_PATH,
        entity_csv_path=ENTITY_CSV,
        entity_min_freq=10,
        entity_min_len=5,
    )
    vocab_df.to_csv(VOCAB_CSV, index=False)
    print(f'  Saved {len(vocab_df)} entries to {VOCAB_CSV}')
    for layer, group in vocab_df.groupby('layer'):
        cat = group['category'].iloc[0]
        print(f'    Layer {layer} ({cat:10}): {len(group):6} entries')
    return vocab_df


# ── Step 3: train and save BPE tokenizer ─────────────────────────────────────

def build_bpe_tokenizer(n_articles=N_ARTICLES):
    from pipeline_bpe import BPEPipeline

    print(f'Collecting text for BPE training ({n_articles} articles)...')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    texts = []
    for i, row in enumerate(dataset):
        if i >= n_articles:
            break
        text = row['text'].strip()
        if text and not text.startswith(' ='):
            texts.append(text)
    print(f'  {len(texts)} articles collected')

    bpe = BPEPipeline(vocab_size=BPE_VOCAB_SIZE)
    bpe.train(texts, save_path=BPE_PATH)
    return bpe


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    steps = sys.argv[1:] or ['entities', 'vocab', 'bpe']

    if 'entities' in steps:
        build_entity_vocab()

    if 'vocab' in steps:
        build_morpheme_vocab()

    if 'bpe' in steps:
        build_bpe_tokenizer()

    print('\nAll done. Run: python main.py test')