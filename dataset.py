"""
dataset.py

PyTorch dataset for language-model training over WikiText-103, fed through
any of the three preprocessing pipelines.

Design (see research_journey.md):
  - Emits integer IDs only. The model owns HybridEmbedding; the dataloader
    never touches vectors.
  - Concatenate-and-chop: all articles are tokenized, concatenated into one
    long id stream, and cut into fixed-length sequences. Document boundaries
    are ignored (standard GPT-2-style chunking).
  - Tokenize ONCE per pipeline, cache the id stream to disk as a flat int32
    array, and memory-map it for training. spaCy + morpheme segmentation is
    far too slow to re-run every epoch.
  - Byte tracking for fair cross-pipeline comparison. Per-token perplexity is
    only comparable WITHIN a pipeline (same tokenization). ACROSS pipelines we
    report bits-per-byte (BPB), normalizing the total loss by the number of
    UTF-8 bytes of the original text rather than by token count. The cache
    therefore stores, alongside the id stream, the total byte count of the
    text those ids encode.
  - Corpus cap is a parameter (max_articles). Training cost depends on model
    size, which we haven't fixed yet, so the cap is left tunable.
"""

import os
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_DIR = 'data/cache'


def _cache_key(pipeline_name, split, max_articles):
    """Stable filename stem for a (pipeline, split, cap) combination."""
    raw = f'{pipeline_name}|{split}|{max_articles}'
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f'{pipeline_name}_{split}_{max_articles}_{h}'


def build_id_stream(pipeline, pipeline_name, split='train',
                    max_articles=None, cache_dir=CACHE_DIR, verbose=True):
    """
    Tokenize a WikiText-103 split through `pipeline` and return a cached,
    memory-mapped id stream plus metadata.

    Returns dict:
        ids        : np.memmap int32  [num_tokens]
        num_tokens : int
        num_bytes  : int   (UTF-8 bytes of the encoded text — for BPB)
        num_chars  : int   (characters — for BPC if ever wanted)

    The cache stores the id stream as <stem>.npy and the counts as <stem>.json.
    If the cache exists, tokenization is skipped entirely.
    """
    os.makedirs(cache_dir, exist_ok=True)
    stem = _cache_key(pipeline_name, split, max_articles)
    npy_path  = os.path.join(cache_dir, stem + '.npy')
    meta_path = os.path.join(cache_dir, stem + '.json')

    if os.path.exists(npy_path) and os.path.exists(meta_path):
        if verbose:
            print(f'  [cache hit] {stem}')
        with open(meta_path) as f:
            meta = json.load(f)
        ids = np.load(npy_path, mmap_mode='r')
        meta['ids'] = ids
        return meta

    if verbose:
        print(f'  [cache miss] tokenizing {split} through {pipeline_name}...')

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

    all_ids = []
    num_bytes = 0
    num_chars = 0
    processed = 0

    for i, row in enumerate(ds):
        if max_articles is not None and processed >= max_articles:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):   # skip blanks and headers
            continue

        ids = pipeline.encode_ids(text)
        if not ids:
            continue

        all_ids.extend(ids)
        # byte/char counts of the ORIGINAL text — the denominator for BPB/BPC.
        # We count the raw article text (what the model is ultimately modelling),
        # independent of how many tokens the pipeline split it into.
        num_bytes += len(text.encode('utf-8'))
        num_chars += len(text)

        processed += 1
        if verbose and processed % 2000 == 0:
            print(f'    {processed} articles, {len(all_ids):,} tokens...')

    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(npy_path, arr)
    meta = {
        'pipeline':     pipeline_name,
        'split':        split,
        'max_articles': max_articles,
        'num_tokens':   int(arr.shape[0]),
        'num_bytes':    int(num_bytes),
        'num_chars':    int(num_chars),
        'num_articles': processed,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f'  [cached] {stem}: {meta["num_tokens"]:,} tokens, '
              f'{meta["num_bytes"]:,} bytes')

    ids = np.load(npy_path, mmap_mode='r')
    meta['ids'] = ids
    return meta


class LMWindowDataset(Dataset):
    """
    Fixed-length next-token-prediction windows over a cached id stream.

    Each item is (input_ids, target_ids), both LongTensors of length seq_len,
    where target is input shifted by one position. Windows are non-overlapping
    and tile the concatenated stream (concatenate-and-chop).

    The dataset also exposes corpus-level byte/char/token counts so the eval
    loop can convert summed loss into bits-per-byte:

        BPB = (total_loss_in_nats / ln(2)) / num_bytes
            = total_cross_entropy_bits / num_bytes

    These counts describe the WHOLE cached stream; at eval time the loss is
    summed over exactly the tokens scored, and divided by the corresponding
    byte count for that split.
    """

    def __init__(self, stream_meta, seq_len=256):
        self.ids        = stream_meta['ids']           # memmap int32
        self.seq_len    = seq_len
        self.num_tokens = stream_meta['num_tokens']
        self.num_bytes  = stream_meta['num_bytes']
        self.num_chars  = stream_meta['num_chars']
        self.pipeline   = stream_meta.get('pipeline')

        # number of non-overlapping (input,target) windows.
        # each window needs seq_len+1 ids (input is [0:L], target is [1:L+1]).
        self.num_windows = (self.num_tokens - 1) // seq_len

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = np.asarray(self.ids[start: start + self.seq_len + 1],
                           dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    # ── metrics helpers ───────────────────────────────────────────────────────

    def bits_per_byte(self, total_loss_nats, num_target_tokens):
        """
        Convert a summed cross-entropy (in nats, over `num_target_tokens`
        scored tokens) into bits-per-byte for THIS corpus.

        We scale the per-token loss by the token/byte ratio of the stream so
        the result is comparable across pipelines with different tokenizations.
        """
        # total bits over all scored tokens
        total_bits = total_loss_nats / np.log(2)
        # bytes corresponding to the scored portion: scale corpus bytes by the
        # fraction of tokens actually scored
        frac = num_target_tokens / max(self.num_tokens, 1)
        scored_bytes = self.num_bytes * frac
        return total_bits / max(scored_bytes, 1)

    def perplexity(self, mean_loss_nats):
        """Per-token perplexity — only comparable WITHIN a pipeline."""
        return float(np.exp(mean_loss_nats))


def make_datasets(pipeline, pipeline_name, seq_len=256,
                  max_articles_train=None, max_articles_val=None,
                  cache_dir=CACHE_DIR, verbose=True):
    """
    Convenience: build train and validation window datasets for one pipeline.
    WikiText-103 ships official 'train' and 'validation' splits — we use them
    directly (no custom splitting).
    """
    train_meta = build_id_stream(
        pipeline, pipeline_name, split='train',
        max_articles=max_articles_train, cache_dir=cache_dir, verbose=verbose)
    val_meta = build_id_stream(
        pipeline, pipeline_name, split='validation',
        max_articles=max_articles_val, cache_dir=cache_dir, verbose=verbose)

    train_ds = LMWindowDataset(train_meta, seq_len=seq_len)
    val_ds   = LMWindowDataset(val_meta,   seq_len=seq_len)

    if verbose:
        print(f'  train: {len(train_ds):,} windows '
              f'({train_ds.num_tokens:,} tokens, {train_ds.num_bytes:,} bytes)')
        print(f'  val:   {len(val_ds):,} windows '
              f'({val_ds.num_tokens:,} tokens, {val_ds.num_bytes:,} bytes)')

    return train_ds, val_ds