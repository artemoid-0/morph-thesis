"""
Offline stand-in for the `datasets` package, for rehearsals in a sandbox that
cannot reach Hugging Face (scripts/README.md, "rehearsal/").

Serves:
  wikitext                    a toy WikiText from $REHEARSAL_WIKITEXT/{train,validation}.txt,
                              one row per line, exactly as `load_dataset('wikitext', ...)` does
  wikimedia/wikipedia .uk     the jsonl of articles at $REHEARSAL_UKWIKI
  anything else               $REHEARSAL_DATA/<dataset with / as __>[__<config>]/<split>.jsonl,
                              one row per line (journey §33.53: corpora as specs)

It is only importable when this directory is put on PYTHONPATH on purpose, so it
can never shadow the real package.
"""
import os
import json


class _Rows(list):
    pass


def _jsonl(path):
    with open(path, encoding='utf-8') as f:
        return _Rows(json.loads(l) for l in f if l.strip())


def load_dataset(name, config=None, split='train', **kw):
    if name == 'wikimedia/wikipedia' and (config is None or str(config).endswith('.uk')):
        path = os.environ.get('REHEARSAL_UKWIKI')
        if not path:
            raise RuntimeError('offline rehearsal: set REHEARSAL_UKWIKI to a jsonl of articles')
        return _jsonl(path)
    if name == 'wikitext':
        root = os.environ.get('REHEARSAL_WIKITEXT')
        if not root:
            raise RuntimeError('offline rehearsal: set REHEARSAL_WIKITEXT for the toy WikiText')
        with open(os.path.join(root, f'{split}.txt'), encoding='utf-8') as f:
            return _Rows({'text': line} for line in f)
    root = os.environ.get('REHEARSAL_DATA')
    if not root:
        raise RuntimeError(f'offline rehearsal: no dataset {name!r} — set REHEARSAL_DATA')
    base = os.path.join(root, name.replace('/', '__') + (f'__{config}' if config else ''))
    path = os.path.join(base, f'{split}.jsonl')
    if not os.path.exists(path):
        raise RuntimeError(f'offline rehearsal: {path} not found')
    return _jsonl(path)


def get_dataset_config_names(name):
    raise RuntimeError('offline rehearsal: use --local for BLiMP')
