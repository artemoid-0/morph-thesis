#!/usr/bin/env python3
"""
cache_pack.py — keep a cache's streams packed, losslessly, and unpack one when it is needed.

A stream's arrays — its token ids and every codes file that describes them — are small integers
and compress well: codes to 3–34% of their size, ids to about a third (zstd at level 9). A machine
that no longer trains from a stream can keep it packed. Each array becomes `<name>.npy.zst` beside
its sidecar `.json`, which stays as it is, and the whole stream is packed or unpacked at once.
Every packed array is checked by SHA-256 against its original before the original goes, and only
with --trash (to the trash, never deleted).

Nothing reads a packed array, and nothing rebuilds one. `morphlm/data/stream.py` stops any run or
script that needs a packed stream and names the command that unpacks it. A machine that trains
keeps its cache unpacked and never meets a packed file. A stream unpacked for use is read as
always; `drop` then sends the unpacked copy to the trash once it is checked against the packed one.

    python scripts/data/cache_pack.py status                        # every stream: packed or not, sizes
    python scripts/data/cache_pack.py pack STEM ... [--trash]       # streams by their ids stem
    python scripts/data/cache_pack.py pack --all --split train [--exclude STEM ...] [--trash] [--dry-run]
    python scripts/data/cache_pack.py unpack STEM ...               # back to .npy, checked
    python scripts/data/cache_pack.py drop STEM ...                 # the unpacked copy to the trash
    python scripts/data/cache_pack.py verify [STEM ...] [--full]    # the packed arrays, against the manifest

The manifest, `<cache>/PACKED.json`, holds each packed array's size and SHA-256 before packing,
its packed size, the level and the time.
"""
import os
import re
import sys
import json
import glob
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from morphlm.data.stream import PACKED, stream_files      # noqa: E402

MANIFEST = 'PACKED.json'
STEM = re.compile(r'(bpe|det|morph)_(train|validation|test)_(None|\d+)_[0-9a-f]{10}')
CHUNK = 64 * 2 ** 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def sha256_unpacked(packed):
    """The SHA-256 of what a packed file unpacks to, without writing it."""
    h = hashlib.sha256()
    p = subprocess.Popen(['zstd', '-d', '-q', '-c', packed], stdout=subprocess.PIPE)
    for b in iter(lambda: p.stdout.read(CHUNK), b''):
        h.update(b)
    if p.wait() != 0:
        raise RuntimeError(f'zstd could not unpack {packed}')
    return h.hexdigest()


def trash(path):
    """Removal is to the trash, never a deletion (the project's rule)."""
    subprocess.run(['gio', 'trash', path], check=True)


def load_manifest(cache_dir):
    try:
        with open(os.path.join(cache_dir, MANIFEST)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_manifest(cache_dir, m):
    path = os.path.join(cache_dir, MANIFEST)
    with open(path + '.tmp', 'w') as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(path + '.tmp', path)


def all_streams(cache_dir):
    """Every stream in the cache, by its ids stem (from the ids' sidecar `.json`)."""
    out = []
    for f in sorted(os.listdir(cache_dir)):
        if f.endswith('.json') and STEM.fullmatch(f[:-5]):
            out.append(f[:-5])
    return out


def arrays(cache_dir, stem):
    """The stream's arrays, packed or not, by their `.npy` names."""
    return sorted(set(stream_files(cache_dir, stem)) | set(stream_files(cache_dir, stem, packed=True)))


def state(cache_dir, stem):
    files = arrays(cache_dir, stem)
    unpacked = [f for f in files if os.path.exists(f)]
    packed = [f for f in files if os.path.exists(f + PACKED)]
    if not files:
        return 'empty', files
    if len(packed) == len(files):
        return ('packed, unpacked too' if len(unpacked) == len(files) else
                'packed' if not unpacked else 'packed, partly unpacked'), files
    if not packed:
        return 'unpacked', files
    return 'partly packed', files


def owners(cache_dir, streams):
    """Which streams claim each array (a build file of a validation split can serve several)."""
    by = {}
    for s in streams:
        for f in arrays(cache_dir, s):
            by.setdefault(f, set()).add(s)
    return by


def labels():
    """A readable name for each BPE stream, from the run configurations that used it."""
    from morphlm.data.stream import cache_stem
    out = {}
    for p in glob.glob(os.path.join(ROOT, 'results', '*', 'config.json')) + \
            glob.glob(os.path.join(ROOT, 'results', 'audits', 'device_twins_models', '*', 'config.json')):
        try:
            d = json.load(open(p))['data']
        except (OSError, ValueError, KeyError):
            continue
        if d.get('pipeline') != 'bpe':
            continue
        tok = os.path.basename(d['bpe_path'])[:-5]
        base = '' if tok == 'bpe_tokenizer' else f'tok:{tok}'
        corpus = d.get('corpus', 'wikitext103')
        name = tok
        if corpus != 'wikitext103':
            frac = float(d.get('corpus_fraction', 1.0))
            base = (base + '|' if base else '') + f'corpus:{corpus}:{frac:g}:{d.get("corpus_holdout", 2000)}:nl'
            name = f'{tok}, {corpus} {frac:g}'
        for split in ('train', 'validation', 'test'):
            out[cache_stem('bpe', split, None, base)] = name
    return out


def gib(n):
    return f'{n / 2 ** 30:6.2f}'


def cmd_status(a):
    lab = labels()
    man = load_manifest(a.cache_dir)
    tot_u = tot_p = 0
    print(f'  {"stream":30} {"what":34} {"state":24} {"unpacked GiB":>12} {"packed GiB":>10}')
    for s in all_streams(a.cache_dir):
        if a.split and STEM.fullmatch(s).group(2) != a.split:
            continue
        st, files = state(a.cache_dir, s)
        if st == 'empty':
            continue
        u = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        p = sum(os.path.getsize(f + PACKED) for f in files if os.path.exists(f + PACKED))
        if not u and p:
            u_full = sum(man.get(os.path.basename(f), {}).get('bytes', 0) for f in files)
            note = f' ({gib(u_full).strip()} when unpacked)'
        else:
            note = ''
        tot_u += u
        tot_p += p
        what = lab.get(s, 'the morpheme arms' if s.startswith(('det_', 'morph_')) else '—')
        print(f'  {s:30} {what[:34]:34} {st:24} {gib(u):>12} {gib(p):>10}{note}')
    print(f'  {"on disk":90} {gib(tot_u):>12} {gib(tot_p):>10}')


def select(a):
    streams = all_streams(a.cache_dir)
    if a.all:
        chosen = [s for s in streams if (not a.split or STEM.fullmatch(s).group(2) == a.split)]
    else:
        unknown = [s for s in a.stems if s not in streams]
        if unknown:
            raise SystemExit(f'no such stream: {" ".join(unknown)}')
        chosen = list(a.stems)
    return [s for s in chosen if s not in set(a.exclude or [])], streams


def cmd_pack(a):
    chosen, streams = select(a)
    by = owners(a.cache_dir, streams)
    man = load_manifest(a.cache_dir)
    done_u = done_p = 0
    for s in chosen:
        st, files = state(a.cache_dir, s)
        if st in ('empty', 'packed'):
            continue
        shared = sorted({o for f in files for o in by.get(f, ()) if o != s and o not in chosen})
        if shared:
            print(f'  {s}: an array of it also belongs to {", ".join(shared)} — pack them together; skipped')
            continue
        todo = [f for f in files if not os.path.exists(f + PACKED)]
        size = sum(os.path.getsize(f) for f in todo)
        free = shutil.disk_usage(a.cache_dir).free
        print(f'  {s}: {len(files)} arrays, {gib(size).strip()} GiB to pack ({gib(free).strip()} GiB free)')
        if a.dry_run:
            continue
        if free < size // 2 + 2 * 2 ** 30:
            raise SystemExit('  not enough free space for the packed copies; stopped')
        for f in todo:
            name = os.path.basename(f)
            digest = sha256(f)
            part = f + PACKED + '.part'
            subprocess.run(['zstd', f'-{a.level}', '-T0', '--check', '-q', '-f', f, '-o', part], check=True)
            if sha256_unpacked(part) != digest:
                raise SystemExit(f'  {name}: the packed copy does not unpack to the original; stopped')
            os.replace(part, f + PACKED)
            man[name] = {'bytes': os.path.getsize(f), 'sha256': digest, 'packed_bytes': os.path.getsize(f + PACKED),
                         'level': a.level, 'packed_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
            save_manifest(a.cache_dir, man)
            done_u += man[name]['bytes']
            done_p += man[name]['packed_bytes']
        # every array of the stream is packed and checked before any original goes
        for f in files:
            name = os.path.basename(f)
            assert os.path.exists(f + PACKED) and name in man, name
        if a.trash:
            for f in files:
                if os.path.exists(f):
                    if os.path.getsize(f) != man[os.path.basename(f)]['bytes']:
                        raise SystemExit(f'  {f} changed since it was packed; kept')
                    trash(f)
        print(f'    packed{" and the originals sent to the trash" if a.trash else ""}')
    if done_u:
        print(f'  {gib(done_u).strip()} GiB packed into {gib(done_p).strip()} GiB '
              f'({100 * done_p / done_u:.1f}%)')


def cmd_unpack(a):
    chosen, _ = select(a)
    man = load_manifest(a.cache_dir)
    for s in chosen:
        _, files = state(a.cache_dir, s)
        todo = [f for f in files if not os.path.exists(f) and os.path.exists(f + PACKED)]
        lost = [os.path.basename(f) for f in todo if os.path.basename(f) not in man]
        if lost:
            raise SystemExit(f'  {s}: packed, but not in the manifest, so it cannot be checked: {" ".join(lost)}')
        need = sum(man[os.path.basename(f)]['bytes'] for f in todo)
        free = shutil.disk_usage(a.cache_dir).free
        if need + 2 * 2 ** 30 > free:
            raise SystemExit(f'  {s}: {gib(need).strip()} GiB to unpack, {gib(free).strip()} GiB free; stopped')
        for f in todo:
            name = os.path.basename(f)
            part = f + '.part'
            subprocess.run(['zstd', '-d', '-q', '-f', f + PACKED, '-o', part], check=True)
            if sha256(part) != man[name]['sha256']:
                raise SystemExit(f'  {name}: unpacked, but not to the array that was packed; left as {part}')
            os.replace(part, f)
        print(f'  {s}: {len(todo)} arrays unpacked ({gib(need).strip()} GiB), each checked')


def cmd_drop(a):
    chosen, _ = select(a)
    man = load_manifest(a.cache_dir)
    for s in chosen:
        _, files = state(a.cache_dir, s)
        both = [f for f in files if os.path.exists(f) and os.path.exists(f + PACKED)]
        if len(both) != len(files):
            print(f'  {s}: not every array is both packed and unpacked; nothing dropped')
            continue
        for f in both:
            if sha256(f) != man[os.path.basename(f)]['sha256']:
                raise SystemExit(f'  {f} differs from the packed array; kept')
        for f in both:
            trash(f)
        print(f'  {s}: the unpacked copy ({len(both)} arrays) sent to the trash; the packed one stays')


def cmd_verify(a):
    streams = a.stems or [s for s in all_streams(a.cache_dir) if stream_files(a.cache_dir, s, packed=True)]
    man = load_manifest(a.cache_dir)
    bad = 0
    for s in streams:
        for f in stream_files(a.cache_dir, s, packed=True):
            name = os.path.basename(f)
            if name not in man:
                print(f'  {name}: packed, but not in the manifest'); bad += 1
                continue
            if a.full:
                ok = sha256_unpacked(f + PACKED) == man[name]['sha256']
            else:
                ok = subprocess.run(['zstd', '-t', '-q', f + PACKED]).returncode == 0
            if not ok:
                print(f'  {name}: FAILED'); bad += 1
    print(f'  {len(streams)} streams checked ({"SHA-256 of the unpacked bytes" if a.full else "zstd integrity"}): '
          + ('all good' if not bad else f'{bad} problems'))
    if bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('status', 'pack', 'unpack', 'drop', 'verify'))
    ap.add_argument('stems', nargs='*', help='streams, by their ids stem (bpe_train_None_25577fdac5)')
    ap.add_argument('--all', action='store_true', help='every stream (of --split)')
    ap.add_argument('--split', choices=('train', 'validation', 'test'), default=None)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--level', type=int, default=9)
    ap.add_argument('--trash', action='store_true', help='pack: send each original to the trash once its stream is packed')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--full', action='store_true', help='verify: unpack each array and compare its SHA-256')
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    a = ap.parse_args()
    if a.command in ('pack', 'unpack', 'drop') and not (a.stems or a.all):
        ap.error('name the streams, or --all')
    {'status': cmd_status, 'pack': cmd_pack, 'unpack': cmd_unpack, 'drop': cmd_drop, 'verify': cmd_verify}[a.command](a)


if __name__ == '__main__':
    main()
