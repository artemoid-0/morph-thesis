#!/usr/bin/env bash
#
# pack.sh — bundle the current project state into a single archive
#           to share with a fresh assistant session.
#
# Usage:
#   ./pack.sh            # writes morph_snapshot_YYYYMMDD_HHMM.tar.gz
#   ./pack.sh --zip      # writes a .zip instead (if you prefer)
#   ./pack.sh --no-csv   # code + docs only, skip the vocab CSVs
#
# Run from the project root (where the .py files live).

set -euo pipefail

# --- locate project root (dir containing this script) ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- parse flags ---
USE_ZIP=0
INCLUDE_CSV=1
for arg in "$@"; do
  case "$arg" in
    --zip)    USE_ZIP=1 ;;
    --no-csv) INCLUDE_CSV=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M)"
# Write the archive to the parent dir (e.g. PycharmProjects/) so it lands
# outside the packed tree and can't be swept into a later snapshot.
OUTDIR="$(dirname "$ROOT")"
NAME="${OUTDIR}/morph_snapshot_${STAMP}"

# --- build the file list ---------------------------------------------------
# Code: every .py in the root, but NOT temp.py / scratch files.
FILES=()

for f in *.py; do
  [ -e "$f" ] || continue
  case "$f" in
    temp.py|scratch*.py) continue ;;   # skip scratch
  esac
  FILES+=("$f")
done

# Docs: the decision history. (HANDOFF.md lives in project knowledge, not on disk.)
for doc in research_journey.md; do
  [ -e "$doc" ] && FILES+=("$doc")
done

# Lightweight, readable, regenerable assets (optional).
if [ "$INCLUDE_CSV" -eq 1 ]; then
  for csv in data/our_final_vocab.csv data/entity_vocab.csv; do
    [ -e "$csv" ] && FILES+=("$csv")
  done
fi

# --- explicitly EXCLUDED (documented here so it's obvious why) -------------
#   data/cache/*            large, regenerable .npy/.json token streams
#   data/MorphoLEX_en.xlsx  binary blob; irreplaceable but not code — back up separately
#   data/bpe_tokenizer.json regenerable, low-signal merge rules
#   __pycache__/            compiled junk
#   temp.py                 scratch

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "error: no files matched — are you in the project root?" >&2
  exit 1
fi

# --- pack ------------------------------------------------------------------
if [ "$USE_ZIP" -eq 1 ]; then
  OUT="${NAME}.zip"
  rm -f "$OUT"
  zip -q "$OUT" "${FILES[@]}"
else
  OUT="${NAME}.tar.gz"
  tar -czf "$OUT" "${FILES[@]}"
fi

# --- report ----------------------------------------------------------------
echo "packed ${#FILES[@]} files -> $OUT"
echo "----"
for f in "${FILES[@]}"; do
  echo "  $f"
done
echo "----"
du -h "$OUT" | cut -f1 | sed 's/^/size: /'