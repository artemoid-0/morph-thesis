#!/usr/bin/env bash
#
# pack.sh — bundle the project for sharing with a fresh assistant session.
#
# Writes ../morph_snapshot.tar.gz, OVERWRITING the previous one (no versioning:
# only the current state is ever wanted).
#
# SELECTION MODEL
#   1. an extension in INCLUDE_EXT is included by default
#   2. anything under an EXCLUDE_DIR is dropped, whatever its extension
#   3. EXCLUDE_GLOB drops specific paths that rule 1 would otherwise catch
#   4. INCLUDE_PATH force-adds specific files whose extension is not in rule 1
#
# So the common case is "by file type", and the exceptions are explicit and
# few, which is what keeps this readable a year from now.
#
# Usage:
#   ./pack.sh              # write ../morph_snapshot.tar.gz
#   ./pack.sh --zip        # ../morph_snapshot.zip instead
#   ./pack.sh --list       # show what would be packed, write nothing
#   ./pack.sh --out DIR    # write somewhere other than the parent directory

set -euo pipefail

# ── selection rules ───────────────────────────────────────────────────────────

# included by extension
INCLUDE_EXT=(py md sh toml cfg ini txt)

# dropped wholesale, at any depth
EXCLUDE_DIR=(__pycache__ .git .idea .vscode .pytest_cache node_modules
             results wandb runs checkpoints)

# dropped even though the extension matched
#   data/cache      large regenerable token streams (.npy + .json)
#   *.egg-info      build noise
EXCLUDE_GLOB=('data/cache/*' '*.egg-info/*' '*.pyc' 'temp.py' 'scratch*.py')

# force-added despite not matching INCLUDE_EXT
#   the vocab CSVs are readable, small, and useful to inspect in-session.
#   NOT included: data/MorphoLEX_en.xlsx    (binary, irreplaceable — back it up
#                                            separately, it is not code)
#                 data/bpe_tokenizer.json   (regenerable merge rules, low signal)
INCLUDE_PATH=(data/our_final_vocab.csv data/entity_vocab.csv)

# ── args ──────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

USE_ZIP=0
LIST_ONLY=0
OUTDIR="$(dirname "$ROOT")"

while [ $# -gt 0 ]; do
  case "$1" in
    --zip)  USE_ZIP=1 ;;
    --list) LIST_ONLY=1 ;;
    --out)  shift; OUTDIR="$1" ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

# ── build the file list ───────────────────────────────────────────────────────

FIND_ARGS=()
for d in "${EXCLUDE_DIR[@]}"; do
  FIND_ARGS+=( -name "$d" -prune -o )
done

EXT_EXPR=()
for e in "${INCLUDE_EXT[@]}"; do
  EXT_EXPR+=( -name "*.${e}" -o )
done
unset 'EXT_EXPR[${#EXT_EXPR[@]}-1]'   # drop trailing -o

mapfile -t CANDIDATES < <(
  find . "${FIND_ARGS[@]}" \( "${EXT_EXPR[@]}" \) -type f -print \
  | sed 's|^\./||' | sort
)

keep() {
  local f="$1"
  for g in "${EXCLUDE_GLOB[@]}"; do
    # shellcheck disable=SC2053
    [[ "$f" == $g ]] && return 1
  done
  return 0
}

FILES=()
for f in "${CANDIDATES[@]}"; do
  keep "$f" && FILES+=("$f")
done

for f in "${INCLUDE_PATH[@]}"; do
  [ -e "$f" ] && FILES+=("$f")
done

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "error: nothing matched — are you in the project root?" >&2
  exit 1
fi

# ── report / pack ─────────────────────────────────────────────────────────────

if [ "$LIST_ONLY" -eq 1 ]; then
  printf '%s\n' "${FILES[@]}"
  echo "---- ${#FILES[@]} files (nothing written)"
  exit 0
fi

if [ "$USE_ZIP" -eq 1 ]; then
  OUT="${OUTDIR}/morph_snapshot.zip"
  rm -f "$OUT"
  zip -q "$OUT" "${FILES[@]}"
else
  OUT="${OUTDIR}/morph_snapshot.tar.gz"
  rm -f "$OUT"
  tar -czf "$OUT" "${FILES[@]}"
fi

echo "packed ${#FILES[@]} files -> $OUT"
printf '  %s\n' "${FILES[@]}" | head -40
[ "${#FILES[@]}" -gt 40 ] && echo "  ... and $(( ${#FILES[@]} - 40 )) more"
echo "size: $(du -h "$OUT" | cut -f1)"
