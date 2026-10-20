#!/usr/bin/env bash
# refresh_exports.sh — bring exports/ up to date with results/: every table and every
# figure the documents and the manuscript draw on (journey §33.72). Run it after every
# chain and every result: the manuscript takes its numbers from exports/, so a stale
# file there is a wrong number in the manuscript. A figure whose planned runs are
# missing is not drawn, and the script says so (demo_figures.py).
#     bash scripts/ops/refresh_exports.sh
#
# Every step runs, even after one fails. exports/REFRESHED.txt is written only when every
# step succeeded and every file in exports/ was rewritten by them (or is on NOT_DRAWN below);
# otherwise the script prints "exports/ NOT refreshed", names the failed steps and the files
# left stale, and exits 1. (On 12 and 14 October export_results.py crashed, "refreshed" was
# printed all the same, and exports/audits.md stayed stale through three commits.)
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
command -v "$PY" >/dev/null || PY=~/miniconda3/envs/morph/bin/python

STEPS=(
    "scripts/analysis/export_results.py --out exports/"
    "scripts/analysis/demo_figures.py --out exports/figures"
    "scripts/analysis/learning_law.py --out exports/figures"
    "scripts/analysis/thesis_figures.py --out exports/figures"
)
# Files in exports/ that no step above draws, so they are never reported stale; each with
# its reason. Nothing else may stay unrewritten.
NOT_DRAWN=(
    # none since 20 October: probe_axis_posprev.png, drawn once on 7 October by
    # scripts/analysis/probe_axis.py and used by no document, left exports/
)

# the start marker: a file the steps leave older than the marker was not rewritten
mkdir -p logs exports
MARK=logs/.refresh_exports.start
touch "$MARK" || exit 1

FAILED=()
for step in "${STEPS[@]}"; do
    # shellcheck disable=SC2086
    "$PY" $step || FAILED+=("${step%% *}")
done

STALE=()
while IFS= read -r f; do
    keep=""
    for n in "${NOT_DRAWN[@]}"; do [ "$f" = "$n" ] && keep=1; done
    [ -n "$keep" ] || STALE+=("$f")
done < <(find exports -type f ! -newer "$MARK" ! -name REFRESHED.txt | sort)

if [ ${#FAILED[@]} -eq 0 ] && [ ${#STALE[@]} -eq 0 ]; then
    date '+  exports/ refreshed %F %T' | tee exports/REFRESHED.txt
else
    echo "  exports/ NOT refreshed — exports/REFRESHED.txt keeps the last full refresh:"
    for s in "${FAILED[@]}"; do echo "    failed:        $s"; done
    for f in "${STALE[@]}"; do echo "    not rewritten: $f"; done
    exit 1
fi
