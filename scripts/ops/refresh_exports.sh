#!/usr/bin/env bash
# refresh_exports.sh — bring exports/ up to date with results/: every table and every
# figure the documents and the manuscript draw on (journey §33.72). Run it after every
# chain and every result: the manuscript takes its numbers from exports/, so a stale
# file there is a wrong number in the manuscript. A figure whose planned runs are
# missing is not drawn, and the script says so (demo_figures.py).
#     bash scripts/ops/refresh_exports.sh
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
command -v "$PY" >/dev/null || PY=~/miniconda3/envs/morph/bin/python
"$PY" scripts/analysis/export_results.py --out exports/ ;
"$PY" scripts/analysis/demo_figures.py --out exports/figures ;
"$PY" scripts/analysis/learning_law.py --out exports/figures ;
"$PY" scripts/analysis/thesis_figures.py --out exports/figures ;
date '+  exports/ refreshed %F %T' | tee exports/REFRESHED.txt
