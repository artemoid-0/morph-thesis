#!/usr/bin/env bash
#
# project_health.sh — diagnose a PyCharm project that shows no files.
#
# Read-only: inspects and reports, changes nothing.
#
#   bash scripts/ops/project_health.sh

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1      # the project root, from scripts/ops/
ROOT="$(pwd)"

echo "=============================================="
echo " project health check"
echo " root: $ROOT"
echo "=============================================="

# ── 1. the usual culprit: missing IDE config ─────────────────────────────────
echo
echo "[1] PyCharm project config (.idea/)"
if [ -d .idea ]; then
  echo "  .idea/ EXISTS"
  for f in modules.xml misc.xml; do
    if [ -s ".idea/$f" ]; then echo "    ok      $f"
    elif [ -e ".idea/$f" ]; then echo "    EMPTY   $f   <-- corrupt"
    else echo "    MISSING $f"; fi
  done
  iml_count=$(find .idea -maxdepth 1 -name '*.iml' | wc -l)
  if [ "$iml_count" -eq 0 ]; then
    echo "    MISSING *.iml   <-- no module definition: panel will look empty"
  else
    find .idea -maxdepth 1 -name '*.iml' | while read -r f; do
      echo "    ok      $(basename "$f")"
    done
  fi
  if [ -f .idea/modules.xml ]; then
    echo "    modules.xml points at:"
    grep -o 'filepath="[^"]*"' .idea/modules.xml | sed 's/^/      /'
  fi
else
  echo "  .idea/ IS MISSING  <-- this is almost certainly the problem"
  echo "     PyCharm stores the content root and module definition here."
  echo "     Without it the project tree renders empty."
fi

# ── 2. are the files actually there? ─────────────────────────────────────────
echo
echo "[2] Files on disk"
for p in morphlm tests scripts data docs/record/journey/README.md; do
  if [ -e "$p" ]; then echo "  ok      $p"; else echo "  MISSING $p"; fi
done
echo "  python modules: $(find morphlm -name '*.py' 2>/dev/null | wc -l) files"

# ── 3. git state ─────────────────────────────────────────────────────────────
echo
echo "[3] Git"
if [ -d .git ]; then
  echo "  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "  branches:"
  git for-each-ref --format='    %(refname:short)  %(objectname:short)  %(contents:subject)' refs/heads 2>/dev/null
  echo "  uncommitted changes: $(git status --porcelain 2>/dev/null | wc -l)"
  if git log --oneline --all --diff-filter=D -- .idea 2>/dev/null | head -1 | grep -q .; then
    echo "  .idea was DELETED in:"
    git log --oneline --all --diff-filter=D -- .idea 2>/dev/null | sed 's/^/    /'
    echo "  recoverable from history (see fix below)"
  fi
else
  echo "  no .git directory"
fi

# ── 4. stale bytecode from deleted modules ───────────────────────────────────
echo
echo "[4] Stale bytecode"
stale=$(find . -name '__pycache__' -type d 2>/dev/null | wc -l)
echo "  __pycache__ dirs: $stale"
for old in common model dataset experiment train frozen_embedding; do
  hit=$(find . -name "${old}.cpython*.pyc" 2>/dev/null | head -1)
  [ -n "$hit" ] && echo "  STALE: $hit  (module was deleted; may shadow imports)"
done

# ── 5. does the package import? ──────────────────────────────────────────────
echo
echo "[5] Package import"
# the project's interpreter, as the chains find it, when no `python` is on PATH
PY=$(command -v python || echo "$HOME/miniconda3/envs/morph/bin/python")
"$PY" -c "import sys; sys.path.insert(0,'.'); import morphlm; print('  ok      morphlm', morphlm.__version__, 'with', sys.executable)" 2>&1 | tail -2

# ── 6. interpreter ───────────────────────────────────────────────────────────
echo
echo "[6] Python environment"
echo "  python:  $(command -v python || echo '<not on PATH>')"
echo "  version: $("$PY" --version 2>&1) ($PY)"
echo "  conda:   ${CONDA_DEFAULT_ENV:-<none active>}"
if [ "${CONDA_DEFAULT_ENV:-}" != "morph" ]; then
  echo "  NOTE: the 'morph' env is not active. PyCharm's interpreter is stored"
  echo "        in .idea/misc.xml — if .idea was regenerated you must re-select it."
fi

echo
echo "=============================================="
echo " done"
echo "=============================================="
