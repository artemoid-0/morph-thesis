#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run this BEFORE the night chain, while you are awake — about 3–5 minutes.
# It exercises on your GPU everything the night relies on that could not be
# run where it was written. Unlike the night chain it STOPS at the first
# failure: if it does not end with "ALL SMOKE CHECKS PASSED", send the output
# back instead of starting the night.
#
#   bash scripts/chains/smoke_2026-09-26.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
set -e

echo "== 1/5  test suites and repository checks"
python scripts/ops/verify.py

echo; echo "== 2/5  every cell the night trains: real model, one forward+backward pass"
python scripts/ops/smoke_model.py --cells mstruct pos mfeatlex mstruct,mfeatlex mfeat mstruct,mfeat

echo; echo "== 3/5  BLiMP scorer on the GPU: encoding, batched-vs-single self-check, pairing"
python scripts/eval/blimp_eval.py --runs a7175857fdf6 648675b80ba0 \
    --paradigms determiner_noun_agreement_1 anaphor_gender_agreement --limit 40

echo; echo "== 4/5  tagger causality harness, 30 sentences"
python scripts/audits/morph_causality_check.py --sentences 30

echo; echo "== 5/5  the rebuild gate itself (tokenization and existing facts reproduce the cache)"
python scripts/data/preflight_rebuild.py

echo; echo "ALL SMOKE CHECKS PASSED — start the night:"
echo "    tmux new -s night"
echo "    bash scripts/chains/night_2026-09-26.sh            # ~10.5 h"
echo "    TAIL=1 bash scripts/chains/night_2026-09-26.sh     # ~12.5 h, adds the seed-4 pair"
