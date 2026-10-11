#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Short block, prepared 9 October 2026 (journey §33.74): G-38's control, the
# title's question — what the sentence's case adds over the word form's:
# context, or the tagger's right-context leak at prepositions?
#
# `ucaseprep` is case in context everywhere except at prepositions, which carry
# their form's majority case (derive_lexical_codes.py --recode-pos ADP; derived
# and gated 9 October). At ~38M tokens (fraction 0.05) and a 10k budget it
# joins the null, case and the word form already recorded there, seeds 0–3:
#
#   1  seed 0 under BPE 17k and under BPE 17k that keeps endings apart   ~24 min
#   2  seeds 1–3, the same                                                ~72 min  P164–P168
#   then the tables and figures (seconds).
#
# Bold, not quiet: a new derived fact, so it runs as a block of its own and is
# reported when it ends. Guard, recorded before it runs: after seed 0, each
# recoded run must end within 0.005 of case at seed 0 under its tokenizer — the
# two inputs differ at 2.65% of words, so a larger gap means codes misread, not
# an effect; otherwise seeds 1–3 are skipped. A smoke test per tokenizer first.
# Every run is a resumable piece (scripts/ops/run_ids.py --done).
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/short_2026-10-09.sh >/dev/null 2>&1 &
# Knobs: WAIT_RUNS (0: no wait), MIN_FREE_GB; UKEXTRA (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/short_2026-10-09.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

TOKS=${TOKS:-bpe_uk_17k bpeend_uk_17k}
MIN_FREE_GB=${MIN_FREE_GB:-1}
UK="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S10="-o train.max_steps=10000 -o train.lr_horizon=10000"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"
args_for() { echo "--facts $2 $UK -o data.bpe_path=data/$1.json $C $S10 -o train.seed=$3 ${UKEXTRA:-}"; }   # TOKENIZER FACT SEED
free_gb() { df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
run() {
    python scripts/ops/run_ids.py --done $(args_for "$@") && return 0
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then stamp "DISK: below $MIN_FREE_GB GB — not starting $*"; return 0; fi
    $RUN $(args_for "$@")
}
bpb_of() { python scripts/ops/run_ids.py --bpb $(args_for "$@") 2>/dev/null; }

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "short block of 9 October: G-38's control — case with prepositions read from their form"

OK_TOKS=""
for t in $TOKS; do
    if python scripts/ops/smoke_model.py --cells ucaseprep --bpe-path "data/$t.json" \
            --set data.corpus=ukwiki data.corpus_fraction=0.05; then OK_TOKS="$OK_TOKS $t"
    else stamp "SMOKE TEST FAILED for ucaseprep under $t — skipped"; fi
done

stamp "1  seed 0:$OK_TOKS"
for t in $OK_TOKS; do run "$t" ucaseprep 0; done
SANE=yes
for t in $OK_TOKS; do
    new=$(bpb_of "$t" ucaseprep 0); ref=$(bpb_of "$t" ucase 0)
    if [ -n "$new" ] && [ -n "$ref" ] && awk -v a="$new" -v b="$ref" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.005) }'; then
        echo "  [guard] $t: ucaseprep $new, case $ref — within 0.005, passes"
    else
        echo "  [guard] $t: ucaseprep ${new:-none}, case ${ref:-none} — NOT within 0.005, fails"; SANE=no
    fi
done
if [ "$SANE" = yes ] && [ -n "$OK_TOKS" ]; then
    stamp "2  seeds 1–3"
    for s in 1 2 3; do for t in $OK_TOKS; do run "$t" ucaseprep "$s"; done; done
else
    stamp "2  SKIPPED — the seed-0 guard did not pass"
fi
stamp "tables and figures"
bash scripts/ops/refresh_exports.sh ;
stamp "SHORT BLOCK DONE"
