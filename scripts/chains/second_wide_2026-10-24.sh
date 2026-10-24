#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 24 October 2026, after second_2026-10-24.sh (journey §33.137): the
# text's request of 24 October, item 1 — width 1024 without repeating the text. At 20k on ~38M tokens
# the BPE 17k null stops improving after 4.3 passes, and the last 1,500 steps add about 0.0025 to the
# tokenizer's gain (§33.124). Does the gain hold at 20k when the text is not repeated?
#
#   wide151   the nulls at width 1024, 20k, on the 151M-token corpus (about 1.1 passes), a 1,000-step
#             warm-up, under BPE 17k and endings apart, seeds 0–3                       P310–P312
#   wide38    the same on ~38M tokens (4.3 passes): the repeated text under the same warm-up, so that
#             the two differ in the repetition alone
#   mbwide    MultiBLiMP on each corpus's eight, once all are here                       P313
#
# The 1,000-step warm-up is the one the width needs (§33.132); the 20k runs of §33.124 had 100 steps.
# Guards: ready; pair (the first seed's two nulls end within SANITY_TOL of each other); gpu, disk,
# END_BY — a run starts only if it can end before END_BY. Every run is a RESUMABLE PIECE. No tables
# here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_wide_2026-10-24.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-wide151 wide38 mbwide}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
null() {        # null FRACTION TOKENIZER SEED — width 1024, 20k, the longer warm-up
    echo "--facts mconst $U -o data.corpus_fraction=$1 -o data.bpe_path=data/$2.json $C $(steps 20000) -o model.d_model=1024 -o train.warmup=1000 -o train.seed=$3"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in wide151|wide38) echo 94 ;; esac        # width 1024, 20k: 88.2–89.2 min (20–21 October)
}

# exits 0 when a MultiBLiMP record of labels ending in its argument is already here
MB_RECORD='import glob, json, sys
for p in glob.glob("results/audits/multiblimp_eval_*.json"):
    try:
        a = json.load(open(p)).get("args") or {}
    except (OSError, ValueError):
        continue
    if any(str(l).endswith(sys.argv[1]) for l in a.get("labels") or []):
        sys.exit(0)
sys.exit(1)'

multiblimp_wide() {     # MultiBLiMP on each corpus's eight nulls: once all eight are here, once
    local fr s t missing
    for fr in 0.2 0.05; do
        stamp "mbwide: MultiBLiMP on width 1024's 20k nulls on ukwiki $fr, seeds 0–3"
        if python -c "$MB_RECORD" _ukwiki${fr}_wu1000_st20k_w1024; then echo "  [skip] mbwide, $fr: a record of it is already here"; continue; fi
        missing=0
        for s in 0 1 2 3; do
            for t in bpe_uk_17k bpeend_uk_17k; do
                python scripts/ops/run_ids.py --done $(null $fr $t $s) >/dev/null || missing=$((missing + 1))
            done
        done
        if [ "$missing" -gt 0 ]; then echo "  [skipped] mbwide, $fr: $missing of the 8 runs are not here"; continue; fi
        if [ -n "${PLAN:-}" ]; then echo "  [would run] mbwide, $fr — minutes"; continue; fi
        gpu_ok || return 0
        python scripts/eval/multiblimp_eval.py --machine second \
            --labels bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki${fr}_wu1000_st20k_w1024 \
                     bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki${fr}_wu1000_st20k_w1024 \
            || echo "  [mbwide] MultiBLiMP on $fr FAILED — the chain goes on"
    done
}

run_piece() {
    case "$1" in
        wide151) seedpiece "wide151: item 1 — the nulls at width 1024, 20k, on 151M tokens, a 1,000-step warm-up" wide151 \
                     "0 1 2 3" pair "null 0.2 bpe_uk_17k" "null 0.2 bpeend_uk_17k" ;;
        wide38)  seedpiece "wide38: item 1 — the nulls at width 1024, 20k, on ~38M tokens, a 1,000-step warm-up" wide38 \
                     "0 1 2 3" pair "null 0.05 bpe_uk_17k" "null 0.05 bpeend_uk_17k" ;;
        mbwide)  multiblimp_wide ;;
    esac
}

chain_main "the chain of 24 October, after the empty slot: width 1024 without repeating the text"
