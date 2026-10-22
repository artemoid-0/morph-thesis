#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 22 October 2026 (journey §33.130): the text's request of 22 October,
# item 1 — G-81's recipe question. At width 1024 two of four endings-apart nulls lose much of the
# tokenizer's gain early in training (§33.120–§33.124); is the 100-step warm-up, the same at every
# width, at fault at this width?
#
#   wu1024   width 1024 (85.3M parameters), ~38M tokens, 10k, a 1,000-step warm-up, seeds 0–3: the
#            null under BPE 17k and under endings apart                              P296, P297
#   wu512    width 512 (30M), the same warm-up: the reference under the same recipe  P297, P298
#   mbwu     MultiBLiMP on each width's eight, once all are here                     P299
#
# Every other run of the thesis has a 100-step warm-up; these are labelled `_wu1000`. Gender and
# number at full training (item 3) follow in second_full_2026-10-22.sh, started after this chain
# lets go of its lock. Guards: ready; pair (the first seed's two nulls end within SANITY_TOL of each
# other); gpu, disk, END_BY — a run starts only if it can end before END_BY. Every run is a RESUMABLE
# PIECE. No tables here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-22.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-wu1024 wu512 mbwu}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
null() {        # null TOKENIZER WIDTH SEED — the null at ~38M tokens and 10k, with the longer warm-up
    echo "--facts mconst $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json $C $(steps 10000) -o model.d_model=$2 -o train.warmup=1000 -o train.seed=$3"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        wu1024) echo 48 ;;              # width 1024, 10k: 44.1–44.4 min (20 October)
        wu512)  echo 19 ;;              # width 512, 10k: ~16 min (19 and 21 October)
    esac
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

multiblimp_wu() {       # MultiBLiMP on each width's eight nulls: once all eight are here, once
    local w s t missing
    for w in 1024 512; do
        stamp "mbwu: MultiBLiMP on width $w's nulls with the longer warm-up, seeds 0–3"
        if python -c "$MB_RECORD" _wu1000_st10k_w$w; then echo "  [skip] mbwu, width $w: a record of it is already here"; continue; fi
        missing=0
        for s in 0 1 2 3; do
            for t in bpe_uk_17k bpeend_uk_17k; do
                python scripts/ops/run_ids.py --done $(null $t $w $s) >/dev/null || missing=$((missing + 1))
            done
        done
        if [ "$missing" -gt 0 ]; then echo "  [skipped] mbwu, width $w: $missing of the 8 runs are not here"; continue; fi
        if [ -n "${PLAN:-}" ]; then echo "  [would run] mbwu, width $w — minutes"; continue; fi
        gpu_ok || return 0
        python scripts/eval/multiblimp_eval.py --machine second \
            --labels bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_wu1000_st10k_w$w \
                     bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_wu1000_st10k_w$w \
            || echo "  [mbwu] MultiBLiMP at width $w FAILED — the chain goes on"
    done
}

run_piece() {
    case "$1" in
        wu1024) seedpiece "wu1024: G-81 — the nulls at width 1024 with a 1,000-step warm-up, ~38M tokens, 10k" wu1024 \
                    "0 1 2 3" pair "null bpe_uk_17k 1024" "null bpeend_uk_17k 1024" ;;
        wu512)  seedpiece "wu512: G-81 — the nulls at width 512 with a 1,000-step warm-up, ~38M tokens, 10k" wu512 \
                    "0 1 2 3" pair "null bpe_uk_17k 512" "null bpeend_uk_17k 512" ;;
        mbwu)   multiblimp_wu ;;
    esac
}

chain_main "the chain of 22 October: G-81's recipe question, a longer warm-up at width 1024"
