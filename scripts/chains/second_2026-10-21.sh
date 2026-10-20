#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 21 October 2026 (journey §33.123): the thesis's list of 19 October,
# after item 1's 20k half (scripts/chains/second_2026-10-20.sh), in the text's order as far as each
# item is ready (item 2, number and gender, follows when its preparation is done):
#
#   mb20k    MultiBLiMP on width 1024's 20k nulls, seeds 0–3, once all eight are here     P279
#   en100k   item 3 — English part of speech off from the start and off at 5,000, 100k, seeds 0–3
#                                                                                   P280–P282
#   uni512   Unigram 17k at width 512, ~38M tokens, 10k, seeds 0–3: the null, case and the empty
#            slot (item 4's cells under the third exposing tokenizer)               P283–P285
#
# Started after item 1's chain has let go of its lock (flock on logs/.second_2026-10-20.lock), so
# the two never share the GPU. Guards: ready; pair (the first seed's first two cells end within
# SANITY_TOL of each other); gpu, disk, END_BY — a run starts only if it can end before END_BY.
# Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-21.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-mb20k en100k uni512}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
tok() { echo "$U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json"; }
# item 1's cell (second_2026-10-20.sh), for mb20k's check that its runs are all here
null() { echo "--facts mconst $(tok $1) $C $(steps $2) -o model.d_model=1024 -o train.seed=$3"; }
# item 3: English part of speech, never fed (OFF0) and switched off at 5,000 (night_2026-10-16.sh
# piece posoff, at twice its budget)
posoff() { echo "--facts pos -o data.bpe_path=data/bpe_17k.json $C $(steps 100000) -o frozen.feature_off_step=$1 -o train.seed=$2"; }
# item 6: second_2026-10-19.sh's cells under Unigram 17k
W="-o model.d_model=512"
null_u_w512()  { echo "--facts mconst $(tok unigram_uk_17k) $C $(steps 10000) $W -o train.seed=$1"; }
case_u_w512()  { echo "--facts ucase $(tok unigram_uk_17k) $C $(steps 10000) $W -o train.seed=$1"; }
empty_u_w512() { echo "--facts mconst $(tok unigram_uk_17k) $C $(steps 10000) $W -o frozen.feature_off_step=0 -o train.seed=$1"; }

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        en100k) echo 46 ;;              # 50k at width 128: 21.0 min (16 October), so ~42 min for 100k
        uni512) echo 18 ;;              # width 512, 10k: ~16 min (19 October)
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

multiblimp_20k() {      # MultiBLiMP on width 1024's 20k nulls: once all eight runs are here, once
    local s t missing=0
    stamp "mb20k: MultiBLiMP on width 1024's 20k nulls, seeds 0–3"
    if python -c "$MB_RECORD" _st20k_w1024; then echo "  [skip] mb20k: a record of it is already here"; return 0; fi
    for s in 0 1 2 3; do
        for t in bpe_uk_17k bpeend_uk_17k; do
            python scripts/ops/run_ids.py --done $(null $t 20000 $s) >/dev/null || missing=$((missing + 1))
        done
    done
    if [ "$missing" -gt 0 ]; then echo "  [skipped] mb20k: $missing of the 8 runs are not here"; return 0; fi
    if [ -n "${PLAN:-}" ]; then echo "  [would run] mb20k — minutes"; return 0; fi
    gpu_ok || return 0
    python scripts/eval/multiblimp_eval.py --machine second \
        --labels bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st20k_w1024 \
                 bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st20k_w1024 \
        || echo "  [mb20k] MultiBLiMP FAILED — the chain goes on"
}

run_piece() {
    case "$1" in
        mb20k)  multiblimp_20k ;;
        en100k) seedpiece "en100k: item 3 — English part of speech off from the start and off at 5,000, 100k" en100k \
                    "0 1 2 3" pair "posoff 0" "posoff 5000" ;;
        uni512) seedpiece "uni512: Unigram 17k at width 512 — the null, case and the empty slot, 10k" uni512 \
                    "0 1 2 3" pair null_u_w512 case_u_w512 empty_u_w512 ;;
    esac
}

chain_main "the chain of 21 October: MultiBLiMP at 20k, items 3 and 6"
