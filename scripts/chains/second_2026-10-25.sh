#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 25 October 2026 (journey §33.140): full training at 30M parameters.
# At full training (151M tokens, 50k) the tokenizer's gain and the features' are measured in the 5M
# model only; at 30M parameters (width 512) the tokenizer ran to 20k on ~38M tokens, and case to 10k,
# where it still gains over its null (§33.118). Do both answers hold at full training at 30M? And does
# English endings apart, measured in the 5M model only, gain at 30M?
#
#   full512   the nulls under BPE 17k and endings apart, and case (tagger) under BPE 17k: width 512,
#             151M tokens, 50k, seeds 0–3                                              P314–P316
#   mb512     MultiBLiMP on the nulls' eight, once all are here                         P317
#   en512     English plain baselines under BPE 17k and endings apart, width 512, 10k, seeds 0–3   P318
#   lex512    ONLY IF case keeps more than 0.002 over its null in full512: case from the word form and
#             case without the leak (prepositions read from their form), under BPE 17k, the same
#   empty512  ONLY IF the tokenizer's gain in full512 is below 0.006: the empty slot (the null
#             switched off from step 0) under both tokenizers, the same
#
# The order and both conditions are the thesis text's (its answer of 25 October): what case's lead
# over the word form is made of matters only if case keeps a gain, and the empty slot only if the
# gain between the nulls could fall to zero over it. The 100-step warm-up of every other run: at width
# 512 the longer one changes neither null (§33.132). Guards: ready; pair (the first seed's first two
# runs end within SANITY_TOL of each other); gpu, disk, END_BY — a run starts only if it can end
# before END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis
# machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-25.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-full512 mb512 en512 lex512 empty512}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
f512() {        # f512 FACT TOKENIZER SEED — width 512, 151M tokens, 50k
    echo "--facts $1 $U -o data.corpus_fraction=0.2 -o data.bpe_path=data/$2.json $C $(steps 50000) -o model.d_model=512 -o train.seed=$3"
}
e512() {        # e512 TOKENIZER SEED — the empty slot: the null switched off from step 0
    echo "$(f512 mconst "$1" "$2") -o frozen.feature_off_step=0"
}
en512() {       # en512 TOKENIZER SEED — English, WikiText-103, a plain baseline, width 512, 10k
    echo "-o data.bpe_path=data/$1.json $(steps 10000) -o model.d_model=512 -o train.seed=$2"
}
NULL_B=bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_w512
NULL_E=bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2_w512
CASE_B=bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_w512

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        full512|lex512|empty512) echo 86 ;;     # width 512: 16.0–16.2 min at 10k, 31.6–32.0 at 20k (19–21 October)
        en512) echo 18 ;;                       # the same width at 10k; English's vocabulary is as large
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

# prints the mean gain of cell B over cell A on this machine (A's bits per byte minus B's, paired by
# seed) once seeds 0–3 of both are here; exits 1 otherwise
GAIN='import sys
sys.path.insert(0, ".")
from morphlm import runs
R = runs.load(machine="second", keep=lambda r: "seed" in r and "bits_per_byte" in r)
pairs, _ = runs.pair(R, sys.argv[1], sys.argv[2])
d = [a["bits_per_byte"] - b["bits_per_byte"] for s, (a, b) in pairs.items() if s in (0, 1, 2, 3)]
if len(d) < 4:
    sys.exit(1)
print("%.5f" % (sum(d) / len(d)))'

decide() {      # decide PIECE gt|lt THRESHOLD A B — 0 if B's mean gain over A is above (gt) or below (lt) it
    local piece=$1 op=$2 t=$3 g
    stamp "$piece: decided on full512 — the mean gain of $5 over $4, $op $t"
    g=$(python -c "$GAIN" "$4" "$5") || g=""
    if [ -z "$g" ]; then
        if [ -n "${PLAN:-}" ]; then echo "  [plan] $piece: decided once full512 is here — planned as if it runs"; return 0; fi
        echo "  [skipped] $piece: the four seeds it is decided on are not all here"; return 1
    fi
    if awk -v g="$g" -v t="$t" -v op="$op" 'BEGIN { exit !((op == "gt" && g > t) || (op == "lt" && g < t)) }'; then
        echo "  [decided] $piece runs: $g ($op $t)"; return 0
    fi
    echo "  [decided] $piece does not run: $g (not $op $t)"; return 1
}

multiblimp_512() {      # MultiBLiMP on the nulls' eight: once all eight are here, once
    local s t missing=0
    stamp "mb512: MultiBLiMP on width 512's nulls at full training, seeds 0–3"
    if python -c "$MB_RECORD" _ukwiki0.2_w512; then echo "  [skip] mb512: a record of it is already here"; return 0; fi
    for s in 0 1 2 3; do
        for t in bpe_uk_17k bpeend_uk_17k; do
            python scripts/ops/run_ids.py --done $(f512 mconst $t $s) >/dev/null || missing=$((missing + 1))
        done
    done
    if [ "$missing" -gt 0 ]; then echo "  [skipped] mb512: $missing of the 8 runs are not here"; return 0; fi
    if [ -n "${PLAN:-}" ]; then echo "  [would run] mb512 — minutes"; return 0; fi
    gpu_ok || return 0
    python scripts/eval/multiblimp_eval.py --machine second --labels "$NULL_B" "$NULL_E" \
        || echo "  [mb512] MultiBLiMP FAILED — the chain goes on"
}

run_piece() {
    case "$1" in
        full512)  seedpiece "full512: the nulls under both tokenizers and case under BPE 17k, width 512, full training" full512 \
                      "0 1 2 3" pair "f512 mconst bpe_uk_17k" "f512 mconst bpeend_uk_17k" "f512 ucase bpe_uk_17k" ;;
        mb512)    multiblimp_512 ;;
        en512)    seedpiece "en512: English plain baselines under BPE 17k and endings apart, width 512, 10k" en512 \
                      "0 1 2 3" pair "en512 bpe_17k" "en512 bpeend_17k" ;;
        lex512)   decide lex512 gt 0.002 "$NULL_B" "$CASE_B" || return 0
                  seedpiece "lex512: case from the word form and without the leak under BPE 17k, width 512, full training" lex512 \
                      "0 1 2 3" pair "f512 ucaselex bpe_uk_17k" "f512 ucaseprep bpe_uk_17k" ;;
        empty512) decide empty512 lt 0.006 "$NULL_B" "$NULL_E" || return 0
                  seedpiece "empty512: the empty slot under both tokenizers, width 512, full training" empty512 \
                      "0 1 2 3" pair "e512 bpe_uk_17k" "e512 bpeend_uk_17k" ;;
    esac
}

chain_main "the chain of 25 October: full training at 30M parameters, and English at 30M"
