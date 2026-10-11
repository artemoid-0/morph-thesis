# guard.sh — sourced at the top of every chain, after its `cd` to the project:
#     source scripts/chains/guard.sh
# A chain refuses to start while another copy of itself runs (30 September:
# a chain started twice trained every run twice, at half speed, into the same
# result folders). The lock is held by the chain AND every process it starts,
# so it is free again only when all of them have exited.
_chain_name=$(basename "$0" .sh)
mkdir -p logs
exec 9>"logs/.${_chain_name}.lock"
if ! flock -n 9; then
    echo "=== $(date '+%F %T')  $_chain_name is already running — this copy does not start"
    echo "    (bash scripts/ops/chain_status.sh shows what is running)"
    exit 1
fi
# The project's Python, whatever shell launched the chain: without the conda
# environment active, `python` is not on PATH and every call fails at once
# (9 October: a chain launched from a plain shell ended in two seconds).
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"
fi
