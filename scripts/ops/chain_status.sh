#!/usr/bin/env bash
# chain_status.sh — where every chain stands, in one command; changes nothing.
#     bash scripts/ops/chain_status.sh
cd "$(dirname "$0")/../.." || exit 1
date
echo "── chains"
pids=$(pgrep -f "^bash scripts/chains/[^ ]*\.sh")
if [ -z "$pids" ]; then echo "   none running"; fi
declare -A seen
for p in $pids; do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); pp=$(ps -o ppid= -p $p | tr -d ' ')
    ppcmd=$(tr '\0' ' ' < /proc/$pp/cmdline 2>/dev/null)
    [ "$ppcmd" = "$cmd" ] && continue           # a chain's own background track
    name=$(basename "${cmd% }" .sh | awk '{print $NF}')
    if [ -n "${seen[$name]}" ]; then echo "   !! DUPLICATE: $name runs twice (pids ${seen[$name]} and $p)"; fi
    seen[$name]="$p"
    echo "   $name  (pid $p, running $(ps -o etime= -p $p | tr -d ' '))"
    log="logs/${name}.log"
    if [ -f "$log" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$log") ))
        echo "     last stamp: $(grep '^=== 20' "$log" | tail -n 1 | cut -c5-110)"
        echo "     last line (${age}s ago): $(tail -n 1 "$log" | cut -c1-110)"
    fi
done
echo "── training"
runs=$(pgrep -af "morphlm.cli run")      # captured first, so later pipes cannot match themselves
if [ -z "$runs" ]; then echo "   nothing"; else
    while read -r line; do
        facts=$(grep -oE -- "--facts [^ ]+" <<< "$line" | cut -d' ' -f2)
        extra=$(grep -oE "corpus_fraction=[0-9.]+|train.seed=[0-9]+|max_steps=[0-9]+|inject=[a-z]+" <<< "$line" | paste -sd' ')
        echo "   ${facts:-baseline}  $extra"
    done <<< "$runs"
fi
echo "── GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/   /' || echo "   (no nvidia-smi)"
echo "── runs finished in the last ${SINCE_HOURS:=24} hours (SINCE_HOURS=… for another window)"
python3 - "$SINCE_HOURS" <<'PY'
import glob, json, os, sys, time
cut = time.time() - float(sys.argv[1]) * 3600
records = {os.path.basename(os.path.dirname(p)): p for p in glob.glob('results/*/result.json')}
# Duplicate seeds (journey §33.48): run ids hash the WHOLE config, so a field added
# since a cell ran gives its re-run a NEW id under the SAME label and seed.
cells = {}
for rid, p in records.items():
    r = json.load(open(p))
    cells.setdefault((r.get('label'), r.get('seed')), set()).add(rid)
done = sorted((os.path.getmtime(p), rid, p) for rid, p in records.items() if os.path.getmtime(p) >= cut)
dups = []
for t, rid, p in done:
    r = json.load(open(p))
    print(f"   {time.strftime('%m-%d %H:%M', time.localtime(t))}  {rid}  {r.get('label')} seed {r.get('seed')}  bpb {r.get('bits_per_byte', float('nan')):.4f}")
    others = sorted(cells[(r.get('label'), r.get('seed'))] - {rid})
    if others:
        dups.append(f"   !! DUPLICATE SEED: {rid} {r.get('label')} seed {r.get('seed')} — also {', '.join(others)}; quarantine one")
print(f"   {len(done)} run(s)" if done else "   none")
print('\n'.join(dups)) if dups else None
stopped = sorted(d for d in glob.glob('results/*/') if os.path.basename(d.rstrip('/')) != 'audits'
                 and not os.path.exists(os.path.join(d, 'result.json')) and os.path.getmtime(d) >= cut)
for d in stopped:
    print(f"   no result.json (in training, or stopped): {os.path.basename(d.rstrip('/'))}")
PY
