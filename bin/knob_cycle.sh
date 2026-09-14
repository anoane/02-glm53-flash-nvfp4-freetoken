#!/usr/bin/env bash
# One experiment cycle for the router/cache knobs: restart the service with the given
# serve.env overrides, wait for READY, drive ~3k greedy decode tokens, report misses/step
# (from the [moe-stats] dump, needs MOE_STATS=1) and decode tok/s (server gen-throughput
# lines, first 3 samples dropped). usage: knob_cycle.sh <label> KEY=VALUE ...
# Overrides are applied to config/serve.env for the cycle and reverted at the end.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
ENVF=$HERE/config/serve.env
LABEL=${1:?label}; shift
cp "$ENVF" "$ENVF.bak.$LABEL"
for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; if grep -q "^$k=" "$ENVF"; then sed -i "s|^$k=.*|$k=$v|" "$ENVF"; else echo "$k=$v" >> "$ENVF"; fi; done
# shellcheck disable=SC1090
. "$ENVF"
LOG=$FT_HOME/serve.log
: > "$LOG"   # truncate BEFORE the restart: ExecStop takes up to 30 s and the old READY line would match
systemctl restart freetoken-glm53.service
sleep 5
t0=$(date +%s)
while :; do
  grep -aq "API server is ready to serve" "$LOG" 2>/dev/null && break
  if grep -aqE "Traceback|OutOfMemory|Backend worker" "$LOG" 2>/dev/null; then echo "[$LABEL] SERVER FAILED"; grep -aE "Error|Traceback" "$LOG" | tail -3; cp "$ENVF.bak.$LABEL" "$ENVF"; exit 2; fi
  [ $(( $(date +%s) - t0 )) -gt 1500 ] && { echo "[$LABEL] READY timeout"; cp "$ENVF.bak.$LABEL" "$ENVF"; exit 3; }
  sleep 10
done
echo "[$LABEL] ready after $(( $(date +%s) - t0 ))s; overrides: $*"
n0=$(tr '\r' '\n' < "$LOG" | grep -ac "Decode batch")
. "$FT_HOME/venv/bin/activate"
python3 "$HERE/bin/stats_run.py" "http://127.0.0.1:$PORT/v1/chat/completions" /root/workspace/bench_c_review/corpus.c.txt "${TOKENS:-3000}" | tail -1
tr '\r' '\n' < "$LOG" | grep -a "Decode batch" | tail -n +$((n0 + 4)) | grep -oE "gen throughput \(token/s\): [0-9.]+" | awk '{s+=$NF; n++} END {if (n) printf "[%s] decode mean of %d samples: %.2f tok/s\n", "'"$LABEL"'", n, s/n}'
tr '\r' '\n' < "$LOG" | grep -a "\[moe-stats\]" | tail -1 | sed "s/.*\[moe-stats\] //" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read()); pl = d[\"per_layer\"]
rows = [r for r in pl if r[\"steps\"]]
miss = sum(r[\"missing_per_step\"] for r in rows); act = sum(r[\"active_per_step\"] for r in rows)
print(f\"[$LABEL] offloaded layers: misses/step {miss:.1f} of active {act:.1f} ({100*miss/act if act else 0:.0f}%), steps {rows[0][\"steps\"] if rows else 0}\")"
cp "$LOG" "$FT_HOME/run_$LABEL.log"
cp "$ENVF.bak.$LABEL" "$ENVF"; rm -f "$ENVF.bak.$LABEL"
echo "[$LABEL] serve.env restored (service still running with the overrides until the next restart)"
