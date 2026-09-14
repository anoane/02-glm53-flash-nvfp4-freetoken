#!/usr/bin/env bash
# Serve GLM-5.3-Flash NVFP4 with FreeToken (b12x sm_120 expert kernel on every layer).
# usage: serve_ft.sh [--foreground|--detach]   (systemd uses --foreground)
# Foreground mode also runs memguard as a child of the same cgroup, so `systemctl stop`
# takes both down. Logs: journal (foreground) and $FT_HOME/serve.log (tee inside the container).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1090
. "${SERVE_ENV:-$HERE/config/serve.env}"
MODE=${1:---foreground}
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
: > "$FT_HOME/serve.log"
DFLAG=""
if [ "$MODE" = "--foreground" ]; then
  [ -s "$FT_HOME/memguard.log" ] && mv -f "$FT_HOME/memguard.log" "$FT_HOME/memguard.log.$(date +%s)"
  bash "$HERE/bin/memguard.sh" "$CONTAINER" "$MIN_AVAIL_GB" "$FT_HOME/memguard.log" >/dev/null 2>&1 &
else
  DFLAG="-d"
fi
# shellcheck disable=SC2086
exec docker run $DFLAG --rm --name "$CONTAINER" --gpus all --ulimit memlock=-1:-1 \
  --memory="$MEMCAP" --memory-swap="$MEMCAP" --shm-size=8g \
  -p 127.0.0.1:"$PORT":"$PORT" -v "$FT_HOME":/ft -v "$MODEL":/model:ro \
  -e HOME=/ft/home -e XDG_CACHE_HOME=/ft/home/.cache -e HF_HUB_OFFLINE=1 \
  -e FREETOKEN_GLM5_RESIDENT_LAYERS="$RES" \
  -e FREETOKEN_GLM5_ATTN_FP8=1 -e FREETOKEN_GLM5_MLP_FP8=1 -e FREETOKEN_GLM5_KDA_FP8=1 \
  -e FREETOKEN_GLM5_MTP=0 -e FREETOKEN_GLM5_SPEC=0 -e FREETOKEN_MOE_STATS="${MOE_STATS:-0}" -e FREETOKEN_TRACE_OUT="${TRACE_OUT:-}" -e FREETOKEN_TRACE_N="${TRACE_N:-3000}" -e FREETOKEN_SCORE_OUT="${SCORE_OUT:-}" -e FREETOKEN_SCORE_N="${SCORE_N:-512}" -e FREETOKEN_SCORE_IDS="${SCORE_IDS:-}" -e FREETOKEN_GLM5_OFFLOAD_TOPK="${OFFLOAD_TOPK:-0}" -e FREETOKEN_GLM5_CACHE_BIAS="${CACHE_BIAS:-0}" \
  --entrypoint bash "$IMAGE" -c ". /ft/venv/bin/activate && cd /ft/src && exec ft serve \
    --model /model --moe-backend offload --nvfp4-backend auto \
    --max-seq-len-override $CTX --num-tokens $CTX --moe-cache-size $CACHE_SLOTS \
    --max-prefill-length $CHUNK --memory-ratio 0.95 \
    --host 0.0.0.0 --port $PORT --max-running-requests 1 --max-output-tokens $MAX_OUT $EXTRA \
    2>&1 | tee /ft/serve.log"
