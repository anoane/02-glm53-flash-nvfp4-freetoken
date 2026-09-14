#!/usr/bin/env bash
# Build (or rebuild) the FreeToken checkout + venv for serving. Idempotent.
# usage: build_venv.sh [--bench]   (--bench runs `ft bench bw`; needs an idle GPU)
# Pins: torch 2.11.0+cu130 (from FreeToken's pyproject), triton 3.6.0, flashinfer 0.6.18,
# sglang-kernel 0.4.5, plus flash-linear-attention (fla), which the GLM-5.3 KDA layers import
# but FreeToken does not declare.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1090
. "${SERVE_ENV:-$HERE/config/serve.env}"
mkdir -p "$FT_HOME/home/.cache"
if [ ! -d "$FT_HOME/src/.git" ]; then git clone -q "$FT_REPO" "$FT_HOME/src"; fi
git -C "$FT_HOME/src" fetch -q origin
git -C "$FT_HOME/src" checkout -q "$FT_REF"
git -C "$FT_HOME/src" reset -q --hard "origin/$FT_REF" 2>/dev/null || true
echo "checkout: $(git -C "$FT_HOME/src" rev-parse --short HEAD) ($(git -C "$FT_HOME/src" log -1 --format=%s | cut -c1-70))"
BENCH=""; [ "${1:-}" = "--bench" ] && BENCH="&& ft bench bw --gpu 0 | tail -22"
docker run --rm --gpus all -v "$FT_HOME":/ft -e HOME=/ft/home -e XDG_CACHE_HOME=/ft/home/.cache \
  --entrypoint bash "$IMAGE" -c "cd /ft/src \
  && ( [ -x /ft/venv/bin/python ] || python3 -m venv /ft/venv ) && . /ft/venv/bin/activate \
  && pip install -q -U pip && pip install -q -e '.[accel]' && pip install -q flash-linear-attention \
  && python -c 'import torch,flashinfer,fla,triton;print(\"torch\",torch.__version__,\"flashinfer\",flashinfer.__version__,\"fla\",fla.__version__,\"triton\",triton.__version__)' \
  && python -c 'from fla.ops.kda import chunk_kda' && echo 'fla.ops.kda ok' $BENCH"
echo "venv: $(du -sh "$FT_HOME/venv" | cut -f1) at $FT_HOME/venv"
