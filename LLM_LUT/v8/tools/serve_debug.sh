#!/usr/bin/env bash
# serve_debug.sh — LUT serve 定位实验（对照 + 插桩，问题记录见 AGENTS.md）。
#
#   bash tools/serve_debug.sh baseline 18004 "4,5" > logs/serve_baseline.log 2>&1 &
#       对照组：不带 LUT，eval 必须 ~0.73+，否则 LUT 从头到尾无辜
#   bash tools/serve_debug.sh lut 18002 "6,7" > logs/serve_lut_debug.log 2>&1 &
#       实验组：LUT 37 38 39 + V8_LUT_DEBUG（逐层 dense vs LUT 实测对比）
#   LAYERS="37" DEBUG=0 bash tools/serve_debug.sh lut 18002 "6,7" > logs/serve_lut_l37.log 2>&1 &
#       层数扫描/并发实验：LAYERS 覆盖层集合，DEBUG=0 关插桩（timing 干净）
set -eu
MODE=$1
PORT=$2
GPUS=$3

# 环境变量覆盖（并发实验用生产配置，见报告 38）：
#   UTIL=0.88 MAXSEQS=16 SLOTS=1024 DEBUG=0 LAYERS="37 38 39" \
#     bash tools/serve_debug.sh lut 18002 "6,7" > logs/serve_lut3_prod_c16.log 2>&1 &
UTIL="${UTIL:-0.70}"
MAXSEQS="${MAXSEQS:-4}"
SLOTS="${SLOTS:-512}"

if [ "$MODE" = "lut" ]; then
  export V8_LUT_LAYERS="${LAYERS:-37 38 39}"
  export V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles
  if [ "${DEBUG:-1}" = "1" ]; then
    export V8_LUT_DEBUG=1
  fi
elif [ "$MODE" != "baseline" ]; then
  echo "usage: serve_debug.sh {baseline|lut} PORT GPUS" >&2
  exit 1
fi
export V8_COMPRESS_SLOTS="$SLOTS"

CUDA_VISIBLE_DEVICES=$GPUS python -m vllm_plugin.serve \
  /home/u/downloads/models/Qwen3.6-35B-A3B \
  --enforce-eager --no-enable-prefix-caching --max-model-len 131072 \
  --tensor-parallel-size 2 --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization "$UTIL" \
  --port "$PORT"
