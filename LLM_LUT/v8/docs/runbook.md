# 格 1：只 L37（DEBUG=0，timing 干净）
LAYERS="37" DEBUG=0 bash tools/serve_debug.sh lut 18002 "6,7" > logs/serve_lut_l37.log 2>&1 &
# health 起来后：
nohup python tools/bench_concurrency.py --base-url http://localhost:18002 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data data/longctx_multi_turn_65536_64docs.jsonl \
  --concurrency 16 --out results/phase1_lut_l37_c16.json \
  > logs/batch_concurrency_l37.log 2>&1 &

# 格 2：L37+L38（重启 再跑，out 改 l37l38）
LAYERS="37 38" DEBUG=0 bash tools/serve_debug.sh lut 18002 "6,7" > logs/serve_lut_l37l38.log 2>&1 &
python tools/bench_concurrency.py ... --out results/phase1_lut_l37l38_c16.json

# serve：生产配置 + LUT 三层，无
UTIL=0.88 MAXSEQS=16 SLOTS=1024 DEBUG=0 LAYERS="37 38 39" \
  bash tools/serve_debug.sh lut 18002 "6,7" > logs/serve_lut3_prod_c16.log 2>&1 &

# health 起来后，bench（与报告 38 完全同参数）
python tools/bench_concurrency.py \
  --base-url http://localhost:18002 \
  --model /home/u/downloads/models/Qwen3.6-35B-A3B \
  --data data/longctx_multi_turn_65536_64docs.jsonl \
  --concurrency 16 --out results/prod_lut3_c16.json