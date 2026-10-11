#!/usr/bin/env bash
# identify_model.sh — 验证一个无名模型目录到底是哪个 Qwen MoE。
# 用法： bash v6/scripts/utils/identify_model.sh /home/u/downloads/models
#
# 步骤：拉官方 Qwen/Qwen3.6-35B-A3B 的 config 做逐字段 diff；
#       再和 Qwen3.5 代候选各 diff 一遍；拉取失败时给出诊断。
set -u

MODEL_DIR="${1:?usage: identify_model.sh <model_dir>}"
MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[identify] $MODEL_DIR 下没有 config.json，不是完整模型目录" >&2
    exit 1
fi

echo "[identify] 本地 config 摘要（$MODEL_DIR/config.json）："
python - "$MODEL_DIR/config.json" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
t = cfg.get("text_config", cfg)
for k in ("architectures", "model_type", "num_hidden_layers",
          "num_experts", "num_experts_per_tok", "hidden_size",
          "full_attention_interval"):
    print(f"  {k} = {cfg.get(k, t.get(k))}")
print(f"  multimodal = {'vision_config' in cfg}")
EOF

check_one () {
    # $1 = HF 仓库 id
    local repo="$1" out="$TMP/$(echo "$1" | tr '/' '_').json"
    echo
    echo "[identify] 对比 $repo ..."
    if ! curl -fsSL "$MIRROR/$repo/raw/main/config.json" -o "$out"; then
        echo "  [跳过] 拉取失败（仓库不存在或网络问题）: $MIRROR/$repo"
        return 1
    fi
    if ! python -m json.tool "$out" > /dev/null 2>&1; then
        echo "  [跳过] 拉到的不是 JSON，疑似镜像错误页"
        return 1
    fi
    if diff <(python -m json.tool "$out") \
            <(python -m json.tool "$MODEL_DIR/config.json") > "$TMP/diff.txt"; then
        echo "  [MATCH] config 与 $repo 完全一致 —— 就是这个模型"
        return 0
    fi
    echo "  [不一致] 差异行数: $(wc -l < "$TMP/diff.txt")（前 20 行：）"
    head -20 "$TMP/diff.txt" | sed 's/^/    /'
    return 1
}

for repo in \
    "Qwen/Qwen3.6-35B-A3B" \
    "Qwen/Qwen3.5-35B-A3B" \
    "Qwen/Qwen3.5-122B-A10B" \
    "Qwen/Qwen3.5-397B-A17B" ; do
    if check_one "$repo"; then
        exit 0
    fi
done

echo
echo "[identify] 以上候选都不匹配。列一下镜像上 Qwen3.5/3.6 系 MoE 的真实仓库名："
curl -fsSL "$MIRROR/api/models?author=Qwen&search=MoE" \
  | python -c "import json,sys; [print(' ', m['id']) for m in json.load(sys.stdin)]" \
  || echo "  （API 查询失败，可浏览器打开 $MIRROR/Qwen 手动找）"
exit 1
