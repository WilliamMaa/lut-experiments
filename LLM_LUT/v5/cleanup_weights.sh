#!/bin/bash
# cleanup_weights.sh
# 删除 LLM_LUT/v5 下老实验产生的权重/输出目录，释放磁盘空间。
# 默认只打印会删什么（DRY_RUN=1），确认后再改成 DRY_RUN=0 真正删除。

set -euo pipefail

DRY_RUN=${DRY_RUN:-1}
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# 要清理的旧实验输出目录
OLD_OUTPUT_DIRS=(
    outputs_o_proj_exp
    outputs_o_proj_l17
    outputs_o_proj_l27
    outputs_sequential_large
    outputs_sequential_small
    outputs_tree_21_23
    outputs_tree_l15_l27
    outputs_independent_down_small
    outputs_independent_o_small
)

# 清理旧的 sensitivity scan 结果（可选）
CLEAN_SCAN=${CLEAN_SCAN:-0}

run_cmd() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY-RUN] would run: $*"
    else
        echo "[DELETE] $*"
        "$@"
    fi
}

echo "=== Cleaning old experiment output directories under $ROOT ==="
for d in "${OLD_OUTPUT_DIRS[@]}"; do
    if [ -d "$d" ]; then
        run_cmd rm -rf "$d"
    fi
done

echo "=== Cleaning any leftover .pt / .bin / .safetensors under outputs_* ==="
shopt -s nullglob
files=(outputs_*/*.pt outputs_*/*.bin outputs_*/*.safetensors)
shopt -u nullglob
for f in "${files[@]}"; do
    [ -e "$f" ] || continue
    run_cmd rm -f "$f"
done

if [ "$CLEAN_SCAN" = "1" ]; then
    if [ -f "results/sensitivity_scan.json" ]; then
        run_cmd rm -f "results/sensitivity_scan.json"
    fi
fi

if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "Dry run finished. Set DRY_RUN=0 to actually delete."
    echo "Example: DRY_RUN=0 bash cleanup_weights.sh"
else
    echo ""
    echo "Cleanup finished."
fi
