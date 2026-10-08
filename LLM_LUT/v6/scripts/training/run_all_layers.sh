#!/usr/bin/env bash
# run_all_layers.sh — 8 卡并行派发逐层 LUT 训练（build_lut_ffn_output_v3_shared_coarse）
# + 训练结束后的 v4 checkpoint 转换。
#
# 用法示例：
#   # 最小调用（LAYERS 必填，其余走默认值）：
#   LAYERS="17 18 19 20" bash scripts/training/run_all_layers.sh
#
#   # 完整覆盖（env 覆盖默认值）：
#   LAYERS="17 18 19 20 21 22 23 24 25" \
#   DATA_ROOT=/data/qwen35b_ffn_collect \
#   TEACHER_DIR=/data/qwen35b_teachers \
#   OUT_PREFIX=outputs_ffn_lut \
#   GPUS="0 1 2 3 4 5 6 7" \
#   CONCURRENT=8 \
#   LOG_DIR=logs/lut_train \
#   SKIP_EXISTING=1 \
#   bash scripts/training/run_all_layers.sh
#
# 环境变量说明（均有默认值，可用 env 覆盖）：
#   LAYERS        空格分隔层号列表（必填，无默认；为空则报错退出）
#   DATA_ROOT     采集数据根目录（含 layer{L}/input 与 layer{L}/output），必填
#   TEACHER_DIR   teacher 权重 .pt 目录（含 qwen_35b_shared_expert_l{L}.pt），必填
#   OUT_PREFIX    输出根前缀，默认 outputs_ffn_lut
#                 训练输出: <OUT_PREFIX>_layer{L}_shared_expert_offpolicy
#                 转换输出: <OUT_PREFIX>_layer{L}_shared_expert_offpolicy_as_v4
#   GPUS          空格分隔 GPU 号列表，默认 "0 1 2 3 4 5 6 7"
#   CONCURRENT    并行训练数上限，默认 = GPUS 个数
#   LOG_DIR       日志目录（每层一个 layer{L}.log），默认 logs/lut_train
#   SKIP_EXISTING 默认 1：若该层 *_as_v4/checkpoints 已存在则整层跳过；
#                 若 v3 checkpoints 存在但 v4 缺失，则跳过训练、只补转换。
#                 设 0 则强制重跑。
#
# 退出码：全部 SUCCESS/SKIPPED -> 0；任一 FAILED -> 1。

set -u

# ---------------- 参数解析（env 覆盖） ----------------
LAYERS="${LAYERS:-}"
DATA_ROOT="${DATA_ROOT:-}"
TEACHER_DIR="${TEACHER_DIR:-}"
OUT_PREFIX="${OUT_PREFIX:-outputs_ffn_lut}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
LOG_DIR="${LOG_DIR:-logs/lut_train}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ -z "$LAYERS" ]; then
    echo "ERROR: LAYERS 为空（必填，空格分隔层号，如 LAYERS='17 18 19'）" >&2
    exit 1
fi
if [ -z "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT 为空（必填，采集数据根目录）" >&2
    exit 1
fi
if [ -z "$TEACHER_DIR" ]; then
    echo "ERROR: TEACHER_DIR 为空（必填，teacher .pt 目录）" >&2
    exit 1
fi

# CONCURRENT 默认 = GPU 个数
if [ -z "${CONCURRENT:-}" ]; then
    # shellcheck disable=SC2086
    set -- $GPUS
    CONCURRENT=$#
fi

# 校验 CONCURRENT 为正整数
case "$CONCURRENT" in
    ''|*[!0-9]*)
        echo "ERROR: CONCURRENT 必须是正整数，得到 '$CONCURRENT'" >&2
        exit 1
        ;;
esac
if [ "$CONCURRENT" -lt 1 ]; then
    echo "ERROR: CONCURRENT 必须 >= 1" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# GPU 列表转数组
# shellcheck disable=SC2206
GPU_ARR=($GPUS)
NUM_GPUS=${#GPU_ARR[@]}
# 层列表转数组
# shellcheck disable=SC2206
LAYER_ARR=($LAYERS)
NUM_LAYERS=${#LAYER_ARR[@]}

# 并发上限不超过 GPU 数（round-robin 槽位一一对应 GPU）
if [ "$CONCURRENT" -gt "$NUM_GPUS" ]; then
    echo "WARN: CONCURRENT=$CONCURRENT > GPU 数 $NUM_GPUS，收敛到 $NUM_GPUS" >&2
    CONCURRENT=$NUM_GPUS
fi

echo "=== run_all_layers ==="
echo "LAYERS       : $LAYERS"
echo "DATA_ROOT    : $DATA_ROOT"
echo "TEACHER_DIR  : $TEACHER_DIR"
echo "OUT_PREFIX   : $OUT_PREFIX"
echo "GPUS         : $GPUS ($NUM_GPUS 个)"
echo "CONCURRENT   : $CONCURRENT"
echo "LOG_DIR      : $LOG_DIR"
echo "SKIP_EXISTING: $SKIP_EXISTING"
echo ""

# ---------------- 状态表 ----------------
# layer -> 状态: PENDING / RUNNING / SUCCESS / FAILED / SKIPPED / CONVERT_FAILED
declare -A STATUS=()
declare -A LAYER_LOG=()
declare -A SLOT_PID=()    # slot -> pid（0 表示空）
declare -A SLOT_LAYER=()  # slot -> layer
declare -A PID_LAYER=()   # pid -> layer

for L in "${LAYER_ARR[@]}"; do
    STATUS[$L]="PENDING"
    LAYER_LOG[$L]="$LOG_DIR/layer${L}.log"
done
for ((s = 0; s < CONCURRENT; s++)); do
    SLOT_PID[$s]=0
done

# ---------------- 单任务执行体 ----------------
train_layer() {
    local L=$1 GPU=$2
    python -u scripts/training/build_lut_ffn_output_v3_shared_coarse.py \
        --teacher_weight_path "${TEACHER_DIR}/qwen_35b_shared_expert_l${L}.pt" \
        --dataset_dir "${DATA_ROOT}/layer${L}/input" \
        --output_dataset_dir "${DATA_ROOT}/layer${L}/output" \
        --output_root "${OUT_PREFIX}_layer${L}_shared_expert_offpolicy" \
        --group_size 64 --group_ids "0-31" \
        --coarse_num_bits 14 --residual_num_bits 16 \
        --tree_candidates 256 --tree_min_samples 4 --tree_max_samples 400000 \
        --calib_size 600000 --eval_size 69000 \
        --finetune_epochs 50 --finetune_loss_mode multi --device "cuda:${GPU}"
}

convert_layer() {
    local L=$1 GPU=$2
    python scripts/conversion/convert_v3_to_v4_checkpoints.py \
        --v3_checkpoint_dir "${OUT_PREFIX}_layer${L}_shared_expert_offpolicy/checkpoints" \
        --output_root "${OUT_PREFIX}_layer${L}_shared_expert_offpolicy_as_v4" \
        --device "cuda:${GPU}"
}

v4_done() {
    [ -d "${OUT_PREFIX}_layer$1_shared_expert_offpolicy_as_v4/checkpoints" ]
}

v3_done() {
    [ -d "${OUT_PREFIX}_layer$1_shared_expert_offpolicy/checkpoints" ]
}

# 后台执行"训练+转换"整个链条，退出码反映链条整体成败
run_chain() {
    local L=$1 GPU=$2
    if v3_done "$L" && [ "$SKIP_EXISTING" = "1" ]; then
        echo "[layer $L] v3 checkpoints 已存在，跳过训练，仅补 v4 转换"
    else
        if ! train_layer "$L" "$GPU"; then
            echo "[layer $L] TRAIN FAILED" >&2
            return 1
        fi
    fi
    if ! convert_layer "$L" "$GPU"; then
        echo "[layer $L] CONVERT FAILED" >&2
        return 1
    fi
    return 0
}

# ---------------- 并发调度（round-robin slot <-> GPU） ----------------
# 从下一个空槽开始，尝试派发一个 PENDING 层；无空槽返回 1
next_layer_to_dispatch() {
    local L
    for L in "${LAYER_ARR[@]}"; do
        if [ "${STATUS[$L]}" = "PENDING" ]; then
            echo "$L"
            return 0
        fi
    done
    return 1
}

dispatch_slot() {
    local slot=$1 L GPU pid
    L=$(next_layer_to_dispatch) || return 1
    GPU=${GPU_ARR[$slot]}
    STATUS[$L]="RUNNING"
    SLOT_LAYER[$slot]=$L
    echo "[dispatch] layer $L -> gpu $GPU (slot $slot), log: ${LAYER_LOG[$L]}"
    ( run_chain "$L" "$GPU" ) > "${LAYER_LOG[$L]}" 2>&1 &
    pid=$!
    SLOT_PID[$slot]=$pid
    PID_LAYER[$pid]=$L
    return 0
}

# 回收已退出的后台任务，更新状态并释放槽位
reap_finished() {
    local slot pid L rc
    for ((slot = 0; slot < CONCURRENT; slot++)); do
        pid=${SLOT_PID[$slot]}
        [ "$pid" = "0" ] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            rc=$?
            L=${SLOT_LAYER[$slot]}
            if [ $rc -eq 0 ]; then
                STATUS[$L]="SUCCESS"
            elif v4_done "$L"; then
                # 链条失败但 v4 已存在（极端情况）：视为可用
                STATUS[$L]="SUCCESS"
            else
                STATUS[$L]="FAILED"
            fi
            echo "[done] layer $L rc=$rc -> ${STATUS[$L]}"
            SLOT_PID[$slot]=0
            unset "PID_LAYER[$pid]"
        fi
    done
}

# 等待至少一个槽位空出
wait_for_slot() {
    while true; do
        reap_finished
        local slot
        for ((slot = 0; slot < CONCURRENT; slot++)); do
            [ "${SLOT_PID[$slot]}" = "0" ] && return 0
        done
        sleep 10
    done
}

# 等所有运行中的任务结束（复用 reap_finished 做状态落账）
wait_all() {
    while :; do
        reap_finished
        local slot busy=0
        for ((slot = 0; slot < CONCURRENT; slot++)); do
            [ "${SLOT_PID[$slot]}" != "0" ] && busy=1
        done
        [ "$busy" = "0" ] && return 0
        sleep 5
    done
}

# 预热 SKIP_EXISTING：v4 完整存在且 SKIP_EXISTING=1 的层直接 SKIPPED；
# v3 存在但 v4 缺失的层保留 PENDING（run_chain 会跳过训练只补转换）。
for L in "${LAYER_ARR[@]}"; do
    if v4_done "$L" && [ "$SKIP_EXISTING" = "1" ]; then
        STATUS[$L]="SKIPPED"
    fi
done

# 主循环：填满槽位 -> 等一个退出 -> 再填
while :; do
    # 尽量填满空槽
    for ((slot = 0; slot < CONCURRENT; slot++)); do
        if [ "${SLOT_PID[$slot]}" = "0" ]; then
            dispatch_slot "$slot" || break 2
        fi
    done
    wait_for_slot
done

wait_all

# ---------------- 汇总 ----------------
echo ""
echo "================ 汇总 ================"
printf "%-10s %-12s %s\n" "LAYER" "STATUS" "LOG"
FAILED_LAYERS=()
for L in "${LAYER_ARR[@]}"; do
    printf "%-10s %-12s %s\n" "$L" "${STATUS[$L]}" "${LAYER_LOG[$L]}"
    if [ "${STATUS[$L]}" = "FAILED" ]; then
        FAILED_LAYERS+=("$L")
    fi
done

if [ ${#FAILED_LAYERS[@]} -gt 0 ]; then
    echo ""
    echo "FAILED 层: ${FAILED_LAYERS[*]}"
    for L in "${FAILED_LAYERS[@]}"; do
        echo "  layer $L -> ${LAYER_LOG[$L]}"
    done
    exit 1
fi

echo ""
echo "全部层完成（SUCCESS/SKIPPED）。"
exit 0
