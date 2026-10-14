#!/usr/bin/env bash
# benchmark_transfer.sh — 评估两台服务器间拷贝 ~70GB 模型权重的耗时。
#
# 用法：
#   1) 接收方（新机器）先起 iperf3 服务：
#        bash benchmark_transfer.sh server
#   2) 发送方（老机器）跑测试：
#        bash benchmark_transfer.sh client <新机器IP> [待测大文件路径]
#      不传文件路径时自动用 dd 造一个 5GB 临时文件做真实传输样例。
#
# 最后按 70GB 估算总耗时：时间 ≈ 70GB / 三者最小值（网络、源盘读、目标盘写、实测传输）。
set -u
MODE="${1:?usage: benchmark_transfer.sh server|client <peer_ip> [test_file]}"
PEER_IP="${2:-}"
TEST_FILE="${3:-}"

hr() { printf '%s\n' "----------------------------------------"; }

case "$MODE" in
server)
    echo "[1/1] iperf3 服务端已启动，Ctrl-C 结束（对端跑 client 模式即可）"
    iperf3 -s
    ;;

client)
    command -v iperf3 >/dev/null || { echo "先装 iperf3: conda install -c conda-forge iperf3 或 yum/apt 装"; exit 1; }
    [ -n "$PEER_IP" ] || { echo "缺对端 IP"; exit 1; }

    echo "[1/4] 网络带宽（iperf3，30 秒，8 线程）"
    iperf3 -c "$PEER_IP" -t 30 -P 8 2>&1 | grep -E "sender|receiver" || \
        echo "iperf3 连不上：确认对端跑了 server 模式、防火墙放行 5201 端口"

    hr
    echo "[2/4] 源盘读速（dd，5GB）"
    SRC_TMP="$(mktemp -u /tmp/diskreadXXXX)"
    dd if=/dev/zero of="$SRC_TMP" bs=1M count=5120 conv=fdatasync 2>/dev/null   # 先造文件
    dd if="$SRC_TMP" of=/dev/null bs=1M 2>&1 | tail -1
    rm -f "$SRC_TMP"

    hr
    echo "[3/4] 目标盘写速（scp 落盘目录所在盘的 dd 写测试，5GB）"
    DST_DIR="/data/mamingyu"
    [ -d "$DST_DIR" ] || DST_DIR="$HOME"
    dd if=/dev/zero of="$DST_DIR/.write_test_tmp" bs=1M count=5120 conv=fdatasync 2>&1 | tail -1
    rm -f "$DST_DIR/.write_test_tmp"

    hr
    echo "[4/4] 真实传输样例（rsync 一个 ~5GB 文件，含 ssh 加密开销）"
    if [ -z "$TEST_FILE" ]; then
        TEST_FILE="$(mktemp -u /tmp/rsynctestXXXX).bin"
        echo "  造 5GB 测试文件 $TEST_FILE ..."
        dd if=/dev/zero of="$TEST_FILE" bs=1M count=5120 conv=fdatasync 2>/dev/null
        CLEANUP=1
    else
        [ -f "$TEST_FILE" ] || { echo "文件不存在: $TEST_FILE"; exit 1; }
        CLEANUP=0
    fi
    START=$(date +%s)
    rsync -avP "$TEST_FILE" "mamingyu@$PEER_IP:/data/mamingyu/.rsync_test_tmp" || \
    rsync -avP "$TEST_FILE" "$PEER_IP:/tmp/.rsync_test_tmp"
    END=$(date +%s)
    [ "$CLEANUP" = 1 ] && rm -f "$TEST_FILE"
    SECS=$((END - START))
    echo "  5GB 用时 ${SECS}s → 实测吞吐 $((5120 / (SECS > 0 ? SECS : 1))) MB/s"

    hr
    echo "估算 70GB 总耗时：用上面四个数里最小的一个当有效吞吐，"
    echo "时间(秒) ≈ 71680 / 吞吐(MB/s)。例如 110MB/s ≈ 11 分钟，1.1GB/s ≈ 1 分钟。"
    ;;

*)
    echo "unknown mode: $MODE"; exit 1;;
esac
