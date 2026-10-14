#!/usr/bin/env bash
# transfer_onpolicy_assets.sh — 老服务器 → 新服务器 传输 onpolicy 资产。
#
# 背景：onpolicy 资产都在老机器 /data/mamingyu/ 下，总量约 10–20GB
#   （三层 v3+v4 checkpoints、三层 datasets、teacher .pt ×3、bundles ×3）。
# 通讯方式：ssh 免密 + rsync（断点续传）。
#
# 执行顺序：
#   步骤 A（只需做一次）：建立两台机器间的 ssh 免密，见本文件头部注释。
#   步骤 B：填好下面两个变量，直接 bash 本脚本。

# ===================== 步骤 A：ssh 免密（只需做一次） =====================
# 在【发送方】（老机器）执行：
#   1) 生成密钥（已有 ~/.ssh/id_ed25519.pub 就跳过）：
#        ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
#   2) 把公钥推到【接收方】（新机器），这一步需要输入一次新机器密码：
#        ssh-copy-id -i ~/.ssh/id_ed25519.pub mamingyu@<新机器IP>
#   3) 验证免密生效：
#        ssh mamingyu@<新机器IP> 'hostname && date'
#      不再提示密码即成功。
#
# 如果连不上，按顺序排查：
#   ping <新机器IP>                        # 网络通不通
#   ssh -p 22 mamingyu@<新机器IP>          # 端口/服务是否正常
#   （新机器上）sudo systemctl status sshd # sshd 是否在跑
#   （新机器上）sudo firewall-cmd --list-ports  # 防火墙是否放行 22
# 两台机器不在同一内网时，需要找网管开互通策略或用一台公网跳板机中转：
#   ssh -J mamingyu@<跳板机> mamingyu@<新机器IP>   # rsync 加 -e 'ssh -J ...' 同理
# =========================================================================

set -u

# ===================== 步骤 B：填这两个变量 =====================
SRC_HOST="mamingyu@u"            # 老机器 ssh 地址（发送方）
SRC_ROOT="/data/mamingyu"        # 老机器资产根目录
DST_ROOT="/data/mamingyu"        # 新机器落地根目录（保持一致最省事）
# ================================================================

command -v rsync >/dev/null || { echo "需要 rsync"; exit 1; }

echo "== 免密检查：ssh $SRC_HOST 'hostname'"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$SRC_HOST" 'hostname' 2>/dev/null; then
    echo "免密未生效，先按文件头部的步骤 A 配置。"
    exit 1
fi

echo "== 先看一眼老机器上有哪些资产、各多大 =="
ssh "$SRC_HOST" "du -sh $SRC_ROOT/qwen_35b_shared_expert_l*.pt \
    $SRC_ROOT/datasets/layer3*_onpolicy* \
    $SRC_ROOT/outputs_ffn_lut_layer3* \
    $SRC_ROOT/lut_bundles 2>/dev/null"

read -r -p "确认开始传输？[y/N] " ans
[ "$ans" = "y" ] || exit 0

mkdir -p "$DST_ROOT/datasets" "$DST_ROOT/lut_bundles"

# teacher 权重（3 个小文件，秒级）
rsync -avP "$SRC_HOST:$SRC_ROOT/qwen_35b_shared_expert_l"*.pt "$DST_ROOT/"

# 三层数据集
rsync -avP "$SRC_HOST:$SRC_ROOT/datasets/layer37_shared_expert_v3_onpolicy" \
           "$DST_ROOT/datasets/"
rsync -avP "$SRC_HOST:$SRC_ROOT/datasets/layer38_shared_expert_v3_onpolicy" \
           "$DST_ROOT/datasets/"
rsync -avP "$SRC_HOST:$SRC_ROOT/datasets/layer39_shared_expert_v3_onpolicy_multilayer" \
           "$DST_ROOT/datasets/"

# 三层 v3 checkpoints + v4 转换结果（v3 与 as_v4 是两个独立目录）
for L in 37 38; do
  rsync -avP "$SRC_HOST:$SRC_ROOT/outputs_ffn_lut_layer${L}_shared_expert_v3_onpolicy" \
             "$DST_ROOT/"
  rsync -avP "$SRC_HOST:$SRC_ROOT/outputs_ffn_lut_layer${L}_shared_expert_v3_onpolicy_as_v4" \
             "$DST_ROOT/"
done
rsync -avP "$SRC_HOST:$SRC_ROOT/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer" \
           "$DST_ROOT/"
rsync -avP "$SRC_HOST:$SRC_ROOT/outputs_ffn_lut_layer39_shared_expert_v3_onpolicy_multilayer_as_v4" \
           "$DST_ROOT/"

# bundles（3 × 626MB）
rsync -avP "$SRC_HOST:$SRC_ROOT/lut_bundles/" "$DST_ROOT/lut_bundles/"

echo "== 传输完成，校验大小 =="
du -sh "$DST_ROOT"/qwen_35b_shared_expert_l*.pt \
       "$DST_ROOT"/datasets/layer3* \
       "$DST_ROOT"/outputs_ffn_lut_layer3* \
       "$DST_ROOT"/lut_bundles
