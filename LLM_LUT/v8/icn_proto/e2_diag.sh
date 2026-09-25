#!/usr/bin/env bash
# E2 矩阵命令 2 (sp-1 x s=1.6) 失败诊断脚本
# 用法: bash icn_proto/e2_diag.sh   (在 LLM_LUT/v8 目录下运行)
set -uo pipefail
cd "$(dirname "$0")/.."

echo "########## 1) 失败 JSON 汇总 ##########"
cat > /tmp/e2_diag.py <<'EOF'
import json, os

base = "results/icn_proto"
fails = ["blkcluster_s8t40_20260925_175925", "blkcluster_s8t40_20260925_180218",
         "blkcluster_s8t40_20260925_180523", "blkcluster_s8t40_20260925_180816"]
for p in fails:
    fp = os.path.join(base, p + ".json")
    if not os.path.exists(fp):
        print("== MISSING", fp)
        continue
    d = json.load(open(fp))
    print("==", p)
    print("   spill dropped:", {w: s.get("dropped_bytes") for w, s in d["spill"]["workers"].items()})
    print("   resident_blocks:", {w: s["resident_blocks"] for w, s in d["workers"].items()})
    print("   failed:", d["failed"])
    for r in d["records"]:
        if not r.get("ok"):
            dec = r.get("decision") or {}
            xfer = r.get("xfer_blocks") or []
            print("   FAIL", r["request_id"], "->", r.get("worker"), "E=", r.get("E"),
                  "mode=", dec.get("mode"), "xfer_n=", len(xfer))
            print("        err:", str(r.get("error"))[:160])
EOF
python /tmp/e2_diag.py

echo
echo "########## 2) 失败块全生命周期 (cell 日志) ##########"
for LOG in results/icn_proto/cell_logs/cell_pp2_z16s1.6_*_sp-1_s2_*.log; do
  [ -e "$LOG" ] || { echo "no log matched: $LOG"; continue; }
  echo "---- $LOG : block 1840-1856 ----"
  grep -n "1840-1856" "$LOG" | head -30
  echo "---- $LOG : worker traceback ----"
  grep -n -B5 -A10 "resume block not resident" "$LOG" | head -80
done
