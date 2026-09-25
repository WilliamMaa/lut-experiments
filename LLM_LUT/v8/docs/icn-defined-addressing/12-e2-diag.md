# 12 — E2 矩阵命令 2 (sp-1 × s=1.6) 失败诊断

日期：2026-09-25
状态：进行中 — 四格 rc=1，部分 turn 失败（failed=3/8/5/8），全部报
`RuntimeError: resume block not resident: <块名>`。

## 1. 一键执行

在 `~/lut-experiments/LLM_LUT/v8` 下运行（git pull 后）：

```bash
bash icn_proto/e2_diag.sh
```

脚本内容见 `icn_proto/e2_diag.sh`，与本文件 §2 逐条等价。

## 2. 分步指令（与脚本等价，供审计）

```bash
cd ~/lut-experiments/LLM_LUT/v8

# (1) 失败 JSON 汇总：spill dropped、resident 块数、每个失败 record 的 worker/E/mode/xfer
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

# (2) 失败块全生命周期 + worker traceback（cell 日志）
for LOG in results/icn_proto/cell_logs/cell_pp2_z16s1.6_*_sp-1_s2_*.log; do
  echo "---- $LOG : block 1840-1856 ----"
  grep -n "1840-1856" "$LOG" | head -30
  echo "---- $LOG : worker traceback ----"
  grep -n -B5 -A10 "resume block not resident" "$LOG" | head -80
done
```

## 3. 待查假设

1. fetch-deliver 路径 `on_message` "delivered" 分支用
   `e = self._tip_end(xfer["names"][-1])` 反推 E —— 目标 worker 分散持有
   （resident∪spilled 交错）时 need 可能是中段，`names[-1]` 未必是 tip。
2. 调度器 `w.resident`/`w.spiled` 视图被 status 滞后或驱逐/清扫消息交错污染，
   导致 assign 的 E 超出 worker 实际持有。

## 4. 输出回来后

- 定罪于 scheduler.py 具体路径 → 修复 + test_e2.py 回归测试 + 本地五测全绿
- 让用户 `--drop-bad` 清 4 格，重跑 runbook §4 矩阵命令 2
- 根因补进本文件 §4c 与 runbook §6 异常表
