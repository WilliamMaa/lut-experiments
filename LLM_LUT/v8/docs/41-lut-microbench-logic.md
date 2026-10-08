# 41 · LUT microbenchmark 逻辑链与路线判决（v3：撤回 v6 死刑，下一道门改为 FFN 边界扫描）

日期：2026-10-08（v2 修订）· 数据：`v6/scripts/benchmarks/micro_lut_vs_gemm.py` 实测（2×A800，bf16）· 前置：docs/39（假设）、docs/40（v6 家底）

## 实测数据

```
[dims] hidden=2048 shared_expert_inter=512
     N   dense ms   naive ms    flat ms  flat/dense
     1     0.0556    56.0003    53.8949     969.22x
    64     0.0635    55.9132    54.0180     850.34x
  8192     0.3992    56.7452    54.7529     137.14x
```

## 逻辑链

### ① eager LUT 不可行（判决成立，归因保留）

`naive ≈ flat ≈ 55ms` 强烈说明拍平没有换来收益，时间被大量细粒度 eager 算子的 CPU/kernel launch 开销吃掉（PyTorch/NVIDIA 对大量微小 kernel 在 eager 下被 launch overhead 卡死有明确文档）。但**精确归因未证明**：54ms = 2600 launches × 20µs 只是估算，还可能混有 Python 循环、index/gather 本体、依赖串行化、低 occupancy、临时张量分配等。**这不影响判决**——850× 差距已经足够判 eager 集成死刑（vLLM serving 恰好 eager）。

### ② fused kernel 的物理账（数字修正，判断不变）

- dense 每 token 权重读取 = 3×2048×512×2B ≈ **6.29MB**（v1 误写 8.4MB），能进 L2，小 batch 时 dense 本身也是 launch-bound；
- LUT 每 token ≈ **8KB 随机读**（coarse 行 4KB + 32×residual 行 128B）；
- 理论访存量差距巨大，但"理论带宽赢"到"GPU kernel 真赢"距离仍远（8KB 粒度随机读有效带宽、in-kernel 顺序遍历的延迟隐藏、triton 实现质量全未知）。需要 triton microbench 才能变成数——**见判决，现在不急着写**。

### ③ 端到端算术（v1 重大错误，按官方结构修正）

Qwen3.6-35B-A3B：40 层，每层 8 routed + 1 shared expert，expert inter=512，hidden=2048，~35B total / **3B activated**（每 token 8 routed + 1 shared）。单个 expert FFN = 3×2048×512 ≈ 3.15M MAC/token：

| 替换对象 | MAC/token | 占 active ~3G 比例 |
|---|---|---|
| v6 现状：L37–39 三个 shared（选定的起点，非能力上限） | 9.4M | **~0.31%** |
| 全部 40 层 shared（流水线层参数化，机制上可做，只是没测） | 126M | **~4%** |
| 全部 routed（8×40 experts） | ~1.01G | **~1/3** |

v1 把"3 层"误写成"shared expert 类别上限 ~1%"——**错误**：类别上限是 ~4%。v1 说 routed 占"90%+"——需限定：是 **expert FFN 内部**的 8/9=88.9%，全模型口径约 33%。

**两个结论分别成立（算术层）**：
- v6 现状（只替 3 层 shared）端到端上限 ~0.3%，够不到 118→125 的 +6% 门。
- 即使 shared 全线扩到 40 层（机制上可行、未测），天花板约 **~4%**（118→~123 理论极限，
  未扣 LUT 自身开销、未验证深层质量）。

**但以上不构成对 v6 的判决。** 两处修正（v3）：

1. **"死刑"只适用于两个具体命题，不适用于 v6 方法本身**：(a) eager 树遍历直接进 serving
   不可行（850×）；(b) 停在 3 层配置没有端到端意义（0.3%）。v6 的树+表方法是小规模
   质量验证过的、项目内唯一有生产级调参积累的 LUT realization，是后续一切工作的基座，
   不是被处决对象。
2. **"+6% 门"是 v8 serving 线的吞吐标尺，不是杀死 FFN 线的唯一标准。** 40 层 ~4% 覆盖、
   kernel 未测、存储可由 v8 headroom 覆盖——这些账没算完之前，"够不到 6%"不能推出
   "别做 FFN"。

## 判决（v3）

**下一道门（唯一）：v6 FFN 层数边界扫描**——按 `v6/docs/plans/15-ffn40-boundary-scan.md`
执行：固定 v6 验证过的表配置，量 8/…/N 层的质量–覆盖–存储边界。出口判据见 runbook §5。

**routed experts 降级为条件 fallback，不是方向**：只有当边界扫描显示 FFN 线在可用层数内
质量崩溃时才重启 routed 议题。理由：v6 是精心调参后仍有退化的成熟方法；routed 是零验证
新课题——router 条件分布、256 张表、长尾 expert 样本稀疏、路由动态性，每一项的风险都
不小于 v6 已暴露的问题。**在 v6 自己的边界没量出来之前，没有任何依据认为 routed 会更容易。**

**eager 死刑仍然成立**：850× 是集成方式问题，与层数扫描正交；扫描质量过关后，fused
kernel microbench 仍是必经之门（本文件 §①）。

**与 v8 headroom 的结合方式不变**（若 FFN 边界扫描过关）：

```text
v8 frees memory → FFN LUT 存储（~12.5 GiB @40层）由 KV headroom 覆盖
→ kernel 过关 → 端到端算总账
```

routed 的 hot-expert 方案仅作为远期备选记录在案，不在当前路径上。

## 复现

`python v6/scripts/benchmarks/micro_lut_vs_gemm.py --model-path <model>`（v6/scripts/benchmarks/micro_lut_vs_gemm.py）
