# v8 调试记录：fact_acc 0/16 → shared_expert gate 根因（2026-10-15 ~ 10-16）

> 原载于 AGENTS.md，2026-10-16 迁入本文件。AGENTS.md 只留结论与红线，排查/实验记录一律写 docs。

## 已确认的事实（不要再重查）

1. **打包链路无辜**。`verify_lut_bundle.py pack`：L37/38/39 bundle 与 checkpoint 逐字段一致。
2. **权重 = 训练原样**。qual（公平 on-policy 链）cos 与训练 summary.json 自评严丝合缝：
   L37 0.75（训练 0.77）、L38 0.85（0.90）、L39 0.74（multilayer 0.80）。左尾烂是训练内在的
   （p10≈0.5），serve/转换没引入额外损伤。
3. **短上下文栈健康**：9 token prompt 下 LUT 输出与 dense 逐字级一致（三层全装）。
4. **64k 长上下文出现退化**：raw 补全下 LUT 有重复循环，dense 没有。
5. serve 环境约束：共享机唯一可用档位 `--gpu-memory-utilization 0.70`；
   `CUDA_VISIBLE_DEVICES=6,7` + `V8_LUT_LAYERS="37 38 39"` + `V8_LUT_BUNDLE_DIR=/data/mamingyu/lut_bundles`；
   模型路径必须写全 `/home/u/downloads/models/Qwen3.6-35B-A3B`（父目录没 tokenizer）；
   bench 参数是 `--concurrency` 不是 `--currency`；404 = 模型名和 serve 路径不一致。
6. triton 两个坑（已修）：kernel 体里的名字只从 `fn.__globals__` 解析（模块级 import，
   函数内 import 旧 triton 会炸 `NameError: tl`）；launch 必须在 `x.device` 的上下文里
   （跨卡时否则报骗人的 "cpu tensor?"）。

## 待验证的假设（用户的）

~~LUT 校准/训练用的 prompt 分布 vs bench 实际 prompt 分布错位。~~
**2026-10-15 被 gen 实验否掉**：32k 长文档×3 轮 QA，LUT 三轮全对（19千万元/王丽芳/30天，
均准确引用原文），短问答也全对。权重和分布在 32k 内都无辜（64k 双文档未测，但 8/512 是
全崩不像渐变）。嫌疑转移到 serve 运行时。

## 根因（2026-10-16 定案）：shared_expert gate 语义边界不一致

**0/16 的根因**：LUT bundle 学的是 **HF 语义**的 shared expert 输出（raw MLP，
gate 在模块外乘），而 vLLM 的 `shared_expert` 模块**内部**已经乘了
`sigmoid(shared_expert_gate(x))`（vLLM 0.19.1 的 `Qwen3NextMLP` 复用
`Qwen2MoeMLP`，`expert_gate` 即 `shared_expert_gate`）。插件替换整个
`shared_expert.forward` 后 gate 丢失 → 输出膨胀 **1/σ(g(x))**。

数值全部吻合（serve 实测 dense/HF dense = σ(g)）：
- L37: 2.1/15.5 = 0.135 → gate logit ≈ -1.86
- L38: 2.4/24.5 = 0.098 → gate logit ≈ -2.22
- L39: 15.2/19 = 0.80 → gate logit ≈ +1.39（所以 L39 碰巧"健康"）

gate 逐 token 波动 → 解释 ratio 逐 token 波动；cos 不受标量影响 → 解释
serve cos 与训练自评吻合。HF 链路 3/3：HF 里 gate 在模块外，raw LUT 语义正确。

**已排除**（全部有实测/审计）：pack、bundle 表（=训练数据，行 norm 13.1/26.2/13.7
vs 数据 15.5/24.5/19）、teacher 权重（与模型逐字节一致）、采集 hook（三层全
shared_expert）、x 尺度（训练 57~68 vs serve 50~60）、均值初始化/转换脚本、
rank 方案（rank1 ydf 与 rank0 全等）、kernel、探针自身。

**修复**（`vllm_plugin/lut_ffn.py`，v2026-10-08l）：LUT 输出乘回
`sigmoid(shared_expert_gate(x))`。表不重训、不重新打包。

**2026-10-16 已验证**：64k 多轮 bench（N=16，64 sessions，三层全装）
**fact_acc = 0.9551（489/512）**，errors=0。对照：修复前 0/16，baseline dense
serve ~0.734。0/16 → 0.955，gate 根因一次性定罪。0.955 > 0.734 超出预期，
可能是该 bench 上的偶然/方差，不要过度解读，但"LUT 替换可用"已实锤。

**并发对照（2026-10-16，同 serve 配置：0.70 mem / max-num-seqs 4 / 512 slots /
N=16，data=longctx_multi_turn_65536_64docs）**：

| 层数 | sess/h | fact_acc | P50 会话 | wall | 备注 |
|---|---|---|---|---|---|
| 0（baseline） | 87.6 | 0.9844 | 647s | 2629s | |
| 1（L37） | 81.5 | 0.9707 | 694s | 2828s | DEBUG=0 |
| 3（L37/38/39） | 99.9 | 0.9551 | 565s | 2306s | **DEBUG=1（ handicap）** |

LUT 组带 V8_LUT_DEBUG=1  handicap 仍胜出，方向信号（doc 42 Phase 1 判据：
compute 真被替掉、吞吐方向正确、质量不崩）初步成立。**但单次跑噪声 ±8~10%
（三次 wall 2306/2629/2828s），已淹没 per-layer 预期收益（几个百分点）——
1 层垫底而 3 层最高只能用噪声解释，层数扫描原协议判读无效。**

遗留：max-num-seqs=4 与 bench N=16 不对齐，两组绝对值都被准入上限压低；
噪声未量化前不给扩层决策。结果文件：results/phase1_*_c16.json。

## 历史排查记录（2026-10-15）

同一 bundle 三层全装：HF 路径 32k QA 3/3，vLLM serve（64k 多轮）**0/16**。
关键推理：**0/16 是主动投毒不是 dense 回退**——插件失败会静默保持 dense（WARNING），
dense 输出是 baseline 质量（~0.734），不可能 0/16。所以 LUT 确实装着、确实在跑、跑得很错。
kernel 基本排除：test_lut_ffn 在 vllm_py310 下验证过 kernel ≡ reference（但那是小 n）。

投毒两类假设：
- **A. 与长度无关**：TP rank0 方案不成立（shared_expert 输出没被 all-reduce → rank 间状态分叉；
  或 is_tp_rank0() 两 rank 都 True → all-reduce 后翻倍）。**探针 v3 已排除**。
- **B. 与长度有关**：kernel 在 serve 大 n（64k prefill 一次前向几万 token）下有 bug，HF 只测到 32k。

定位靠实测不靠猜（probe 猜类别已废弃）：
1. ~~对照组~~ **跳过**：0.734 红线就是这套 eval 在 baseline serve 上的实测值，对照已存在。
2. **插桩组**：`bash tools/serve_debug.sh lut 18002 "6,7"`（V8_LUT_DEBUG=1，插件前 4 次调用对真实
   输入同时算 dense(x) 和 LUT(x)）。`grep v8_debug logs/serve_lut_debug.log` 拿逐层 rank cos：
   cos ≈ HF 的 0.75~0.85 → LUT 给定输入是对的，毒在 rank 方案/下游；cos ≈ 0 → 输入/kernel 现形。
