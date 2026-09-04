# KV Cache 压缩方法全景

> 目标：在 Qwen3.6-35B-A3B 上验证 KV cache 压缩，为 CIM 设备降低推理存储/带宽。
> 已跑通统一 eval 接口（`kv_cache/eval_kv_cache.py`），支持 multi-turn 长对话压力测试。

## 已实现的 3 个方法

| 方法 | 核心思路 | 状态 | 关键结果 |
|---|---|---|---|
| **KIVI** | K per-channel + V per-token 量化 | ✅ 已跑 | 2-bit 文本崩；4-bit 用户判"炸了" |
| **Retention** | sink + recent 位置驱动 eviction | ✅ 已跑 | 1024 = 无损（125x）；256 = 明显幻觉（500x） |
| **Heavy-Hitter** | sink + recent + 中间高 importance token | ✅ 已实现，待跑 | importance 用 key L2 norm proxy |

## 其余方法（按思路分组）

### 1. 量化类（不丢 token，只降精度）

**思路**：所有 token 保留，但 K/V 用低 bit 表示。

| 方法 | 特点 | 对我们的价值 |
|---|---|---|
| **KVQuant** | per-channel + 非均匀量化 + q-order 保持敏感通道 | KIVI 的直接加强版，值得做 |
| **ZeroQuant / LLM.int8()** | 混合精度，outlier 通道保留 FP16 | 工程复杂，提升有限 |
| **QuaRot / SpinQuant** | 旋转变换让分布更均匀，量化误差更小 | 需要改权重，侵入性大 |

**评价**：量化类天花板明显（2-bit 大概率崩），4-bit 是安全区。和 KIVI 属同一族，不建议重复投入。

### 2. Eviction 类（丢 token，保留子集）

**思路**：只保留部分 token 的 KV，其余丢弃。

| 方法 | 特点 | 对我们的价值 |
|---|---|---|
| **StreamingLLM** | sink + recent，无中间保留 | = 我们的 Retention，已覆盖 |
| **H2O** | 按 cumulative attention score 淘汰 | = Heavy-Hitter 的 attention 版 |
| **Scissorhands** | 低 attention + 高 retrieval 双标准 | 需要真实 attention score |
| **SnapKV** | prefill 后按 attention 聚类选重要 token 池 | 实现简单，适合我们的场景 |

**评价**：Eviction 类的关键差异在 **importance 度量**。我们的 Heavy-Hitter 用 key norm proxy，SnapKV/H2O 用真实 attention score。**下一步优先把 importance 换成真实 attention score**。

### 3. 分层缓存类（Hot/Warm/Cold）

**思路**：token 分档，不同精度/策略。

| 方法 | 特点 | 对我们的价值 |
|---|---|---|
| **PyramidKV** | 层数越深保留 token 越少（金字塔形） |  layer-adaptive，和我们多层实验思路一致 |
| **MiniCache** | 相邻层 KV 做差分 + 量化 | 跨层信息冗余，适合 CIM |
| **Ada-KV** | 按层动态分配 cache budget | 和 PyramidKV 类似 |

**评价**：分层类是目前学术界 KV cache 压缩的主流方向，**PyramidKV / Ada-KV 是最值得跟进的**。

### 4. Token 合并 / 聚合类

**思路**：相似 token 的 KV 合并成一个代表。

| 方法 | 特点 | 对我们的价值 |
|---|---|---|
| **Token Merging (ToMe)** | 语义相似 token 合并 | 压缩率有限，视觉任务为主 |
| **ClusteredKV** | K-means 聚类，每簇保留 centroid | 需要维护 codebook，和 LUT 思路契合 |
| **CaM** | 按注意力簇合并 value | 适合长文档 |

**评价**：ClusteredKV 的 codebook 思路和我们的 LUT 方向天然契合，**值得作为"LUT-KV"的候选设计**。

### 5. 结构压缩类（改模型结构）

| 方法 | 特点 | 对我们的价值 |
|---|---|---|
| **GQA / MQA / MLA** | 减少 KV head 数 | 模型已固定（Qwen3.6 已是 GQA），不适用 |
| **Cross-layer KV sharing** | 相邻层共享 KV | MiniCache 已覆盖类似思路 |
| **YOCO (You Only Cache Once)** | 只缓存一层 KV，其余层复用 | 架构级改动，侵入性大 |

**评价**：结构类侵入性太大，不符合我们"plug-in 替换"的实验方式，跳过。

### 6. 与 GDN / Linear Attention 结合

**思路**：Qwen3.6 大部分层是 GDN（recurrent，无 KV cache），只有 full-attention 层有 KV。

| 方法 | 特点 | 对我们的价值 |
|---|---|---|
| **Mamba-in-the-middle** | 中间层用 linear attention | 模型已如此 |
| **Linear KV cache for full-attn layers** | 把 full-attn 层的 KV 也改成 recurrent | 这是我们的 GDN 方向，已确认"都很好" |

**评价**：Qwen3.6 天然是 GDN + 少量 full-attention，KV cache 压缩只需要针对少数 full-attention 层。这意味着**实际可压缩的 KV 量本来就小，评估时要算清楚总收益**。

## 推荐优先级

| 优先级 | 方法 | 理由 |
|---|---|---|
| **P0** | Heavy-Hitter 换成真实 attention score | 当前实现的 key norm proxy 太弱，真实 attention 是正确做法 |
| **P0** | retention 512 结果 | 确认 retention 边界，已有基线 |
| **P1** | PyramidKV / Ada-KV | 分层类主流方向，layer-adaptive budget |
| **P1** | SnapKV | 实现简单，attention-based eviction 的代表 |
| **P2** | ClusteredKV（LUT-KV） | codebook 思路，与项目 LUT 主题契合 |
| **P2** | KVQuant | KIVI 加强版，但天花板有限 |

## 当前实验下一步

1. 跑完 `retention_l512_s4_multiturn.json`（边界确认）
2. 跑完 `heavy_hitter_l256_s4_r128_multiturn.json`（importance-aware vs 位置驱动）
3. 把 Heavy-Hitter 的 importance 从 key norm 换成 **prefill attention score**（需要 hook attention 层）
4. 根据 2/3 的结果决定走 PyramidKV 还是 SnapKV 方向
