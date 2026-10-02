# vLLM 集成 Spike 报告（docs/26 §4 Route A 第 ①② 步）

日期：2026-10-02
源码：`LLM_LUT/v8/spike/vllm-src`，vLLM main @ 58b32984（2026-10-02，浅克隆，
只读）。所有行号以该 commit 为准。

## 结论：GO。两条原以为高风险的事都不成立

1. **不需要 fork vLLM**。两个官方扩展点正好覆盖我们的需求：
   - `@register_kv_cache_spec`（`vllm/v1/kv_cache_spec_registry.py:8`）注册自定义
     KVCacheSpec + manager；
   - `register_backend(Backend.CUSTOM, ...)`（`vllm/v1/attention/backends/registry.py:152-170`）
     注册自定义 attention backend。集成代码全部写在我们自己的包里，vLLM 侧零改动。
2. **per-key scores 风险解除，且不需要任何调度器钩子**：v8 的 importance 是
   "prefill 末尾 obs_window(64) 个 query 对每个 key 的注意力质量"（`attention_scores.py`
   的 stash 语义）。backend 的 `forward()` 在 prefill 期间**每个 chunk 都能看到
   该请求的 K/V**（写路径必经），可以在 impl 内部用一个 fp32 累加器
   `[num_keys]` 自己累计分数，淘汰决策完全在 backend 内部完成。scheduler /
   model runner 都不用碰。

## 关键机制（文件:行号）

**模型**：`Qwen3_5MoeForCausalLM` → `vllm/model_executor/models/qwen3_5.py`（注册
registry.py:200），复用 Qwen3NextDecoderLayer，按 `config.layer_types` 分流
（qwen3_5.py:147/155）：linear_attention 层走 GDN，full_attention 层走
Qwen3NextAttention。**那 10 个 full-attn 层就是要替换 KV 的层**（GDN 层状态不动）。

**"每请求固定 KV"的杠杆**——`KVCacheSpec.max_memory_usage_bytes(vllm_config)`：
- FullAttentionSpec 的实现是 `cdiv(max_model_len, block_size) × page_size_bytes`
  （kv_cache_interface.py:~578）——按最长上下文给每请求留满血空间，这就是
  full 配置并发上限低的根源；
- MambaSpec 的实现是**常数**（kv_cache_interface.py:1083）——这就是 GDN 层
  不管上下文多长每请求只占定长的机制；
- **自定义 spec 重载这一个方法** → allocator（`vllm/v1/core/kv_cache_manager.py:371`
  allocate_slots）自动按 budget 给每请求算账 → 总块数由剩余显存定
  （kv_cache_utils.py get_kv_cache_configs，gpu_memory_utilization 口径）→
  "KV 压缩 → 更多并发"的传导链是 vLLM 原生机制，不用我们造。
  512 slots/请求 ≈ 512×2×256×2B×2×10层 ≈ 10.5MB（bf16），对 128k 满血
  (~2.6GB/请求) 是 ~250×。

**写钩子**——`AttentionImpl.do_kv_cache_update(layer, key, value, kv_cache,
slot_mapping)`（backend.py:1113 默认实现；flash_attn.py:1518 实现）：
scatter 写 `reshape_and_cache_flash`（flash_attn.py:1536）。**override 这个方法**
= 在写路径上做"每 chunk 累计分数 + 超 budget 时淘汰/折叠/量化后写"，即
prefill 内滚动淘汰的实现点。kv_cache 张量 layout 由 spec 的 shapes 决定，
我们可以定义自己的 page 布局。

**读路径**：`forward()` 调 `flash_attn_varlen_func`（flash_attn.py:1536 附近
的 paged 变体），block_table 来自 attn_metadata；GQA 原生（kernel 直接收
num_kv_heads，无 repeat_kv 物化）。新版 FA 调用已带 `s_aux=self.sinks` 参数
（flash_attn.py:1504）——sink 通道已 plumbing 进 kernel，v8 的 sink 段可能
可以直接复用。**定制点**：注意力前把压缩页内的 K/V 反量化/展开成 kernel 要的
paged 视图，或让 kernel 直接读我们的布局（工作量大时选前者）。

**hybrid KV group**：`UniformTypeKVCacheSpecs`（kv_cache_interface.py:~1200）
把同类层打包成 group，hybrid 模型天然是"full-attn group + mamba group"两
组，各组 block_size 可不同（kv_cache_utils.py 的 group 逻辑）——我们的压缩
group 与 GDN group 互不干扰。

## 集成面（最小改动，估计 5 个新文件 ~800-1200 行，vLLM 零改动）

| # | 组件 | 做法 |
|---|---|---|
| 1 | `CompressedKVSpec(AttentionSpec)` | 重载 `max_memory_usage_bytes` = cdiv(budget, block_size)×page_size_bytes；`merge` 类方法；`@register_kv_cache_spec` 注册；prefix_cacheable=False |
| 2 | `CompressedAttnBackend` | 实现 AttentionBackend 接口（照抄 flash_attn.py 骨架），`@register_backend(Backend.CUSTOM)`；get_kv_cache_spec 返回 #1 |
| 3 | `CompressedAttnImpl(AttentionImpl)` | 继承 FlashAttentionImpl；override `do_kv_cache_update`（分数累计+淘汰+写）与 `forward`（读压缩页）；维护 per-request 分数累加器与小元数据 |
| 4 | 淘汰/折叠/量化移植 | 从 `heavy_hitter_cache.py` 移植选择/凸折叠/k8v8 到 #3 的写路径（纯 torch，kernel 无关） |
| 5 | 启动 glue | `python -m vllm... --attention-backend CUSTOM` + 注册模块 import；先 `--enforce-eager` 跑通再管 CUDA graph |

## 遗留确认项（spike ③ 之前要查，均为代码阅读，不占 GPU）

1. **block table 语义**：full-attn 的 block table 是"seq position → block"
   的映射；我们压缩布局内位置会随淘汰变动，是用"请求私有页 + backend 内部
   slot 重映射"（不动 scheduler 的表）还是 MambaSpec 式 state 页——读
   `single_type_kv_cache_manager.py` 定案。**倾向前者**：scheduler 照常按
   budget 分配块，块内布局 backend 自管。
2. **CUDA graph 兼容性**：decode 步的自定义元数据是否在 capture 路径上
   可序列化（gdn_attn.py 是现成通过样例）。
3. **prefix caching / spec decode** 对该 group 显式禁用（config 层面）。
4. obs_window 分数在 chunked prefill 边界的对齐方式与 docs/16 档案的
   一致性标定（B=1 回归门重跑）。

## 建议的 spike ③ 执行

单机（1×A800 即可）：vLLM 装好后 `--enforce-eager`、单请求、64k 负载，
输出与旧 harness 512 档对拍（turn 0 逐字一致门槛，参照 runbook step 2 的
判定先例）。环境注意：vLLM 仓库 AGENTS.md 要求 python 全部走 `uv`。
