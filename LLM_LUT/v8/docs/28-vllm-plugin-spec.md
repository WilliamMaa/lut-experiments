# docs/28: vLLM 压缩 KV backend — 实施规格（spike③ 开工文档）

> 状态：代码已写完（按 vLLM **0.19.1** 接口，py_compile 通过），待远程实机验证。
> 目标版本变更记录：本文最初按 main@58b32984（0.30 时代 nightly）设计，但远程机器驱动 550.90.07（CUDA 12.4）
> 无法升级，torch 2.11+ 硬性拒绝该驱动，只有 torch ≤2.10 的 minor 版本兼容能跑 → 目标版本降为 **vllm 0.19.1 + torch 2.10.0+cu128**。
> 因此 §2 架构图和 §4 代码清单里凡与 0.30 接口相关的描述（customize_spec、registry、pool [nb,H,bs,2D] 等）**全部作废，
> 以 `v8/vllm_plugin/` 代码和 §7 的 0.19.1 接口表为准**。vLLM 源码参考副本：`v8/spike/vllm-src-019`（v0.19.1 浅克隆，只读）。
>
> 上游结论见 docs/26（计划）、docs/27（spike 报告 GO）。

## 1. 目标与验证口径

把 v8 heavy-hitter KV 压缩（`kv_cache/heavy_hitter_cache.py` 的 attn_score + 凸折叠版）接入 vLLM V1 serving 栈，
回答产品问题：**在 fixed HBM budget 下，compressed serving 能撑多少并发 / 什么 QPS**。

spike③ 验证口径（达成即 GO）：

1. 单请求、`--enforce-eager`、64k 负载、512 slots；
2. 输出与旧 harness `results/budget_512_32k` 对拍 —— 不要求逐字一致（attention kernel 从 sdpa 换成 mem-efficient SDPA，
   有 ~1e-3 级舍入差，docs/27 已确认），要求 turn-0 答案正确性同水平（fact acc 一致、无语义崩坏）；
3. `nvidia-smi` 确认 per-request KV 物理占用 ≈ 10.5MB（512 slots × 2 heads × 256 dim × 2B × 2(KV) × 10 层），
   而不是 64k 满血的 ~2.6GB。

## 2. 总体架构

### 2.1 核心不变量

| # | 不变量 | 理由 |
|---|--------|------|
| 1 | 每请求 KV 物理 footprint = `blocks_per_request` 块，**常量**，与 seq_len 无关 | 这是整个实验的产品价值；靠 `CompressedKVSpec.max_memory_usage_bytes` 让 allocator 只分这么多 |
| 2 | scheduler 的 slot_mapping 物理槽位**全部忽略**，backend 内部自己做 slot 重映射 | 请求 position 60000 的 token 在物理上只占用 512 槽之一；scheduler 的 block table 只做"页预留" |
| 3 | 压缩布局永远是连续紧凑的：`[sink | heavy-hitter | recent]`，长度 L ≤ retention_tokens | 读路径可以当普通连续 KV 用 |
| 4 | **prefill 每个 chunk 先淘汰再写入**，保证 chunk 内所有 query 能看到全部旧 compact 键 | 旧 compact 键 orig_pos 都 < chunk 起点 → 对所有 chunk query 因果可见，mask 只需管 chunk 内部 |
| 5 | 淘汰/折叠算法逐行移植 v8（§6），禁止任何"改进" | v8 的每个坑都是实测买来的 |

### 2.2 数据流

```
scheduler (零改动)
  └─ KVCacheManager.allocate_slots: 按 CompressedKVSpec 每请求固定分 blocks_per_request 块
  └─ builder.build(): 拼标准 metadata + 挂 per-request 状态 (req_states)
       │
impl.forward(layer, q, k, v, kv_cache, attn_metadata, output)   ← 重写，不调父类
  ├─ q_len > 1 (prefill chunk):
  │   1. 从 pool gather compact K/V [L slots]
  │   2. obs window (末 W=min(obs,C) 行) 算 column mass：
  │      逐 q-head fp32: softmax(q_h @ [compact_k; chunk_k]^T) 列和 → total [L+C], per_head [H_kv, L+C]
  │      （chunk 列加因果 mask；compact 列无 mask —— 不变量 4）
  │   3. 更新 snap 表（按 orig position 覆盖 [0, L+C)）
  │   4. 若 L+C > budget：选保留集（stable argsort + obs-window 排除 [+ span pool]）
  │      → gather 保留 K/V → 凸折叠被淘汰的 V（one-hot matmul, fp32）→ 新 compact L' ≤ budget
  │   5. 写回 pool 的请求私有块（自己的 slot 重映射）
  │   6. attention: SDPA(mem-efficient) q=[H,C,D], kv=[concat(compact,chunk)], additive mask
  │      compact 列=0，chunk 列=因果 -inf；逐请求循环（v1，N 小）
  └─ q_len == 1 (decode):
      1. append token，L+1 > budget 则同样淘汰一轮（decode token 用 prefill 均值参赛，v8 规则）
      2. attention: SDPA q=[H,1,D] vs compact k [H,L',D]，无 mask
```

CUDA graph：**不支持，v1 强制 `--enforce-eager`**（per-request 状态是 Python 对象 + 变长 compact 长度，进不了 graph 捕获；后续工作项）。

## 3. 文件清单（`v8/vllm_plugin/`）

| 文件 | 内容 | 依赖 |
|------|------|------|
| `__init__.py` | env 配置 + monkeypatch `Qwen3NextAttention.__init__` 注入 backend | vLLM model 层 |
| `spec.py` | `CompressedKVSpec`，照抄 `HiSparseHotSpec` 结构 | `vllm/v1/kv_cache_interface.py` |
| `backend.py` | `CompressedKVBackend` + `CompressedKVMetadataBuilder` + metadata dataclass | flash_attn backend 系 |
| `eviction.py` | 纯函数移植：obs 打分 / 选择 / 凸折叠。不 import vLLM | 只依赖 torch |
| `impl.py` | `CompressedKVImpl.forward` 全重写（数据流 §2.2） | eviction.py |
| `serve.py` | 入口包装：装 patch → 转 vllm CLI | — |

## 4. 代码

### 4.1 `eviction.py`（纯移植，可直接完工）

```python
#!/usr/bin/env python3
"""v8 heavy-hitter eviction, ported to pure functions from
kv_cache/heavy_hitter_cache.py (attn_score + convex-fold variant).

DO NOT "improve": additive folding inflated values 2-9x and caused
repetition loops (2026-09-10); CUDA topk is not run-to-run stable
(2026-09-12); index_add_ atomics flip near-tie decode tokens
(2026-09-15). All three are forbidden here.
"""
import torch

NEG_INF = float("-inf")


def obs_window_scores(q_lastW, k_comp, k_chunk, scale, causal_mask):
    """Column attention mass for the last W query rows.

    q_lastW: [H, W, D] fp32 (detached). k_comp: [H_kv, L, D]. k_chunk: [H_kv, C, D].
    causal_mask: additive [W, C] fp32 (0 / -inf), applied to chunk columns only.
    Returns total [L+C] fp32, per_head [H_kv, L+C] fp32.
    Memory: loops q-heads, keeps only [W, L+C] fp32 alive (v8 attention_scores.py
    lesson: single-tensor B*H*W*K fp32 >100GB at N=64/128k).
    """
    H, W, _ = q_lastW.shape
    H_kv, L, _ = k_comp.shape
    C = k_chunk.shape[1]
    n_rep = H // H_kv
    k_all = torch.cat([k_comp, k_chunk], dim=1)          # [H_kv, L+C, D]
    total = torch.zeros(L + C, device=q_lastW.device, dtype=torch.float32)
    per_head = torch.zeros(H_kv, L + C, device=q_lastW.device, dtype=torch.float32)
    for h in range(H):
        s = torch.matmul(q_lastW[h], k_all[h // n_rep].transpose(-1, -2)) * scale
        if C > 0:
            s[:, L:] += causal_mask                       # compact cols: no mask
        p = torch.softmax(s, dim=-1)                      # [W, L+C]
        ph = p.sum(dim=0)                                 # [L+C]
        total += ph
        per_head[h // n_rep] += ph
        del s, p, ph
    return total, per_head


def select_kept(scores_middle, middle_orig, snap_len, hh_budget,
                obs_window=0, span_window=0):
    """Deterministic heavy-hitter selection.

    scores_middle: [M] fp32. middle_orig: [M] long (original positions).
    Returns kept middle-relative indices [hh_budget], ascending (temporal order).
    """
    scores = scores_middle
    if obs_window > 0:
        scores = torch.where(
            middle_orig >= snap_len - obs_window,
            torch.full_like(scores, NEG_INF), scores)
    if span_window > 0:
        pooled = torch.nn.functional.max_pool1d(
            scores.unsqueeze(0).unsqueeze(0),
            kernel_size=2 * span_window + 1, stride=1,
            padding=span_window).squeeze()
        scores = pooled
    sorted_idx = torch.argsort(scores, dim=-1, descending=True, stable=True)
    kept = sorted_idx[:hh_budget].sort().values            # stable ties → low pos wins
    return kept


def fold_evicted_values(hh_values, middle_values, kept, middle_orig, per_head):
    """Convex mass-weighted folding of evicted V into nearest kept successor.

    hh_values: [H_kv, hh, D] fresh gathered tensor (safe to modify).
    middle_values: [H_kv, M, D] read-only. kept: [hh] ascending. middle_orig: [M].
    per_head: [H_kv, snap_len] fp32 attention-mass table.
    Returns folded [H_kv, hh, D] in hh_values.dtype.
    """
    H, hh, D = hh_values.shape
    M = middle_values.shape[1]
    device = middle_values.device
    ev_mask = torch.ones(M, dtype=torch.bool, device=device)
    ev_mask[kept] = False
    ev_pos = ev_mask.nonzero(as_tuple=True)[0]             # [E]
    E = ev_pos.shape[0]
    if E == 0:
        return hh_values
    kept = kept.clamp(max=M - 1).long()
    ev_pos = ev_pos.clamp(max=M - 1).long()
    # First kept slot at a later position; fold FORWARD (later queries look back).
    tgt_slot = torch.searchsorted(kept, ev_pos, right=True).clamp(max=hh - 1)
    tgt_kept = kept[tgt_slot]
    K = per_head.shape[-1]
    ev_orig = middle_orig[ev_pos].clamp(max=K - 1).long()
    tgt_orig = middle_orig[tgt_kept].clamp(max=K - 1).long()
    own_orig = middle_orig[kept].clamp(max=K - 1).long()

    p_ev = per_head[:, ev_orig].float()                    # [H, E]
    p_own = per_head[:, own_orig].float()                  # [H, hh]
    ev_vals = middle_values[:, ev_pos, :]                  # [H, E, D]
    # Deterministic reduction via one-hot matmul — NOT index_add_ (atomics).
    onehot = torch.zeros(E, hh, dtype=torch.float32, device=device)
    onehot[torch.arange(E, device=device), tgt_slot] = 1.0
    contrib = p_ev.unsqueeze(-1) * ev_vals.float()         # [H, E, D]
    num = hh_values.float() * p_own.unsqueeze(-1)
    num = num + torch.einsum("es,hed->hsd", onehot, contrib)
    den = p_own + torch.einsum("es,he->hs", onehot, p_ev)
    return (num / den.clamp(min=1e-8).unsqueeze(-1)).to(hh_values.dtype)
```

### 4.2 `spec.py`

```python
import os
from dataclasses import dataclass

from vllm.v1.kv_cache_interface import FullAttentionSpec


def _env_int(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True, kw_only=True)
class CompressedKVSpec(FullAttentionSpec):
    """Per-request constant-footprint KV (v8 heavy-hitter compression).

    Structure mirrors HiSparseHotSpec (kv_cache_interface.py): the allocator
    reads max_memory_usage_bytes() / max_num_blocks_per_req() and budgets a
    FIXED number of blocks per request, independent of max_model_len.
    """
    retention_tokens: int      # V8_COMPRESS_SLOTS (512 / 1024, Pareto 甜点)
    sink_tokens: int = 4       # V8_SINK_TOKENS
    recent_tokens: int = 32    # V8_RECENT_TOKENS
    obs_window: int = 64       # V8_OBS_WINDOW
    span_window: int = 4       # V8_SPAN_WINDOW
    blocks_per_request: int = 0  # set by customize_spec

    # TODO(verify-1): field/property names below must match HiSparseHotSpec
    # exactly (page_size_bytes, prefix_cacheable). 验证方法:
    #   grep -n "class HiSparseHotSpec" -A 60 spike/vllm-src/vllm/v1/kv_cache_interface.py
    # 逐字段对齐；frozen dataclass 派生值在 __post_init__ 里用 object.__setattr__ 写。

    def __post_init__(self):
        bs = self.block_size
        n = (self.retention_tokens + bs - 1) // bs
        object.__setattr__(self, "blocks_per_request", n)

    @property
    def prefix_cacheable(self) -> bool:
        return False

    def max_memory_usage_bytes(self, vllm_config) -> int:
        return self.blocks_per_request * self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        return self.blocks_per_request
```

注意：margin。`blocks_per_request` 按 retention 向上取整即可（淘汰后 L ≤ retention 严格成立），
但**建议 +1 块 margin**（decode append-then-evict 的瞬态、以及 debug 期防越界）：
`n = (retention + bs) // bs + 1` —— 10 层每请求就多 0.3MB，可接受。开工时取带 margin 的版本。

### 4.3 `backend.py`

```python
from dataclasses import dataclass
from typing import Any, Optional

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadataBuilder

from .spec import CompressedKVSpec, _env_int


@dataclass
class CompressedKVMetadata:
    """FlashAttentionMetadata 的姊妹结构（不继承 —— v1 自己管一切）。"""
    req_states: list                 # RequestKVState，batch 对齐，由 builder 挂进来
    query_start_loc: Any             # 透传，attention 用
    num_computed_tokens_cpu: Any     # 每个请求的 chunk 起始 orig position
    # TODO(verify-3): builder.build 里能拿到什么（ForwardBatch 字段、block_tables
    # 访问方式），以远程实际签名为准补齐。


class CompressedKVBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "V8_COMPRESSED"

    @staticmethod
    def get_impl_cls():
        from .impl import CompressedKVImpl     # 延迟 import 防循环
        return CompressedKVImpl

    @staticmethod
    def get_builder_cls():
        return CompressedKVMetadataBuilder

    @classmethod
    def customize_spec(cls, spec) -> CompressedKVSpec:
        # 只对 FullAttentionSpec 出手；同模型的 Mamba/GDN spec 原样返回。
        if not isinstance(spec, FullAttentionSpec) or isinstance(spec, CompressedKVSpec):
            return spec
        return CompressedKVSpec(
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            block_size=spec.block_size,          # TODO(verify-2): 字段名以
            sliding_window=spec.sliding_window,  #   HiSparseHotSpec 用法为准
            retention_tokens=_env_int("V8_COMPRESS_SLOTS", 512),
            sink_tokens=_env_int("V8_SINK_TOKENS", 4),
            recent_tokens=_env_int("V8_RECENT_TOKENS", 32),
            obs_window=_env_int("V8_OBS_WINDOW", 64),
            span_window=_env_int("V8_SPAN_WINDOW", 4),
        )


class CompressedKVMetadataBuilder(FlashAttentionMetadataBuilder):
    """v1 策略：尽量复用父类 build 的标准字段，再挂上 per-request 状态。

    per-request 状态所有权在 builder（req_id -> RequestKVState），forward 只读 +
    更新其中的 tensor；请求结束时由 builder 在下一轮 build 里发现 req 消失而删除
    （v1 允许一个 step 的延迟释放）。
    """

    def __init__(self, kv_cache_spec, vllm_config):
        super().__init__(kv_cache_spec, vllm_config)
        self.spec = kv_cache_spec
        self.states: dict[str, "RequestKVState"] = {}

    def build(self, *args, **kwargs):
        # TODO(verify-3): 父类 build 返回 FlashAttentionMetadata；我们构造
        # CompressedKVMetadata，标准字段（query_start_loc 等）从父类结果里拷，
        # req_states 按 batch 顺序从 self.states 取（新请求在此创建：
        # num_computed_tokens==0 → slot 基址取自 block_tables）。
        raise NotImplementedError  # 开工时按 verify-3 结果填实
```

### 4.4 `impl.py`（核心，伪代码标出唯一两处待定）

```python
import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from . import eviction


class RequestKVState:
    """每 layer 每请求。tensor 全在 GPU；Python 对象壳仅 eager v1 用。"""
    __slots__ = ("req_id", "slot_base_blocks", "compact_len", "orig_idx",
                 "snap", "snap_len", "snap_per_head")

    def __init__(self, req_id, max_model_len, h_kv, device):
        self.req_id = req_id
        self.slot_base_blocks: list[int] = []   # 请求私有的 blocks_per_request 个块号
        self.compact_len = 0
        self.orig_idx = torch.zeros(max_model_len, dtype=torch.long, device=device)
        self.snap = torch.zeros(max_model_len, dtype=torch.float32, device=device)
        self.snap_per_head = torch.zeros(h_kv, max_model_len, dtype=torch.float32, device=device)
        self.snap_len = 0


class CompressedKVImpl(FlashAttentionImpl):
    forward_includes_kv_cache_update = True   # forward 自己写 KV，不走默认路径

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
        # query/key/value: [1, total_q, H(_kv), D] (vLLM 标准 4-D 布局, B=1 已拍平)
        # kv_cache: 本层 pool；我们不按 slot_mapping 写，用自己的块号算物理槽。
        q = query[0].transpose(0, 1)            # [H, q_len, D]
        k_new = key[0].transpose(0, 1)          # [H_kv, C, D]
        v_new = value[0].transpose(0, 1)
        H, C, D = q.shape
        H_kv = k_new.shape[0]
        scale = self.scale if hasattr(self, "scale") else D ** -0.5
        n_rep = H // H_kv

        for i, st in enumerate(attn_metadata.req_states):
            qs = attn_metadata.query_start_loc[i].item()
            qe = attn_metadata.query_start_loc[i + 1].item()   # TODO(verify-4)
            chunk_orig0 = attn_metadata.num_computed_tokens_cpu[i]  # 本 chunk 起点
            q_i = q[:, qs:qe].float()                          # [H, C, D]
            self._update_and_attend(st, kv_cache, q_i, k_new, v_new,
                                    chunk_orig0, scale, n_rep, output, qs, qe)

    def _update_and_attend(self, st, kv_cache, q, k_new, v_new,
                           chunk_orig0, scale, n_rep, output, qs, qe):
        C = q.shape[1]
        L = st.compact_len
        budget = self.spec.retention_tokens
        sink_n = min(self.spec.sink_tokens, budget)
        recent_n = min(self.spec.recent_tokens, budget - sink_n)
        hh_budget = budget - sink_n - recent_n
        dev = q.device
        phys = self._phys_slots(st)                        # [L] 物理槽（自己的块映射）

        # --- 1) gather 现有 compact K/V ---
        k_comp = kv_cache[0].index_select(0, phys) if L else None   # 布局 TODO(verify-5)
        v_comp = kv_cache[1].index_select(0, phys) if L else None   # [L, H_kv, D]

        # --- 2) obs 打分 ---
        W = min(self.spec.obs_window, C)
        k_all_h = None  # [H_kv, L+C, D]
        total = per_head = None
        if C > 1:  # prefill chunk：末 W 行做 obs window
            k_cat = torch.cat([k_comp, k_new], 1) if L else k_new
            rows = torch.arange(C - W, C, device=dev)[:, None]
            cols = torch.arange(C, device=dev)[None, :]
            causal = torch.zeros(W, C, device=dev)
            causal.masked_fill_(cols > rows, torch.finfo(torch.float32).min)
            total, per_head = eviction.obs_window_scores(
                q[:, -W:], k_comp if L else k_new[:, :0], k_new, scale, causal)
            st.snap[chunk_orig0:chunk_orig0 + L + C] = total
            st.snap_per_head[:, chunk_orig0:chunk_orig0 + L + C] = per_head
            st.snap_len = chunk_orig0 + L + C

        # --- 3) 拼接全量（orig 空间）并淘汰 ---
        orig_all = torch.cat([st.orig_idx[:L],
                              torch.arange(chunk_orig0, chunk_orig0 + C, device=dev)])
        k_all = torch.cat([k_comp, k_new], 1) if L else k_new     # [H_kv, L+C, D]
        v_all = torch.cat([v_comp, v_new], 1) if L else v_new
        total_len = L + C

        if total_len > budget:
            mid_end = total_len - recent_n
            mid_orig = orig_all[sink_n:mid_end]
            # decode 写入的 token（orig >= snap_len）用 prefill 均值参赛（v8 规则）
            valid = mid_orig < st.snap_len
            scores_mid = torch.where(valid, st.snap[mid_orig.clamp(max=st.snap_len - 1)],
                                     st.snap[:st.snap_len].mean())
            kept = eviction.select_kept(
                scores_mid, mid_orig, st.snap_len, hh_budget,
                obs_window=self.spec.obs_window if C > 1 else 0,
                span_window=self.spec.span_window)
            hh_k = k_all[:, sink_n:mid_end, :][:, kept, :]
            hh_v = v_all[:, sink_n:mid_end, :][:, kept, :]
            hh_v = eviction.fold_evicted_values(
                hh_v, v_all[:, sink_n:mid_end, :], kept, mid_orig, st.snap_per_head)
            new_orig = torch.cat([orig_all[:sink_n], mid_orig[kept], orig_all[-recent_n:]])
            k_all = torch.cat([k_all[:, :sink_n, :], hh_k, k_all[:, -recent_n:, :]], 1)
            v_all = torch.cat([v_all[:, :sink_n, :], hh_v, v_all[:, -recent_n:, :]], 1)
            st.orig_idx[:k_all.shape[1]] = new_orig
            st.compact_len = k_all.shape[1]
        else:
            st.orig_idx[:total_len] = orig_all
            st.compact_len = total_len

        # --- 4) 写回 pool（slot 重映射：紧凑槽 s -> 私有块 s//bs, 偏移 s%bs）---
        L2 = st.compact_len
        phys2 = self._phys_slots(st, L2)
        # TODO(verify-5): kv_cache 确切维度顺序（[2, num_blocks, bs, H_kv, D] 还是
        #   每层分开），以远程一次 print(kv_cache.shape) 为准改下面两行。
        kv_cache[0][phys2] = k_all.permute(1, 0, 2)   # -> [L2, H_kv, D]
        kv_cache[1][phys2] = v_all.permute(1, 0, 2)

        # --- 5) attention ---
        k_attn = k_all.repeat_interleave(n_rep, 0)     # [H, L2(+C), D]
        v_attn = v_all.repeat_interleave(n_rep, 0)
        q_h = q if C == 1 else q                        # decode: compact 全可见
        if C > 1:
            # compact 列无 mask + chunk 列因果：
            # 直接用 k_all（含 chunk 尾部）的话因果会被 compact 区段打乱，
            # 所以 attention 的 K = [compact; chunk]，mask 只遮 chunk 未来列。
            k_attn = torch.cat([k_attn, k_new.repeat_interleave(n_rep, 0)], 1)
            v_attn = torch.cat([v_attn, v_new.repeat_interleave(n_rep, 0)], 1)
            rows = torch.arange(C, device=dev)[:, None]
            cols = torch.arange(C, device=dev)[None, :]
            mask = torch.zeros(1, 1, C, C, device=dev)
            mask.masked_fill_(cols > rows, torch.finfo(torch.float32).min)
            mask_full = torch.zeros(1, 1, C, C, device=dev)
            mask_full[..., :L2] = 0
            mask_full[..., L2:] = mask[..., :]
            attn_mask = mask_full
        else:
            attn_mask = None
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0), k_attn.unsqueeze(0), v_attn.unsqueeze(0),
            attn_mask=attn_mask)                        # [1, H, C, D]
        output[qs:qe] = o[0].transpose(0, 1)            # TODO(verify-6): output 布局

    def _phys_slots(self, st, n=None):
        n = st.compact_len if n is None else n
        bs = self.spec.block_size
        blocks = torch.tensor(st.slot_base_blocks, device=self._dev())
        blk_of = torch.arange(n, device=blocks.device) // bs
        off = torch.arange(n, device=blocks.device) % bs
        return blocks[blk_of] * bs + off
```

上面的 forward 有两个已知的 v1 简化，开工时保持，不要"顺手优化"：

1. **逐请求循环**：N=64 × 10 层 × per-step Python 开销在 eager 下可接受（spike 只验证正确性与内存）；
2. **attention 用 mem-efficient SDPA**：probe 已验证该 kernel 吃 additive mask 且数值与 sdpa 一致
   （max|diff| ~2e-4）；flash varlen 不支持非空 mask，math kernel 大 shape 直接 OOM。

### 4.5 `__init__.py`

```python
import os

V8_COMPRESS_SLOTS = int(os.environ.get("V8_COMPRESS_SLOTS", "512"))
V8_SINK_TOKENS = int(os.environ.get("V8_SINK_TOKENS", "4"))
V8_RECENT_TOKENS = int(os.environ.get("V8_RECENT_TOKENS", "32"))
V8_OBS_WINDOW = int(os.environ.get("V8_OBS_WINDOW", "64"))
V8_SPAN_WINDOW = int(os.environ.get("V8_SPAN_WINDOW", "4"))

_patched = False

def patch():
    """把 CompressedKVBackend 注入 Qwen3.6 的 full-attention 层。"""
    global _patched
    if _patched:
        return
    from vllm.model_executor.models import qwen3_next

    orig_init = qwen3_next.Qwen3NextAttention.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["attn_backend"] = None          # 占位：见 TODO(verify-7)
        orig_init(self, *args, **kwargs)
        # 真正注入点在 verify-7 确认 Qwen3NextAttention 如何透传 attn_backend 后填实：
        #   kwargs["attn_backend"] = CompressedKVBackend

    qwen3_next.Qwen3NextAttention.__init__ = patched_init
    _patched = True
```

### 4.6 `serve.py`

```python
import sys
from . import patch

def main():
    patch()
    from vllm.entrypoints.openai.api_server import run_server  # TODO(verify-8): 实际入口函数名
    run_server(sys.argv[1:])

if __name__ == "__main__":
    main()
```

启动（远程 runbook 全文见 §8）：
`python -m vllm_plugin.serve /home/u/downloads/models/Qwen3.6-35B-A3B --enforce-eager --max-model-len 131072 ...`

## 5. 与 v8 harness 的语义差异（预期内，写报告时要列）

| 点 | v8 harness | vLLM 插件 | 影响 |
|----|-----------|-----------|------|
| attention kernel | torch sdpa | mem-efficient SDPA | ~1e-3 舍入差，不对拍逐字 |
| 并发形态 | 整 batch 同步 step、同长 | continuous batching、乱长短 | 指标不可直接比，只看 per-request 质量 |
| 淘汰时机 | 整段 prefill 后、第一个 decode step 一次性压缩 | ~~forward 内、chunk 前~~ → **v2026-10-04l 起同为 deferred**（prefill 只累积打分，首 decode 压缩） | v2026-10-04k 及之前的 per-chunk 淘汰是 64k 全崩（0/64）的直接原因：问题 token 从未参与打分 |
| per-layer 选择 | 是 | 是（shared_selection 留 TODO） | 与 v8 m4_k8v8 配置对齐 |

## 6. 移植清单（来自 heavy_hitter_cache.py / attention_scores.py 的实测教训）

**必须保留**：
- obs window = 末 min(W, chunk) 个 query 行；窗口内 token 排除出 heavy-hitter 候选（chat 模板 token 抢 sink 注意力）；
- decode 写入 token 用 prefill 均值参赛，**+inf 禁止**（一轮 generation 就把 prefill HH 全冲掉）；
- 凸组合折叠 V_t' = (p_t·V_t + Σp_i·V_i)/(p_t + Σp_i)，折进第一个位置更靠后的保留槽；
- stable argsort（descending, stable=True）选择，不用 topk；
- one-hot matmul 归约，不用 index_add_；
- 全程 fp32 打分/折叠。

**v1 不做**（后续工作项，别顺手加）：k8v8 量化（Pareto 已证 INT8 中性但 v1 先 bf16）、shared_selection、
CUDA graph、prefix caching、SLA/排队模型。

## 7. 接口确认状态（2026-10-09 更新：已按 vLLM 0.19.1 重写完毕，见 `v8/vllm_plugin/`）

**本地对照 spike/vllm-src-019（v0.19.1 浅克隆）已确认并写入代码**：

| 点 | 0.19.1 的结论 | 与 0.30 草案的差异 |
|----|---------------|---------------------|
| spec 注册 | **0.19 没有 spec registry**，不注册，直接用 | 0.30 要 `register_kv_cache_spec` |
| spec 替换钩子 | **0.19 没有 customize_spec**。monkeypatch `Attention.get_kv_cache_spec`：精确 `type(spec) is FullAttentionSpec` 时换成 CompressedKVSpec（排除 SlidingWindowSpec） | 0.30 走 backend.customize_spec 中心式调用 |
| blocks cap | 0.19.1 allocator 按 token 数要块，spec 无钩子 → monkeypatch `SingleTypeKVCacheManager.get_num_blocks_to_allocate`，对 CompressedKVSpec 把 num_tokens clamp 到 `blocks_per_request*block_size` | 0.30 草案靠 max_memory_usage_bytes |
| backend 注入 | monkeypatch `Attention.__init__` 注入 backend 类，按 TARGET_ARCH 门控 | 相同思路 |
| builder `__init__` | 签名 `(kv_cache_spec, layer_names, vllm_config, device)` 四参，定义 `__init__(*args, **kwargs)` 兼容 | — |
| per-request 键 | 0.19 的 CommonAttentionMetadata 有 query_start_loc_cpu 但**无 req_ids** → 用 block id 元组当请求键 | 0.30 有 req_ids |
| pool 布局 | **0.19.1 是 `[2, num_blocks, block_size, H_kv, D]`**：`kv_cache.unbind(0)` 得 K/V | 0.30 是 `[nb, bs, H, 2D]`，K=`[...,:D]` |
| merge | `FullAttentionSpec.merge` 会丢子类字段 → CompressedKVSpec 自己 override（校验全等后 deepcopy） | 相同 |
| serve 入口 | 优先 0.30 launchers 入口（model_tag→--model 映射），ImportError fallback：runpy 跑 `vllm.entrypoints.openai.api_server` 的 `__main__`。**0.19 的 api_server 自己不做 model_tag→--model 映射**（在 cli/serve.py 里，被 bypass），serve.py 已手动补 | 实机踩过：不补会回落默认模型 `Qwen/Qwen3-0.6B` 去连 HF |
| **子进程启动方式** | **0.19 的 OpenAI 入口强制 `VLLM_WORKER_MULTIPROC_METHOD=spawn`**（entrypoints/utils.py）。Qwen3.6 是多模态模型，mm 初始化在 API server 进程就初始化了 CUDA → `_maybe_force_spawn` 以 "CUDA is initialized" 为由**强制覆盖 fork**。spawn 的子进程全新解释器，API server 里的 monkeypatch 到不了 EngineCore/Worker → 症状：patch() 日志在、注入日志全无、服务正常跑 stock attention | 实机踩过（fork 方案被覆盖）。**解法：sitecustomize 自举**——serve.py 把插件根目录和 `_bootstrap/`（含 sitecustomize.py）注入 PYTHONPATH 并设 `V8_PLUGIN_AUTOPATCH=1`；spawn 子进程解释器启动时自动 import sitecustomize → 各进程自己 patch 自己。patch 失败时 sitecustomize 直接 sys.exit(1)，拒绝静默回退 stock attention |
| **AttentionBackendEnum** | `Attention.__init__` 会执行 `AttentionBackendEnum[backend.get_name()]`（attention.py:350），枚举是闭集，`V8_COMPRESSED` 不在其中 → ValueError。**解法：运行时给枚举注入成员**（`backend.register_backend_enum()`，value 按惯例填类路径，get_path/get_class 无需 override）；setattr 会被 Enum.__setattr__ 拒，用 `type.__setattr__` | 实机踩过 |
| **spec_manager_map** | EngineCore 的 coordinator 用 `spec_manager_map[type(spec)]` 选管理器类（single_type_kv_cache_manager.py:1129），自定义 spec 不在表里 → KeyError。**解法：patch() 里 `spec_manager_map.setdefault(CompressedKVSpec, FullAttentionManager)`**（行为上=预算被 clamp 的 full attention） | 实机踩过 |
| **get_kv_cache_spec 时序** | worker 在 `_initialize_kv_caches` 里对每个 Attention 实例调 `get_kv_cache_spec()`（gpu_model_runner.py:6926），此时**可能不在 `set_current_vllm_config` 上下文内** → `get_current_vllm_config()` 抛异常；若转换函数里做 arch 门控并吞异常会静默跳过（症状：patch/注入日志都在、并发上限数字不变）。**解法：转换处不做 config 依赖的门控**（backend 名字已足以证明身份） | 实机踩过（g 版修复，采纳后并发上限 192x→729x） |

| **prefill chunk 长度** | vLLM 按 8192 chunk 一次喂入（C 可达数千），arange 辅助张量只按 `budget+bs`（544）开 → `arange_cap[:C]` 静默截断 → `index_copy_` 报 `Number of indices (544) != source.size(2065)`（v2026-10-04j 实机踩过）。**解法：`need_ar = max(n_computed + C + 1, budget + bs)`，不够大就重建** | 同类坑：凡按 token 开表的辅助张量都要按 C 校验 |
| **async scheduling 元数据滞后** | vLLM async scheduling 下 chunk 的调度元数据可滞后一个 chunk：chunk3 到来时 `n_computed`=8192（本应是 16384），`seq_lens = num_computed + scheduled`。on-demand 重建条件 `need_ar=max(n_computed+C+1, budget+bs)` 算出 16385 恰不大于现有 16385 → 不重建 → `arange_cap[:L2]` 静默截断 → 64k 多 chunk 第 3 chunk 必崩 `RuntimeError: value [24576,1,256] vs target [16385,1,256]`（16385=上一 chunk 的 need_ar，2026-10-05 实机踩过；单 chunk smoke 永不暴露）。**解法（v2026-10-04n）：① arange 一次性开满 `V8_MAX_SEQ_TOKENS+bs+16`，不再按需重建；② prefill 元数据自愈 `if C>1 and n_computed < st.snap_len: n_computed = st.snap_len`（decode C==1 不走此路），触发时日志打 `stale metadata healed`** |
| **prefill 瞬态 OOM** | 手写 attention 路径的大张量瞬态不被 vLLM 的 activation profiling 计入：`repeat_interleave` 展开 GQA（[H,L,D]×2，64k 时各 1.7 GiB）+ fp32 [C,L2+C] mask 再 .to(bf16)（共 ~5 GiB），0.9 util 下只剩 1.3 GiB 头部 → 64k prompt ~98k 处 OOM（2026-10-05 实机踩过；baseline 走 flash kernel 无此问题）。**解法（v2026-10-04o）：① SDPA `enable_gqa=True`（kernel 内广播，不物化重复 K/V，启动时小探针一次，不支持则回退 repeat_interleave）；② mask 直接建为 query dtype（省 fp32 副本）；③ 插件服务用 `--gpu-memory-utilization 0.88` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** |
| **并发下 reset 误触发（质量 0/64）** | 新请求判定用 `n_computed == 0`，但 async scheduling 在请求上一步**执行完**前一直把 `num_computed` 停在 0（插件 Python 循环慢、调度器跑得越靠前 lag 越深；单请求时 lag 只有 1 chunk，并发时是'全程'）。后果：每个 prefill chunk 都被判成新请求 → 快照表每步清空 → 首 decode 压缩时 heavy hitter 全选错。并发 sweep 实测 `state reset` 触发 2916 次（正常应为每会话 ≤1 次），N=1 质量 0.84、N≥8 全档 0/64（2026-10-05 实机）。**解法（v2026-10-04p）：reset 判定从元数据挪到 builder 块表身份——进行中请求物理块 append-only、前缀稳定，按 `ceil(seq_len/bs)` 切掉 padding 后比对前缀，对不上即 `blocks[0]` 被重新发放 → 重建状态；删除 impl 里的 n_computed==0 启发式** |

**远程验证状态（2026-10-04 smoke ALL PASS 后更新）**：R1–R6 全部实机通过——0.19.1+cu128 import 正常、builder/metadata 无 TypeError、无 MTP 第二路径、GDN 组与 full-attn 组 spec 共存未炸、pool 布局确为 [2,nb,bs,H_kv,D]、TP=2 正常。原表留档：

| # | 待确认 | 失败时的症状 |
|---|--------|--------------|
| R1 | vllm 0.19.1 wheel 的 CUDA flavor 是 cu128（与 torch 2.10.0+cu128 ABI 匹配） | import vllm 报 `libcudart.so` / `undefined symbol` → 换 0.18.x 或查 wheel flavor |
| R2 | builder 初始化参不匹配 / metadata setattr 被后续流程拒绝 | 启动期 TypeError/AttributeError |
| R3 | Qwen3.6 config 是否带 MTP/spec-decode 层（第二条 attention 路径） | 启动挂 speculative 栈 → 查 0.19 arg_utils 加禁用 flag |
| R4 | GDN/Mamba 组与 full-attn 组的 spec 分组冲突（hybrid 模型两类 spec 共存） | 启动期 spec merge/grouping 报错 |
| R5 | pool 视图在 spec `has_layer_views` 默认下确实按 [2,nb,bs,H,D] 给到 forward | 第一次 forward 的 gather 形状 assert（日志打 kv_cache.shape） |
| R6 | TP=2 下 num_kv_heads=2 的切分与本插件的交互 | 启动期字段缺失 TypeError 或 forward 形状错 |

## 8. 远程 runbook（只写操作：跑什么、怎么算过）

### 8.0 环境（已装好，勿动）
- conda env `vllm_py310`；vllm==0.19.1 + torch 2.10.0+cu128；驱动 550.90.07（共享机不能动）。
- 插件服务固定 GPU 6,7 / 端口 18002；基准固定 GPU 0,1 / 端口 18003。
- 模型 `/home/u/downloads/models/Qwen3.6-35B-A3B`。
- 所有起服务的动作都在脚本内部完成（环境变量写死在脚本里），不要在终端手搓 serve 命令。

### 8.1 生成数据（缺哪个跑哪个）
```bash
python tools/gen_longctx_multiturn.py --target-tokens 65536 --num-docs 8 --tokenizer-path /home/u/downloads/models/Qwen3.6-35B-A3B --out data/longctx_multi_turn_65536.jsonl
python tools/gen_longctx_multiturn.py --target-tokens 32768 --num-docs 8 --tokenizer-path /home/u/downloads/models/Qwen3.6-35B-A3B --out data/longctx_multi_turn_32768.jsonl
```

### 8.2 并发修复验证（换代码后先跑这个，约 15 分钟）
```bash
bash tools/repro_concurrency.sh
```
通过标准（脚本自行打印）：版本 = 2026-10-06a；无 `[v8_plugin]` raise；fact_acc ≥ 0.7。

### 8.3 slots 扫描（约 2 小时）
```bash
bash tools/run_slots_sweep.sh
```
通过标准：每档无 ANCHOR FAIL / 500，fact_acc ≥ 0.734。
换档/换数据：`SLOTS_LIST="512 8192" DATA=data/longctx_multi_turn_32768.jsonl bash tools/run_slots_sweep.sh`
产出：`results/eval_64k_slots<S>.json`

### 8.4 并发扫描（约 2 小时）
```bash
bash tools/run_concurrency_sweep.sh
```
通过标准：N=1 的 fact_acc 与 8.3 一致（±1 题）；N≥8 不崩（fact_acc ≥ 0.7、errors = 0）。
换档：`SLOTS_LIST="512 2048" N_LIST="1 8 16 32" bash tools/run_concurrency_sweep.sh`
产出：`results/bench_c<N>_slots<S>.json`

### 8.5 基准对拍（无插件，GPU 0,1 / 18003）
```bash
bash tools/start_baseline.sh
python tools/eval_longctx_server.py --base-url http://localhost:18003 --model /home/u/downloads/models/Qwen3.6-35B-A3B --data data/longctx_multi_turn_65536.jsonl --out results/eval_64k_baseline.json
python tools/bench_concurrency.py --base-url http://localhost:18003 --model /home/u/downloads/models/Qwen3.6-35B-A3B --data data/longctx_multi_turn_65536.jsonl --concurrency 8 --out results/bench_c8_baseline.json
pkill -f "vllm serve" ; sleep 3
```

### 8.6 判废标准（任一命中即停下回报，不硬撑）
- 单请求 64k fact_acc 显著低于 0.73；
- allocator 仍随 seq_len 线性涨显存（spec 没被采纳）；
- TPOT 比 full-KV 慢 5 倍以上。

### 8.7 故障速查
- 启动失败：`grep -n -A30 "Traceback" <log>`，根因在 "Engine core initialization failed" 包装错误之上。
- 启动报 `Free memory ... less than desired`：孤儿 worker 占卡，`nvidia-smi` 找自己 uid 的进程杀掉（各 sweep 脚本的 cleanup() 已内置此逻辑）。
- 评测全 ANCHOR FAIL：服务没起或端口错，先 `curl localhost:<port>/health`。
- 日志看不到 `[v8_plugin]` 运行时标记：worker stdout 块缓冲，代码已全 flush=True；看不到不等于没跑到。

（历史诊断与各版本根因见代码注释和 §7 坑表，本手册不重复。）
