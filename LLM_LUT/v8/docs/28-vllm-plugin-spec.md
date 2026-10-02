# docs/28: vLLM 压缩 KV backend — 实施规格（spike③ 开工文档）

> 状态：待开工。本文档自包含：读完即可写代码，不需要再回 vLLM 源码调研。
> 唯一例外是 §7 的 TODO 表 —— 那些点必须在远程环境用一条 grep/一次 smoke run 确认，每条都给了验证方法。
>
> 上游结论见 docs/26（计划）、docs/27（spike 报告 GO）。vLLM 源码参考副本：`v8/spike/vllm-src`（commit 58b32984，只读）。

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
`python -m vllm_plugin.serve serve /home/u/downloads/models/Qwen3.6-35B-A3B --enforce-eager --max-model-len 131072 ...`

## 5. 与 v8 harness 的语义差异（预期内，写报告时要列）

| 点 | v8 harness | vLLM 插件 | 影响 |
|----|-----------|-----------|------|
| attention kernel | torch sdpa | mem-efficient SDPA | ~1e-3 舍入差，不对拍逐字 |
| 并发形态 | 整 batch 同步 step、同长 | continuous batching、乱长短 | 指标不可直接比，只看 per-request 质量 |
| 淘汰时机 | update() 内、chunk 后 | forward 内、chunk 前（不变量 4） | 数学等价（因果性论证见 §2.1-4），数值路径不同 |
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

## 7. 接口确认状态（2026-10-08 更新：代码已按确认接口写完，见 `v8/vllm_plugin/`）

**本地对照 spike/vllm-src（commit 58b32984）已确认并写入代码**：

| 点 | 结论 |
|----|------|
| spec 结构 | 照 `HiSparseHotSpec`：`blocks_per_request` 字段 + `max_memory_usage_bytes = blocks × page_size_bytes` + `max_num_blocks_per_req` 常数 + `prefix_cacheable=False` + `block_table_token_alignment=None`；`page_size_bytes` 直接继承 AttentionSpec（整块页公式） |
| 注册 | `register_kv_cache_spec(CompressedKVSpec)` 命令式调用（spec.py 末尾） |
| customize_spec 调用点 | 中心式：worker `attn_utils.py` 对每层 spec 调 `attn_module.get_attn_backend().customize_spec(spec)`，模型层不用改 |
| backend 注入 | `Attention.__init__(..., attn_backend=None)`（attention.py:289）；monkeypatch 注入类即可。GDN 层不构造 Attention，所以按 arch 门控足够 |
| impl 接口 | `forward(layer, q, k, v, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None)`；q/k/v 是 **3-D** [tokens, heads, D]（不是 4-D）；output 预分配 [T, H, D]，返回 [T, H*D] |
| pool 布局 | kv_cache 每层视图 `[num_blocks, H_kv, block_size, 2*D]`，K=`[..., :D]`，V=`[..., D:]`（flash do_kv_cache_update 同款切法） |
| metadata | `FlashAttentionMetadata`（query_start_loc GPU）；builder.build(common_prefix_len, common_attn_metadata) 吃 CommonAttentionMetadata（有 query_start_loc_cpu，无 req_ids → 用 block id 元组当请求键） |
| merge | `FullAttentionSpec.merge` 会丢子类字段 → CompressedKVSpec 自己 override（校验全等后 deepcopy） |

**仍需远程验证（代码里已带防御/日志，按顺序跑 smoke 即可暴露）**：

| # | 待确认 | 失败时的症状 |
|---|--------|--------------|
| R1 | `register_kv_cache_spec` 签名 | import 即报错，按 registry 文件改一行 |
| R2 | builder `__init__` 签名 / `super().build()` 返回后 setattr 是否被后续流程接受 | 启动期 TypeError |
| R3 | Qwen3.6 config 是否带 MTP/spec-decode 层（第二条 attention 路径） | 若启动挂了 speculative 相关栈，加 `--speculative-config` 禁用 |
| R4 | `api_server.main` 入口名 | serve.py 有 fallback 分支，都缺则按文件改一行 |
| R5 | pool 视图在 spec `has_layer_views` 默认下确实按 [nb, H_kv, bs, 2D] 给到 forward | 第一次 forward 的 gather 形状 assert（日志里打 kv_cache.shape） |
| R6 | 单卡 TP 下 num_kv_heads=2 与 customize_spec 字段透传 | 启动期字段缺失 TypeError |

## 8. 远程 runbook

```bash
# 0) 环境（vLLM 源码目录有自家 AGENTS.md：跑 python 必须用 uv）
cd ~/lut-experiments/LLM_LUT/v8/spike/vllm-src
uv venv --seed && uv pip install -e .            # 或按该目录 AGENTS.md 的指引
# 确认版本支持 Qwen3_5Moe（registry 里有 qwen3_5）
uv run python -c "from vllm.model_executor.models.registry import ModelRegistry; print('Qwen3_5MoeForCausalLM' in ModelRegistry.get_supported_archs())"

# 1) 静态验证（不用 GPU，先过一遍 TODO 表 1/2/3/4/7/8/10）
cd ~/lut-experiments/LLM_LUT/v8
python -m py_compile vllm_plugin/*.py

# 2) smoke：单请求 8k，确认 patch 生效 + spec 生效
V8_COMPRESS_SLOTS=512 python -m vllm_plugin.serve serve \
  /home/u/downloads/models/Qwen3.6-35B-A3B \
  --enforce-eager --max-model-len 16384 --max-num-seqs 4 \
  > logs/vllm_smoke.log 2>&1 &
# 期待日志: customize_spec 打出 CompressedKVSpec；nvidia-smi 显存不随 prompt 长度涨

# 3) 正确性对拍：64k 单请求 vs results/budget_512_32k 的答案集
#    （turn-0 fact 正确性同水平即可，不逐字）

# 4) 并发扫描：N ∈ {1, 8, 16, 32} × slots ∈ {512, 1024}，出 Pareto
#    指标：max concurrent seqs（allocator 不再 OOM 的上限）、TTFT/TPOT、
#    fact acc（复用 tools/ 里的判定脚本口径）
```

**判废标准**（任一命中即回 docs/27 复盘，不硬撑）：单请求 64k 出现语义崩坏（fact acc 显著低于 0.73——
budget_512_32k 实测 0.734）；allocator 层面仍随 seq_len 涨显存（说明 spec 没被采纳）；
`--enforce-eager` 下 TPOT 比 full 慢 5 倍以上（逐请求循环的 Python 开销失控，需先优化形态再继续）。
