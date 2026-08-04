#!/usr/bin/env python3
"""
collect_onpolicy_data.py

方案 1 实现：Prompt 选择 + On-Policy 长序列数据 Pipeline（修订版）

核心原则：
  1. Prompt 只带客观元数据（language / task / format / target_length），不预标困难组。
  2. 短 rollout 结束后用 teacher expert 对 LUT 实际访问到的 x 打标签，得到真实 token-level 误差。
  3. 困难度主要由 teacher-LUT 误差决定，输出异常只是补充。
  4. 先全局 PCA 投影 activation，再分别计算 activation / leaf / behavior / metadata 四类相似度。
  5. 两级筛选：1000×256 → 160×1024 → 64×2048。
  6. 长 rollout 采样按位置 + 困难度 + leaf 新颖性联合加权。
  7. 预留 held-out 长生成验证集，不进入选择和训练。

JSONL 格式示例：
  {"prompt": "分析第一次鸦片战争的原因和影响。", "language": "zh", "task": "analysis", "format": "essay", "target_length": 2048}
  {"prompt": "Explain overfitting and regularization.", "language": "en", "task": "explanation", "format": "markdown", "target_length": 2048}

输出目录结构：
  output_root/
    ├── candidate_features.jsonl          # 所有候选 prompt 的短 rollout 特征
    ├── selected_stage1.json              # 第一轮 160 条
    ├── selected_stage2.json              # 第二轮复筛后的 160 条（如果启用）
    ├── selected_final.json               # 最终 64 条
    ├── held_out_prompts.json             # 不参与选择/训练的验证 prompt
    ├── global_pca.pt                     # 全局 PCA 投影
    ├── short_rollout/                    # 短 rollout 原始记录
    ├── medium_rollout/                   # 中 rollout 原始记录（如果启用）
    ├── long_rollout/                     # 长 rollout + teacher 标注
    └── train_samples.pt                  # 最终合并的训练对 (x, y_teacher)
"""

import os
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v6_replacement_engine import V6ReplacementEngine


# =============================================================================
# 1. Prompt 加载：JSONL with metadata
# =============================================================================

def load_prompts(prompt_file: str, max_prompts: Optional[int] = None) -> List[Dict]:
    """
    Load prompts from JSONL. Each line must contain:
      - prompt: str
      - language: str (e.g. zh, en, ja, mixed)
      - task: str (e.g. analysis, coding, math, dialogue, summary)
      - format: str (e.g. essay, markdown, list, code, multi-turn)
      - target_length: int (desired generation length, e.g. 512/1024/2048)
    """
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    print(f"Loaded {len(prompts)} candidate prompts from {prompt_file}")
    return prompts


def split_held_out(prompts: List[Dict], n_held_out: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split prompts into selection pool and held-out evaluation set."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(prompts))
    rng.shuffle(indices)
    held_indices = indices[:n_held_out].tolist()
    select_indices = indices[n_held_out:].tolist()
    held = [prompts[i] for i in held_indices]
    select = [prompts[i] for i in select_indices]
    return select, held


# =============================================================================
# 2. 可记录的 V6 Engine
# =============================================================================

class RecordableV6Engine(V6ReplacementEngine):
    """
    在 V6ReplacementEngine 基础上增加轨迹记录能力。
    记录每次 hook 调用时的 FFN input x 和 LUT 输出 pred（CPU）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: List[Dict[str, torch.Tensor]] = []
        self._record_enabled = False
        self._batch_size: Optional[int] = None

    def reset_records(self):
        self.records = []

    def set_record(self, enabled: bool):
        self._record_enabled = enabled
        if enabled:
            self.reset_records()

    def set_batch_size(self, batch_size: int):
        """设置当前 rollout 的 batch size，用于在 2D flatten 输入下重建 [B, S, hidden]。"""
        self._batch_size = batch_size

    def _hook(self, module, inputs, output):
        out = super()._hook(module, inputs, output)
        if not self._record_enabled:
            return out

        x = inputs[0] if isinstance(inputs, tuple) else inputs
        pred = out[0] if isinstance(out, tuple) else out

        # MoE expert/shared_expert 经常被输入 2D [N, hidden]。
        # 如果调用方通过 set_batch_size 提供了 B，就可以恢复 [B, S, hidden]，
        # 从而支持 batch generation；否则退化成 [1, N, hidden]。
        if x.dim() == 2:
            N, hidden = x.shape
            if self._batch_size is not None and self._batch_size > 0:
                B = self._batch_size
                if N % B != 0:
                    raise ValueError(
                        f"2D flattened input has N={N} tokens, not divisible by batch_size={B}. "
                        f"Cannot reconstruct [B, S, hidden]."
                    )
                S = N // B
                x_view = x.view(B, S, hidden)
                pred_view = pred.view(B, S, hidden)
            else:
                x_view = x.unsqueeze(0)
                pred_view = pred.unsqueeze(0)
        else:
            x_view = x
            pred_view = pred

        B, S, hidden = x_view.shape
        # 只保留最后一个位置（新生成的 token）
        x_save = x_view[:, -1:, :].detach().cpu()
        pred_save = pred_view[:, -1:, :].detach().cpu()

        self.records.append({
            "ffn_input": x_save,      # [B, 1, hidden]
            "lut_output": pred_save,  # [B, 1, hidden]
            "seq_len": S,
        })
        return out

    def concat_records(self) -> Dict[str, torch.Tensor]:
        if not self.records:
            return {}
        ffn_inputs = torch.cat([r["ffn_input"] for r in self.records], dim=1)
        lut_outputs = torch.cat([r["lut_output"] for r in self.records], dim=1)
        return {"ffn_input": ffn_inputs, "lut_output": lut_outputs}


# =============================================================================
# 3. Teacher Expert 加载
# =============================================================================

def _extract_module_state(state_dict: Dict[str, torch.Tensor], module_path: str) -> Dict[str, torch.Tensor]:
    """
    从完整模型 state_dict 里按模块路径提取子模块权重。
    module_path 支持两种写法：
      - model.model.layers[39].mlp.shared_expert
      - model.model.layers.39.mlp.shared_expert
    返回去掉前缀后的子模块 state_dict。
    """
    # 统一转成点分隔前缀。
    # 例如 model.model.layers[39].mlp.shared_expert -> model.model.layers.39.mlp.shared_expert
    import re
    prefix = module_path.strip()
    prefix = re.sub(r"\[(\d+)\]", r".\1.", prefix)
    prefix = re.sub(r"\.{2,}", ".", prefix)
    while prefix.startswith("."):
        prefix = prefix[1:]
    while prefix.endswith("."):
        prefix = prefix[:-1]
    prefix_dot = prefix + "."

    # 有些 checkpoint 会多包一层 "model." 或 module 前缀，优先精确匹配
    matched = {k: v for k, v in state_dict.items() if k.startswith(prefix_dot)}

    # 如果完全没匹配到，尝试把 prefix 最外层可能多出的 'model.' 去掉或加上
    if not matched:
        alt_prefixes = []
        if prefix.startswith("model."):
            alt_prefixes.append(prefix[6:])
        else:
            alt_prefixes.append("model." + prefix)
        for alt in alt_prefixes:
            alt_dot = alt + "."
            matched = {k: v for k, v in state_dict.items() if k.startswith(alt_dot)}
            if matched:
                prefix_dot = alt_dot
                break

    if not matched:
        raise KeyError(
            f"Could not find any keys matching module_path='{module_path}' "
            f"(tried prefix='{prefix_dot[:-1]}'). "
            f"Available top-level keys (first 20): {list(state_dict.keys())[:20]}"
        )

    return {k[len(prefix_dot):]: v for k, v in matched.items()}


def load_real_teacher(teacher_weight_path: str, device: torch.device, module_path: Optional[str] = None):
    """
    加载单个 expert FFN 作为 teacher。
    如果 teacher_weight_path 是完整模型权重，则通过 module_path 提取对应子模块。
    """
    import torch.nn as nn

    raw_state = torch.load(teacher_weight_path, map_location="cpu", weights_only=False)

    # 先尝试当单 expert state_dict 使用
    new_state = {}
    for k, v in raw_state.items():
        if k.startswith("expert."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    # 如果没有 down_proj.weight，说明不是单 expert；尝试从 module_path 提取
    if "down_proj.weight" not in new_state:
        if module_path is None:
            raise KeyError(
                "'down_proj.weight' not found in checkpoint; this looks like a full model "
                "checkpoint rather than a single-expert state dict. "
                "Please pass --teacher_module_path (e.g. 'model.model.layers[39].mlp.shared_expert')."
            )
        new_state = _extract_module_state(raw_state, module_path)
        if "down_proj.weight" not in new_state:
            raise KeyError(
                f"Extracted module '{module_path}' does not contain 'down_proj.weight'. "
                f"Extracted keys (first 20): {list(new_state.keys())[:20]}"
            )

    # 过滤掉非 FFN 权重（如 shared_expert_gate.weight）
    new_state = {k: v for k, v in new_state.items() if k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}

    # 确定 hidden/intermediate：与 build_lut_ffn_output.py 保持一致，
    # gate_proj.weight shape 为 [intermediate_size, hidden_size]
    gate_key = next(k for k in new_state.keys() if "gate_proj" in k and "weight" in k)
    intermediate_size, hidden_size = new_state[gate_key].shape

    class Expert(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    expert = Expert(hidden_size, intermediate_size)
    expert.load_state_dict(new_state)
    expert.to(device).eval()
    return expert, hidden_size, intermediate_size


# =============================================================================
# 3b. Teacher label 生成器：支持从加载的 expert 或原始 hook 模块生成
# =============================================================================

class TeacherLabeler:
    """
    生成 teacher 标签的封装。

    mode="loaded_expert": 用 load_real_teacher 加载的独立 expert 前向。
    mode="original_module": 临时关掉 LUT hook，用模型原始的 hook_mod（如完整 mlp）前向。
                            用于 LUT 目标是替换完整模块（如整个 MoE block）的场景。
    """

    def __init__(
        self,
        mode: str,
        device: torch.device,
        teacher: Optional[torch.nn.Module] = None,
        hook_mod: Optional[torch.nn.Module] = None,
        engine: Optional[RecordableV6Engine] = None,
    ):
        assert mode in ("loaded_expert", "original_module")
        self.mode = mode
        self.device = device
        self.teacher = teacher
        self.hook_mod = hook_mod
        self.engine = engine

        if mode == "loaded_expert" and teacher is None:
            raise ValueError("mode='loaded_expert' requires teacher")
        if mode == "original_module" and hook_mod is None:
            raise ValueError("mode='original_module' requires hook_mod")

    def _module_device_dtype(self):
        param = next(self.hook_mod.parameters())
        return param.device, param.dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, hidden] or [B, S, hidden]；返回 float32 CPU tensor。"""
        if self.mode == "loaded_expert":
            teacher_dtype = next(self.teacher.parameters()).dtype
            x = x.to(self.device).to(teacher_dtype)
            with torch.no_grad():
                y = self.teacher(x).float().cpu()
            return y

        # original_module: 临时卸载 LUT hook，跑原始模块
        module_device, module_dtype = self._module_device_dtype()
        x = x.to(module_device).to(module_dtype)
        if self.engine is not None and self.engine._hook_handle is not None:
            self.engine.uninstall()
        try:
            with torch.no_grad():
                y = self.hook_mod(x)
            # 有些模块返回 tuple
            if isinstance(y, tuple):
                y = y[0]
            return y.float().cpu()
        finally:
            if self.engine is not None:
                self.engine.install()


# =============================================================================
# 4. Rollout：短 / 中 / 长
# =============================================================================

def rollout_prompts(
    model,
    tokenizer,
    engine: RecordableV6Engine,
    prompts: List[Dict],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    desc: str = "rollout",
) -> List[Dict]:
    """
    对一批 prompt 做 autoregressive rollout，记录每个生成 token 的 FFN input 和 LUT 输出。
    返回每个 prompt 的 dict，包含 prompt 元数据、ffn_input [T, hidden]、lut_output [T, hidden]。

    支持 batch generation：调用方需通过 engine.set_batch_size(len(batch)) 提供 batch size，
    当 hook 模块接收 2D flatten 输入（如 MoE shared_expert）时，engine 可据此重建 [B, S, hidden]。
    取最后 T 条记录（跳过 prefill 阶段）对齐到实际生成的 T 个 token。
    """
    results = []

    for start_idx in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[start_idx:start_idx + batch_size]
        batch_texts = [p["prompt"] for p in batch]
        engine.reset_records()
        engine.set_record(True)
        engine.set_batch_size(len(batch))

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_lens = attention_mask.sum(dim=1).cpu()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        records = engine.concat_records()
        engine.set_record(False)
        engine.set_batch_size(None)

        if records is None or "ffn_input" not in records:
            warnings.warn(f"Batch {start_idx}: no records captured")
            continue

        ffn_input = records["ffn_input"]    # [B, num_records, hidden]
        lut_output = records["lut_output"]  # [B, num_records, hidden]
        B = len(batch)

        for b in range(B):
            # 所有 batch 序列在 output_ids 中总长度相同：input_len + max_new_tokens
            gen_start = input_ids.shape[1]
            gen_ids = output_ids[b, gen_start:].cpu()
            T = gen_ids.shape[0]
            if T > max_new_tokens:
                raise RuntimeError(
                    f"Sequence {b}: generated {T} tokens but max_new_tokens={max_new_tokens}."
                )
            # 严格对齐：records 最后 T 条必须对应生成的 T 个 token
            if ffn_input.shape[1] < T:
                raise RuntimeError(
                    f"Sequence {b}: need {T} records but only have {ffn_input.shape[1]}. "
                    f"This usually means the hook did not fire for every generation step. "
                    f"Try reducing rollout batch_size to 1."
                )
            item = dict(batch[b])
            item["generated_token_ids"] = gen_ids
            item["generated_text"] = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # 最后 T 条记录对应实际生成的 T 个 token（前面可能有一条 prefill 记录）
            item["ffn_input"] = ffn_input[b, -T:].cpu().float()
            item["lut_output"] = lut_output[b, -T:].cpu().float()
            item["T"] = T
            results.append(item)

    return results


def add_teacher_labels(items: List[Dict], labeler: TeacherLabeler, batch_size: int = 256):
    """
    对 rollout 结果批量生成 teacher 标签。
    注意：这里只在 LUT 访问到的状态上打标签，不重新运行完整 LLM。
    """
    # original_module 模式下，批量处理前临时卸载 LUT hook，处理完再装回
    uninstalled = False
    if labeler.mode == "original_module" and labeler.engine is not None:
        labeler.engine.uninstall()
        uninstalled = True

    try:
        for item in tqdm(items, desc="teacher label"):
            x = item["ffn_input"]
            ys = []
            with torch.no_grad():
                for start in range(0, x.shape[0], batch_size):
                    yb = labeler(x[start:start + batch_size])
                    ys.append(yb)
            item["teacher_output"] = torch.cat(ys, dim=0)
    finally:
        if uninstalled:
            labeler.engine.install()


def compute_token_metrics(item: Dict) -> Dict:
    """计算 token-level 和 prompt-level 困难度指标。"""
    x = item["ffn_input"]
    y_teacher = item["teacher_output"]
    y_lut = item["lut_output"]

    cos = F.cosine_similarity(y_teacher, y_lut, dim=-1)           # [T]
    rel_l2 = torch.norm(y_teacher - y_lut, dim=-1) / (torch.norm(y_teacher, dim=-1) + 1e-12)
    residual_norm = torch.norm(y_teacher - y_lut, dim=-1)

    T = cos.shape[0]
    low_cos = (cos < 0.6).float()

    # 连续低 cosine 最长区间
    max_bad_run = 0
    cur = 0
    for c in low_cos:
        if c:
            cur += 1
            max_bad_run = max(max_bad_run, cur)
        else:
            cur = 0

    return {
        "T": T,
        "cos_mean": cos.mean().item(),
        "cos_p10": torch.quantile(cos, 0.10).item(),
        "cos_p5": torch.quantile(cos, 0.05).item(),
        "rel_l2_mean": rel_l2.mean().item(),
        "residual_norm_mean": residual_norm.mean().item(),
        "residual_norm_p90": torch.quantile(residual_norm, 0.90).item(),
        "pr_cos_lt_0.6": low_cos.mean().item(),
        "max_bad_run": max_bad_run,
    }


def add_output_instability(item: Dict) -> Dict:
    """从生成文本侧补充不稳定性指标。"""
    text = item.get("generated_text", "")
    total = max(len(text), 1)
    return {
        "newline_rate": text.count("\n") / total,
        "repetition_rate": _compute_repetition(text),
        "anomaly_rate": _compute_anomaly_rate(text),
    }


def _compute_repetition(text: str, n: int = 10) -> float:
    if len(text) < n:
        return 0.0
    ngrams = set()
    total = 0
    for i in range(len(text) - n + 1):
        ngrams.add(text[i:i + n])
        total += 1
    return 1.0 - len(ngrams) / max(total, 1)


def _compute_anomaly_rate(text: str) -> float:
    if not text:
        return 0.0
    unusual = sum(1 for c in text if ord(c) < 32 and c not in "\n\t")
    return unusual / len(text)


# =============================================================================
# 5. 全局 PCA 与 Leaf 直方图
# =============================================================================

def fit_global_pca(items: List[Dict], n_components: int = 64, max_samples: int = 50000) -> Dict:
    """
    从所有候选短 rollout token 中采样，训练全局 PCA 投影。
    返回：{"mean": [hidden], "V": [hidden, n_components], "singular_values": [n_components]}
    """
    all_x = []
    for item in items:
        all_x.append(item["ffn_input"])
    all_x = torch.cat(all_x, dim=0)  # [N_total, hidden]
    N = all_x.shape[0]
    if N > max_samples:
        perm = torch.randperm(N)[:max_samples]
        all_x = all_x[perm]

    mean = all_x.mean(dim=0, keepdim=True)
    centered = all_x - mean
    # 用 SVD 做 PCA
    u, s, vh = torch.svd(centered.float())
    V = vh[:, :n_components]  # [hidden, n_components]
    return {
        "mean": mean.squeeze(0),
        "V": V,
        "singular_values": s[:n_components],
    }


def compute_leaf_ids_and_histograms(engine: RecordableV6Engine, ffn_input: torch.Tensor):
    """
    计算给定 FFN input 上所有 address 的 per-token leaf ID 和聚合直方图。

    返回：
      leaf_ids: Dict[str, Tensor[T]]，每个 address type 每个 token 访问的 leaf index
      hists:    Dict[str, Tensor[num_bins]]，去重后的聚合访问直方图
                （coarse 所有 group 共享同一 address，只算一次）
    """
    device = next(engine.hook_mod.parameters()).device
    x = ffn_input.unsqueeze(0).to(device)
    leaf_ids: Dict[str, torch.Tensor] = {}
    hists: Dict[str, torch.Tensor] = {}
    seen_coarse_addr = None

    with torch.no_grad():
        for gid, spec in engine.group_specs.items():
            for addr_idx, addr in enumerate(spec["addresses"]):
                indices = addr.compute_indices(x).view(-1, addr.num_tables).cpu()[:, 0]  # [T]
                if addr_idx == 0:
                    key = "coarse"
                    # shared coarse address 只去重统计一次
                    if seen_coarse_addr is not None and id(addr) != seen_coarse_addr:
                        continue
                    seen_coarse_addr = id(addr)
                elif addr_idx == 1:
                    key = f"residual_g{gid}"
                else:
                    key = "hard"
                leaf_ids[key] = indices
                hist = torch.bincount(indices.long(), minlength=addr.num_entries).float()
                hists[key] = hists.get(key, torch.zeros_like(hist)) + hist
    return leaf_ids, hists


# 保留旧名字兼容 build_prompt_features
def compute_leaf_histograms(engine: RecordableV6Engine, ffn_input: torch.Tensor) -> Dict[str, torch.Tensor]:
    """兼容接口：返回去重后的聚合直方图。"""
    _, hists = compute_leaf_ids_and_histograms(engine, ffn_input)
    return hists


# =============================================================================
# 6. Prompt-Level 特征向量
# =============================================================================

def build_prompt_features(
    item: Dict,
    engine: RecordableV6Engine,
    pca_state: Dict,
    global_leaf_hists: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    """
    为单条 prompt 构建四类特征：
      - activation: 基于全局 PCA 投影后的统计
      - leaf: 各 tree 的访问直方图
      - behavior: 文本生成行为 + teacher-LUT 误差
      - metadata: 语言 / 任务 / 格式 one-hot
    """
    x = item["ffn_input"].float()  # [T, hidden]
    metrics = compute_token_metrics(item)
    instability = add_output_instability(item)

    # 1. activation features via global PCA
    pca_mean = pca_state["mean"].to(x.device)
    V = pca_state["V"].to(x.device)
    z = (x - pca_mean) @ V  # [T, n_components]

    activation_feat = np.concatenate([
        z.mean(dim=0).cpu().numpy(),
        z.std(dim=0).cpu().numpy(),
        torch.quantile(z, 0.10, dim=0).cpu().numpy(),
        torch.quantile(z, 0.90, dim=0).cpu().numpy(),
    ]).astype(np.float32)

    # token displacement in PCA space
    if z.shape[0] > 1:
        disp = torch.norm(z[1:] - z[:-1], dim=-1)
        disp_feat = np.array([disp.mean().item(), disp.std().item(), torch.quantile(disp, 0.90).item()], dtype=np.float32)
    else:
        disp_feat = np.zeros(3, dtype=np.float32)

    activation_feat = np.concatenate([activation_feat, disp_feat])

    # 2. leaf histogram features
    leaf_hists = compute_leaf_histograms(engine, x)
    leaf_feat_list = []
    for key in sorted(leaf_hists.keys()):
        h = leaf_hists[key].float()
        # 归一化频率 + coverage + entropy
        p = h / (h.sum() + 1e-12)
        coverage = (h > 0).sum().item()
        entropy = -(p * (p + 1e-12).log()).sum().item()
        leaf_feat_list.append(p.numpy())
        leaf_feat_list.append(np.array([coverage, entropy], dtype=np.float32))
    leaf_feat = np.concatenate(leaf_feat_list) if leaf_feat_list else np.zeros(1, dtype=np.float32)

    # 3. behavior features
    behavior_feat = np.array([
        1.0 - metrics["cos_mean"],         # 越高越困难
        1.0 - metrics["cos_p10"],
        metrics["rel_l2_mean"],
        metrics["residual_norm_mean"],
        metrics["residual_norm_p90"],
        metrics["pr_cos_lt_0.6"],
        metrics["max_bad_run"] / max(metrics["T"], 1),
        instability["repetition_rate"],
        instability["anomaly_rate"],
        instability["newline_rate"],
    ], dtype=np.float32)

    # 4. metadata features
    metadata = {
        "language": item.get("language", "unknown"),
        "task": item.get("task", "unknown"),
        "format": item.get("format", "unknown"),
    }

    # 困难度分数：主要来自 teacher-LUT 误差
    difficulty = (
        0.40 * (1.0 - metrics["cos_mean"]) +
        0.20 * metrics["pr_cos_lt_0.6"] +
        0.20 * (metrics["max_bad_run"] / max(metrics["T"], 1)) +
        0.10 * instability["repetition_rate"] +
        0.10 * instability["anomaly_rate"]
    )

    return {
        **item,
        "activation_feat": activation_feat,
        "leaf_feat": leaf_feat,
        "behavior_feat": behavior_feat,
        "metadata": metadata,
        "metrics": metrics,
        "instability": instability,
        "difficulty": float(difficulty),
        "leaf_hists": leaf_hists,
    }


# =============================================================================
# 7. 多视角相似度 + 选择
# =============================================================================

def normalize_features(feats: np.ndarray) -> np.ndarray:
    """L2 normalize per feature dimension."""
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-12
    return (feats - mean) / std


def jensen_shannon_similarity(hists: np.ndarray) -> np.ndarray:
    """Compute pairwise JS similarity between normalized histograms."""
    N = hists.shape[0]
    sim = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        p = hists[i] + 1e-12
        for j in range(i, N):
            q = hists[j] + 1e-12
            m = 0.5 * (p + q)
            jsd = 0.5 * (_kl(p, m) + _kl(q, m))
            sim[i, j] = sim[j, i] = 1.0 - jsd
    return sim


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return np.sum(p * np.log(p / q))


def compute_combined_similarity(
    activation_feats: np.ndarray,
    leaf_feats: np.ndarray,
    behavior_feats: np.ndarray,
    metadata: List[Dict],
    alpha: float = 0.45,
    beta: float = 0.35,
    gamma: float = 0.15,
    delta: float = 0.05,
) -> np.ndarray:
    """
    组合四类相似度：
      - activation: cosine on normalized features
      - leaf: Jensen-Shannon similarity
      - behavior: RBF / negative L2 after normalization
      - metadata: exact match reward
    """
    N = activation_feats.shape[0]

    # activation cosine
    a_norm = activation_feats / (np.linalg.norm(activation_feats, axis=1, keepdims=True) + 1e-12)
    sim_act = a_norm @ a_norm.T

    # leaf JS similarity
    # leaf_feats 中前一半是概率向量（交替出现），我们只取概率部分
    hist_dim = leaf_feats.shape[1] // 2  # 另一半是 coverage/entropy 统计
    sim_leaf = jensen_shannon_similarity(leaf_feats[:, :hist_dim])

    # behavior RBF
    b_norm = normalize_features(behavior_feats)
    sq_dists = np.sum((b_norm[:, None, :] - b_norm[None, :, :]) ** 2, axis=2)
    sim_beh = np.exp(-sq_dists / 2.0)

    # metadata match reward
    sim_meta = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            score = 0.0
            for key in ["language", "task", "format"]:
                if metadata[i].get(key) == metadata[j].get(key):
                    score += 1.0 / 3.0
            sim_meta[i, j] = sim_meta[j, i] = score

    return alpha * sim_act + beta * sim_leaf + gamma * sim_beh + delta * sim_meta


def select_prompts_constrained(
    items: List[Dict],
    n_select: int,
    min_by_language: Dict[str, int],
    min_by_task: Dict[str, int],
    min_by_format: Dict[str, int],
    difficulty_lambda: float = 0.5,
    seed: int = 42,
) -> List[int]:
    """
    带配额约束的贪心选择。

    策略：
      1. 先选 difficulty 最高且彼此不重复的 k 条（默认 n_select // 4）。
      2. 剩下的位置用 facility location 从候选中补足。
      3. 在贪心过程中检查语言/任务/格式最低配额，必要时优先补配额。
    """
    N = len(items)
    rng = np.random.RandomState(seed)

    activation = np.stack([it["activation_feat"] for it in items])
    leaf = np.stack([it["leaf_feat"] for it in items])
    behavior = np.stack([it["behavior_feat"] for it in items])
    metadata = [it["metadata"] for it in items]
    difficulty = np.array([it["difficulty"] for it in items])

    sim_matrix = compute_combined_similarity(activation, leaf, behavior, metadata)

    # 配额计数器
    selected = []
    counts_lang = defaultdict(int)
    counts_task = defaultdict(int)
    counts_format = defaultdict(int)
    max_sim = np.full(N, -np.inf)

    n_difficult = max(1, n_select // 4)

    def _satisfied() -> bool:
        for k, v in min_by_language.items():
            if counts_lang[k] < v:
                return False
        for k, v in min_by_task.items():
            if counts_task[k] < v:
                return False
        for k, v in min_by_format.items():
            if counts_format[k] < v:
                return False
        return True

    def _priority_mask() -> np.ndarray:
        """返回哪些索引可以帮助满足未完成的配额。"""
        mask = np.zeros(N, dtype=bool)
        need_lang = {k for k, v in min_by_language.items() if counts_lang[k] < v}
        need_task = {k for k, v in min_by_task.items() if counts_task[k] < v}
        need_format = {k for k, v in min_by_format.items() if counts_format[k] < v}
        for i in range(N):
            meta = metadata[i]
            if meta["language"] in need_lang or meta["task"] in need_task or meta["format"] in need_format:
                mask[i] = True
        return mask

    def _pick_idx(candidates: np.ndarray, use_difficulty: bool) -> int:
        if len(candidates) == 0:
            candidates = np.arange(N)
        # coverage gain
        gains = sim_matrix[candidates].max(axis=1) - max_sim[candidates]
        # difficulty bonus
        if use_difficulty:
            score = gains + difficulty_lambda * difficulty[candidates]
        else:
            score = gains
        # 配额未满足时，优先满足配额的候选
        mask = _priority_mask()[candidates]
        if mask.any():
            masked_score = np.where(mask, score, score - 1e6)
        else:
            masked_score = score
        idx_local = int(masked_score.argmax())
        return int(candidates[idx_local])

    available = np.arange(N)

    # 阶段 1：选困难样本
    for _ in range(n_difficult):
        idx = _pick_idx(available, use_difficulty=True)
        selected.append(idx)
        available = available[available != idx]
        max_sim = np.maximum(max_sim, sim_matrix[idx])
        counts_lang[metadata[idx]["language"]] += 1
        counts_task[metadata[idx]["task"]] += 1
        counts_format[metadata[idx]["format"]] += 1

    # 阶段 2：facility location 补足
    while len(selected) < n_select:
        idx = _pick_idx(available, use_difficulty=False)
        selected.append(idx)
        available = available[available != idx]
        max_sim = np.maximum(max_sim, sim_matrix[idx])
        counts_lang[metadata[idx]["language"]] += 1
        counts_task[metadata[idx]["task"]] += 1
        counts_format[metadata[idx]["format"]] += 1

    # 如果配额仍未满足，可以最后强制替换，但这里先打印警告
    if not _satisfied():
        warnings.warn(f"Quota not fully satisfied: lang={dict(counts_lang)}, task={dict(counts_task)}, format={dict(counts_format)}")

    return selected


# =============================================================================
# 8. 长 rollout 采样：位置 + 困难度 + 新颖性
# =============================================================================

def sample_long_rollout_positions(
    T: int,
    cos: torch.Tensor,
    leaf_ids: Dict[str, torch.Tensor],
    global_seen_leaves: Optional[Dict[str, Set[int]]] = None,
    base_rates: Optional[Dict[Tuple[int, int], float]] = None,
    difficulty_boost: float = 0.3,
    novelty_boost: float = 0.2,
) -> torch.Tensor:
    """
    对长 rollout 的每个位置决定是否保留。

    leaf_ids: Dict[str, Tensor[T]]，每个 address type 每个 token 访问的 leaf index。
    base_rates: 默认按位置分桶：
      0–256: 0.1, 256–512: 0.2, 512–1024: 0.4, 1024+: 0.6
    """
    if base_rates is None:
        base_rates = {
            (0, 256): 0.10,
            (256, 512): 0.20,
            (512, 1024): 0.40,
            (1024, float("inf")): 0.60,
        }

    if cos.shape[0] != T:
        raise RuntimeError(
            f"sample_long_rollout_positions: T={T} but cos has {cos.shape[0]} elements"
        )

    # 验证 leaf_ids 长度
    for key, ids in leaf_ids.items():
        if ids.shape[0] != T:
            raise RuntimeError(
                f"leaf_ids['{key}'] length {ids.shape[0]} != T={T}"
            )

    probs = np.zeros(T, dtype=np.float32)
    for i in range(T):
        rate = 0.0
        for (lo, hi), r in base_rates.items():
            if lo <= i < hi:
                rate = r
                break

        # 困难度提升
        if cos[i] < 0.7:
            rate += difficulty_boost * (0.7 - cos[i].item()) / 0.7

        # 新颖性提升：如果访问了之前没见过的 leaf
        if global_seen_leaves is not None:
            novelty = 0.0
            for key, ids in leaf_ids.items():
                leaf_id = ids[i].item()
                if leaf_id not in global_seen_leaves[key]:
                    novelty += 1.0
            # 平均到每个地址类型
            rate += novelty_boost * (novelty / max(len(leaf_ids), 1))

        probs[i] = min(rate, 1.0)

    keep = np.random.rand(T) < probs
    indices = np.where(keep)[0]

    # 强制保留崩溃前最后 128 token 的连续低 cosine 窗口
    low_cos = (cos < 0.6).float().numpy()
    for i in range(T - 1, -1, -1):
        if low_cos[i]:
            start = max(0, i - 127)
            window = np.arange(start, i + 1)
            indices = np.union1d(indices, window)
            break

    # 至少保留最后一个位置
    if len(indices) == 0:
        indices = np.array([T - 1])

    return torch.tensor(indices, dtype=torch.long)


# =============================================================================
# 9. 长 rollout + teacher 标注
# =============================================================================

def collect_long_rollout(
    model,
    tokenizer,
    engine: RecordableV6Engine,
    labeler: TeacherLabeler,
    selected_prompts: List[Dict],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    resume: bool = False,
):
    """
    对最终选中的 prompt 做长 rollout，按位置/困难度/新颖性采样，保存 teacher 标注。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 全局已见 leaf 集合（用于 novelty 计算）
    global_seen = defaultdict(set)

    for start_idx in tqdm(range(0, len(selected_prompts), batch_size), desc="long rollout"):
        batch = selected_prompts[start_idx:start_idx + batch_size]
        batch_texts = [p["prompt"] for p in batch]
        engine.reset_records()
        engine.set_record(True)
        engine.set_batch_size(len(batch))

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_lens = attention_mask.sum(dim=1).cpu()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        records = engine.concat_records()
        engine.set_record(False)
        engine.set_batch_size(None)

        ffn_input = records["ffn_input"]
        lut_output = records["lut_output"]
        B = len(batch)

        for b in range(B):
            prompt_idx = start_idx + b
            subdir = output_dir / f"prompt_{prompt_idx:06d}"

            if resume and subdir.exists() and (subdir / "ffn_input.pt").exists():
                continue

            gen_start = input_ids.shape[1]
            gen_ids = output_ids[b, gen_start:].cpu()
            T = gen_ids.shape[0]
            if T > max_new_tokens:
                raise RuntimeError(
                    f"Prompt {prompt_idx}: generated {T} tokens but max_new_tokens={max_new_tokens}."
                )
            if ffn_input.shape[1] < T:
                raise RuntimeError(
                    f"Prompt {prompt_idx}: need {T} records but only have {ffn_input.shape[1]}. "
                    f"This usually means the hook did not fire for every generation step. "
                    f"Try reducing rollout batch_size to 1."
                )
            # 最后 T 条记录对应实际生成的 T 个 token（跳过 prefill）
            x = ffn_input[b, -T:].cpu().float()
            lut_y = lut_output[b, -T:].cpu().float()

            # teacher forward on LUT-visited states
            teacher_y = labeler(x)

            cos = F.cosine_similarity(teacher_y, lut_y, dim=-1)

            # compute per-token leaf IDs and histograms for novelty
            leaf_ids, leaf_hists = compute_leaf_ids_and_histograms(engine, x.cpu())

            keep_indices = sample_long_rollout_positions(
                T, cos, leaf_ids, global_seen_leaves=global_seen
            )

            # update global seen leaves
            for key, ids in leaf_ids.items():
                global_seen[key].update(ids.tolist())

            subdir.mkdir(parents=True, exist_ok=True)

            torch.save(x[keep_indices].cpu(), subdir / "ffn_input.pt")
            torch.save(lut_y[keep_indices].cpu(), subdir / "lut_ffn_out.pt")
            torch.save(teacher_y[keep_indices], subdir / "teacher_ffn_out.pt")
            torch.save(gen_ids[keep_indices], subdir / "tokens.pt")
            torch.save({k: v for k, v in leaf_ids.items()}, subdir / "leaf_ids.pt")

            with open(subdir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({
                    "prompt": batch[b].get("prompt", ""),
                    "metadata": batch[b].get("metadata", {}),
                    "T": T,
                    "kept": len(keep_indices),
                    "keep_indices": keep_indices.tolist(),
                    "cos_mean": cos.mean().item(),
                    "cos_p10": torch.quantile(cos, 0.10).item(),
                    "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
                }, f, ensure_ascii=False, indent=2)


def merge_to_train_samples(long_rollout_dir: Path, output_path: Path):
    """合并所有长 rollout 采样结果为训练对。"""
    xs, ys = [], []
    for subdir in sorted(long_rollout_dir.glob("prompt_*")):
        if not (subdir / "ffn_input.pt").exists():
            continue
        x = torch.load(subdir / "ffn_input.pt", map_location="cpu")
        y = torch.load(subdir / "teacher_ffn_out.pt", map_location="cpu")
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"No samples found in {long_rollout_dir}")
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    torch.save({"x": xs, "y_teacher": ys}, output_path)
    print(f"Merged {xs.shape[0]} training samples -> {output_path}")


# =============================================================================
# 10. Resume helpers
# =============================================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidate_features(path: Path) -> List[Dict]:
    """从 candidate_features.jsonl 加载所有 feature dict。"""
    features = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            features.append(json.loads(line))
    return features


def save_candidate_features(path: Path, features: List[Dict]):
    """保存 candidate_features.jsonl。"""
    with open(path, "w", encoding="utf-8") as f:
        for it in features:
            f.write(json.dumps({
                "prompt": it["prompt"],
                "metadata": it["metadata"],
                "metrics": it["metrics"],
                "instability": it["instability"],
                "difficulty": it["difficulty"],
            }, ensure_ascii=False) + "\n")


def merge_features(old_features: List[Dict], new_features: List[Dict]) -> List[Dict]:
    """按 prompt 文本去重合并，新特征覆盖旧特征。"""
    seen = {}
    for it in old_features:
        seen[it["prompt"]] = it
    for it in new_features:
        seen[it["prompt"]] = it
    return list(seen.values())


def load_held_out_prompts(output_root: Path, all_prompts: List[Dict], n_held_out: int, seed: int):
    """加载已有 held-out；不存在则重新拆分并保存。"""
    held_path = output_root / "held_out_prompts.json"
    if held_path.exists():
        held = load_json(held_path)
        select = [p for p in all_prompts if p not in held]
        print(f"[Resume] loaded held-out: {len(held)}, selection pool: {len(select)}")
        return select, held
    select, held = split_held_out(all_prompts, n_held_out, seed)
    with open(held_path, "w", encoding="utf-8") as f:
        json.dump(held, f, ensure_ascii=False, indent=2)
    print(f"Held-out: {len(held)}, selection pool: {len(select)}")
    return select, held


def load_selected_basic(path: Path) -> List[Dict]:
    """从 selected_stage*.json 读取 prompt 列表（resume 用，不包含完整特征）。"""
    obj = load_json(path)
    items = []
    for it in obj.get("selected", []):
        items.append({
            "prompt": it["prompt"],
            "metadata": it.get("metadata", {}),
            "difficulty": it.get("difficulty", 0.0),
        })
    return items


def load_selected_full(path: Path) -> List[Dict]:
    """从 *_full.pt 读取完整 feature dict（保留 tensor）。"""
    return torch.load(path, map_location="cpu", weights_only=False)


def save_selected_stage1(output_root: Path, selected: List[Dict]):
    """保存 Stage 1 初选结果：基本版 JSON + 完整版 PT。"""
    with open(output_root / "selected_stage1.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_select": len(selected),
            "selected_indices": list(range(len(selected))),
            "selected": [{"prompt": it["prompt"], "metadata": it["metadata"], "difficulty": it["difficulty"]} for it in selected],
        }, f, ensure_ascii=False, indent=2)
    torch.save(selected, output_root / "selected_stage1_full.pt")


def save_selected_stage2(output_root: Path, selected: List[Dict]):
    """保存 Stage 2 复筛结果：基本版 JSON + 完整版 PT。"""
    with open(output_root / "selected_stage2.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_select": len(selected),
            "selected_indices": list(range(len(selected))),
            "selected": [{"prompt": it["prompt"], "metadata": it["metadata"], "difficulty": it["difficulty"]} for it in selected],
        }, f, ensure_ascii=False, indent=2)
    torch.save(selected, output_root / "selected_stage2_full.pt")


# =============================================================================
# 11. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scheme 1: prompt selection + on-policy data collection")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--teacher_weight_path", type=str, default=None,
                        help="mode='loaded_expert' 时必填；mode='original_module' 时不需要。")
    parser.add_argument("--teacher_module_path", type=str, default=None,
                        help="当 teacher_weight_path 是完整模型权重时，用此路径提取单个 expert。"
                             "例如：model.model.layers[39].mlp.shared_expert")
    parser.add_argument("--teacher_mode", type=str, default="loaded_expert",
                        choices=["loaded_expert", "original_module"],
                        help="teacher 标签来源。"
                             "loaded_expert: 从 teacher_weight_path 加载独立 expert；"
                             "original_module: 临时关掉 LUT hook，用模型原始 hook_mod 输出作 teacher。"
                             "当 LUT 目标是替换完整模块（如整个 MoE block）时用 original_module。")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--layer_idx", type=int, default=39)
    parser.add_argument("--hook_path", type=str, default=None)
    parser.add_argument("--device_map", type=str, default="balanced_low_0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_candidate_prompts", type=int, default=1000)
    parser.add_argument("--short_max_new_tokens", type=int, default=256)
    parser.add_argument("--enable_medium_stage", action="store_true")
    parser.add_argument("--medium_max_new_tokens", type=int, default=1024)
    parser.add_argument("--long_max_new_tokens", type=int, default=2048)
    parser.add_argument("--short_batch_size", type=int, default=1)
    parser.add_argument("--medium_batch_size", type=int, default=1)
    parser.add_argument("--long_batch_size", type=int, default=1)
    parser.add_argument("--n_select_stage1", type=int, default=160)
    parser.add_argument("--n_select_final", type=int, default=64)
    parser.add_argument("--n_held_out", type=int, default=64)
    parser.add_argument("--pca_components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：检测到已完成的阶段输出时跳过对应阶段。")
    parser.add_argument("--resume_stage1_from", type=str, default=None,
                        help="增量扩展 Stage 1：从已有输出目录加载 candidate_features.jsonl 和 global_pca.pt，"
                             "复用老特征的 PCA，只对新 prompt 跑 short rollout，然后合并。")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 加载模型和 tokenizer
    print(f"Loading model from {args.model_path}")
    dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 模型生成时必须左对齐，否则 right-padding 会导致生成错位
    tokenizer.padding_side = "left"

    engine = RecordableV6Engine(
        model=model,
        layer_idx=args.layer_idx,
        checkpoint_dir=args.checkpoint_dir,
        hook_path=args.hook_path,
    )
    engine.install()

    device = torch.device(args.device)

    if args.teacher_mode == "loaded_expert":
        if not args.teacher_weight_path:
            raise ValueError("--teacher_weight_path is required when --teacher_mode=loaded_expert")
        teacher, hidden_size, _ = load_real_teacher(
            args.teacher_weight_path, device, module_path=args.teacher_module_path
        )
        labeler = TeacherLabeler(mode="loaded_expert", device=device, teacher=teacher)
    else:
        # original_module: teacher 是安装 LUT hook 之前的原始 hook_mod
        teacher = None
        hidden_size = None
        labeler = TeacherLabeler(
            mode="original_module",
            device=device,
            hook_mod=engine.hook_mod,
            engine=engine,
        )
        print(f"[TeacherLabeler] Using original module output as teacher: {engine.hook_mod}")

    # 加载并拆分候选 prompt
    all_prompts = load_prompts(args.prompt_file, max_prompts=args.max_candidate_prompts)
    select_prompts_list, held_out = load_held_out_prompts(
        output_root, all_prompts, args.n_held_out, args.seed
    )

    # =====================================================================
    # Stage 1: 短 rollout + teacher 标注 + 初选
    # =====================================================================
    stage1_full_path = output_root / "selected_stage1_full.pt"
    stage1_path = output_root / "selected_stage1.json"
    pca_path = output_root / "global_pca.pt"

    # 增量扩展模式：复用已有 candidate_features.jsonl + global_pca.pt，只处理新 prompt
    if args.resume_stage1_from is not None:
        old_root = Path(args.resume_stage1_from)
        old_features_path = old_root / "candidate_features.jsonl"
        old_pca_path = old_root / "global_pca.pt"
        if not old_features_path.exists():
            raise FileNotFoundError(f"--resume_stage1_from: {old_features_path} not found")
        if not old_pca_path.exists():
            raise FileNotFoundError(f"--resume_stage1_from: {old_pca_path} not found")

        old_features = load_candidate_features(old_features_path)
        pca_state = torch.load(old_pca_path, map_location="cpu", weights_only=False)
        print(f"\n[Stage 1 Incremental] loaded {len(old_features)} old features from {old_root}")

        old_prompts = {it["prompt"] for it in old_features}
        new_prompts = [p for p in select_prompts_list if p["prompt"] not in old_prompts]
        print(f"[Stage 1 Incremental] {len(new_prompts)} new prompts to process")

        if new_prompts:
            print("[Stage 1 Incremental] Short rollout on new prompts ...")
            short_items = rollout_prompts(
                model, tokenizer, engine, new_prompts,
                max_new_tokens=args.short_max_new_tokens,
                batch_size=args.short_batch_size,
                device=device,
                desc="short rollout (incremental)",
            )
            add_teacher_labels(short_items, labeler)

            print("[Stage 1 Incremental] Building features with old global PCA ...")
            new_features = [build_prompt_features(it, engine, pca_state) for it in tqdm(short_items, desc="features")]
            all_features = merge_features(old_features, new_features)
        else:
            all_features = old_features

        save_candidate_features(output_root / "candidate_features.jsonl", all_features)
        torch.save(pca_state, pca_path)
        print(f"[Stage 1 Incremental] saved {len(all_features)} combined features")

        # 默认配额
        languages = set(it["metadata"]["language"] for it in all_features)
        tasks = set(it["metadata"]["task"] for it in all_features)
        formats = set(it["metadata"]["format"] for it in all_features)
        min_by_language = {lang: 2 for lang in languages}
        min_by_task = {task: 3 for task in tasks}
        min_by_format = {fmt: 2 for fmt in formats}

        print("[Stage 1 Incremental] Selecting top prompts ...")
        selected_stage1_idx = select_prompts_constrained(
            all_features,
            n_select=min(args.n_select_stage1, len(all_features)),
            min_by_language=min_by_language,
            min_by_task=min_by_task,
            min_by_format=min_by_format,
            seed=args.seed,
        )
        selected_stage1 = [all_features[i] for i in selected_stage1_idx]
        save_selected_stage1(output_root, selected_stage1)

    elif args.resume and stage1_full_path.exists():
        selected_stage1 = load_selected_full(stage1_full_path)
        print(f"\n[Resume] Stage 1: loaded {len(selected_stage1)} full items from {stage1_full_path.name}")
    elif args.resume and stage1_path.exists():
        selected_stage1 = load_selected_basic(stage1_path)
        print(f"\n[Resume] Stage 1: loaded {len(selected_stage1)} basic items from {stage1_path.name}")
    else:
        print("\n[Stage 1] Short rollout (256 tokens) ...")
        short_items = rollout_prompts(
            model, tokenizer, engine, select_prompts_list,
            max_new_tokens=args.short_max_new_tokens,
            batch_size=args.short_batch_size,
            device=device,
            desc="short rollout",
        )
        add_teacher_labels(short_items, labeler)

        print("[Stage 1] Fitting global PCA ...")
        pca_state = fit_global_pca(short_items, n_components=args.pca_components)
        torch.save(pca_state, pca_path)

        print("[Stage 1] Building prompt features ...")
        short_features = [build_prompt_features(it, engine, pca_state) for it in tqdm(short_items, desc="features")]
        save_candidate_features(output_root / "candidate_features.jsonl", short_features)

        # 默认配额
        languages = set(it["metadata"]["language"] for it in short_features)
        tasks = set(it["metadata"]["task"] for it in short_features)
        formats = set(it["metadata"]["format"] for it in short_features)
        min_by_language = {lang: 2 for lang in languages}
        min_by_task = {task: 3 for task in tasks}
        min_by_format = {fmt: 2 for fmt in formats}

        print("[Stage 1] Selecting top prompts ...")
        selected_stage1_idx = select_prompts_constrained(
            short_features,
            n_select=min(args.n_select_stage1, len(short_features)),
            min_by_language=min_by_language,
            min_by_task=min_by_task,
            min_by_format=min_by_format,
            seed=args.seed,
        )
        selected_stage1 = [short_features[i] for i in selected_stage1_idx]
        save_selected_stage1(output_root, selected_stage1)

    # 加载 PCA（Stage 2 复筛 feature 需要）
    if not pca_path.exists():
        raise FileNotFoundError(
            f"global_pca.pt not found at {pca_path}. "
            "Cannot resume Stage 2/3 without Stage 1 PCA state."
        )
    if args.resume_stage1_from is None:
        pca_state = torch.load(pca_path, map_location="cpu", weights_only=False)

    # 默认配额复用（Stage 2 selection 仍需要）
    languages = set(it["metadata"]["language"] for it in selected_stage1)
    tasks = set(it["metadata"]["task"] for it in selected_stage1)
    formats = set(it["metadata"]["format"] for it in selected_stage1)
    min_by_language = {lang: 2 for lang in languages}
    min_by_task = {task: 3 for task in tasks}
    min_by_format = {fmt: 2 for fmt in formats}

    # =====================================================================
    # Stage 2 (可选): 中等 rollout 复筛
    # =====================================================================
    stage2_full_path = output_root / "selected_stage2_full.pt"
    stage2_path = output_root / "selected_stage2.json"

    if args.enable_medium_stage:
        if args.resume and stage2_full_path.exists():
            selected_final = load_selected_full(stage2_full_path)
            print(f"\n[Resume] Stage 2: loaded {len(selected_final)} full items from {stage2_full_path.name}")
        elif args.resume and stage2_path.exists():
            selected_final = load_selected_basic(stage2_path)
            print(f"\n[Resume] Stage 2: loaded {len(selected_final)} basic items from {stage2_path.name}")
        else:
            print("\n[Stage 2] Medium rollout (1024 tokens) on selected candidates ...")
            medium_items = rollout_prompts(
                model, tokenizer, engine, selected_stage1,
                max_new_tokens=args.medium_max_new_tokens,
                batch_size=args.medium_batch_size,
                device=device,
                desc="medium rollout",
            )
            add_teacher_labels(medium_items, labeler)
            medium_features = [build_prompt_features(it, engine, pca_state) for it in tqdm(medium_items, desc="medium features")]

            selected_stage2_idx = select_prompts_constrained(
                medium_features,
                n_select=min(args.n_select_final * 2, len(medium_features)),
                min_by_language=min_by_language,
                min_by_task=min_by_task,
                min_by_format=min_by_format,
                seed=args.seed,
            )
            selected_final = [medium_features[i] for i in selected_stage2_idx]
            save_selected_stage2(output_root, selected_final)
    else:
        selected_final = selected_stage1[:args.n_select_final]

    # =====================================================================
    # Stage 3: 长 rollout + teacher 标注 + 采样
    # =====================================================================
    print("\n[Stage 3] Long rollout (2048 tokens) and on-policy collection ...")
    collect_long_rollout(
        model, tokenizer, engine, labeler,
        selected_final[:args.n_select_final],
        max_new_tokens=args.long_max_new_tokens,
        batch_size=args.long_batch_size,
        device=device,
        output_dir=output_root / "long_rollout",
        resume=args.resume,
    )

    # =====================================================================
    # Stage 4: 合并训练样本
    # =====================================================================
    print("\n[Stage 4] Merging to train_samples.pt ...")
    merge_to_train_samples(output_root / "long_rollout", output_root / "train_samples.pt")

    engine.uninstall()
    print("\nAll done.")


if __name__ == "__main__":
    main()
