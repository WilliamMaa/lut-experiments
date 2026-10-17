"""
v9 逐层推进（layer-wise propagation）公用工具。

复用 v6 代码的方式：直接把 v6 的 scripts/utils、scripts/training、archive
加入 sys.path，保持 pickle 类定义路径与 v6 checkpoint 一致（不拷贝 v6 代码）。

核心原理：校准文本固定、batch=1、无 padding 时，对完整序列依次手动执行
decoder layers [a, b) 与一次完整 forward 完全等价——每个 decoder layer 在
位置 t 的输出只依赖 <= t 的 hidden states，causal attention 自洽，无需 KV cache。
因此"逐层推进"就是把"从 tokens 重跑前缀"换成"在边界 hidden state 上跑一段 block"，
第 N 层输入仍然经过真实 LUT（on-policy），但前缀只算一次。
"""

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "v6"))
V6_ROOT = _V6_ROOT
V6_UTILS = os.path.join(_V6_ROOT, "scripts", "utils")
V6_TRAINING = os.path.join(_V6_ROOT, "scripts", "training")
V6_EVAL = os.path.join(_V6_ROOT, "scripts", "evaluation")
V6_ARCHIVE = os.path.join(_V6_ROOT, "archive")

for _p in (V6_UTILS, V6_TRAINING, V6_EVAL, V6_ARCHIVE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def shared_expert_path(layer_idx: int) -> str:
    return f"model.model.layers[{layer_idx}].mlp.shared_expert"


def parse_layers(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def load_prompt_texts(prompt_file: str, max_prompts: Optional[int] = None) -> List[str]:
    """支持 .jsonl（prompt/text 字段）或 .txt（每行一条）。"""
    texts: List[str] = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        if prompt_file.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("prompt") or obj.get("text")
                if t:
                    texts.append(t)
        else:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    if max_prompts is not None:
        texts = texts[:max_prompts]
    if not texts:
        raise ValueError(f"No prompts loaded from {prompt_file}")
    return texts


def load_model(model_path: str, device_map: Optional[str], torch_dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_map == "auto":
        raise ValueError("device_map='auto' is forbidden (project red line). Use e.g. balanced_low_0.")

    dtype = getattr(torch, torch_dtype)
    print(f"[lw] Loading model: {model_path} (dtype={torch_dtype}, device_map={device_map})")
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True, device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        )
        model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


class BoundaryCapture:
    """
    捕获指定层的两类边界张量（CPU fp32）：
      - residual_in: decoder layer block 的输入 hidden states（forward_pre_hook）
      - ffn_in:      shared_expert 的输入（forward_pre_hook）

    用法：cap = BoundaryCapture(model, layers); cap.reset(); model(...); cap.get(idx, kind)
    """

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = list(layer_indices)
        self._handles = []
        self._bufs: Dict[int, Dict[str, torch.Tensor]] = {}
        for idx in self.layer_indices:
            block = model.model.layers[idx]
            self._handles.append(
                block.register_forward_pre_hook(self._make_hook(idx, "residual_in"))
            )
            self._handles.append(
                block.mlp.shared_expert.register_forward_pre_hook(
                    self._make_hook(idx, "ffn_in")
                )
            )

    def _make_hook(self, idx: int, kind: str):
        def hook(module, inputs):
            x = inputs[0]
            if isinstance(x, tuple):
                x = x[0]
            if x is not None:
                self._bufs.setdefault(idx, {})[kind] = x.detach().float().cpu()
        return hook

    def reset(self):
        self._bufs = {}

    def get(self, idx: int, kind: str) -> torch.Tensor:
        try:
            return self._bufs[idx][kind]
        except KeyError:
            raise RuntimeError(
                f"Boundary capture missing layer {idx} '{kind}'. "
                f"Captured: {list(self._bufs.get(idx, {}))}"
            )

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _block_kwargs(block, hidden: torch.Tensor, position_ids: torch.Tensor,
                  position_embeddings):
    """按 block forward 签名构造 kwargs（兼容不同 transformers 版本）。"""
    params = inspect.signature(block.forward).parameters
    kwargs = {}
    if "attention_mask" in params:
        kwargs["attention_mask"] = None  # batch=1 无 padding -> 纯 causal
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if "use_cache" in params:
        kwargs["use_cache"] = False
    if "cache_position" in params:
        kwargs["cache_position"] = torch.arange(
            hidden.shape[1], device=hidden.device
        )
    if "position_embeddings" in params:
        kwargs["position_embeddings"] = position_embeddings
    return kwargs


def run_decoder_slice(model, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """
    在完整序列 hidden states 上依次执行 decoder layers [start, end)。

    hidden: [1, S, H]（bf16/fp16，CPU 或 GPU）。返回 end-1 层输出的 residual 流。
    与完整 forward 精确等价：每个 decoder layer 在位置 t 的输出只依赖 <= t 的
    hidden states，causal attention 在完整序列上自洽，无需 KV cache。
    调用方如需捕获边界，用 BoundaryCapture（hook 对任何调用路径生效）。
    """
    layers = model.model.layers
    n_layers = len(layers)
    if not (0 <= start < end <= n_layers):
        raise ValueError(f"invalid slice [{start}, {end}) for {n_layers} layers")

    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(layers[start].self_attn, "rotary_emb", None)

    h = hidden.to(dtype=next(layers[start].parameters()).dtype)
    for i in range(start, end):
        block = layers[i]
        dev = next(block.parameters()).device
        h = h.to(dev)
        position_ids = torch.arange(h.shape[1], device=dev).unsqueeze(0)
        pos_emb = rotary(h, position_ids) if rotary is not None else None
        kwargs = _block_kwargs(block, h, position_ids, pos_emb)
        out = block(h, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
    return h


def save_shard(path: str, tensor: torch.Tensor):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def shard_paths(shard_dir: str) -> List[str]:
    return sorted(
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.endswith(".pt")
    )
