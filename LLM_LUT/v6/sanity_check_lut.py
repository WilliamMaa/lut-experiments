#!/usr/bin/env python3
"""
sanity_check_lut.py

对单个固定输入做 4-way sanity check：
  - x: hook 目标模块（如 shared_expert）的输入
  - y_original: 原始模块输出（无 LUT 替换）
  - y_teacher: 加载的 teacher 输出
  - y_offline_lut: V6 engine 独立 forward（lut_forward）输出
  - y_runtime_lut: 模型带 hook 运行时实际返回的 LUT 输出

用法：
  python -u sanity_check_lut.py \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v4_tail/checkpoints \
    --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
    --teacher_module_path "shared_expert" \
    --hook_path "model.model.layers[39].mlp.shared_expert" \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16 \
    --device cuda:0 \
    --input_text "请你简述明朝灭亡的原因" \
    --max_length 512 \
    --output_json ./sanity_check_report.json \
    > sanity_check.log 2>&1 &
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v6_replacement_engine import V6ReplacementEngine


def load_teacher(teacher_weight_path: str, module_path: str, device: torch.device):
    """从完整 checkpoint 提取指定子模块作为 teacher。"""
    raw_state = torch.load(teacher_weight_path, map_location="cpu", weights_only=False)

    # 先尝试单 expert state dict
    new_state = {}
    for k, v in raw_state.items():
        if k.startswith("expert."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    if "down_proj.weight" not in new_state:
        # 从完整模型提取子模块
        prefix = module_path.strip().replace("[", ".").replace("]", ".")
        while prefix.endswith("."):
            prefix = prefix[:-1]
        prefix_dot = prefix + "."
        matched = {k: v for k, v in raw_state.items() if k.startswith(prefix_dot)}
        if not matched:
            alt_prefix = prefix[6:] if prefix.startswith("model.") else "model." + prefix
            alt_dot = alt_prefix + "."
            matched = {k: v for k, v in raw_state.items() if k.startswith(alt_dot)}
            if matched:
                prefix_dot = alt_dot
        if not matched:
            raise KeyError(f"Cannot find module {module_path} in {teacher_weight_path}")
        new_state = {k[len(prefix_dot):]: v for k, v in matched.items()}

    gate_key = next(k for k in new_state.keys() if "gate_proj" in k and "weight" in k)
    intermediate_size, hidden_size = new_state[gate_key].shape

    class Expert(torch.nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
            self.act = torch.nn.SiLU()
        def forward(self, x):
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    expert = Expert(hidden_size, intermediate_size)
    expert.load_state_dict(new_state)
    expert.to(device).eval()
    return expert, hidden_size


def cosine_and_norm(y_a: torch.Tensor, y_b: torch.Tensor):
    cos = F.cosine_similarity(y_a, y_b, dim=-1)
    norm_ratio = torch.norm(y_b, dim=-1) / (torch.norm(y_a, dim=-1) + 1e-12)
    return cos, norm_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--teacher_weight_path", required=True)
    parser.add_argument("--teacher_module_path", type=str, default=None)
    parser.add_argument("--hook_path", required=True)
    parser.add_argument("--layer_idx", type=int, default=39)
    parser.add_argument("--device_map", type=str, default="balanced_low_0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.torch_dtype)

    print("Loading model and tokenizer ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading teacher from {args.teacher_weight_path} module={args.teacher_module_path}")
    teacher, _ = load_teacher(args.teacher_weight_path, args.teacher_module_path, device)

    print("Loading LUT engine ...")
    engine = V6ReplacementEngine(
        model=model,
        layer_idx=args.layer_idx,
        checkpoint_dir=args.checkpoint_dir,
        hook_path=args.hook_path,
        device=device,
    )

    inputs = tokenizer(
        args.input_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(f"Input shape: {tuple(input_ids.shape)}")

    # -------------------------------------------------------------------------
    # Pass 1: capture original module input/output without LUT
    # -------------------------------------------------------------------------
    original_hook = {"x": None, "y": None}
    hook_module = eval(args.hook_path, {"model": model})

    def capture_hook(module, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        y = out[0] if isinstance(out, tuple) else out
        original_hook["x"] = x.detach().cpu().float()
        original_hook["y"] = y.detach().cpu().float()
        return out

    handle = hook_module.register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
    handle.remove()

    x_orig = original_hook["x"]
    y_orig = original_hook["y"]

    # 统一处理 2D/3D：只保留最后两个维度 [..., hidden]
    if x_orig.dim() > 2:
        x_orig = x_orig.reshape(-1, x_orig.shape[-1])
    if y_orig.dim() > 2:
        y_orig = y_orig.reshape(-1, y_orig.shape[-1])

    print(f"Captured original: x={tuple(x_orig.shape)}, y={tuple(y_orig.shape)}")

    # -------------------------------------------------------------------------
    # Pass 2: capture runtime LUT output with engine installed
    # -------------------------------------------------------------------------
    engine.install()
    runtime_hook = {"x": None, "y": None}

    def capture_runtime_hook(module, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        y = out[0] if isinstance(out, tuple) else out
        runtime_hook["x"] = x.detach().cpu().float()
        runtime_hook["y"] = y.detach().cpu().float()
        return out

    handle2 = hook_module.register_forward_hook(capture_runtime_hook)
    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)
    handle2.remove()
    engine.uninstall()

    x_rt = runtime_hook["x"]
    y_rt = runtime_hook["y"]
    if x_rt.dim() > 2:
        x_rt = x_rt.reshape(-1, x_rt.shape[-1])
    if y_rt.dim() > 2:
        y_rt = y_rt.reshape(-1, y_rt.shape[-1])

    print(f"Captured runtime LUT: x={tuple(x_rt.shape)}, y={tuple(y_rt.shape)}")

    # -------------------------------------------------------------------------
    # Teacher 和 offline LUT 都在 original x 上计算
    # -------------------------------------------------------------------------
    x_eval = x_orig.to(device)
    with torch.no_grad():
        y_teacher = teacher(x_eval).cpu().float()
        y_offline = engine.lut_forward(x_eval).cpu().float()

    # -------------------------------------------------------------------------
    # 统计比较
    # -------------------------------------------------------------------------
    comparisons = {
        "original vs teacher": cosine_and_norm(y_orig, y_teacher),
        "original vs offline_lut": cosine_and_norm(y_orig, y_offline),
        "original vs runtime_lut": cosine_and_norm(y_orig, y_rt),
        "offline_lut vs runtime_lut": cosine_and_norm(y_offline, y_rt),
        "teacher vs offline_lut": cosine_and_norm(y_teacher, y_offline),
    }

    report = {}
    print("\n" + "=" * 60)
    for name, (cos, nr) in comparisons.items():
        report[name] = {
            "cos_mean": float(cos.mean()),
            "cos_min": float(cos.min()),
            "cos_max": float(cos.max()),
            "norm_ratio_mean": float(nr.mean()),
            "norm_ratio_min": float(nr.min()),
            "norm_ratio_max": float(nr.max()),
        }
        print(f"{name}")
        print(f"  cosine: mean={report[name]['cos_mean']:.4f}, "
              f"min={report[name]['cos_min']:.4f}, max={report[name]['cos_max']:.4f}")
        print(f"  norm_ratio: mean={report[name]['norm_ratio_mean']:.4f}, "
              f"min={report[name]['norm_ratio_min']:.4f}, max={report[name]['norm_ratio_max']:.4f}")
    print("=" * 60)

    # x 是否一致
    x_diff = torch.norm(x_orig - x_rt, dim=-1) / (torch.norm(x_orig, dim=-1) + 1e-12)
    print(f"\noriginal x vs runtime x: rel_l2 mean={x_diff.mean():.4f}")
    report["x_runtime_vs_original"] = {
        "rel_l2_mean": float(x_diff.mean()),
        "rel_l2_max": float(x_diff.max()),
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
