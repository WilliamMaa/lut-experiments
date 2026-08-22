#!/usr/bin/env python3
"""
diagnose_leaf_coverage.py

对 v3 shared-coarse + residual checkpoint 做 leaf coverage 诊断：
  1. 从 calibration dataset 统计每棵 address tree 的 leaf 访问频率
  2. 对单条样本（.pt 或 prompt）计算访问的 leaf
  3. 对比样本访问的 leaf 是否在 calibration 中出现过、出现过多少次
  4. 输出 LUT / teacher 预测和 cosine

用法：
  # 从 .pt 文件加载单条样本
  python -u diagnose_leaf_coverage.py \
    --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
    --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
    --sample_input_path /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input/xxx.pt \
    --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
    --teacher_module_path "shared_expert" \
    --device cuda:0

  # 从 prompt 通过模型 forward 捕获输入
  python -u diagnose_leaf_coverage.py \
    --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
    --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --prompt "请你简述明朝灭亡的原因" \
    --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
    --teacher_module_path "shared_expert" \
    --hook_path "model.model.layers[39].mlp.shared_expert" \
    --device cuda:0

python -u diagnose_leaf_coverage.py \
  --checkpoint_dir ./outputs_ffn_lut_layer39_full_moe_v3_shared/checkpoints \
  --dataset_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2/input \
  --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
  --prompt "请你简述明朝灭亡的原因" \
  --hook_path "model.model.layers[39].mlp.shared_expert" \
  --teacher_weight_path /root/data1/rce/OLMo-core/tmp/qwen_35b_last_moe.pt \
  --teacher_module_path "shared_expert" \
  --n_calib_files 5 \
  --device cuda:0 \
  > diagnose_leaf_coverage.log 2>&1 &
"""

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lut_ffn_output_v3_shared_coarse as v3


def _inject_v3_classes():
    import __main__ as _main_mod
    for _name in ("AddressGreedyTree", "_TreeNode", "LUTGroup", "QwenMoEExpert"):
        _cls = getattr(v3, _name, None)
        if _cls is not None and not hasattr(_main_mod, _name):
            setattr(_main_mod, _name, _cls)


def load_v3_base(ckpt_dir, hidden_size, group_size, device):
    _inject_v3_classes()
    ckpt_dir = Path(ckpt_dir)
    coarse_ckpt = torch.load(ckpt_dir / "shared_coarse.pt", map_location="cpu", weights_only=False)
    coarse_address = coarse_ckpt["address"]
    coarse_lut = v3.LUTGroup(
        num_tables=coarse_address.num_tables,
        num_entries=coarse_address.num_entries,
        output_dim=hidden_size,
        init_table=coarse_ckpt["table"],
        device=device,
    )
    residual_addresses = {}
    residual_luts = {}
    max_group = hidden_size // group_size
    for gid in range(max_group):
        residual_path = ckpt_dir / f"residual_g{gid}.pt"
        if not residual_path.exists():
            continue
        res_ckpt = torch.load(residual_path, map_location="cpu", weights_only=False)
        residual_addresses[gid] = res_ckpt["address"]
        residual_luts[gid] = v3.LUTGroup(
            num_tables=residual_addresses[gid].num_tables,
            num_entries=residual_addresses[gid].num_entries,
            output_dim=group_size,
            init_table=res_ckpt["table"],
            device=device,
        )
    group_ids = sorted(residual_luts.keys())
    return coarse_address, coarse_lut, residual_addresses, residual_luts, group_ids


def load_teacher(pt_path, module_path, device, dtype):
    raw_state = torch.load(pt_path, map_location="cpu", weights_only=False)
    new_state = {}
    for k, v in raw_state.items():
        if k.startswith("expert."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    if "down_proj.weight" not in new_state:
        prefix = module_path.strip().replace("[", ".").replace("]", ".")
        while prefix.endswith("."):
            prefix = prefix[:-1]
        prefix_dot = prefix + "."
        matched = {k: v for k, v in raw_state.items() if k.startswith(prefix_dot)}
        if not matched:
            alt_prefix = prefix[6:] if prefix.startswith("model.") else "model." + prefix
            alt_dot = alt_prefix + "."
            matched = {k: v for k, v in raw_state.items() if k.startswith(alt_dot)}
        if not matched:
            raise KeyError(f"Cannot find module {module_path}")
        new_state = {k[len(prefix_dot):]: v for k, v in matched.items()}
    gate_key = next(k for k in new_state.keys() if "gate_proj" in k and "weight" in k)
    intermediate_size, hidden_size = new_state[gate_key].shape
    expert = v3.QwenMoEExpert(hidden_size, intermediate_size)
    expert.load_state_dict(new_state, strict=False)
    expert.to(device).to(dtype).eval()
    return expert


def collect_leaf_histogram(address, x_samples, max_samples=100000):
    """从 calibration x 统计每个 leaf 的访问次数。"""
    if x_samples.shape[0] > max_samples:
        perm = torch.randperm(x_samples.shape[0])[:max_samples]
        x_samples = x_samples[perm]
    indices = address.compute_indices(x_samples.unsqueeze(0)).view(-1, address.num_tables)
    counts = torch.bincount(indices[:, 0].long(), minlength=address.num_entries).float()
    return counts


def predict_base(coarse_address, coarse_lut, residual_addresses, residual_luts,
                 group_ids, group_size, x, device):
    was_2d = (x.dim() == 2)
    if was_2d:
        x = x.unsqueeze(0)
    x = x.to(device)
    coarse_lut.to(device)
    coarse_indices = coarse_address.compute_indices(x).view(-1, coarse_address.num_tables)
    coarse_full = coarse_lut(coarse_indices)
    pred_y = torch.zeros_like(coarse_full)
    for gid in group_ids:
        g_start = gid * group_size
        g_end = g_start + group_size
        residual_luts[gid].to(device)
        residual_indices = residual_addresses[gid].compute_indices(x).view(
            -1, residual_addresses[gid].num_tables
        )
        residual_group = residual_luts[gid](residual_indices)
        pred_y[:, g_start:g_end] = coarse_full[:, g_start:g_end] + residual_group
    pred_y = pred_y.view(x.shape[0], x.shape[1], -1)
    if was_2d:
        pred_y = pred_y.squeeze(0)
    return pred_y


def diagnose(checkpoint_dir, dataset_dir, sample_x, teacher, device, group_size=64,
             n_calib_files=5, max_calib_samples=100000, n_sample_groups=8):
    _inject_v3_classes()
    hidden_size = sample_x.shape[-1]
    coarse_address, coarse_lut, residual_addresses, residual_luts, group_ids = load_v3_base(
        checkpoint_dir, hidden_size, group_size, device
    )

    # 1. Load calibration samples and build histograms
    print("\n" + "=" * 60)
    print("Building calibration leaf histograms ...")
    input_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))[:n_calib_files]
    calib_x = []
    for f in input_files:
        x = torch.load(f, map_location="cpu", weights_only=False)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        calib_x.append(x)
    calib_x = torch.cat(calib_x, dim=0)
    if calib_x.shape[0] > max_calib_samples:
        perm = torch.randperm(calib_x.shape[0])[:max_calib_samples]
        calib_x = calib_x[perm]
    print(f"  Calibration samples: {tuple(calib_x.shape)}")

    coarse_hist = collect_leaf_histogram(coarse_address, calib_x)
    residual_hists = {}
    for gid in group_ids:
        residual_hists[gid] = collect_leaf_histogram(residual_addresses[gid], calib_x)

    # 2. Compute sample leaf indices and coverage
    if sample_x.dim() == 2:
        sample_x_input = sample_x.unsqueeze(0)
    else:
        sample_x_input = sample_x

    sample_coarse_indices = coarse_address.compute_indices(sample_x_input.to(device)).view(-1, coarse_address.num_tables)[:, 0].cpu()

    print("\n" + "=" * 60)
    print(f"Sample shape: {tuple(sample_x.shape)}, tokens: {sample_x.shape[0]}")
    print(f"Coarse tree: {coarse_address.num_entries} entries, {coarse_address.num_bits} bits")
    print(f"Visited coarse leaves: {sample_coarse_indices.unique().tolist()}")
    print(f"\nCoarse leaf coverage per token:")
    for t in range(sample_coarse_indices.shape[0]):
        leaf = sample_coarse_indices[t].item()
        count = coarse_hist[leaf].item()
        print(f"  token {t}: leaf={leaf}, calib_count={count:.0f}")

    print(f"\nCoarse coverage summary:")
    total_visited = sample_coarse_indices.shape[0]
    unseen = (coarse_hist[sample_coarse_indices] == 0).sum().item()
    print(f"  total tokens: {total_visited}")
    print(f"  unseen leaves: {unseen} ({unseen/total_visited:.1%})")
    print(f"  median calib count of visited leaves: {coarse_hist[sample_coarse_indices].median().item():.0f}")
    print(f"  mean calib count of visited leaves: {coarse_hist[sample_coarse_indices].mean().item():.1f}")

    # 3. Residual coverage for a few groups
    print("\n" + "=" * 60)
    print("Residual leaf coverage (sample groups):")
    sample_groups = group_ids[::max(1, len(group_ids) // n_sample_groups)]
    for gid in sample_groups:
        addr = residual_addresses[gid]
        sample_res_idx = addr.compute_indices(sample_x_input.to(device)).view(-1, addr.num_tables)[:, 0].cpu()
        hist = residual_hists[gid]
        unseen = (hist[sample_res_idx] == 0).sum().item()
        print(f"  Group {gid}: {addr.num_entries} entries, unseen={unseen}/{sample_res_idx.shape[0]} ({unseen/sample_res_idx.shape[0]:.1%}), median_count={hist[sample_res_idx].median().item():.0f}")

    # 4. LUT vs teacher prediction
    print("\n" + "=" * 60)
    print("Computing LUT and teacher predictions ...")
    with torch.no_grad():
        lut_pred = predict_base(
            coarse_address, coarse_lut, residual_addresses, residual_luts,
            group_ids, group_size, sample_x, device
        ).float().cpu()
        teacher_pred = teacher(sample_x.to(device).to(dtype)).float().cpu()

    cos = F.cosine_similarity(lut_pred, teacher_pred, dim=-1)
    nr = torch.norm(lut_pred, dim=-1) / (torch.norm(teacher_pred, dim=-1) + 1e-12)
    print(f"\nPer-token LUT vs teacher:")
    for t in range(cos.shape[0]):
        print(f"  token {t}: cos={cos[t]:.4f}, norm_ratio={nr[t]:.4f}")
    print(f"\nAggregate: cos_mean={cos.mean():.4f}, cos_min={cos.min():.4f}, norm_ratio_mean={nr.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--sample_input_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--hook_path", type=str, default="model.model.layers[39].mlp.shared_expert")
    parser.add_argument("--layer_idx", type=int, default=39)
    parser.add_argument("--teacher_weight_path", required=True)
    parser.add_argument("--teacher_module_path", type=str, default="shared_expert")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--n_calib_files", type=int, default=5)
    parser.add_argument("--max_calib_samples", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", type=str, default="balanced_low_0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.torch_dtype)

    if args.sample_input_path:
        sample_x = torch.load(args.sample_input_path, map_location="cpu", weights_only=False)
        if sample_x.dim() == 1:
            sample_x = sample_x.unsqueeze(0)
        print(f"Loaded sample from {args.sample_input_path}: {tuple(sample_x.shape)}")
    elif args.model_path and args.prompt:
        print(f"Loading model from {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=args.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        hook_mod = eval(args.hook_path, {"model": model})
        captured = {"x": None}

        def capture_hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            captured["x"] = x.detach().cpu().float()
            return out

        inputs = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        handle = hook_mod.register_forward_hook(capture_hook)
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
        handle.remove()

        sample_x = captured["x"]
        if sample_x.dim() > 2:
            sample_x = sample_x.reshape(-1, sample_x.shape[-1])
        print(f"Captured sample from prompt: {tuple(sample_x.shape)}")
    else:
        raise ValueError("Must provide either --sample_input_path or both --model_path and --prompt")

    teacher = load_teacher(args.teacher_weight_path, args.teacher_module_path, device, dtype)

    diagnose(
        args.checkpoint_dir, args.dataset_dir, sample_x, teacher, device,
        group_size=args.group_size,
        n_calib_files=args.n_calib_files,
        max_calib_samples=args.max_calib_samples,
    )


if __name__ == "__main__":
    main()
