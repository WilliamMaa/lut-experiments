"""
多层联合微调：让多个层的 down_proj 权重同时适应 LUT 的存在。

用法:
    cd LLM_LUT/v4
    python finetune_multi_layer.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layers "19,20,21,22,23" --groups_per_layer 8 \
        --checkpoint_root ../v3/outputs \
        --epochs 3 --lr 1e-5 \
        --output_dir results/finetune_all_layers_half

或显式指定每层 group 数:
    python finetune_multi_layer.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --configs "19:8,20:8,21:8,22:8,23:8" \
        --checkpoint_root ../v3/outputs \
        --epochs 3 --lr 1e-5 \
        --output_dir results/finetune_all_layers_half
"""

import os
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"

import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Insert v3 and v0 into path so we can reuse their modules without copying.
V3_DIR = os.path.join(os.path.dirname(__file__), "..", "v3")
V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)
sys.path.insert(0, V3_DIR)

from data import prepare_data, load_jsonl, TextDataset
from metrics import compute_model_metrics, compute_baseline_probs
from finetune_with_lut import TrainableV3PartialEngine, load_model_and_data, collect_baseline_logits


def parse_layer_configs(arg_str: str) -> List[Tuple[int, int]]:
    """Parse '19:8,20:8,21:8' -> [(19,8), (20,8), (21,8)]."""
    configs = []
    for part in arg_str.split(","):
        layer_str, count_str = part.strip().split(":")
        configs.append((int(layer_str.strip()), int(count_str.strip())))
    return configs


def load_groups_for_layer(checkpoint_root: str, layer_id: int, group_count: int) -> List[Tuple[int, str]]:
    """Return list of (group_id, checkpoint_path) for a layer config."""
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    prefix = f"replacement_l{layer_id}g"
    suffix = ".pt"
    pattern = os.path.join(ckpt_dir, f"{prefix}*{suffix}")
    paths = sorted(glob.glob(pattern))
    groups = []
    for p in paths:
        name = os.path.basename(p)
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        gid = int(name[len(prefix):-len(suffix)])
        groups.append((gid, p))
    return groups


def build_engine_for_layer(model, layer_id: int, group_count: int, checkpoint_root: str,
                           lut_dtype: str = "fp32") -> TrainableV3PartialEngine:
    """Build a TrainableV3PartialEngine for one layer from v3 checkpoints."""
    groups = load_groups_for_layer(checkpoint_root, layer_id, group_count)
    if not groups:
        raise ValueError(f"No checkpoints found for L{layer_id} G{group_count} in {checkpoint_root}")
    if len(groups) != group_count:
        print(f"[WARN] L{layer_id} G{group_count}: expected {group_count} checkpoints, found {len(groups)}")

    engine = TrainableV3PartialEngine(model, layer_id, group_size=64, num_bins=64)
    for gid, path in groups:
        ckpt = torch.load(path, map_location="cpu")
        table = ckpt["table"]
        # Training requires a floating-point table. If a quantized table is
        # provided, dequantize it on the fly. The quantized engine is used for
        # eval-only paths.
        if not table.dtype.is_floating_point:
            scale = ckpt.get("scale")
            if scale is None:
                raise ValueError(f"Quantized checkpoint missing 'scale': {ckpt_path}")
            zero_point = ckpt.get("zero_point", 0.0)
            quantization = ckpt.get("quantization", "int8")
            if quantization == "symmetric_int8":
                table = table.float() * scale
            else:
                table = (table.float() - zero_point) * scale
        engine.add_group(
            group_id=gid,
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            table=table,
        )
    return engine


def compute_mac_reduction(configs: List[Tuple[int, int]], hidden_size: int,
                          intermediate_size: int, num_layers: int) -> float:
    """Full-model major-linear MAC reduction ratio."""
    per_layer_total = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    full_model_total = num_layers * per_layer_total
    eliminated = sum(count * 64 * intermediate_size for _, count in configs)
    return eliminated / full_model_total


def compute_lut_storage(configs: List[Tuple[int, int]], checkpoint_root: str,
                        lut_dtype: str = "fp32") -> int:
    """Total LUT table bytes for a multi-layer config."""
    bytes_per_el = {"fp32": 4, "fp16": 2, "int8": 1}.get(lut_dtype, 4)
    total = 0
    for layer_id, group_count in configs:
        groups = load_groups_for_layer(checkpoint_root, layer_id, group_count)
        for _, path in groups:
            ckpt = torch.load(path, map_location="cpu")
            table = ckpt["table"]
            total += table.numel() * bytes_per_el
    return total


def format_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def finetune_multi_layer(model, calib_loader, eval_loader, engines: List[TrainableV3PartialEngine],
                         epochs: int, lr: float, output_dir: str, baseline_eval_probs,
                         configs: List[Tuple[int, int]]) -> Tuple[Dict, List[Dict]]:
    """Joint fine-tune multiple layers' down_proj weights with LUT installed."""
    device = model.device

    # Collect all target down_proj modules and make their weights trainable.
    down_projs = []
    for engine in engines:
        layer = model.model.layers[engine.layer_id]
        down_projs.append(layer.mlp.down_proj)

    for p in model.parameters():
        p.requires_grad_(False)
    for dp in down_projs:
        dp.weight.requires_grad_(True)

    trainable_params = [dp.weight for dp in down_projs]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0, eps=1e-8)

    # Pre-compute baseline logits (original model, no LUT).
    print("\n[Pre-compute] Collecting baseline logits on calibration set...")
    baseline_logits = collect_baseline_logits(model, calib_loader)

    # Cast trainable weights to fp32 for stable optimization.
    original_dtypes = [dp.weight.dtype for dp in down_projs]
    for dp in down_projs:
        dp.weight.data = dp.weight.data.float()

    # Install all engines.
    for engine in engines:
        engine.install()

    # Pre-train eval.
    print("\n[Pre-train eval] Baseline evaluation (LUT model, before fine-tuning)...")
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            baseline_metrics = compute_model_metrics(
                model, eval_loader, reference_probs_list=baseline_eval_probs
            )
    print(f"  Before: KL={baseline_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    results = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] Fine-tuning {len(engines)} layers...")
        model.train()
        for dp in down_projs:
            dp.train()

        total_loss = 0.0
        num_batches = 0

        for bi, batch in enumerate(tqdm(calib_loader, desc=f"Train epoch {epoch}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            if not torch.isfinite(logits).all():
                print(f"[WARN] Batch {bi}: non-finite logits")
                optimizer.zero_grad(set_to_none=True)
                continue

            target_logits = baseline_logits[bi].to(device)
            pred = logits[:, :-1, :].reshape(-1, logits.size(-1)).float()
            target = target_logits[:, :-1, :].reshape(-1, target_logits.size(-1)).to(device=device, dtype=torch.float32)

            log_probs = F.log_softmax(pred, dim=-1)
            target_log_probs = F.log_softmax(target, dim=-1)
            target_probs = target_log_probs.exp()

            loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
            if not torch.isfinite(loss):
                print(f"[WARN] Batch {bi}: non-finite loss")
                continue

            loss.backward()

            grads_finite = True
            for dp in down_projs:
                grad = dp.weight.grad
                if grad is None or not torch.isfinite(grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                print(f"[WARN] Batch {bi}: non-finite gradient, skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            for dp in down_projs:
                if not torch.isfinite(dp.weight).all():
                    raise RuntimeError(f"down_proj.weight became NaN/Inf after batch {bi}")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg KL loss: {avg_loss:.6f}")

        # Eval.
        print(f"  Evaluating...")
        model.eval()
        for dp in down_projs:
            dp.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)

        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        # Save per-layer weights.
        epoch_paths = {}
        for dp, (layer_id, _), orig_dtype in zip(down_projs, configs, original_dtypes):
            ckpt_name = f"l{layer_id}_epoch{epoch}_down_proj.pt"
            ckpt_path = os.path.join(output_dir, ckpt_name)
            torch.save(dp.weight.data.to(orig_dtype).cpu(), ckpt_path)
            epoch_paths[layer_id] = ckpt_path

        results.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "checkpoints": epoch_paths,
        })

    for engine in engines:
        engine.uninstall()
    return baseline_metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", type=str, default=None,
                       help="Comma-separated layer:count pairs, e.g. '19:8,20:8,21:8'")
    group.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer IDs; use with --groups_per_layer")

    parser.add_argument("--groups_per_layer", type=int, default=8,
                        help="Used with --layers to assign the same group count to every layer")
    parser.add_argument("--checkpoint_root", default="../v3/outputs",
                        help="Root directory containing checkpoints/l{layer}/g{count}")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="results/finetune_all_layers_half")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0)")
    parser.add_argument("--lut_dtype", default="fp32", choices=["fp32", "fp16", "int8"],
                        help="Dtype of LUT checkpoints on disk. Training itself still uses float.")
    args = parser.parse_args()

    if args.configs is not None:
        configs = parse_layer_configs(args.configs)
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]
        configs = [(lid, args.groups_per_layer) for lid in layers]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Multi-Layer Fine-Tune with LUT")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Layer configs: {configs}")
    print(f"Epochs={args.epochs}, LR={args.lr}, LUT dtype={args.lut_dtype}")
    print("=" * 70)

    # Load model and data.
    print("\n[1/4] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size,
        device_str=args.device,
    )
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers

    # Build engines.
    print("\n[2/4] Building V3PartialEngines...")
    engines = []
    for layer_id, group_count in configs:
        engine = build_engine_for_layer(model, layer_id, group_count, args.checkpoint_root, lut_dtype=args.lut_dtype)
        engines.append(engine)

    # Pre-compute baseline eval probabilities (original model, no LUT).
    print("\n[3/4] Collecting baseline eval probabilities (original model, no LUT)...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)

    # Fine-tune.
    print("\n[4/4] Fine-tuning...")
    baseline_metrics, results = finetune_multi_layer(
        model, calib_loader, eval_loader, engines,
        args.epochs, args.lr, args.output_dir,
        baseline_eval_probs=baseline_eval_probs,
        configs=configs,
    )

    mac_reduction = compute_mac_reduction(configs, hidden_size, intermediate_size, num_layers)
    lut_storage = compute_lut_storage(configs, args.checkpoint_root, lut_dtype=args.lut_dtype)

    summary = {
        "model": args.model,
        "configs": configs,
        "epochs": args.epochs,
        "lr": args.lr,
        "lut_dtype": args.lut_dtype,
        "mac_reduction_ratio": mac_reduction,
        "lut_storage_bytes": lut_storage,
        "lut_storage_human": format_bytes(lut_storage),
        "before": {
            "kl": baseline_metrics.get("avg_kl", 0.0),
            "ppl": baseline_metrics["ppl"],
            "acc": baseline_metrics["next_token_acc"],
        },
        "after": results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("MULTI-LAYER FINE-TUNE COMPLETE")
    print("=" * 70)
    print(f"Results saved to {args.output_dir}")
    print(f"  l*_epoch*_down_proj.pt  — fine-tuned down_proj weights")
    print(f"  summary.json            — before/after metrics")


if __name__ == "__main__":
    main()
