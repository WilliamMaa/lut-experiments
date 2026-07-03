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

# Parse --device / --isolate_gpu before importing torch.  By default we do NOT
# manipulate CUDA_VISIBLE_DEVICES; the user passes the exact device (e.g. cuda:1)
# and every function uses that device explicitly.  If --isolate_gpu is set, we
# restore the old behaviour of hiding other GPUs via CUDA_VISIBLE_DEVICES before
# torch is imported.
_earliest_parser = argparse.ArgumentParser(add_help=False)
_earliest_parser.add_argument("--device", default="cuda:0")
_earliest_parser.add_argument("--isolate_gpu", action="store_true")
_earliest_args, _ = _earliest_parser.parse_known_args()

if _earliest_args.isolate_gpu and _earliest_args.device.startswith("cuda:"):
    _gpu_id = _earliest_args.device.split(":", 1)[1]
    # Respect an explicit external CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
    _canonical_device = "cuda:0"
else:
    _canonical_device = _earliest_args.device

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import prepare_data, load_jsonl, TextDataset
from metrics import compute_model_metrics, compute_baseline_probs
from trainable_engine import TrainableV3PartialEngine, load_model_and_data, collect_baseline_logits
from o_proj_engine import TrainableOProjPartialEngine


def parse_layer_configs(arg_str: str) -> List[Tuple[int, int]]:
    """Parse '19:8,20:8,21:8' -> [(19,8), (20,8), (21,8)]."""
    configs = []
    for part in arg_str.split(","):
        layer_str, count_str = part.strip().split(":")
        configs.append((int(layer_str.strip()), int(count_str.strip())))
    return configs


def load_layer_summary(checkpoint_root: str, layer_id: int) -> Dict:
    """Load v3 expand_ratio summary JSON for a layer."""
    path = os.path.join(checkpoint_root, "summaries", f"expand_ratio_l{layer_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary not found: {path}. Run v3/expand_ratio.py first.")
    with open(path, "r") as f:
        return json.load(f)


def get_group_ids_for_count(summary: Dict, count: int) -> List[int]:
    """Return the group_ids used for a specific num_groups in the summary."""
    progressive = summary.get("progressive", [])
    for item in progressive:
        if isinstance(item, dict) and item.get("num_groups") == count and "group_ids" in item:
            return [int(g) for g in item["group_ids"]]
    return []


def load_groups_for_layer(checkpoint_root: str, layer_id: int, group_count: int,
                          group_ids: List[int] = None) -> List[Tuple[int, str]]:
    """Return list of (group_id, checkpoint_path) for a layer config.

    If group_ids is provided, only load those groups.
    """
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
        if group_ids is not None and gid not in group_ids:
            continue
        groups.append((gid, p))
    return groups


def build_engine_for_layer(model, layer_id: int, group_count: int, checkpoint_root: str,
                           lut_dtype: str = "fp32", summary: Dict = None) -> TrainableV3PartialEngine:
    """Build a TrainableV3PartialEngine for one layer from v3 checkpoints."""
    group_ids = get_group_ids_for_count(summary, group_count) if summary else None
    groups = load_groups_for_layer(checkpoint_root, layer_id, group_count, group_ids=group_ids)
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


def build_o_proj_engine_for_layer(model, layer_id: int, group_count: int,
                                  checkpoint_root: str,
                                  lut_dtype: str = "fp32") -> TrainableOProjPartialEngine:
    """Build a TrainableOProjPartialEngine for one layer from v3-style checkpoints."""
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No o_proj checkpoints found for L{layer_id} G{group_count} in {ckpt_dir}")

    engine = TrainableOProjPartialEngine(model, layer_id, group_size=64, num_bins=64)
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        table = ckpt["table"]
        if not table.dtype.is_floating_point:
            scale = ckpt.get("scale")
            if scale is None:
                raise ValueError(f"Quantized o_proj checkpoint missing 'scale': {p}")
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


def compute_mac_reduction(down_configs: List[Tuple[int, int]], hidden_size: int,
                          intermediate_size: int, num_layers: int,
                          o_proj_configs: List[Tuple[int, int]] = None) -> float:
    """Full-model major-linear MAC reduction ratio."""
    o_proj_configs = o_proj_configs or []
    per_layer_total = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    full_model_total = num_layers * per_layer_total
    eliminated = sum(count * 64 * intermediate_size for _, count in down_configs)
    eliminated += sum(count * 64 * hidden_size for _, count in o_proj_configs)
    return eliminated / full_model_total


def compute_lut_storage(down_configs: List[Tuple[int, int]], checkpoint_root: str,
                        lut_dtype: str = "fp32",
                        o_proj_configs: List[Tuple[int, int]] = None,
                        o_proj_checkpoint_root: str = None) -> int:
    """Total LUT table bytes for a multi-layer config."""
    o_proj_configs = o_proj_configs or []
    bytes_per_el = {"fp32": 4, "fp16": 2, "int8": 1}.get(lut_dtype, 4)
    total = 0
    for layer_id, group_count in down_configs:
        groups = load_groups_for_layer(checkpoint_root, layer_id, group_count)
        for _, path in groups:
            ckpt = torch.load(path, map_location="cpu")
            table = ckpt["table"]
            total += table.numel() * bytes_per_el
    for layer_id, group_count in o_proj_configs:
        engine = build_o_proj_engine_for_layer(
            None, layer_id, group_count, o_proj_checkpoint_root, lut_dtype=lut_dtype
        )
        for _, cfg in engine.group_configs.items():
            total += cfg[3].numel() * bytes_per_el
    return total


def format_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def finetune_multi_layer(model, calib_loader, eval_loader,
                         down_engines: List[TrainableV3PartialEngine],
                         o_proj_engines: List[TrainableOProjPartialEngine],
                         epochs: int, lr: float, output_dir: str, baseline_eval_probs,
                         configs: List[Tuple[int, int]],
                         o_proj_configs: List[Tuple[int, int]] = None,
                         freeze_layer_set: set = None,
                         train_o_proj_only: bool = False) -> Tuple[Dict, List[Dict]]:
    if freeze_layer_set is None:
        freeze_layer_set = set()
    if o_proj_configs is None:
        o_proj_configs = []
    """Joint fine-tune down_proj and/or o_proj weights with LUT installed."""
    device = model.device

    # Collect all target modules.
    down_projs = []
    for engine in down_engines:
        layer = model.model.layers[engine.layer_id]
        down_projs.append(layer.mlp.down_proj)

    o_projs = []
    for engine in o_proj_engines:
        layer = model.model.layers[engine.layer_id]
        o_projs.append(layer.self_attn.o_proj)

    for p in model.parameters():
        p.requires_grad_(False)

    for engine in down_engines:
        layer_id = engine.layer_id
        dp = model.model.layers[layer_id].mlp.down_proj
        if train_o_proj_only or layer_id in freeze_layer_set:
            dp.weight.requires_grad_(False)
            print(f"  [Freeze] L{layer_id} down_proj is frozen")
        else:
            dp.weight.requires_grad_(True)

    for engine in o_proj_engines:
        layer_id = engine.layer_id
        op = model.model.layers[layer_id].self_attn.o_proj
        if layer_id in freeze_layer_set:
            op.weight.requires_grad_(False)
            print(f"  [Freeze] L{layer_id} o_proj is frozen")
        else:
            op.weight.requires_grad_(True)

    trainable_down_projs = [model.model.layers[e.layer_id].mlp.down_proj for e in down_engines
                            if e.layer_id not in freeze_layer_set and not train_o_proj_only]
    trainable_o_projs = [model.model.layers[e.layer_id].self_attn.o_proj for e in o_proj_engines
                         if e.layer_id not in freeze_layer_set]
    trainable_params = [p.weight for p in trainable_down_projs + trainable_o_projs]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0, eps=1e-8)

    # Pre-compute baseline logits (original model, no LUT).
    print("\n[Pre-compute] Collecting baseline logits on calibration set...")
    baseline_logits = collect_baseline_logits(model, calib_loader)

    # Cast trainable weights to fp32 for stable optimization.
    original_down_dtypes = [dp.weight.dtype for dp in down_projs]
    for dp in down_projs:
        dp.weight.data = dp.weight.data.float()
    original_o_dtypes = [op.weight.dtype for op in o_projs]
    for op in o_projs:
        op.weight.data = op.weight.data.float()

    # Install all down_proj and o_proj engines.
    for engine in down_engines:
        engine.install()
    for engine in o_proj_engines:
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
        print(f"\n[Epoch {epoch}/{epochs}] Fine-tuning {len(down_engines)} down_proj + "
              f"{len(o_proj_engines)} o_proj layers...")
        model.train()
        for dp in down_projs:
            dp.train()
        for op in o_projs:
            op.train()

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

            # Only check gradients for trainable layers; frozen layers have None grad.
            grads_finite = True
            for mod in trainable_down_projs + trainable_o_projs:
                grad = mod.weight.grad
                if grad is None or not torch.isfinite(grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                print(f"[WARN] Batch {bi}: non-finite gradient, skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            for mod in down_projs + o_projs:
                if not torch.isfinite(mod.weight).all():
                    raise RuntimeError(f"{mod.__class__.__name__}.weight became NaN/Inf after batch {bi}")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg KL loss: {avg_loss:.6f}")

        # Eval.
        print(f"  Evaluating...")
        model.eval()
        for dp in down_projs:
            dp.eval()
        for op in o_projs:
            op.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)

        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        # Save per-layer weights (all layers, including frozen ones).
        epoch_paths = {}
        for engine, (layer_id, _), orig_dtype in zip(down_engines, configs, original_down_dtypes):
            dp = model.model.layers[layer_id].mlp.down_proj
            ckpt_name = f"l{layer_id}_epoch{epoch}_down_proj.pt"
            ckpt_path = os.path.join(output_dir, ckpt_name)
            torch.save(dp.weight.data.to(orig_dtype).cpu(), ckpt_path)
            epoch_paths[layer_id] = ckpt_path
        for engine, (layer_id, _), orig_dtype in zip(o_proj_engines, o_proj_configs, original_o_dtypes):
            op = model.model.layers[layer_id].self_attn.o_proj
            ckpt_name = f"l{layer_id}_epoch{epoch}_o_proj.pt"
            ckpt_path = os.path.join(output_dir, ckpt_name)
            torch.save(op.weight.data.to(orig_dtype).cpu(), ckpt_path)
            epoch_paths[f"o_{layer_id}"] = ckpt_path

        results.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "checkpoints": epoch_paths,
        })

    for engine in down_engines:
        engine.uninstall()
    for engine in o_proj_engines:
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
    parser.add_argument("--summary_root", default=None,
                        help="Root directory containing summaries/expand_ratio_l*.json. "
                             "Defaults to --checkpoint_root.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="results/finetune_all_layers_half")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use, e.g. cuda:1. The exact device string is passed through to all functions.")
    parser.add_argument("--isolate_gpu", action="store_true", help="Set CUDA_VISIBLE_DEVICES before importing torch so that cuda:0 maps to the requested GPU. Old behaviour; off by default.")
    parser.add_argument("--lut_dtype", default="fp32", choices=["fp32", "fp16", "int8"],
                        help="Dtype of LUT checkpoints on disk. Training itself still uses float.")
    parser.add_argument("--o_proj_configs", type=str, default=None,
                        help="Comma-separated layer:count pairs for o_proj LUT replacement, "
                             "e.g. '17:4,15:4'. Loaded from --o_proj_checkpoint_root.")
    parser.add_argument("--o_proj_checkpoint_root", default="../v3/o_proj_outputs",
                        help="Root directory containing o_proj checkpoints/l{layer}/g{count}")
    parser.add_argument("--train_o_proj_only", action="store_true",
                        help="Freeze all down_proj weights and only train o_proj weights. "
                             "Useful for the first stage where down_proj is already tuned.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Directory containing l*_epoch*_down_proj.pt checkpoints to resume from. "
                             "If provided, the latest epoch is loaded as the starting point. "
                             "Missing layers are left uninitialized (useful for staged training).")
    parser.add_argument("--resume_best_epoch", action="store_true",
                        help="When resuming, load the epoch with the lowest PPL from summary.json "
                             "instead of the latest epoch.")
    parser.add_argument("--freeze_layers", type=str, default=None,
                        help="Comma-separated layer IDs whose down_proj weights are loaded/frozen and "
                             "excluded from training. Useful for staged training.")
    args = parser.parse_args()

    # Use the canonical device derived before torch was imported.
    args.device = _canonical_device

    if args.summary_root is None:
        args.summary_root = args.checkpoint_root

    freeze_layer_set = set()
    if args.freeze_layers is not None:
        freeze_layer_set = {int(x.strip()) for x in args.freeze_layers.split(",")}

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
    print(f"Down-proj configs: {configs}")
    if o_proj_configs:
        print(f"o_proj configs: {o_proj_configs}")
    print(f"Epochs={args.epochs}, LR={args.lr}, LUT dtype={args.lut_dtype}")
    if args.train_o_proj_only:
        print("Mode: down_proj FROZEN, only o_proj weights trained")
    print("=" * 70)

    # Load model and data.
    print("\n[1/4] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model,
        eval_size=args.eval_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device_str=args.device,
        calib_size=args.calib_size,
    )
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers

    # Build engines.
    print("\n[2/4] Building V3PartialEngines...")
    summaries = {lid: load_layer_summary(args.summary_root, lid) for lid, _ in configs}
    down_engines = []
    for layer_id, group_count in configs:
        engine = build_engine_for_layer(
            model, layer_id, group_count, args.checkpoint_root,
            lut_dtype=args.lut_dtype, summary=summaries[layer_id],
        )
        down_engines.append(engine)

    o_proj_configs = []
    o_proj_engines = []
    if args.o_proj_configs is not None:
        print("\n[2a/4] Building o_proj partial engines...")
        o_proj_configs = parse_layer_configs(args.o_proj_configs)
        for layer_id, group_count in o_proj_configs:
            engine = build_o_proj_engine_for_layer(
                model, layer_id, group_count, args.o_proj_checkpoint_root,
                lut_dtype=args.lut_dtype,
            )
            o_proj_engines.append(engine)

    # Resume from previous down_proj checkpoints if requested.
    # Missing layers are left uninitialized (useful for staged training where
    # some layers are added in the current stage).
    # When --freeze_layers is specified, only resume the frozen layers; trainable
    # layers are re-initialized from the base model / v3 checkpoint. This is
    # necessary when the current stage changes group counts for some layers.
    if args.resume is not None:
        print(f"\n[Resume] Loading down_proj weights from {args.resume}...")
        resume_epochs = {}
        for layer_id, _ in configs:
            if freeze_layer_set and layer_id not in freeze_layer_set:
                print(f"  [Skip] L{layer_id} is trainable in this stage; using base weight")
                continue
            pattern = os.path.join(args.resume, f"l{layer_id}_epoch*_down_proj.pt")
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"  [WARN] No resume checkpoint for L{layer_id}; leaving as current weight")
                continue

            # Determine target epoch: latest or best PPL from summary.json.
            target_epoch = None
            if args.resume_best_epoch:
                summary_path = os.path.join(args.resume, "summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r") as f:
                            summary = json.load(f)
                        best_entry = min(summary.get("after", []), key=lambda e: e.get("ppl", float("inf")))
                        target_epoch = best_entry.get("epoch")
                        print(f"  [Resume] Using best epoch {target_epoch} (PPL={best_entry.get('ppl', 0):.2f}) from summary.json")
                    except Exception as e:
                        print(f"  [WARN] Failed to load summary.json for best epoch selection: {e}; falling back to latest epoch")

            # Extract epoch number and pick the target or largest one.
            best_path = None
            best_epoch = -1
            for p in paths:
                name = os.path.basename(p)
                prefix = f"l{layer_id}_epoch"
                suffix = "_down_proj.pt"
                if name.startswith(prefix) and name.endswith(suffix):
                    epoch_str = name[len(prefix):-len(suffix)]
                    try:
                        epoch = int(epoch_str)
                        if target_epoch is not None and epoch == target_epoch:
                            best_epoch = epoch
                            best_path = p
                            break
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_path = p
                    except ValueError:
                        continue
            if best_path is None:
                print(f"  [WARN] No resume checkpoint for L{layer_id}; leaving as current weight")
                continue
            resume_epochs[layer_id] = best_epoch
            ckpt = torch.load(best_path, map_location="cpu")
            down_proj = model.model.layers[layer_id].mlp.down_proj
            target_device = down_proj.weight.device
            target_dtype = down_proj.weight.dtype
            down_proj.weight.data = ckpt.to(device=target_device, dtype=target_dtype)
        if resume_epochs:
            print(f"  Resumed from epoch {max(resume_epochs.values())}: " +
                  ", ".join(f"L{lid}:ep{ep}" for lid, ep in sorted(resume_epochs.items())))
        else:
            print("  No layers were resumed.")

    # Resume o_proj weights if present in the resume directory.
    if args.resume is not None and o_proj_configs:
        print(f"\n[Resume] Loading o_proj weights from {args.resume}...")
        for layer_id, _ in o_proj_configs:
            pattern = os.path.join(args.resume, f"l{layer_id}_epoch*_o_proj.pt")
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"  [WARN] No o_proj resume checkpoint for L{layer_id}")
                continue
            target_epoch = None
            if args.resume_best_epoch and os.path.exists(os.path.join(args.resume, "summary.json")):
                try:
                    with open(os.path.join(args.resume, "summary.json"), "r") as f:
                        summary = json.load(f)
                    best_entry = min(summary.get("after", []), key=lambda e: e.get("ppl", float("inf")))
                    target_epoch = best_entry.get("epoch")
                except Exception as e:
                    print(f"  [WARN] best-epoch resume failed for o_proj: {e}")
            best_path = None
            best_epoch = -1
            for p in paths:
                name = os.path.basename(p)
                prefix = f"l{layer_id}_epoch"
                suffix = "_o_proj.pt"
                if name.startswith(prefix) and name.endswith(suffix):
                    epoch_str = name[len(prefix):-len(suffix)]
                    try:
                        epoch = int(epoch_str)
                        if target_epoch is not None and epoch == target_epoch:
                            best_epoch = epoch
                            best_path = p
                            break
                        if epoch > best_epoch:
                            best_epoch = epoch
                            best_path = p
                    except ValueError:
                        continue
            if best_path is None:
                print(f"  [WARN] No usable o_proj resume checkpoint for L{layer_id}")
                continue
            ckpt = torch.load(best_path, map_location="cpu")
            o_proj = model.model.layers[layer_id].self_attn.o_proj
            o_proj.weight.data = ckpt.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype)
            print(f"  [Resume] L{layer_id} o_proj epoch {best_epoch}")

    # Pre-compute baseline eval probabilities (original model, no LUT).
    print("\n[3/4] Collecting baseline eval probabilities (original model, no LUT)...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)

    # Fine-tune.
    print("\n[4/4] Fine-tuning...")
    baseline_metrics, results = finetune_multi_layer(
        model, calib_loader, eval_loader, down_engines, o_proj_engines,
        args.epochs, args.lr, args.output_dir,
        baseline_eval_probs=baseline_eval_probs,
        configs=configs,
        o_proj_configs=o_proj_configs,
        freeze_layer_set=freeze_layer_set,
        train_o_proj_only=args.train_o_proj_only,
    )

    mac_reduction = compute_mac_reduction(
        configs, hidden_size, intermediate_size, num_layers,
        o_proj_configs=o_proj_configs
    )
    lut_storage = compute_lut_storage(
        configs, args.checkpoint_root, lut_dtype=args.lut_dtype,
        o_proj_configs=o_proj_configs, o_proj_checkpoint_root=args.o_proj_checkpoint_root
    )

    summary = {
        "model": args.model,
        "configs": configs,
        "o_proj_configs": o_proj_configs,
        "epochs": args.epochs,
        "lr": args.lr,
        "lut_dtype": args.lut_dtype,
        "train_o_proj_only": args.train_o_proj_only,
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
    print(f"  l*_epoch*_o_proj.pt     — fine-tuned o_proj weights")
    print(f"  summary.json            — before/after metrics")


if __name__ == "__main__":
    main()
