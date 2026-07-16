"""
v5 joint fine-tuning for down_proj + o_proj hybrid LUT replacement.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune_joint.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --down_configs "21:8,22:8,23:8" \
        --down_checkpoint_root ../v5/outputs_tree_21_23 \
        --o_configs "17:8" \
        --o_checkpoint_root ../v5/outputs_o_proj_l17 \
        --epochs 5 --lr 5e-5 --calib_size 512 --eval_size 128 \
        --output_dir results/finetune_joint_down_o
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from engine import HybridPartialEngine, HybridOProjEngine
from hybrid_gate_proj_engine import HybridGateProjEngine
from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup
from utils import load_model_and_data, collect_baseline_logits
from metrics import compute_model_metrics, compute_baseline_probs, format_bytes


def parse_configs(arg_str: str) -> List[Tuple[int, int]]:
    configs = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        layer_str, count_str = part.split(":")
        # Allow group ids after count, e.g. layer:count;id1;id2 or layer:count:id1;id2
        count_str = count_str.split(";")[0]
        configs.append((int(layer_str), int(count_str)))
    return configs


def build_address(ckpt: dict):
    """Build an address generator from a checkpoint dict."""
    if ckpt["address_type"] == "2d":
        address = Address2D(
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            num_bins=ckpt["num_bins"],
            addr_clip=ckpt.get("addr_clip", 3.0),
        )
    elif ckpt["address_type"] == "high_order":
        address = AddressHighOrderRandom(
            input_dim=1,
            num_tables=ckpt["num_tables"],
            num_bits=ckpt["num_bits"],
            channels_per_bit=ckpt["channels_per_bit"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
        )
        address.channel_idx = ckpt["channel_idx"]
        address.signs = ckpt["signs"]
        address.input_dim = int(ckpt.get("input_dim", ckpt["channel_idx"].max().item() + 1))
    elif ckpt["address_type"] == "tree":
        address = AddressGreedyTree(
            input_dim=1,
            num_bits=ckpt["num_bits"],
            channels_per_bit=ckpt["channels_per_bit"],
            tree_state=ckpt["tree_state"],
        )

        def max_ch(node):
            if "leaf_index" in node:
                return 0
            return max(max(node["channel_idx"]) + 1, max_ch(node["left"]), max_ch(node["right"]))

        address.input_dim = int(ckpt.get("input_dim", max_ch(ckpt["tree_state"]["tree"])))
    else:
        raise ValueError(f"Unknown address type: {ckpt['address_type']}")
    return address


def build_down_engine_for_layer(model, layer_id: int, group_count: int,
                                checkpoint_root: str, group_size: int) -> HybridPartialEngine:
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", "down_proj", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No down_proj checkpoints for L{layer_id} G{group_count} in {ckpt_dir}")

    engine = HybridPartialEngine(model, layer_id, group_size=group_size)
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        address = build_address(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(model.device)
        engine.add_group(gid, address, lut_group)
    return engine


def build_o_proj_engine_for_layer(model, layer_id: int, group_count: int,
                                  checkpoint_root: str, group_size: int) -> HybridOProjEngine:
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", "o_proj", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No o_proj checkpoints for L{layer_id} G{group_count} in {ckpt_dir}")

    first_ckpt = torch.load(paths[0], map_location="cpu")
    mode = first_ckpt.get("mode", "direct")
    engine = HybridOProjEngine(model, layer_id, group_size=group_size, mode=mode)
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        address = build_address(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(model.device)
        engine.add_group(gid, address, lut_group)
    return engine


def build_gate_proj_engine_for_layer(model, layer_id: int, group_count: int,
                                     checkpoint_root: str, group_size: int) -> HybridGateProjEngine:
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", "gate_proj", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No gate_proj checkpoints for L{layer_id} G{group_count} in {ckpt_dir}")

    engine = HybridGateProjEngine(model, layer_id, group_size=group_size)
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        address = build_address(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(model.device)
        engine.add_group(gid, address, lut_group)
    return engine


def finetune(model, calib_loader, eval_loader, down_engines, gate_engines, o_engines, epochs, lr, output_dir,
             baseline_eval_probs, freeze_down=False, freeze_gate=False, freeze_o=False):
    device = model.device

    down_projs = [model.model.layers[e.layer_id].mlp.down_proj for e in down_engines]
    gate_projs = [model.model.layers[e.layer_id].mlp.gate_proj for e in gate_engines]
    o_projs = [model.model.layers[e.layer_id].self_attn.o_proj for e in o_engines]

    for p in model.parameters():
        p.requires_grad_(False)

    trainable_params = []
    if not freeze_down:
        for dp in down_projs:
            dp.weight.requires_grad_(True)
            trainable_params.append(dp.weight)
    else:
        for dp in down_projs:
            dp.weight.requires_grad_(False)
    if not freeze_gate:
        for gp in gate_projs:
            gp.weight.requires_grad_(True)
            trainable_params.append(gp.weight)
    else:
        for gp in gate_projs:
            gp.weight.requires_grad_(False)
    if not freeze_o:
        for op in o_projs:
            op.weight.requires_grad_(True)
            trainable_params.append(op.weight)
    else:
        for op in o_projs:
            op.weight.requires_grad_(False)
    for engine in down_engines + gate_engines + o_engines:
        if (engine in down_engines and not freeze_down) or \
           (engine in gate_engines and not freeze_gate) or \
           (engine in o_engines and not freeze_o):
            trainable_params.extend(engine.trainable_parameters())
        else:
            for _, lut_group in engine.group_configs.values():
                lut_group.table.requires_grad_(False)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0, eps=1e-8)

    print("\n[Pre-compute] Collecting baseline logits on calibration set...")
    baseline_logits = collect_baseline_logits(model, calib_loader)

    down_original_dtypes = [dp.weight.dtype for dp in down_projs]
    gate_original_dtypes = [gp.weight.dtype for gp in gate_projs]
    o_original_dtypes = [op.weight.dtype for op in o_projs]
    for dp in down_projs:
        dp.weight.data = dp.weight.data.float()
    for gp in gate_projs:
        gp.weight.data = gp.weight.data.float()
    for op in o_projs:
        op.weight.data = op.weight.data.float()

    for engine in down_engines + gate_engines + o_engines:
        engine.install()

    print("\n[Pre-train eval] Baseline evaluation (LUT model, before fine-tuning)...")
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
    print(f"  Before: KL={baseline_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    results = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] Fine-tuning {len(down_engines)} down + {len(gate_engines)} gate + {len(o_engines)} o layers...")
        model.train()
        for dp in down_projs:
            if not freeze_down:
                dp.train()
            else:
                dp.eval()
        for gp in gate_projs:
            if not freeze_gate:
                gp.train()
            else:
                gp.eval()
        for op in o_projs:
            if not freeze_o:
                op.train()
            else:
                op.eval()

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
            for p in trainable_params:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                print(f"[WARN] Batch {bi}: non-finite gradient, skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            if not freeze_down:
                for dp in down_projs:
                    if not torch.isfinite(dp.weight).all():
                        raise RuntimeError(f"down_proj.weight became NaN/Inf after batch {bi}")
            if not freeze_gate:
                for gp in gate_projs:
                    if not torch.isfinite(gp.weight).all():
                        raise RuntimeError(f"gate_proj.weight became NaN/Inf after batch {bi}")
            if not freeze_o:
                for op in o_projs:
                    if not torch.isfinite(op.weight).all():
                        raise RuntimeError(f"o_proj.weight became NaN/Inf after batch {bi}")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg KL loss: {avg_loss:.6f}")

        model.eval()
        for dp in down_projs:
            dp.eval()
        for gp in gate_projs:
            gp.eval()
        for op in o_projs:
            op.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        epoch_paths = {}
        for engine, orig_dtype in zip(down_engines, down_original_dtypes):
            lid = engine.layer_id
            dp = model.model.layers[lid].mlp.down_proj
            w_path = os.path.join(output_dir, f"l{lid}_epoch{epoch}_down_proj.pt")
            torch.save(dp.weight.data.to(orig_dtype).cpu(), w_path)
            epoch_paths[f"l{lid}_down_proj"] = w_path
            lut_dir = os.path.join(output_dir, f"l{lid}_epoch{epoch}_down_lut")
            engine.save_group_checkpoints(lut_dir)
            epoch_paths[f"l{lid}_down_lut"] = lut_dir

        for engine, orig_dtype in zip(gate_engines, gate_original_dtypes):
            lid = engine.layer_id
            gp = model.model.layers[lid].mlp.gate_proj
            w_path = os.path.join(output_dir, f"l{lid}_epoch{epoch}_gate_proj.pt")
            torch.save(gp.weight.data.to(orig_dtype).cpu(), w_path)
            epoch_paths[f"l{lid}_gate_proj"] = w_path
            lut_dir = os.path.join(output_dir, f"l{lid}_epoch{epoch}_gate_lut")
            engine.save_group_checkpoints(lut_dir)
            epoch_paths[f"l{lid}_gate_lut"] = lut_dir

        for engine, orig_dtype in zip(o_engines, o_original_dtypes):
            lid = engine.layer_id
            op = model.model.layers[lid].self_attn.o_proj
            w_path = os.path.join(output_dir, f"l{lid}_epoch{epoch}_o_proj.pt")
            torch.save(op.weight.data.to(orig_dtype).cpu(), w_path)
            epoch_paths[f"l{lid}_o_proj"] = w_path
            lut_dir = os.path.join(output_dir, f"l{lid}_epoch{epoch}_o_lut")
            engine.save_group_checkpoints(lut_dir)
            epoch_paths[f"l{lid}_o_lut"] = lut_dir

        results.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "checkpoints": epoch_paths,
        })

    for engine in down_engines + gate_engines + o_engines:
        engine.uninstall()
    return baseline_metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--down_configs", default="",
                        help="Comma-separated layer:count for down_proj, e.g. '21:8,22:8,23:8'")
    parser.add_argument("--down_checkpoint_root", default="../v5/outputs_tree_21_23")
    parser.add_argument("--o_configs", default="",
                        help="Comma-separated layer:count for o_proj, e.g. '17:8'")
    parser.add_argument("--o_checkpoint_root", default="../v5/outputs_o_proj_l17")
    parser.add_argument("--gate_configs", default="",
                        help="Comma-separated layer:count for gate_proj, e.g. '20:16,21:16'")
    parser.add_argument("--gate_checkpoint_root", default="../v5/outputs_gate_proj")
    parser.add_argument("--freeze_gate", action="store_true",
                        help="Install gate_proj engines but do not update their weights/LUTs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--output_dir", default="results/finetune_joint_down_o")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isolate_gpu", action="store_true")
    parser.add_argument("--freeze_down", action="store_true",
                        help="Install down_proj engines but do not update their weights/LUTs")
    parser.add_argument("--freeze_o", action="store_true",
                        help="Install o_proj engines but do not update their weights/LUTs")
    args = parser.parse_args()

    if args.isolate_gpu and args.device.startswith("cuda:"):
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        args.device = "cuda:0"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    down_configs = parse_configs(args.down_configs) if args.down_configs else []
    gate_configs = parse_configs(args.gate_configs) if args.gate_configs else []
    o_configs = parse_configs(args.o_configs) if args.o_configs else []

    print("=" * 70)
    print("v5 Joint down_proj + gate_proj + o_proj Hybrid LUT Fine-Tune")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"down_proj configs: {down_configs}")
    print(f"gate_proj configs: {gate_configs}")
    print(f"o_proj configs: {o_configs}")
    print(f"Epochs={args.epochs}, LR={args.lr}")
    print("=" * 70)

    print("\n[1/3] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model, eval_size=args.eval_size, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, device_str=args.device, calib_size=args.calib_size,
    )
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers

    print("\n[2/3] Building hybrid LUT engines...")
    down_engines = []
    for layer_id, group_count in down_configs:
        engine = build_down_engine_for_layer(
            model, layer_id, group_count, args.down_checkpoint_root, args.group_size
        )
        down_engines.append(engine)

    gate_engines = []
    for layer_id, group_count in gate_configs:
        engine = build_gate_proj_engine_for_layer(
            model, layer_id, group_count, args.gate_checkpoint_root, args.group_size
        )
        gate_engines.append(engine)

    o_engines = []
    for layer_id, group_count in o_configs:
        engine = build_o_proj_engine_for_layer(
            model, layer_id, group_count, args.o_checkpoint_root, args.group_size
        )
        o_engines.append(engine)

    print("\n[3/3] Collecting baseline eval probabilities...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)

    baseline_metrics, results = finetune(
        model, calib_loader, eval_loader, down_engines, gate_engines, o_engines,
        args.epochs, args.lr, args.output_dir, baseline_eval_probs,
        freeze_down=args.freeze_down, freeze_gate=args.freeze_gate, freeze_o=args.freeze_o
    )

    # Full-model MAC reduction: major linear layers
    per_layer_total = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    full_model_total = num_layers * per_layer_total
    down_eliminated = sum(count * args.group_size * intermediate_size for _, count in down_configs)
    gate_eliminated = sum(count * args.group_size * hidden_size for _, count in gate_configs)
    o_eliminated = sum(count * args.group_size * hidden_size for _, count in o_configs)
    mac_reduction = (down_eliminated + gate_eliminated + o_eliminated) / full_model_total

    # LUT storage from all roots
    total_bytes = 0
    for layer_id, group_count in down_configs:
        ckpt_dir = os.path.join(args.down_checkpoint_root, "checkpoints", f"l{layer_id}", "down_proj", f"g{group_count}")
        for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            ckpt = torch.load(p, map_location="cpu")
            total_bytes += ckpt["lut_table"].numel() * 2
    for layer_id, group_count in gate_configs:
        ckpt_dir = os.path.join(args.gate_checkpoint_root, "checkpoints", f"l{layer_id}", "gate_proj", f"g{group_count}")
        for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            ckpt = torch.load(p, map_location="cpu")
            total_bytes += ckpt["lut_table"].numel() * 2
    for layer_id, group_count in o_configs:
        ckpt_dir = os.path.join(args.o_checkpoint_root, "checkpoints", f"l{layer_id}", "o_proj", f"g{group_count}")
        for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            ckpt = torch.load(p, map_location="cpu")
            total_bytes += ckpt["lut_table"].numel() * 2

    summary = {
        "model": args.model,
        "down_configs": down_configs,
        "gate_configs": gate_configs,
        "o_configs": o_configs,
        "epochs": args.epochs,
        "lr": args.lr,
        "mac_reduction_ratio": mac_reduction,
        "lut_storage_bytes": total_bytes,
        "lut_storage_human": format_bytes(total_bytes),
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
    print("v5 JOINT FINE-TUNE COMPLETE")
    print("=" * 70)
    print(f"MAC reduction ratio: {mac_reduction*100:.3f}%")
    print(f"LUT storage: {format_bytes(total_bytes)}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
