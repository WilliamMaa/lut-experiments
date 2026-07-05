"""
v5 o_proj multi-layer fine-tuning with trainable hybrid LUT.

Usage:
    cd LLM_LUT/v5
    LD_LIBRARY_PATH="" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python finetune_o_proj.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --configs "27:8" \
        --checkpoint_root ../v5/outputs_o_proj \
        --epochs 5 --lr 5e-5 --calib_size 512 --eval_size 128 \
        --output_dir results/finetune_o_proj_l27
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

from engine import HybridOProjEngine
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
        configs.append((int(layer_str), int(count_str)))
    return configs


def build_o_proj_engine_for_layer(model, layer_id: int, group_count: int,
                                  checkpoint_root: str) -> HybridOProjEngine:
    ckpt_dir = os.path.join(checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
    pattern = os.path.join(ckpt_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No o_proj checkpoints for L{layer_id} G{group_count} in {ckpt_dir}")

    first_ckpt = torch.load(paths[0], map_location="cpu")
    mode = first_ckpt.get("mode", "direct")
    engine = HybridOProjEngine(model, layer_id, group_size=64, mode=mode)

    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")

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

        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(model.device)
        engine.add_group(gid, address, lut_group)
    return engine


def finetune(model, calib_loader, eval_loader, engines, epochs, lr, output_dir,
             baseline_eval_probs):
    device = model.device

    o_projs = [model.model.layers[e.layer_id].self_attn.o_proj for e in engines]

    for p in model.parameters():
        p.requires_grad_(False)

    trainable_params = []
    for op in o_projs:
        op.weight.requires_grad_(True)
        trainable_params.append(op.weight)
    for engine in engines:
        trainable_params.extend(engine.trainable_parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0, eps=1e-8)

    print("\n[Pre-compute] Collecting baseline logits on calibration set...")
    baseline_logits = collect_baseline_logits(model, calib_loader)

    original_dtypes = [op.weight.dtype for op in o_projs]
    for op in o_projs:
        op.weight.data = op.weight.data.float()

    for engine in engines:
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
        print(f"\n[Epoch {epoch}/{epochs}] Fine-tuning {len(engines)} o_proj layers...")
        model.train()
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

            for op in o_projs:
                if not torch.isfinite(op.weight).all():
                    raise RuntimeError(f"o_proj.weight became NaN/Inf after batch {bi}")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg KL loss: {avg_loss:.6f}")

        model.eval()
        for op in o_projs:
            op.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                metrics = compute_model_metrics(model, eval_loader, reference_probs_list=baseline_eval_probs)
        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        epoch_paths = {}
        for engine, orig_dtype in zip(engines, original_dtypes):
            lid = engine.layer_id
            op = model.model.layers[lid].self_attn.o_proj
            w_path = os.path.join(output_dir, f"l{lid}_epoch{epoch}_o_proj.pt")
            torch.save(op.weight.data.to(orig_dtype).cpu(), w_path)
            epoch_paths[f"l{lid}_o_proj"] = w_path
            # Save updated LUT tables
            lut_dir = os.path.join(output_dir, f"l{lid}_epoch{epoch}_lut")
            engine.save_group_checkpoints(lut_dir)
            epoch_paths[f"l{lid}_lut"] = lut_dir

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
    parser.add_argument("--configs", required=True,
                        help="Comma-separated layer:count pairs, e.g. '27:8'")
    parser.add_argument("--checkpoint_root", default="../v5/outputs_o_proj")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="results/finetune_o_proj_first")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--isolate_gpu", action="store_true")
    args = parser.parse_args()

    if args.isolate_gpu and args.device.startswith("cuda:"):
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        args.device = "cuda:0"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)

    print("=" * 70)
    print("v5 Hybrid LUT o_proj Fine-Tune")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Configs: {configs}")
    print(f"Epochs={args.epochs}, LR={args.lr}")
    print("=" * 70)

    print("\n[1/3] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model, eval_size=args.eval_size, max_seq_len=args.max_seq_len,
        batch_size=args.batch_size, device_str=args.device, calib_size=args.calib_size,
    )
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    print("\n[2/3] Building hybrid LUT engines for o_proj...")
    engines = []
    for layer_id, group_count in configs:
        engine = build_o_proj_engine_for_layer(model, layer_id, group_count, args.checkpoint_root)
        engines.append(engine)

    print("\n[3/3] Collecting baseline eval probabilities...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)

    baseline_metrics, results = finetune(
        model, calib_loader, eval_loader, engines,
        args.epochs, args.lr, args.output_dir, baseline_eval_probs
    )

    # Approximate MAC reduction: each replaced group saves group_size * hidden_size MACs per token
    total_replaced = sum(group_count for _, group_count in configs)
    total_o_proj_mac = hidden_size * hidden_size * num_layers
    mac_reduction = total_replaced * args.group_size * hidden_size / total_o_proj_mac

    # LUT storage
    total_bytes = 0
    for layer_id, group_count in configs:
        ckpt_dir = os.path.join(args.checkpoint_root, "checkpoints", f"l{layer_id}", f"g{group_count}")
        for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            ckpt = torch.load(p, map_location="cpu")
            total_bytes += ckpt["lut_table"].numel() * 2  # assume fp16 on disk

    summary = {
        "model": args.model,
        "configs": configs,
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
    print("v5 O_PROJ FINE-TUNE COMPLETE")
    print("=" * 70)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
