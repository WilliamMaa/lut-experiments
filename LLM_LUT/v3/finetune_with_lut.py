"""
端到端联合微调：让 down_proj 权重适应 LUT 的存在。

核心思路：
1. 安装 v3 partial engine（选定 groups）
2. 冻结所有层，仅让 target layer 的 down_proj.weight 可训练
3. 在 calibration 数据上做端到端 KL 微调
4. 每 epoch 评估 KL/PPL/Acc

用法:
    cd /data/mingyu/LLM_LUT/v3
    python finetune_with_lut.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layer 21 --groups "26,50,51,4,7,40" \
        --checkpoint_dir ../v2/results/7B_l21_6group_ckpt \
        --epochs 3 --lr 1e-5 \
        --output_dir results/finetune_l21
"""

import os
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"

import sys
import json
import glob
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

from data import prepare_data, load_jsonl, TextDataset
from metrics import compute_model_metrics, compute_baseline_probs
from partial_linear import V3PartialEngine
from triton_kernels import lut_fill


class TrainableV3PartialEngine(V3PartialEngine):
    """V3PartialEngine variant that reads active_weight directly from down_proj.weight
    on every forward, so gradients flow back to the original weight during fine-tuning.
    """

    def _patched_down_proj_forward(self, hidden):
        B, S, intermediate_size = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        normed_x = self._cached_normed_x
        if normed_x is None:
            raise RuntimeError("normed_x not cached!")

        hidden_size = normed_x.shape[-1]
        replaced_groups = sorted(self.group_configs.keys())

        # Compute active matmul and LUT fill in fp32 for numerical stability,
        # then cast the assembled output back to the model dtype.
        # Disable autocast so fp32 inputs are not silently cast back to fp16.
        compute_dtype = torch.float32
        with torch.autocast(device_type=device.type, enabled=False):
            hidden_f32 = hidden.to(compute_dtype)
            normed_x_f32 = normed_x.to(compute_dtype)

            # --- Always slice from original weight (no cache) for trainability ---
            active_weight = self.down_proj.weight[self._active_indices, :].to(compute_dtype)
            active_bias = None
            if self.down_proj.bias is not None:
                active_bias = self.down_proj.bias[self._active_indices].to(compute_dtype)
            active_out = F.linear(hidden_f32, active_weight, active_bias)
            if not torch.isfinite(active_out).all():
                raise RuntimeError(
                    f"active_out contains NaN/Inf: "
                    f"hidden finite={torch.isfinite(hidden).all().item()}, "
                    f"weight finite={torch.isfinite(self.down_proj.weight).all().item()}, "
                    f"active abs max={torch.nan_to_num(active_out.detach()).abs().max().item():.2e}"
                )

            # --- LUT fill in fp32 ---
            if len(replaced_groups) == 0:
                lut_outputs = torch.empty(B, S, 0, device=device, dtype=compute_dtype)
            elif self._batched_tables is not None and self._cached_bin_idx_tensor is not None:
                M = B * S
                normed_x_flat = normed_x_f32.view(M, hidden_size)
                try:
                    tables_f32 = self._batched_tables.to(compute_dtype)
                    lut_outputs_flat = lut_fill(
                        self._cached_bin_idx_tensor,
                        tables_f32,
                        normed_x_flat,
                        self._group_starts,
                    )
                    lut_outputs = lut_outputs_flat.view(B, S, -1)
                except Exception as e:
                    print(f"[V3] LUT fill error ({e}), falling back to per-group loop")
                    lut_outputs = self._lut_fill_loop(B, S, normed_x_f32, device, compute_dtype)
            else:
                lut_outputs = self._lut_fill_loop(B, S, normed_x_f32, device, compute_dtype)

            if not torch.isfinite(lut_outputs).all():
                raise RuntimeError(
                    f"lut_outputs contains NaN/Inf: "
                    f"normed_x finite={torch.isfinite(normed_x).all().item()}, "
                    f"lut abs max={torch.nan_to_num(lut_outputs.detach()).abs().max().item():.2e}"
                )

            # --- Assemble in fp32 and cast back ---
            full_out = torch.zeros(B, S, hidden_size, device=device, dtype=compute_dtype)
            full_out = full_out.index_copy_(2, self._active_indices, active_out)
            if lut_outputs.shape[-1] > 0:
                full_out = full_out.index_copy_(2, self._replaced_indices, lut_outputs)

        return full_out.to(dtype)


def load_model_and_data(model_name, calib_size, eval_size, max_seq_len, batch_size, device_str="cuda:0"):
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device_str, low_cpu_mem_usage=True,
    )
    model.eval()

    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            print(f"[WARN] GPU {i} has allocated memory; proceeding because device_map={device_str} is explicit single-GPU.")

    calib_path = "../v0/data/calib.jsonl"
    eval_path = "../v0/data/eval.jsonl"
    prepare_data(tokenizer, calib_path, eval_path, calib_size=calib_size, eval_size=eval_size, max_seq_len=max_seq_len)
    calib_texts = load_jsonl(calib_path)
    eval_texts = load_jsonl(eval_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len=max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=batch_size, shuffle=False)
    eval_loader = eval_dataset.make_loader(batch_size=batch_size, shuffle=False)

    return model, tokenizer, calib_loader, eval_loader


def collect_baseline_logits(model, data_loader):
    """Pre-compute baseline logits on calibration set, per batch.

    Returns a list of [B, S, vocab] tensors so batches with different sequence
    lengths (due to dynamic padding) do not need to be concatenated.
    """
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collect baseline logits", leave=False):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
    return all_logits


def _check_weight_finite(weight, tag=""):
    if not torch.isfinite(weight).all():
        raise RuntimeError(f"{tag}down_proj weight is not finite")
    w = weight.detach().float()
    print(f"  [DEBUG] {tag}down_proj weight: finite=True, min={w.min().item():.4e}, max={w.max().item():.4e}, mean={w.mean().item():.4e}")


def finetune(model, calib_loader, eval_loader, engine, epochs, lr, output_dir, baseline_eval_probs):
    """Fine-tune down_proj.weight to adapt to LUT presence."""
    device = model.device
    layer = model.model.layers[engine.layer_id]
    down_proj = layer.mlp.down_proj

    # Make only down_proj.weight trainable
    for p in model.parameters():
        p.requires_grad_(False)
    down_proj.weight.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [down_proj.weight],
        lr=lr,
        weight_decay=0.0,
        eps=1e-8,
    )

    # Pre-compute baseline logits on calibration set (original model, no LUT)
    print("\n[Pre-compute] Collecting baseline logits on calibration set...")
    baseline_logits = collect_baseline_logits(model, calib_loader)

    # Convert target weight to fp32 for stable fine-tuning (after baseline, before installing LUT hook)
    original_dtype = down_proj.weight.dtype
    down_proj.weight.data = down_proj.weight.data.float()

    # Install engine
    engine.install()

    # Pre-train eval: LUT model vs original model
    print("\n[Pre-train eval] Baseline evaluation (LUT model, before fine-tuning)...")
    model.eval()
    _check_weight_finite(down_proj.weight)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_metrics = compute_model_metrics(
                model,
                eval_loader,
                reference_probs_list=baseline_eval_probs,
            )
    print(f"  Before: KL={baseline_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    results = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] Fine-tuning...")
        model.train()
        down_proj.train()  # only this matters

        total_loss = 0.0
        num_batches = 0

        for bi, batch in enumerate(tqdm(calib_loader, desc=f"Train epoch {epoch}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward with LUT hook installed
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B, S, vocab]

            if not torch.isfinite(logits).all():
                print(f"[WARN] Batch {bi}: non-finite logits, "
                      f"weight finite={torch.isfinite(down_proj.weight).all().item()}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Target: baseline logits for the same batch
            target_logits = baseline_logits[bi].to(device)

            # KL divergence loss on next-token prediction
            # logits[:, :-1] predict input_ids[:, 1:]
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

            grad = down_proj.weight.grad
            if grad is None or not torch.isfinite(grad).all():
                print(f"[WARN] Batch {bi}: non-finite gradient, skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [down_proj.weight],
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            optimizer.step()

            if not torch.isfinite(down_proj.weight).all():
                raise RuntimeError(f"down_proj.weight became NaN/Inf after batch {bi}")

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg KL loss: {avg_loss:.6f}")

        # Evaluate
        print(f"  Evaluating...")
        model.eval()
        _check_weight_finite(down_proj.weight)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                metrics = compute_model_metrics(
                    model,
                    eval_loader,
                    reference_probs_list=baseline_eval_probs,
                )

        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"epoch{epoch}_down_proj.pt")
        torch.save(down_proj.weight.data.to(original_dtype).cpu(), ckpt_path)

        results.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "ckpt": ckpt_path,
        })

    engine.uninstall()
    return baseline_metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--groups", default="26,50,51,4,7,40", help="Comma-separated group IDs, or 'auto' to infer from checkpoint_dir")
    parser.add_argument("--checkpoint_dir", default="results/expand_ratio_l21/checkpoints/ckpt_g16", help="Directory containing replacement_l{layer}g{gid}.pt files")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="outputs/finetune_l21")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.groups.strip().lower() == "auto":
        pattern = os.path.join(args.checkpoint_dir, f"replacement_l{args.layer}g*.pt")
        ckpt_paths = sorted(glob.glob(pattern))
        if not ckpt_paths:
            raise ValueError(f"No checkpoints found for layer {args.layer} in {args.checkpoint_dir}")
        group_list = []
        for p in ckpt_paths:
            name = os.path.basename(p)
            prefix = f"replacement_l{args.layer}g"
            suffix = ".pt"
            if not (name.startswith(prefix) and name.endswith(suffix)):
                raise ValueError(f"Unexpected checkpoint filename: {name}")
            gid = int(name[len(prefix):-len(suffix)])
            group_list.append(gid)
        group_list = sorted(group_list)
        print(f"[auto] Inferred groups from {args.checkpoint_dir}: {group_list}")
    else:
        group_list = [int(g.strip()) for g in args.groups.split(",")]

    print("=" * 70)
    print(f"Fine-tune with LUT: L{args.layer}, groups={group_list}")
    print(f"Epochs={args.epochs}, LR={args.lr}")
    print("=" * 70)

    # Load
    print("\n[1/3] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )

    # Build engine
    print(f"\n[2/3] Building V3PartialEngine...")
    engine = TrainableV3PartialEngine(model, args.layer, group_size=64, num_bins=64)
    for gid in group_list:
        ckpt_path = os.path.join(args.checkpoint_dir, f"replacement_l{args.layer}g{gid}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        engine.add_group(
            group_id=gid,
            addr_idx=ckpt["addr_idx"],
            addr_mean=ckpt["addr_mean"],
            addr_std=ckpt["addr_std"],
            table=ckpt["table"],
        )

    # Pre-compute baseline eval probabilities on original model (no LUT)
    print("\n[3/3] Collecting baseline eval probabilities (original model, no LUT)...")
    model.eval()
    with torch.no_grad():
        baseline_eval_probs = compute_baseline_probs(model, eval_loader)

    # Fine-tune
    print("\n[4/4] Fine-tuning...")
    baseline_metrics, results = finetune(
        model, calib_loader, eval_loader, engine,
        args.epochs, args.lr, args.output_dir,
        baseline_eval_probs=baseline_eval_probs,
    )

    # Save summary
    summary = {
        "model": args.model,
        "layer": args.layer,
        "groups": group_list,
        "epochs": args.epochs,
        "lr": args.lr,
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
    print("FINE-TUNE COMPLETE")
    print("=" * 70)
    print(f"Results saved to {args.output_dir}")
    print(f"  epoch*_down_proj.pt  — fine-tuned down_proj weights")
    print(f"  summary.json         — before/after metrics")


if __name__ == "__main__":
    main()
