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
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
V2_DIR = os.path.join(os.path.dirname(__file__), "..", "v2")
sys.path.insert(0, V0_DIR)
sys.path.insert(0, V2_DIR)

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

        # --- Always slice from original weight (no cache) for trainability ---
        active_weight = self.down_proj.weight[self._active_indices, :]
        active_bias = None
        if self.down_proj.bias is not None:
            active_bias = self.down_proj.bias[self._active_indices]
        active_out = F.linear(hidden, active_weight, active_bias)

        # --- LUT fill ---
        if len(replaced_groups) == 0:
            lut_outputs = torch.empty(B, S, 0, device=device, dtype=dtype)
        elif self._batched_tables is not None and self._cached_bin_idx_tensor is not None:
            M = B * S
            normed_x_flat = normed_x.view(M, hidden_size)
            try:
                lut_outputs_flat = lut_fill(
                    self._cached_bin_idx_tensor,
                    self._batched_tables,
                    normed_x_flat,
                    self._group_starts,
                    self._batched_addr_mean,
                    self._batched_addr_std,
                )
                lut_outputs = lut_outputs_flat.view(B, S, -1)
            except Exception:
                lut_outputs = self._lut_fill_loop(B, S, normed_x, device, dtype)
        else:
            lut_outputs = self._lut_fill_loop(B, S, normed_x, device, dtype)

        # --- Assemble ---
        full_out = torch.zeros(B, S, hidden_size, device=device, dtype=dtype)
        full_out = full_out.index_copy_(2, self._active_indices, active_out)
        if lut_outputs.shape[-1] > 0:
            full_out = full_out.index_copy_(2, self._replaced_indices, lut_outputs)
        return full_out


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
    """Pre-compute baseline logits on calibration set."""
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collect baseline logits", leave=False):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Collect logits for all positions
            all_logits.append(outputs.logits.cpu())
    return torch.cat(all_logits, dim=0)


def finetune(model, calib_loader, eval_loader, engine, epochs, lr, output_dir):
    """Fine-tune down_proj.weight to adapt to LUT presence."""
    device = model.device
    layer = model.model.layers[engine.layer_id]
    down_proj = layer.mlp.down_proj

    # Make only down_proj.weight trainable
    for p in model.parameters():
        p.requires_grad_(False)
    down_proj.weight.requires_grad_(True)

    optimizer = torch.optim.AdamW([down_proj.weight], lr=lr)

    # Pre-compute baseline logits
    print("\n[Pre-compute] Collecting baseline logits on calibration set...")
    baseline_logits = collect_baseline_logits(model, calib_loader)

    # Install engine
    engine.install()

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

            # Forward with LUT hook installed
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, S, vocab]

            # Target: baseline logits (shifted for next-token prediction)
            B, S = input_ids.shape
            start_idx = bi * calib_loader.batch_size
            end_idx = start_idx + B
            target_logits = baseline_logits[start_idx:end_idx].to(device)

            # KL divergence loss on next-token prediction
            # logits[:, :-1] predict input_ids[:, 1:]
            pred = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            target = target_logits[:, :-1, :].contiguous().view(-1, target_logits.size(-1))

            log_probs = F.log_softmax(pred, dim=-1)
            target_probs = F.softmax(target, dim=-1)
            loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg KL loss: {avg_loss:.6f}")

        # Evaluate
        print(f"  Evaluating...")
        model.eval()
        reference_probs = compute_baseline_probs(model, eval_loader)
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)

        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"epoch{epoch}_down_proj.pt")
        torch.save(down_proj.weight.data.cpu(), ckpt_path)

        results.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "ckpt": ckpt_path,
        })

    engine.uninstall()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--groups", default="26,50,51,4,7,40")
    parser.add_argument("--checkpoint_dir", default="../v2/results/7B_l21_6group_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="results/finetune_l21")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
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

    # Baseline eval
    print("\n[3/3] Baseline evaluation (before fine-tuning)...")
    engine.install()
    reference_probs = compute_baseline_probs(model, eval_loader)
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    print(f"  Before: KL={baseline_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")
    engine.uninstall()

    # Fine-tune
    results = finetune(model, calib_loader, eval_loader, engine, args.epochs, args.lr, args.output_dir)

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
