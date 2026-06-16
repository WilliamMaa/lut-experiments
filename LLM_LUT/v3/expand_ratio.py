"""
扩展 replacement ratio：扫描 L21 未使用 groups + 渐进 multi-group 评估。

用法:
    cd /data/mingyu/LLM_LUT/v3
    python expand_ratio.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layer 21 \
        --used_groups "26,50,51,4,7,40" \
        --output_root outputs

流程:
    1. Zero ablation 所有未使用 groups（快速筛选）
    2. Top 20 候选做 2D joint bucket 评估
    3. 逐步构建 8/10/12/14/16 group 配置，评估累积效果
    4. 保存 checkpoint、summary、scan 和 generation 样本
"""

import os
os.environ["ACCELERATE_USE_DEVICE_MAP"] = "false"

import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from metrics import compute_baseline_probs, compute_model_metrics
from hooks import PerturbationHook
from config import get_hook_target
from table_builder import collect_teacher_targets, build_joint_bucket_table
from replacement_engine import ReplacementEngine
from generation import generate_outputs, AUTO_PROMPTS


def load_model_and_data(model_name, calib_size, eval_size, max_seq_len, batch_size, device_str="cuda:0"):
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        device_map=device_str, low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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


def eval_zero_group(model, eval_loader, reference_probs, layer_id, group_id, group_size):
    """Evaluate a single group with zero ablation."""
    hook = PerturbationHook(
        candidate_type="mlp_delta",
        group_size=group_size,
        group_id=group_id,
        mode="zero",
        num_bins=64,
    )
    target_mod = get_hook_target(model, layer_id, "mlp_delta")
    handle = target_mod.register_forward_hook(hook)
    try:
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        handle.remove()
    return metrics


def eval_bucket_group(model, eval_loader, reference_probs, engine):
    """Evaluate a single group with ReplacementEngine (2D bucket table)."""
    engine.install()
    try:
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        engine.uninstall()
    return metrics


def build_replacement_engine(model, calib_loader, layer_id, group_id, group_size, calib_results, num_bins=64):
    """Build a ReplacementEngine for a single group with 2D joint bucket table."""
    calib = calib_results[(layer_id, "mlp_delta")]
    addr_idx = calib["addr_idx"][group_id]
    addr_mean = calib["addr_mean"][group_id]
    addr_std = calib["addr_std"][group_id]

    bin_idx, targets, _ = collect_teacher_targets(
        model, calib_loader, layer_id, "mlp_delta", group_id, group_size,
        addr_idx, addr_mean, addr_std, num_bins=num_bins,
    )
    joint_table = build_joint_bucket_table(bin_idx, targets, num_bins, group_size)

    engine = ReplacementEngine(
        model=model, layer_id=layer_id, group_id=group_id,
        group_size=group_size, addr_idx=addr_idx, addr_mean=addr_mean,
        addr_std=addr_std, table=joint_table, num_bins=num_bins,
    )
    return engine


def eval_multi_group(model, eval_loader, reference_probs, engines):
    """Install all engines, evaluate, then uninstall."""
    for e in engines:
        e.install()
    try:
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        for e in engines:
            e.uninstall()
    return metrics


def save_generation_samples(model, tokenizer, engines, output_path, num_prompts=3, device="cuda:0"):
    """Generate samples for drift detection."""
    for e in engines:
        e.install()
    try:
        outputs = generate_outputs(
            model, tokenizer, prompts=AUTO_PROMPTS[:num_prompts],
            num_samples=1, max_new_tokens=64,
            device=device,
        )
    finally:
        for e in engines:
            e.uninstall()

    with open(output_path, "w", encoding="utf-8") as f:
        for item, samples in zip(AUTO_PROMPTS[:num_prompts], outputs):
            f.write(f"**Prompt**: {item['prompt']}\n\n")
            f.write(f"**Output**: {samples[0]}\n\n")
            f.write("---\n\n")
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--used_groups", default="26,50,51,4,7,40")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--zero_eval_size", type=int, default=64, help="Fast eval size for zero ablation")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_root", default="outputs", help="Root output directory for experiments")
    parser.add_argument("--experiment_name", default=None, help="Experiment name prefix (default: expand_ratio_l{layer})")
    parser.add_argument("--top_k_zero", type=int, default=25, help="How many top zero-ablation groups to evaluate with bucket")
    parser.add_argument("--target_counts", default="8,10,12,14,16", help="Progressive group counts to evaluate")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = f"expand_ratio_l{args.layer}"

    # Output paths by artifact type
    summaries_dir = os.path.join(args.output_root, "summaries")
    scans_dir = os.path.join(args.output_root, "scans")
    generation_dir = os.path.join(args.output_root, "generation")
    checkpoints_root = os.path.join(args.output_root, "checkpoints")
    for d in [summaries_dir, scans_dir, generation_dir, checkpoints_root]:
        Path(d).mkdir(parents=True, exist_ok=True)

    used_groups = set(int(g.strip()) for g in args.used_groups.split(",") if g.strip())
    layer_id = args.layer
    target_counts = [int(x) for x in args.target_counts.split(",")]

    print("=" * 70)
    print(f"Expand Ratio: L{layer_id}")
    print(f"Used groups: {sorted(used_groups)}")
    print(f"Target counts: {target_counts}")
    print("=" * 70)

    # 1. Load model and data
    print("\n[1/5] Loading model and data...")
    model, tokenizer, calib_loader, eval_loader = load_model_and_data(
        args.model, args.calib_size, args.eval_size, args.max_seq_len, args.batch_size, device_str=args.device
    )
    # Fast eval loader for zero ablation
    fast_eval_texts = load_jsonl("../v0/data/eval.jsonl")[:args.zero_eval_size]
    fast_eval_dataset = TextDataset(fast_eval_texts, tokenizer, max_seq_len=args.max_seq_len)
    fast_eval_loader = fast_eval_dataset.make_loader(batch_size=args.batch_size, shuffle=False)

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    num_groups = hidden_size // args.group_size

    # 2. Calibrate
    print(f"\n[2/5] Calibrating layer {layer_id}...")
    calib_results = calibrate_llm_address(
        model, tokenizer, calib_loader,
        layer_ids=(layer_id,),
        candidate_types=("mlp_delta",),
        hidden_group_size=args.group_size,
        intermediate_group_size=args.group_size * 2,
        heads=2,
    )

    # 3. Zero ablation for all unused groups
    print(f"\n[3/5] Zero ablation for {num_groups - len(used_groups)} unused groups...")
    reference_probs = compute_baseline_probs(model, fast_eval_loader)

    zero_results = []
    for gid in tqdm(range(num_groups), desc="Zero ablation"):
        if gid in used_groups:
            continue
        metrics = eval_zero_group(
            model, fast_eval_loader, reference_probs,
            layer_id, gid, args.group_size,
        )
        zero_results.append({
            "group": gid,
            "kl_zero": metrics.get("avg_kl", 0.0),
            "ppl_zero": metrics["ppl"],
            "acc_zero": metrics["next_token_acc"],
        })

    zero_results.sort(key=lambda x: x["kl_zero"], reverse=True)

    # Save zero results
    zero_scan_path = os.path.join(scans_dir, f"{args.experiment_name}_zero_scan.json")
    with open(zero_scan_path, "w") as f:
        json.dump(zero_results, f, indent=2)

    print(f"\n[Zero Scan] Top 10 candidates by zero KL:")
    for i, r in enumerate(zero_results[:10]):
        print(f"  G{r['group']:2d}: KL={r['kl_zero']:.4f}, PPL={r['ppl_zero']:.2f}, Acc={r['acc_zero']:.4f}")

    # 4. Bucket eval for top candidates
    print(f"\n[4/5] Bucket eval for top {args.top_k_zero} candidates...")
    top_candidates = zero_results[:args.top_k_zero]
    bucket_results = []

    for item in tqdm(top_candidates, desc="Bucket eval"):
        gid = item["group"]
        calib = calib_results[(layer_id, "mlp_delta")]
        addr_idx = calib["addr_idx"][gid]
        addr_mean = calib["addr_mean"][gid]
        addr_std = calib["addr_std"][gid]
        group_means = calib["group_means"][gid]

        # Mean replacement baseline (use PerturbationHook for mean mode)
        hook_mean = PerturbationHook(
            candidate_type="mlp_delta", group_size=args.group_size, group_id=gid,
            mode="mean", mean_vec=group_means, num_bins=64,
        )
        target_mod = get_hook_target(model, layer_id, "mlp_delta")
        handle_mean = target_mod.register_forward_hook(hook_mean)
        try:
            m_mean = compute_model_metrics(model, fast_eval_loader, reference_probs_list=reference_probs)
        finally:
            handle_mean.remove()
        item["kl_mean"] = m_mean.get("avg_kl", 0.0)
        item["ppl_mean"] = m_mean["ppl"]
        item["acc_mean"] = m_mean["next_token_acc"]

        # Build 2D joint bucket table
        bin_idx, targets, _ = collect_teacher_targets(
            model, calib_loader, layer_id, "mlp_delta", gid, args.group_size,
            addr_idx, addr_mean, addr_std, num_bins=args.num_bins,
        )
        joint_table = build_joint_bucket_table(bin_idx, targets, args.num_bins, args.group_size)
        coverage = (joint_table.abs().sum(dim=-1) > 0).sum().item() / (args.num_bins ** 2)

        # Bucket eval (use ReplacementEngine with 2D table)
        bucket_engine = ReplacementEngine(
            model=model, layer_id=layer_id, group_id=gid,
            group_size=args.group_size, addr_idx=addr_idx, addr_mean=addr_mean,
            addr_std=addr_std, table=joint_table, num_bins=args.num_bins,
        )
        m_bucket = eval_bucket_group(model, fast_eval_loader, reference_probs, bucket_engine)
        item["kl_bucket"] = m_bucket.get("avg_kl", 0.0)
        item["ppl_bucket"] = m_bucket["ppl"]
        item["acc_bucket"] = m_bucket["next_token_acc"]
        item["coverage"] = coverage
        item["recovery"] = (item["kl_zero"] - item["kl_bucket"]) / item["kl_zero"] if item["kl_zero"] > 0 else 0.0
        item["table"] = joint_table.cpu()

        bucket_results.append(item)

    # Sort by bucket KL
    bucket_results.sort(key=lambda x: x["kl_bucket"])

    bucket_eval_path = os.path.join(scans_dir, f"{args.experiment_name}_bucket_eval.json")
    with open(bucket_eval_path, "w") as f:
        # Don't save tensors to JSON
        json.dump([{k: v for k, v in r.items() if k != "table"} for r in bucket_results], f, indent=2)

    print(f"\n[Bucket Eval] Top 10 candidates by bucket KL:")
    for i, r in enumerate(bucket_results[:10]):
        print(f"  G{r['group']:2d}: KL={r['kl_bucket']:.4f}, PPL={r['ppl_bucket']:.2f}, "
              f"Acc={r['acc_bucket']:.4f}, Recovery={r['recovery']:.1%}")

    # 5. Progressive multi-group evaluation
    print(f"\n[5/5] Progressive multi-group evaluation...")

    # Build base engines (already used groups)
    base_engines = []
    for gid in sorted(used_groups):
        engine = build_replacement_engine(model, calib_loader, layer_id, gid, args.group_size, calib_results, args.num_bins)
        base_engines.append(engine)

    # Reference probs on full eval set
    full_reference_probs = compute_baseline_probs(model, eval_loader)

    # Evaluate baseline
    baseline_metrics = compute_model_metrics(model, eval_loader, reference_probs_list=full_reference_probs)
    print(f"\n[Baseline] KL={baseline_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={baseline_metrics['ppl']:.2f}, Acc={baseline_metrics['next_token_acc']:.4f}")

    # Evaluate current 6-group
    cur_metrics = eval_multi_group(model, eval_loader, full_reference_probs, base_engines)
    print(f"[Current 6g] KL={cur_metrics.get('avg_kl', 0):.4f}, "
          f"PPL={cur_metrics['ppl']:.2f}, Acc={cur_metrics['next_token_acc']:.4f}")

    # Build candidate engines
    candidate_engines = []
    for r in bucket_results:
        gid = r["group"]
        calib = calib_results[(layer_id, "mlp_delta")]
        engine = ReplacementEngine(
            model=model, layer_id=layer_id, group_id=gid,
            group_size=args.group_size,
            addr_idx=calib["addr_idx"][gid],
            addr_mean=calib["addr_mean"][gid],
            addr_std=calib["addr_std"][gid],
            table=r["table"].to(device),
            num_bins=args.num_bins,
        )
        candidate_engines.append(engine)

    progressive_results = []
    for target in target_counts:
        current_total = len(used_groups) + len(candidate_engines)
        if target > current_total:
            print(f"[Skip] target={target} > available groups={current_total}")
            continue

        selected = base_engines + candidate_engines[:target - len(used_groups)]
        group_ids = [e.group_id for e in selected]

        print(f"\n[Eval {target} groups] {group_ids}")
        metrics = eval_multi_group(model, eval_loader, full_reference_probs, selected)
        print(f"  KL={metrics.get('avg_kl', 0):.4f}, PPL={metrics['ppl']:.2f}, Acc={metrics['next_token_acc']:.4f}")

        # Generation check
        gen_path = os.path.join(generation_dir, f"{args.experiment_name}_g{target}.md")
        save_generation_samples(model, tokenizer, selected, gen_path, num_prompts=3, device=args.device)

        progressive_results.append({
            "num_groups": target,
            "group_ids": group_ids,
            "kl": metrics.get("avg_kl", 0.0),
            "ppl": metrics["ppl"],
            "acc": metrics["next_token_acc"],
            "gen_file": gen_path,
        })

        # Save checkpoint in v3-compatible format (one file per group)
        ckpt_dir = os.path.join(checkpoints_root, f"l{layer_id}", f"g{target}")
        os.makedirs(ckpt_dir, exist_ok=True)
        for e in selected:
            torch.save({
                "layer_id": e.layer_id,
                "group_id": e.group_id,
                "group_size": e.group_size,
                "addr_idx": e.addr_idx.cpu(),
                "addr_mean": e.addr_mean.cpu(),
                "addr_std": e.addr_std.cpu(),
                "table": e.table.cpu(),
                "num_bins": e.num_bins,
                "addr_clip": e.addr_clip,
            }, os.path.join(ckpt_dir, f"replacement_l{e.layer_id}g{e.group_id}.pt"))
        progressive_results[-1]["checkpoint_dir"] = ckpt_dir

    # Save summary
    summary = {
        "model": args.model,
        "layer": layer_id,
        "used_groups": sorted(used_groups),
        "baseline": {
            "kl": baseline_metrics.get("avg_kl", 0.0),
            "ppl": baseline_metrics["ppl"],
            "acc": baseline_metrics["next_token_acc"],
        },
        "current_6group": {
            "kl": cur_metrics.get("avg_kl", 0.0),
            "ppl": cur_metrics["ppl"],
            "acc": cur_metrics["next_token_acc"],
        },
        "progressive": progressive_results,
    }
    summary_path = os.path.join(summaries_dir, f"{args.experiment_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EXPAND RATIO COMPLETE")
    print("=" * 70)
    print(f"Output root: {args.output_root}")
    print(f"  summaries/  — {args.experiment_name}.json")
    print(f"  scans/      — {args.experiment_name}_zero_scan.json, {args.experiment_name}_bucket_eval.json")
    print(f"  generation/ — {args.experiment_name}_g*.md")
    print(f"  checkpoints/l{layer_id}/g*/ — per-group .pt files")


if __name__ == "__main__":
    main()
