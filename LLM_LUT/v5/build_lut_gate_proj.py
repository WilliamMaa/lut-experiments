"""
Build LUT for partial gate_proj replacement.

Build target is the **post-SiGLU output**:
    y = SiLU(gate_proj(x)) * up_proj(x)

The LUT itself predicts the pre-activation gate_proj output for the replaced
group; the post-SiGLU error is used to guide address construction and evaluation.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data import load_jsonl
from address import Address2D, AddressHighOrderRandom, AddressGreedyTree
from lut import LUTGroup
from hybrid_gate_proj_engine import HybridGateProjEngine


V0_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v0", "data")


def tokenize_texts(tokenizer, texts, max_seq_len):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )


def _run_forward_batches(model, tokens, batch_size, device):
    """Run model forward in small batches to reduce peak GPU memory."""
    n = tokens["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        batch = {k: v[i:i + batch_size].to(device) for k, v in tokens.items()}
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(**batch)


def capture_gate_proj_residual(model, tokenizer, layer_id, calib_texts, eval_texts,
                               max_seq_len, device, batch_size=32):
    """Capture gate_proj input, gate_proj output, and up_proj output for a layer."""
    layer = model.model.layers[layer_id]
    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    intermediate_size = gate_proj.weight.shape[0]

    captured = {
        "x": [], "gate_out": [], "up_out": [],
    }

    def gate_hook(module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        captured["x"].append(x.detach())
        captured["gate_out"].append(output.detach())

    def up_hook(module, input, output):
        captured["up_out"].append(output.detach())

    calib_texts_plain = [t["text"] if isinstance(t, dict) else t for t in calib_texts]
    eval_texts_plain = [t["text"] if isinstance(t, dict) else t for t in eval_texts]
    calib_tok = tokenize_texts(tokenizer, calib_texts_plain, max_seq_len)
    eval_tok = tokenize_texts(tokenizer, eval_texts_plain, max_seq_len)

    handle_gate = gate_proj.register_forward_hook(gate_hook)
    handle_up = up_proj.register_forward_hook(up_hook)

    try:
        model.eval()
        _run_forward_batches(model, calib_tok, batch_size, device)
        handle_gate.remove()
        handle_up.remove()
        handle_gate = gate_proj.register_forward_hook(gate_hook)
        handle_up = up_proj.register_forward_hook(up_hook)
        _run_forward_batches(model, eval_tok, batch_size, device)
    finally:
        handle_gate.remove()
        handle_up.remove()

    def flatten(seq):
        return torch.cat([t.view(-1, t.shape[-1]) for t in seq], dim=0)

    # captured order: calib batches first, then eval batches
    n_calib = len(captured["x"]) // 2
    return {
        "calib_x": flatten(captured["x"][:n_calib]),
        "calib_gate": flatten(captured["gate_out"][:n_calib]),
        "calib_up": flatten(captured["up_out"][:n_calib]),
        "eval_x": flatten(captured["x"][n_calib:]),
        "eval_gate": flatten(captured["gate_out"][n_calib:]),
        "eval_up": flatten(captured["up_out"][n_calib:]),
        "intermediate_size": intermediate_size,
    }


def build_tree_address(calib_x, group_target, num_bits, channels_per_bit, seed,
                       num_candidates, min_samples, max_samples):
    address = AddressGreedyTree(
        input_dim=calib_x.shape[-1],
        num_bits=num_bits,
        channels_per_bit=channels_per_bit,
        seed=seed,
    )
    address.build(
        calib_x,
        group_target,
        num_candidates=num_candidates,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    return address


def evaluate_gate_group(eval_x, eval_gate, eval_up, address, lut_group, gid, group_size):
    """Evaluate a gate_proj group using post-SiGLU MSE."""
    with torch.no_grad():
        indices = address.compute_indices(eval_x.unsqueeze(0)).view(-1, address.num_tables)
        lut_out = lut_group(indices)
        g_start = gid * group_size
        g_end = g_start + group_size
        up_group = eval_up[:, g_start:g_end]
        gate_true = eval_gate[:, g_start:g_end]
        post_true = F.silu(gate_true) * up_group
        post_pred = F.silu(lut_out) * up_group
        mse = F.mse_loss(post_pred, post_true, reduction="mean").item()
        var = post_true.var().item()
        return {
            "mse": mse,
            "rmse": mse ** 0.5,
            "relative_mse": mse / (var + 1e-8),
        }


def select_2d_address(calib_x, target, group_size, num_bins, num_candidates=8):
    """Pick a good 2-channel address pair using the group target."""
    import itertools
    hidden_size = calib_x.shape[-1]
    channel_var = calib_x.var(dim=0)
    top_channels = torch.topk(channel_var, k=min(num_candidates * 2, hidden_size)).indices.tolist()
    pairs = list(itertools.combinations(top_channels[:num_candidates], 2))[:20]

    best_rmse = float("inf")
    best_pair = None
    for c1, c2 in pairs:
        addr_idx = torch.tensor([c1, c2], device=calib_x.device)
        address = Address2D(
            addr_idx,
            calib_x[:, addr_idx].mean(dim=0),
            calib_x[:, addr_idx].std(dim=0),
            num_bins=num_bins,
        )
        indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, 1)
        lut = LUTGroup(1, address.num_entries, group_size, device=calib_x.device)
        lut.initialize_from_calibration(indices, target)
        rec = lut(indices).squeeze(1)
        rmse = F.mse_loss(rec, target, reduction="mean").item() ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_pair = (c1, c2)

    addr_idx = torch.tensor(best_pair, device=calib_x.device)
    return Address2D(
        addr_idx,
        calib_x[:, addr_idx].mean(dim=0),
        calib_x[:, addr_idx].std(dim=0),
        num_bins=num_bins,
    )


def build_gate_proj_layer(model, tokenizer, layer_id, group_ids, group_size,
                          address_mode, num_bins, num_bits, channels_per_bit,
                          tree_candidates, tree_min_samples, tree_max_samples,
                          calib_texts, eval_texts, max_seq_len, device, output_root,
                          capture_batch_size=32):
    """Build gate_proj LUTs for a single layer on the current student model."""
    data = capture_gate_proj_residual(
        model, tokenizer, layer_id, calib_texts, eval_texts,
        max_seq_len, device, batch_size=capture_batch_size,
    )
    calib_x = data["calib_x"]
    calib_gate = data["calib_gate"]
    calib_up = data["calib_up"]
    eval_x = data["eval_x"]
    eval_gate = data["eval_gate"]
    eval_up = data["eval_up"]
    intermediate_size = data["intermediate_size"]
    hidden_size = calib_x.shape[-1]

    group_results = []
    save_dir = os.path.join(output_root, "checkpoints", f"l{layer_id}", "gate_proj", f"g{len(group_ids)}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gid in tqdm(group_ids, desc=f"L{layer_id} gate_proj groups", leave=False):
        g_start = gid * group_size
        g_end = g_start + group_size
        # Address is built on post-SiGLU target
        post_target = F.silu(calib_gate[:, g_start:g_end]) * calib_up[:, g_start:g_end]
        # LUT is initialized with pre-activation gate output
        gate_target = calib_gate[:, g_start:g_end]

        seed = layer_id * 1000 + gid
        if address_mode == "2d":
            addr_idx, _ = select_2d_address(calib_x, post_target, group_size, num_bins)
            address = Address2D(
                addr_idx=addr_idx,
                addr_mean=calib_x[:, addr_idx].mean(dim=0),
                addr_std=calib_x[:, addr_idx].std(dim=0),
                num_bins=num_bins,
            )
        elif address_mode == "high_order":
            address = AddressHighOrderRandom(
                input_dim=hidden_size,
                num_tables=1,
                num_bits=num_bits,
                channels_per_bit=channels_per_bit,
                seed=seed,
            )
            address.fit_calibration(calib_x.unsqueeze(0))
        else:  # tree
            address = build_tree_address(
                calib_x, post_target, num_bits, channels_per_bit, seed,
                tree_candidates, tree_min_samples, tree_max_samples,
            )

        indices = address.compute_indices(calib_x.unsqueeze(0)).view(-1, address.num_tables)
        lut_group = LUTGroup(address.num_tables, address.num_entries, group_size, device=calib_x.device)
        lut_group.initialize_from_calibration(indices, gate_target)

        eval_metrics = evaluate_gate_group(
            eval_x, eval_gate, eval_up, address, lut_group, gid, group_size,
        )
        print(f"  [L{layer_id} gate g{gid:3d}] rel_mse={eval_metrics['relative_mse']:.4f}, "
              f"rmse={eval_metrics['rmse']:.4f}")

        ckpt_path = os.path.join(save_dir, f"replacement_l{layer_id}g{gid}.pt")
        state = {
            "layer_id": layer_id,
            "group_id": gid,
            "group_size": group_size,
            "lut_table": lut_group.table.detach().cpu(),
        }
        if isinstance(address, Address2D):
            state.update({
                "address_type": "2d",
                "addr_idx": address.addr_idx,
                "addr_mean": address.addr_mean,
                "addr_std": address.addr_std,
                "num_bins": address.num_bins,
                "addr_clip": address.addr_clip,
            })
        elif isinstance(address, AddressHighOrderRandom):
            state.update({
                "address_type": "high_order",
                "input_dim": address.input_dim,
                "num_tables": address.num_tables,
                "num_bits": address.num_bits,
                "channels_per_bit": address.channels_per_bit,
                "channel_idx": address.channel_idx,
                "signs": address.signs,
                "addr_mean": address.addr_mean,
                "addr_std": address.addr_std,
            })
        elif isinstance(address, AddressGreedyTree):
            state.update({
                "address_type": "tree",
                "input_dim": address.input_dim,
                "num_bits": address.num_bits,
                "channels_per_bit": address.channels_per_bit,
                "tree_state": address.serialize(),
            })
        torch.save(state, ckpt_path)

        group_results.append({"group_id": gid, **eval_metrics})

    return group_results, save_dir


def parse_configs(arg_str: str) -> List[Tuple[int, int, List[int]]]:
    configs = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        layer_id = int(tokens[0])
        rest_after_layer = tokens[1].split(";")
        count = int(rest_after_layer[0])
        if len(rest_after_layer) >= 2:
            group_ids = [int(x) for x in rest_after_layer[1:]]
        elif len(tokens) >= 3:
            group_ids = [int(x) for x in tokens[2].split(";")]
        else:
            group_ids = list(range(count))
        if len(group_ids) != count:
            raise ValueError(f"Config {part}: group id count mismatch")
        configs.append((layer_id, count, group_ids))
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--configs", required=True,
                        help="layer:count[:group_ids] for gate_proj")
    parser.add_argument("--address_mode", default="tree", choices=["2d", "high_order", "tree"])
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--num_bits", type=int, default=10)
    parser.add_argument("--channels_per_bit", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--calib_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="../v5/outputs_gate_proj")
    parser.add_argument("--tree_candidates", type=int, default=32)
    parser.add_argument("--tree_min_samples", type=int, default=32)
    parser.add_argument("--tree_max_samples", type=int, default=16384)
    parser.add_argument("--capture_batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    configs = parse_configs(args.configs)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_texts = load_jsonl(os.path.join(V0_DATA_DIR, "calib.jsonl"))[:args.calib_size]
    eval_texts = load_jsonl(os.path.join(V0_DATA_DIR, "eval.jsonl"))[:args.eval_size]

    output_root = args.output_root
    Path(output_root).mkdir(parents=True, exist_ok=True)

    all_results = []
    for layer_id, count, group_ids in configs:
        print(f"\n[Layer {layer_id}] Building gate_proj groups {group_ids}")
        layer_results, save_dir = build_gate_proj_layer(
            model, tokenizer, layer_id, group_ids, args.group_size,
            args.address_mode, args.num_bins, args.num_bits, args.channels_per_bit,
            args.tree_candidates, args.tree_min_samples, args.tree_max_samples,
            calib_texts, eval_texts, args.max_seq_len, device, output_root,
            capture_batch_size=args.capture_batch_size,
        )
        all_results.append({
            "layer_id": layer_id,
            "group_ids": group_ids,
            "save_dir": save_dir,
            "groups": layer_results,
        })

    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "address_mode": args.address_mode,
            "num_bits": args.num_bits,
            "channels_per_bit": args.channels_per_bit,
            "group_size": args.group_size,
            "configs": configs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved summary: {summary_path}")
