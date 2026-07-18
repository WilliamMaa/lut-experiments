#!/usr/bin/env python3
"""
Generation evaluation for v5 joint fine-tuned checkpoints.

Load a specific epoch's trained weights + LUTs, run generation on a set of
prompts, and save the outputs for side-by-side comparison.

Example:
    cd LLM_LUT/v5
    python generate_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --down_configs "15:9,16:3,..." \
        --o_configs "15:16,..." \
        --gate_configs "15:97,..." \
        --checkpoint_dir results/finetune_joint_phase4_down_o_gate_5pct \
        --epochs 8,10 \
        --prompts prompts.txt \
        --max_new_tokens 80 \
        --output generation_phase4_epoch_8_10.json
"""

import os
import glob
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_joint import parse_configs, build_address
from engine import HybridPartialEngine, HybridOProjEngine
from hybrid_gate_proj_engine import HybridGateProjEngine
from lut import LUTGroup


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def get_proj_module(model, layer_id, proj_type):
    layer = model.model.layers[layer_id]
    if proj_type == "down_proj":
        return layer.mlp.down_proj
    elif proj_type == "gate_proj":
        return layer.mlp.gate_proj
    elif proj_type == "o_proj":
        return layer.self_attn.o_proj
    raise ValueError(f"Unknown proj_type: {proj_type}")


def make_engine(model, layer_id, proj_type, group_size):
    if proj_type == "down_proj":
        return HybridPartialEngine(model, layer_id, group_size=group_size)
    elif proj_type == "gate_proj":
        return HybridGateProjEngine(model, layer_id, group_size=group_size)
    elif proj_type == "o_proj":
        return HybridOProjEngine(model, layer_id, group_size=group_size)
    raise ValueError(f"Unknown proj_type: {proj_type}")


def load_proj_weight(module, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    module.weight.data = ckpt.to(module.weight.device, dtype=module.weight.dtype)


def load_lut_groups(engine, layer_id, lut_dir, device):
    pattern = os.path.join(lut_dir, f"replacement_l{layer_id}g*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No LUT checkpoints in {lut_dir}")
    for p in paths:
        name = os.path.basename(p)
        prefix = f"replacement_l{layer_id}g"
        suffix = ".pt"
        gid = int(name[len(prefix):-len(suffix)])
        ckpt = torch.load(p, map_location="cpu")
        address = build_address(ckpt)
        table = ckpt["lut_table"]
        lut_group = LUTGroup(table.shape[0], table.shape[1], table.shape[2], init_table=table)
        lut_group = lut_group.to(device)
        engine.add_group(gid, address, lut_group)


def install_epoch(model, configs, proj_type, epoch, checkpoint_dir, group_size, device):
    engines = []
    for layer_id, _ in configs:
        module = get_proj_module(model, layer_id, proj_type)
        weight_path = os.path.join(
            checkpoint_dir, f"l{layer_id}_epoch{epoch}_{proj_type}.pt"
        )
        load_proj_weight(module, weight_path)

        engine = make_engine(model, layer_id, proj_type, group_size)
        if proj_type == "down_proj":
            lut_dir = os.path.join(checkpoint_dir, f"l{layer_id}_epoch{epoch}_down_lut")
        elif proj_type == "gate_proj":
            lut_dir = os.path.join(checkpoint_dir, f"l{layer_id}_epoch{epoch}_gate_lut")
        elif proj_type == "o_proj":
            lut_dir = os.path.join(checkpoint_dir, f"l{layer_id}_epoch{epoch}_o_lut")
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")
        load_lut_groups(engine, layer_id, lut_dir, device)
        engine.install()
        engines.append(engine)
    return engines


def generate_for_prompts(model, tokenizer, prompts, max_new_tokens):
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type=model.device.type, dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "output": generated})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--down_configs", default="")
    parser.add_argument("--o_configs", default="")
    parser.add_argument("--gate_configs", default="")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", required=True,
                        help="Comma-separated epochs to evaluate, e.g. '8,10'")
    parser.add_argument("--prompts", default="",
                        help="Text file with one prompt per line")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--output", default="generation_results.json")
    parser.add_argument("--baseline", action="store_true",
                        help="Also generate with the original unmodified model")
    args = parser.parse_args()

    device = torch.device(args.device)

    down_configs = parse_configs(args.down_configs) if args.down_configs else []
    o_configs = parse_configs(args.o_configs) if args.o_configs else []
    gate_configs = parse_configs(args.gate_configs) if args.gate_configs else []

    prompts = []
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "请简单介绍一下自己。",
            "The future of artificial intelligence is",
            "如何学习机器学习？",
            "Explain the concept of neural networks in one paragraph.",
        ]

    print("[1/2] Loading base model...")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Save original weights so we can restore them between epochs.
    original_weights = {}
    for layer_id, _ in down_configs:
        original_weights[(layer_id, "down_proj")] = get_proj_module(model, layer_id, "down_proj").weight.data.clone()
    for layer_id, _ in gate_configs:
        original_weights[(layer_id, "gate_proj")] = get_proj_module(model, layer_id, "gate_proj").weight.data.clone()
    for layer_id, _ in o_configs:
        original_weights[(layer_id, "o_proj")] = get_proj_module(model, layer_id, "o_proj").weight.data.clone()

    epochs = [int(x.strip()) for x in args.epochs.split(",")]
    all_results = {"prompts": prompts, "baseline": None, "epochs": {}}

    if args.baseline:
        print("\n[Baseline] Generating with original model...")
        baseline_results = generate_for_prompts(model, tokenizer, prompts, args.max_new_tokens)
        all_results["baseline"] = baseline_results
        for item in baseline_results:
            print(f"    Prompt: {item['prompt'][:60]}...")
            print(f"    Output: {item['output'][:200]}")

    print(f"\n[2/2] Generating for epochs {epochs}...")
    for epoch in epochs:
        print(f"\n  Epoch {epoch}")
        engines = []
        engines.extend(install_epoch(model, down_configs, "down_proj", epoch, args.checkpoint_dir, args.group_size, device))
        engines.extend(install_epoch(model, gate_configs, "gate_proj", epoch, args.checkpoint_dir, args.group_size, device))
        engines.extend(install_epoch(model, o_configs, "o_proj", epoch, args.checkpoint_dir, args.group_size, device))

        epoch_results = generate_for_prompts(model, tokenizer, prompts, args.max_new_tokens)
        all_results["epochs"][str(epoch)] = epoch_results

        for item in epoch_results:
            print(f"    Prompt: {item['prompt'][:60]}...")
            print(f"    Output: {item['output'][:200]}")

        # Uninstall engines and restore original weights.
        for engine in engines:
            engine.uninstall()
        for (layer_id, proj_type), w in original_weights.items():
            module = get_proj_module(model, layer_id, proj_type)
            module.weight.data.copy_(w)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
