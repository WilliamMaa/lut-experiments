#!/usr/bin/env python3
"""
Collect full MoE block input/output data from pre-computed .pt files.

This script loads existing FFN input tensors (e.g., from layer_39_ffn_3000w_0721)
and captures the COMPLETE MLP output (router + routed experts + shared expert).

Usage:
python collect_moe_block_data_from_pt.py \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --layer_idx 39 \
    --input_pt_dir /data/ai2/datasets/lut_distill_dataset/input_qwen3_layer_39_ffn_3000w_0721/39 \
    --output_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe_v2 \
    --max_samples 200000 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import glob

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def load_pt_files(pt_dir: str, max_samples: int) -> List[torch.Tensor]:
    """Load .pt files and return list of tensors."""
    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {pt_dir}")

    print(f"Found {len(pt_files)} .pt files")

    tensors = []
    total_tokens = 0

    for pt_file in tqdm(pt_files, desc="Loading .pt files"):
        if total_tokens >= max_samples:
            break

        try:
            tensor = torch.load(pt_file, map_location="cpu")

            # Handle different tensor shapes
            if tensor.dim() == 1:
                # [hidden_size] -> [1, hidden_size]
                tensor = tensor.unsqueeze(0)
                n_tokens = tensor.shape[0]
            elif tensor.dim() == 2:
                # [seq_len, hidden_size] or [batch, hidden_size]
                n_tokens = tensor.shape[0]
            elif tensor.dim() == 3:
                # [batch, seq_len, hidden] -> [batch*seq_len, hidden]
                batch, seq_len, hidden = tensor.shape
                tensor = tensor.view(-1, hidden)
                n_tokens = batch * seq_len
            else:
                print(f"Warning: {pt_file} has unexpected shape {tensor.shape}, skipping")
                continue

            tensors.append(tensor)
            total_tokens += n_tokens

        except Exception as e:
            print(f"Warning: Failed to load {pt_file}: {e}")
            continue

    print(f"Loaded {len(tensors)} tensors, total {total_tokens} tokens")
    return tensors


class MoEBlockCapture:
    """Hook to capture input and output of a specific MoE block."""

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def __call__(self, module, input, output):
        """Forward hook handler."""
        x = input[0] if isinstance(input, tuple) else input
        if isinstance(output, tuple):
            y = output[0]
        else:
            y = output

        self.inputs.append(x.detach().cpu())
        self.outputs.append(y.detach().cpu())

    def clear(self):
        """Clear captured data."""
        self.inputs.clear()
        self.outputs.clear()


def collect_moe_data_from_pt(
    model,
    input_tensors: List[torch.Tensor],
    layer_idx: int,
    output_dir: Path,
    input_pt_dir: str,
    device: torch.device,
    model_dtype: torch.dtype,
    batch_size: int = 1,
):
    """
    Collect MoE block input/output data from pre-computed .pt tensors.

    Args:
        model: The loaded model
        input_tensors: List of input tensors (each [seq_len, hidden_size])
        layer_idx: Which layer's MLP to capture
        output_dir: Where to save the .pt files
        device: Device to run inference on
        batch_size: Batch size (use 1 for large models to avoid OOM)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = output_dir / "input"
    output_moe_dir = output_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_moe_dir.mkdir(exist_ok=True)

    # Setup hook
    capture = MoEBlockCapture()

    # Get the MLP module and register hook
    try:
        mlp_module = model.model.layers[layer_idx].mlp
    except AttributeError:
        try:
            mlp_module = model.transformer.layers[layer_idx].mlp
        except AttributeError:
            raise ValueError(f"Cannot find MLP module for layer {layer_idx}")

    handle = mlp_module.register_forward_hook(capture)
    print(f"Registered hook on layer {layer_idx} MLP: {type(mlp_module).__name__}")

    model.eval()
    file_counter = 0
    total_tokens = 0

    with torch.no_grad():
        for tensor in tqdm(input_tensors, desc="Processing tensors"):
            # Clear previous captures
            capture.clear()

            # Move tensor to device and convert dtype
            tensor = tensor.to(device=device, dtype=model_dtype)

            # Forward through the model up to the target layer
            # We need to pass through all previous layers first
            # For efficiency, we'll do a forward pass and let the hook capture

            try:
                # tensor is now [N, hidden_size] from load_pt_files
                # Reshape to [1, N, hidden] for layernorm (expects 3D)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)  # [1, N, hidden]

                # Best approach: manually call the layer components
                layer = model.model.layers[layer_idx]

                # Apply input layernorm
                normalized = layer.input_layernorm(tensor)

                # Pass through MLP (this triggers the hook)
                mlp_output = layer.mlp(normalized)

                # The hook should have captured normalized and mlp_output
                if capture.inputs and capture.outputs:
                    x_captured = torch.cat(capture.inputs, dim=0)
                    y_captured = torch.cat(capture.outputs, dim=0)

                    # Validate shapes
                    assert x_captured.shape == y_captured.shape, \
                        f"Shape mismatch: input {x_captured.shape} vs output {y_captured.shape}"

                    # Reshape to 2D [total_tokens, hidden_size] if needed
                    if x_captured.dim() == 3:
                        # [batch, seq_len, hidden] -> [batch*seq_len, hidden]
                        batch, seq_len, hidden = x_captured.shape
                        x_captured = x_captured.view(-1, hidden)
                        y_captured = y_captured.view(-1, hidden)

                    # Save
                    input_path = input_dir / f"sample_{file_counter:06d}.pt"
                    output_path = output_moe_dir / f"sample_{file_counter:06d}.pt"

                    torch.save(x_captured, input_path)
                    torch.save(y_captured, output_path)

                    file_counter += 1
                    total_tokens += x_captured.shape[0]

                capture.clear()

            except Exception as e:
                print(f"Warning: Failed to process tensor: {e}")
                continue

            # Periodic cleanup
            if file_counter % 100 == 0:
                torch.cuda.empty_cache()

    # Remove hook
    handle.remove()
    print(f"\nCollected {file_counter} samples, {total_tokens} total tokens")
    print(f"Input files saved to: {input_dir}")
    print(f"Output files saved to: {output_moe_dir}")

    # Save metadata
    metadata = {
        "layer_idx": layer_idx,
        "num_samples": file_counter,
        "total_tokens": total_tokens,
        "input_dir": str(input_dir),
        "output_dir": str(output_moe_dir),
        "model_class": type(model).__name__,
        "mlp_class": type(mlp_module).__name__,
        "source_pt_dir": str(input_pt_dir),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return file_counter


def main():
    parser = argparse.ArgumentParser(
        description="Collect full MoE block data from pre-computed .pt files"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the model (HF name or local path)")
    parser.add_argument("--layer_idx", type=int, required=True,
                        help="Layer index to capture (e.g., 39 for last layer)")
    parser.add_argument("--input_pt_dir", required=True,
                        help="Directory containing pre-computed .pt input files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for collected data")
    parser.add_argument("--max_samples", type=int, default=200000,
                        help="Maximum number of samples (in tokens)")
    parser.add_argument("--device_map", default="balanced_low_0",
                        help="Device map for model loading")
    parser.add_argument("--torch_dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model")
    args = parser.parse_args()

    if args.device_map == "auto":
        raise ValueError("device_map='auto' is forbidden. Use 'balanced_low_0'.")

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading model: {args.model_path}")
    print(f"  dtype: {args.torch_dtype}")
    print(f"  device_map: {args.device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    model.eval()

    # Print model info
    print(f"Model loaded: {type(model).__name__}")
    if hasattr(model.config, "num_hidden_layers"):
        print(f"Total layers: {model.config.num_hidden_layers}")
    if hasattr(model.config, "hidden_size"):
        print(f"Hidden size: {model.config.hidden_size}")

    # Determine device (for device_map models)
    device = next(model.parameters()).device
    print(f"Model device: {device}")

    # Load input tensors from .pt files
    input_tensors = load_pt_files(args.input_pt_dir, args.max_samples)

    # Collect MoE data
    num_collected = collect_moe_data_from_pt(
        model=model,
        input_tensors=input_tensors,
        layer_idx=args.layer_idx,
        output_dir=Path(args.output_dir),
        input_pt_dir=args.input_pt_dir,
        device=device,
        model_dtype=dtype,
    )

    print(f"\n{'='*60}")
    print(f"Data collection complete!")
    print(f"Total samples: {num_collected}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

    print("\nNext steps:")
    print("1. Run build_lut_ffn_output.py with:")
    print(f"   --dataset_dir {args.output_dir}/input")
    print(f"   --output_dataset_dir {args.output_dir}/output")


if __name__ == "__main__":
    main()
