#!/usr/bin/env python3
"""
Collect full MoE block input/output data for LUT training.

This script captures the COMPLETE MLP output (router + routed experts + shared expert)
instead of a single expert approximation.

Usage:
  python collect_moe_block_data.py \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --layer_idx 39 \
    --calib_file /path/to/calib.jsonl \
    --output_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe \
    --max_samples 200000 \
    --device_map balanced_low_0
python collect_moe_block_data.py \
    --model_path /data/downloads/Qwen3.6/models/Qwen3.6-35B-A3B \
    --layer_idx 39 \
    --calib_file /data/ai2/datasets/calib_data/c4_sample_300k.jsonl \
    --output_dir /data/ai2/datasets/lut_distill_dataset/layer39_full_moe \
    --max_samples 200000 \
    --max_seq_length 2048 \
    --device_map balanced_low_0 \
    --torch_dtype bfloat16
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_calibration_texts(calib_file: str, max_samples: int) -> List[str]:
    """Load calibration texts from JSONL file."""
    texts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Try common text fields
            text = obj.get("text", obj.get("content", obj.get("sentence", obj.get("input", ""))))
            if text:
                texts.append(text)
            if len(texts) >= max_samples:
                break
    print(f"Loaded {len(texts)} calibration texts")
    return texts


class MoEBlockCapture:
    """Hook to capture input and output of a specific MoE block."""

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def __call__(self, module, input, output):
        """Forward hook handler."""
        # input is a tuple, extract the first element
        x = input[0] if isinstance(input, tuple) else input

        # Handle different output formats
        if isinstance(output, tuple):
            # Some MoE implementations return (output, aux_loss)
            y = output[0]
        else:
            y = output

        # Detach and move to CPU to save GPU memory
        self.inputs.append(x.detach().cpu())
        self.outputs.append(y.detach().cpu())

    def clear(self):
        """Clear captured data."""
        self.inputs.clear()
        self.outputs.clear()


def collect_moe_data(
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    output_dir: Path,
    batch_size: int = 1,
    max_seq_length: int = 2048,
):
    """
    Collect MoE block input/output data.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        texts: List of calibration texts
        layer_idx: Which layer's MLP to capture
        output_dir: Where to save the .pt files
        batch_size: Batch size for processing (use 1 for large models)
        max_seq_length: Maximum sequence length
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
    # For Qwen3 models, the path is typically model.model.layers[layer_idx].mlp
    try:
        mlp_module = model.model.layers[layer_idx].mlp
    except AttributeError:
        # Fallback: try different paths
        try:
            mlp_module = model.transformer.layers[layer_idx].mlp
        except AttributeError:
            raise ValueError(f"Cannot find MLP module for layer {layer_idx}. "
                           f"Please check model architecture.")

    handle = mlp_module.register_forward_hook(capture)
    print(f"Registered hook on layer {layer_idx} MLP: {type(mlp_module).__name__}")

    model.eval()
    file_counter = 0

    with torch.no_grad():
        for i, text in enumerate(tqdm(texts, desc="Collecting MoE data")):
            # Clear previous captures
            capture.clear()

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,  # No padding for single sequences
            )

            # Move to model's device
            input_ids = inputs["input_ids"]
            if hasattr(model, "device"):
                input_ids = input_ids.to(model.device)
            else:
                # For device_map models, let accelerate handle it
                pass

            # Forward pass - this will trigger the hook
            try:
                _ = model(input_ids, use_cache=False)
            except Exception as e:
                print(f"Warning: Failed to process text {i}: {e}")
                continue

            # Save captured data
            if capture.inputs and capture.outputs:
                # Concatenate all tokens from this sample
                x_tensor = torch.cat(capture.inputs, dim=0)  # [seq_len, hidden_size]
                y_tensor = torch.cat(capture.outputs, dim=0)  # [seq_len, hidden_size]

                # Validate shapes match
                assert x_tensor.shape == y_tensor.shape, \
                    f"Shape mismatch: input {x_tensor.shape} vs output {y_tensor.shape}"

                # Save as .pt files (save each sequence separately)
                input_path = input_dir / f"sample_{file_counter:06d}.pt"
                output_path = output_moe_dir / f"sample_{file_counter:06d}.pt"

                torch.save(x_tensor, input_path)
                torch.save(y_tensor, output_path)

                file_counter += 1

                # Clear to free memory
                capture.clear()
                del x_tensor, y_tensor

            # Periodic cleanup
            if i % 100 == 0:
                torch.cuda.empty_cache()

    # Remove hook
    handle.remove()
    print(f"\nCollected {file_counter} samples")
    print(f"Input files saved to: {input_dir}")
    print(f"Output files saved to: {output_moe_dir}")

    # Save metadata
    metadata = {
        "layer_idx": layer_idx,
        "num_samples": file_counter,
        "input_dir": str(input_dir),
        "output_dir": str(output_moe_dir),
        "model_class": type(model).__name__,
        "mlp_class": type(mlp_module).__name__,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return file_counter


def main():
    parser = argparse.ArgumentParser(
        description="Collect full MoE block input/output data for LUT training"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the model (HF name or local path)")
    parser.add_argument("--layer_idx", type=int, required=True,
                        help="Layer index to capture (e.g., 39 for last layer)")
    parser.add_argument("--calib_file", required=True,
                        help="Path to calibration JSONL file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for collected data")
    parser.add_argument("--max_samples", type=int, default=200000,
                        help="Maximum number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (recommend 1 for large models)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--device_map", default="balanced_low_0",
                        help="Device map for model loading (e.g., balanced_low_0, auto)")
    parser.add_argument("--torch_dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model")
    args = parser.parse_args()

    if args.device_map == "auto":
        raise ValueError("device_map='auto' is forbidden. Use 'balanced_low_0' or explicit map.")

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading model: {args.model_path}")
    print(f"  dtype: {args.torch_dtype}")
    print(f"  device_map: {args.device_map}")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
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

    # Load calibration texts
    texts = load_calibration_texts(args.calib_file, args.max_samples)

    # Collect data
    num_collected = collect_moe_data(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=args.layer_idx,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    print(f"\n{'='*60}")
    print(f"Data collection complete!")
    print(f"Total samples: {num_collected}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

    # Print next steps
    print("\nNext steps:")
    print("1. Run build_lut_ffn_output.py with:")
    print(f"   --dataset_dir {args.output_dir}/input")
    print(f"   --output_dataset_dir {args.output_dir}/output")
    print("2. Use single expert checkpoint as teacher (for FFN structure)")
    print("3. LUT will learn to approximate full MoE block output")


if __name__ == "__main__":
    main()
