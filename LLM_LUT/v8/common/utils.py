#!/usr/bin/env python3
"""Utility helpers for v8 evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: str = "bfloat16",
    device: str = "cuda:0",
    device_map: str = None,
):
    """Load a causal LM and tokenizer for v8 evaluation.

    Args:
        model_path: HuggingFace model name or local path.
        torch_dtype: "float16", "bfloat16", or "float32".
        device: Single device when device_map is None.
        device_map: HuggingFace device_map string. Must NOT be "auto".

    Returns:
        (model, tokenizer, effective_device)
    """
    if device_map == "auto":
        raise ValueError(
            "device_map='auto' is forbidden by project red line. "
            "Use an explicit map like 'balanced_low_0'."
        )

    dtype = getattr(torch, torch_dtype)

    print(f"Loading model: {model_path}")
    if device_map is not None:
        print(f"  device_map={device_map}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        effective_device = next(model.parameters()).device
        print(f"  first-layer device is {effective_device}")
    else:
        effective_device = torch.device(device)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(effective_device)
        print(f"  loaded on {effective_device}")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, effective_device
