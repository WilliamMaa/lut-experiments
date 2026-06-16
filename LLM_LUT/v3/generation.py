"""Standalone generation utilities for v3.

This module duplicates the minimal functionality from v2/r2_auto_eval.py
so that v3 does not depend on v2 source files.
"""

import torch


GENERATION_PROMPTS = [
    "What is the capital of Japan?",
    "Explain the concept of overfitting in machine learning.",
    "If a train travels at 60 km/h for 2 hours, how far does it go?",
    "Write a haiku about autumn.",
    "Write a Python function to reverse a string.",
    "Summarize the theory of evolution in three sentences.",
    "请介绍一下长城的历史。",
    "如何学习一门新的编程语言？",
    "What are the main differences between TCP and UDP?",
]


AUTO_PROMPTS = [{"prompt": p} for p in GENERATION_PROMPTS]


def generate_outputs(model, tokenizer, prompts, num_samples=10, max_new_tokens=128, device="cuda:0"):
    """Generate outputs for a list of prompt dicts.

    Args:
        prompts: list of dicts with key "prompt"
        num_samples: number of generations per prompt
        max_new_tokens: generation length
        device: target device string

    Returns:
        list of lists: outputs[i][j] = j-th sample for i-th prompt
    """
    model.eval()
    all_outputs = []
    for item in prompts:
        messages = [{"role": "user", "content": item["prompt"]}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = item["prompt"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            samples.append(generated)
        all_outputs.append(samples)
    return all_outputs
