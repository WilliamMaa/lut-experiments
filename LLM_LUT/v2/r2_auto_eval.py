"""Automatic generation evaluation for Scaling-R1.

Paired original vs replacement, with simple exact-match checks.
No subjective analysis — only pass/fail / same/worse/collapse.
"""

import torch
import re


AUTO_PROMPTS = [
    {
        "id": "capital_japan",
        "prompt": "What is the capital of Japan? Answer with one word.",
        "checks": [
            lambda t: "tokyo" in t.lower(),
            lambda t: "东京" in t,
        ],
    },
    {
        "id": "train_distance",
        "prompt": "A train travels at 60 km/h for 2 hours. How far does it go?",
        "checks": [
            lambda t: "120" in t,
            lambda t: "km" in t.lower(),
        ],
    },
    {
        "id": "reverse_string",
        "prompt": "Write a Python one-liner to reverse a string s.",
        "checks": [
            lambda t: "[::-1]" in t,
            lambda t: "reversed" in t.lower(),
        ],
    },
    {
        "id": "evolution",
        "prompt": "Summarize the theory of evolution in one sentence.",
        "checks": [
            lambda t: len(t) > 20 and len(t) < 300,
        ],
    },
    {
        "id": "multiplication",
        "prompt": "What is 17 times 6?",
        "checks": [
            lambda t: "102" in t,
        ],
    },
    {
        "id": "great_wall",
        "prompt": "请用一句话介绍长城。",
        "checks": [
            lambda t: len(t) > 10,
            lambda t: any(kw in t for kw in ["长城", "古代", "建筑", "中国", "防御"]),
        ],
    },
]


def check_prompt(item, text):
    """Run all checks for a prompt. Returns dict with results."""
    results = []
    for check in item["checks"]:
        try:
            results.append(check(text))
        except Exception:
            results.append(False)
    return {
        "prompt_id": item["id"],
        "prompt": item["prompt"],
        "output": text,
        "checks_passed": sum(results),
        "checks_total": len(results),
        "passed": all(results) if results else False,
    }


def compute_repetition_rate(text, n=3):
    """Compute n-gram repetition rate."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = {}
    total = 0
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i+n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    repeats = sum(1 for v in ngrams.values() if v > 1)
    return repeats / len(ngrams)


def compute_avg_length(outputs):
    """Average output length in characters."""
    if not outputs:
        return 0
    return sum(len(o) for o in outputs) / len(outputs)


def run_auto_eval(original_outputs, replacement_outputs):
    """
    Args:
        original_outputs: list of list of strings (N samples per prompt)
        replacement_outputs: list of list of strings
    Returns:
        dict with comparison stats
    """
    assert len(original_outputs) == len(replacement_outputs) == len(AUTO_PROMPTS)
    num_samples = len(original_outputs[0])

    per_prompt_stats = []
    for item, orig_samples, repl_samples in zip(AUTO_PROMPTS, original_outputs, replacement_outputs):
        orig_pass_rate = sum(1 for t in orig_samples if check_prompt(item, t)["passed"]) / num_samples
        repl_pass_rate = sum(1 for t in repl_samples if check_prompt(item, t)["passed"]) / num_samples
        orig_reps = [compute_repetition_rate(t) for t in orig_samples]
        repl_reps = [compute_repetition_rate(t) for t in repl_samples]
        orig_lens = [len(t) for t in orig_samples]
        repl_lens = [len(t) for t in repl_samples]

        per_prompt_stats.append({
            "prompt_id": item["id"],
            "orig_pass_rate": orig_pass_rate,
            "repl_pass_rate": repl_pass_rate,
            "orig_avg_rep": sum(orig_reps) / num_samples,
            "repl_avg_rep": sum(repl_reps) / num_samples,
            "orig_avg_len": sum(orig_lens) / num_samples,
            "repl_avg_len": sum(repl_lens) / num_samples,
        })

    avg_orig_pass = sum(s["orig_pass_rate"] for s in per_prompt_stats) / len(per_prompt_stats)
    avg_repl_pass = sum(s["repl_pass_rate"] for s in per_prompt_stats) / len(per_prompt_stats)
    avg_orig_rep = sum(s["orig_avg_rep"] for s in per_prompt_stats) / len(per_prompt_stats)
    avg_repl_rep = sum(s["repl_avg_rep"] for s in per_prompt_stats) / len(per_prompt_stats)

    # Determine status
    if avg_repl_pass < avg_orig_pass - 0.3:
        status = "WORSE"
    elif avg_repl_rep > 0.5 or avg_repl_pass < 0.3:
        status = "COLLAPSE"
    else:
        status = "SAME"

    return {
        "status": status,
        "num_samples": num_samples,
        "original": {
            "avg_pass_rate": avg_orig_pass,
            "avg_rep": avg_orig_rep,
            "avg_len": sum(s["orig_avg_len"] for s in per_prompt_stats) / len(per_prompt_stats),
        },
        "replacement": {
            "avg_pass_rate": avg_repl_pass,
            "avg_rep": avg_repl_rep,
            "avg_len": sum(s["repl_avg_len"] for s in per_prompt_stats) / len(per_prompt_stats),
        },
        "per_prompt": per_prompt_stats,
    }


def generate_outputs(model, tokenizer, prompts, num_samples=10, max_new_tokens=128, device="cuda:0"):
    """
    Generate outputs for a list of prompt dicts.

    Args:
        num_samples: number of generations per prompt
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
