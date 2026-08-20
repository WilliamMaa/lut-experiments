#!/usr/bin/env python3
"""Metric computation helpers for v8 evaluation.

Includes model-quality, functional, attention-behavior and system-level metrics.
Designed to be used by both VQK and KV Cache Compression experiments.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def compute_ppl(model, tokenizer, texts, device, max_length: int = 512):
    """Compute perplexity over a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= 1:
            continue
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.shape[1]
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def compute_logit_metrics(
    teacher_model,
    student_model,
    tokenizer,
    texts,
    device,
    max_length: int = 512,
    temperature: float = 1.0,
):
    """Compute logit-based functional metrics between a teacher and a compressed student.

    Returns dict with:
        - avg_kl: average KL(teacher || student)
        - top1_agreement: fraction of positions where argmax matches
        - top5_agreement: fraction of positions where top-5 sets overlap
        - avg_teacher_entropy, avg_student_entropy
    """
    teacher_model.eval()
    student_model.eval()

    total_kl = 0.0
    total_top1_match = 0
    total_top5_match = 0
    total_positions = 0
    total_teacher_entropy = 0.0
    total_student_entropy = 0.0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] <= 1:
                continue

            teacher_logits = teacher_model(input_ids).logits / temperature
            student_logits = student_model(input_ids).logits / temperature

            # Shift by one position to compare next-token distributions
            # teacher_logits[:, :-1] predicts input_ids[:, 1:]
            teacher_dist = F.log_softmax(teacher_logits[:, :-1], dim=-1)
            student_dist = F.log_softmax(student_logits[:, :-1], dim=-1)
            teacher_probs = torch.exp(teacher_dist)

            kl = F.kl_div(student_dist, teacher_probs, reduction="none").sum(dim=-1)
            total_kl += kl.sum().item()

            teacher_top1 = teacher_logits[:, :-1].argmax(dim=-1)
            student_top1 = student_logits[:, :-1].argmax(dim=-1)
            total_top1_match += (teacher_top1 == student_top1).sum().item()

            teacher_top5 = teacher_logits[:, :-1].topk(5, dim=-1).indices
            student_top5 = student_logits[:, :-1].topk(5, dim=-1).indices
            # Count overlap of top-5 sets per position
            for b in range(input_ids.shape[0]):
                for pos in range(teacher_top5.shape[1]):
                    t_set = set(teacher_top5[b, pos].tolist())
                    s_set = set(student_top5[b, pos].tolist())
                    if len(t_set & s_set) > 0:
                        total_top5_match += 1

            teacher_entropy = -(teacher_probs * teacher_dist).sum(dim=-1)
            student_entropy = -(torch.exp(student_dist) * student_dist).sum(dim=-1)
            total_teacher_entropy += teacher_entropy.sum().item()
            total_student_entropy += student_entropy.sum().item()

            n_pos = input_ids.shape[1] - 1
            total_positions += n_pos

    if total_positions == 0:
        return {
            "avg_kl": float("inf"),
            "top1_agreement": 0.0,
            "top5_agreement": 0.0,
            "avg_teacher_entropy": 0.0,
            "avg_student_entropy": 0.0,
        }

    return {
        "avg_kl": total_kl / total_positions,
        "top1_agreement": total_top1_match / total_positions,
        "top5_agreement": total_top5_match / total_positions,
        "avg_teacher_entropy": total_teacher_entropy / total_positions,
        "avg_student_entropy": total_student_entropy / total_positions,
    }


def run_generation(model, tokenizer, prompts, device, max_new_tokens: int = 128):
    """Run generation for a list of prompts and collect outputs + basic stats."""
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "output": generated,
            "output_length": generated_ids.shape[0],
            "ended_with_eos": generated_ids[-1].item() == tokenizer.eos_token_id,
        })
    return results


def compute_generation_metrics(generations: List[Dict]) -> Dict:
    """Compute EOS success rate and simple repetition stats from generation results."""
    if not generations:
        return {"eos_success_rate": 0.0, "avg_output_length": 0.0, "repetition_rate": 0.0}

    n = len(generations)
    eos_count = sum(1 for g in generations if g.get("ended_with_eos", False))
    avg_len = sum(g["output_length"] for g in generations) / n

    # Simple repetition: count 4-gram repeats within each generation
    repetition_count = 0
    for g in generations:
        text = g["output"]
        words = text.split()
        if len(words) < 8:
            continue
        ngrams = set()
        repeats = 0
        for i in range(len(words) - 4 + 1):
            gram = tuple(words[i:i + 4])
            if gram in ngrams:
                repeats += 1
            ngrams.add(gram)
        if repeats > 0:
            repetition_count += 1

    return {
        "eos_success_rate": eos_count / n,
        "avg_output_length": avg_len,
        "repetition_rate": repetition_count / n,
    }


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> float:
    """Compute mean cosine similarity between two tensors along a dimension."""
    a_norm = F.normalize(a.float(), dim=dim, eps=1e-8)
    b_norm = F.normalize(b.float(), dim=dim, eps=1e-8)
    return (a_norm * b_norm).sum(dim=dim).mean().item()


def compute_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a.float(), b.float()).item()


def compute_relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2).mean().sqrt().item()
    denom = a.float().pow(2).mean().sqrt().item()
    if denom == 0:
        return float("inf")
    return diff / denom


def measure_peak_memory_mb() -> float:
    """Measure peak allocated GPU memory in MB across all devices."""
    peak_mb = 0.0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            peak_mb += torch.cuda.max_memory_allocated(i) / (1024 ** 2)
    return peak_mb


def reset_peak_memory_stats():
    """Reset CUDA peak memory counters."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
