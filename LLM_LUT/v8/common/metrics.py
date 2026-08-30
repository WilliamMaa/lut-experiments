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


def _logits_for_trajectory(model, tokenizer, input_ids, attention_mask, traj):
    """Force-feed a fixed token trajectory and collect per-step next-token logits.

    traj: 1-D tensor of token ids (length T).
    Returns logits of shape (T, vocab_size). Logits[t] is the distribution over
    the token at position t given the prefix (prompt + traj[:t]).
    """
    model.eval()
    with torch.no_grad():
        # Prefill.
        out = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=None,
        )
        past_key_values = out.past_key_values
        logits = [out.logits[:, -1, :]]  # distribution for traj[0]

        for i in range(len(traj) - 1):
            token = traj[i].unsqueeze(0).unsqueeze(0)
            out = model(
                token,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits.append(out.logits[:, -1, :])

    return torch.cat(logits, dim=0)  # (T, vocab_size)


def compute_decode_divergence_metrics(
    teacher_model,
    student_model,
    tokenizer,
    prompts,
    device,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
):
    """Fixed-trajectory decode divergence metrics.

    1. Greedy-generate a fixed trajectory from the teacher.
    2. Force both teacher and student to consume the exact same trajectory.
    3. Compare next-token distributions at every decode step.

    Returns:
        avg_decode_kl: mean KL(student || teacher) per decode position.
        decode_top1_agreement: fraction where argmax matches.
        decode_top5_agreement: fraction where top-5 sets overlap.
        avg_teacher_greedy_token_prob_under_student: mean probability the student
            assigns to the teacher's greedy token.
        total_decode_positions: total number of positions evaluated.
    """
    teacher_model.eval()
    student_model.eval()

    all_kl = []
    all_top1_match = []
    all_top5_match = []
    all_teacher_token_probs = []
    total_positions = 0

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # 1. Fixed trajectory from teacher greedy decoding.
            gen_ids = teacher_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = input_ids.shape[1]
            traj = gen_ids[0, prompt_len:]
            if traj.numel() == 0:
                continue

            # 2. Both models consume the same trajectory.
            teacher_logits = _logits_for_trajectory(
                teacher_model, tokenizer, input_ids, attention_mask, traj
            )
            student_logits = _logits_for_trajectory(
                student_model, tokenizer, input_ids, attention_mask, traj
            )

            # 3. Compare distributions.
            teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
            student_logp = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_p = torch.exp(teacher_logp)

            kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
            all_kl.extend(kl.tolist())

            teacher_top1 = teacher_logits.argmax(dim=-1)
            student_top1 = student_logits.argmax(dim=-1)
            all_top1_match.extend((teacher_top1 == student_top1).tolist())

            teacher_top5 = teacher_logits.topk(5, dim=-1).indices
            student_top5 = student_logits.topk(5, dim=-1).indices
            for t in range(traj.shape[0]):
                if len(set(teacher_top5[t].tolist()) & set(student_top5[t].tolist())) > 0:
                    all_top5_match.append(1)
                else:
                    all_top5_match.append(0)

            student_p = torch.exp(student_logp)
            teacher_token_probs = student_p[torch.arange(traj.shape[0]), traj]
            all_teacher_token_probs.extend(teacher_token_probs.tolist())

            total_positions += traj.shape[0]

    if total_positions == 0:
        return {
            "avg_decode_kl": float("inf"),
            "decode_top1_agreement": 0.0,
            "decode_top5_agreement": 0.0,
            "avg_teacher_greedy_token_prob_under_student": 0.0,
            "total_decode_positions": 0,
        }

    return {
        "avg_decode_kl": sum(all_kl) / len(all_kl),
        "decode_top1_agreement": sum(all_top1_match) / len(all_top1_match),
        "decode_top5_agreement": sum(all_top5_match) / len(all_top5_match),
        "avg_teacher_greedy_token_prob_under_student": sum(all_teacher_token_probs) / len(all_teacher_token_probs),
        "total_decode_positions": total_positions,
    }
