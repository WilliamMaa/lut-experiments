"""Metrics for LLM-LUT v0.

Local (layer-level) and model-level metrics.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_local_metrics(original, perturbed):
    """
    Args:
        original:   Tensor[N, group_size]
        perturbed:  Tensor[N, group_size]
    Returns:
        dict with mse, cos_sim, rel_err_reduction (vs zero)
    """
    mse = F.mse_loss(perturbed, original).item()
    
    # Cosine similarity over flattened vectors
    orig_flat = original.float().view(-1)
    pert_flat = perturbed.float().view(-1)
    cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), pert_flat.unsqueeze(0), dim=1).item()
    
    # Relative to zero
    zero_mse = F.mse_loss(torch.zeros_like(original), original).item()
    rel_err_reduction = (zero_mse - mse) / (zero_mse + 1e-8)
    
    return {
        "mse": mse,
        "cos_sim": cos_sim,
        "rel_err_reduction": rel_err_reduction,
    }


def compute_model_metrics(model, eval_loader, reference_probs_list=None):
    """
    Run model on eval_loader and compute metrics.
    
    If reference_probs_list is provided, compute KL divergence online
    (per-batch) to avoid storing all logits in CPU memory.
    
    Args:
        reference_probs_list: list of CPU Tensor[batch_seq_len, vocab] 
                              (softmax probs from baseline, flattened over batch+seq)
    
    Returns:
        dict with avg_kl, avg_ppl, next_token_acc
    """
    model.eval()
    device = next(model.parameters()).device
    total_kl = 0.0
    total_nll = 0.0
    total_tokens = 0
    correct = 0
    num_batches = 0
    
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(eval_loader, desc="Eval", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, seq, vocab]
            
            # Next-token accuracy
            preds = logits[:, :-1, :].argmax(dim=-1)  # [B, seq-1]
            targets = input_ids[:, 1:]                # [B, seq-1]
            if attention_mask is not None:
                mask = attention_mask[:, 1:].bool()
                correct += (preds[mask] == targets[mask]).sum().item()
                total_tokens += mask.sum().item()
            else:
                correct += (preds == targets).sum().item()
                total_tokens += targets.numel()
            
            # NLL for perplexity
            if attention_mask is not None:
                token_mask = attention_mask[:, 1:].bool().reshape(-1)
                token_nll = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                )
                total_nll += token_nll[token_mask].sum().item()
            else:
                total_nll += F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            
            # KL divergence (online, per-batch)
            if reference_probs_list is not None:
                cur_logp = F.log_softmax(logits, dim=-1)  # [B, seq, vocab]
                # Flatten over batch and seq
                B, seq, V = cur_logp.shape
                cur_logp_flat = cur_logp.reshape(-1, V).cpu()  # [B*seq, vocab]
                ref_p = reference_probs_list[bi]  # [B*seq, vocab] on CPU
                
                kl = (ref_p * (torch.log(ref_p + 1e-10) - cur_logp_flat)).sum(dim=-1)
                total_kl += kl.sum().item()
            
            num_batches += 1
    
    result = {
        "next_token_acc": correct / max(total_tokens, 1),
        "ppl": torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item(),
    }
    
    if reference_probs_list is not None:
        # Count total tokens for KL (same as eval tokens but over full seq incl first token)
        kl_tokens = sum(p.shape[0] for p in reference_probs_list)
        result["avg_kl"] = total_kl / max(kl_tokens, 1)
    else:
        result["avg_kl"] = None
    
    return result


def compute_baseline_probs(model, eval_loader):
    """
    Compute baseline softmax probabilities for KL divergence.
    Returns a list of CPU tensors, one per batch, each flattened over batch+seq.
    
    This avoids storing full logits (which are bf16/float32 and huge).
    Storing probs (float32) is still large but we do it only once for baseline.
    """
    model.eval()
    device = next(model.parameters()).device
    probs_list = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Baseline", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().float()
            # Flatten over batch and seq: [B*seq, vocab]
            probs_list.append(probs.reshape(-1, probs.size(-1)))
    
    return probs_list
