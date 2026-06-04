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


def compute_model_metrics(model, eval_loader, reference_logits_list=None):
    """
    Run model on eval_loader and compute metrics vs reference (if provided).
    
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
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, seq, vocab]
            all_logits.append(logits.cpu())
            
            # Next-token accuracy (predict next token from current)
            # Shift: predict token t+1 from position t
            preds = logits[:, :-1, :].argmax(dim=-1)  # [B, seq-1]
            targets = input_ids[:, 1:]                # [B, seq-1]
            if attention_mask is not None:
                mask = attention_mask[:, 1:].bool()
                correct += (preds[mask] == targets[mask]).sum().item()
                total_tokens += mask.sum().item()
            else:
                correct += (preds == targets).sum().item()
                total_tokens += targets.numel()
            
            # NLL for perplexity (on non-padding tokens)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            nll = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)),
                targets.reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )
            # Since we can't easily ignore padding in NLL without labels, 
            # we compute token-level cross entropy manually with mask
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
            
            num_batches += 1
    
    result = {
        "next_token_acc": correct / max(total_tokens, 1),
        "ppl": torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item(),
    }
    
    # KL divergence vs reference
    if reference_logits_list is not None:
        kl_sum = 0.0
        kl_tokens = 0
        for ref_logits, cur_logits in zip(reference_logits_list, all_logits):
            # Both on CPU
            ref_logp = F.log_softmax(ref_logits, dim=-1)
            cur_p = F.softmax(cur_logits, dim=-1)
            kl = (cur_p * (torch.log(cur_p + 1e-10) - ref_logp)).sum(dim=-1)
            kl_sum += kl.sum().item()
            kl_tokens += kl.numel()
        result["avg_kl"] = kl_sum / max(kl_tokens, 1)
    else:
        result["avg_kl"] = None
    
    return result, all_logits


def compute_kl_between_runs(logits_a_list, logits_b_list):
    """Compute KL(A || B) given lists of logits tensors."""
    kl_sum = 0.0
    tokens = 0
    for a, b in zip(logits_a_list, logits_b_list):
        log_a = F.log_softmax(a, dim=-1)
        p_b = F.softmax(b, dim=-1)
        kl = (p_b * (torch.log(p_b + 1e-10) - log_a)).sum(dim=-1)
        kl_sum += kl.sum().item()
        tokens += kl.numel()
    return kl_sum / max(tokens, 1)
