"""Training and evaluation for LLM-LUT v1.

Phase 1: collect (address_bin, target_delta) teacher pairs from frozen model.
Phase 2: train LUT table via local MSE + cosine loss.
Phase 3: evaluate with model-level metrics.
"""

import sys
import os

V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import get_hook_target
from metrics import compute_model_metrics, compute_baseline_probs
from lut_hook import CaptureInputHook, CaptureOutputHook


def collect_teacher_targets(
    model,
    calib_loader,
    layer_id: int,
    candidate_type: str,
    group_id: int,
    group_size: int,
    addr_idx: torch.Tensor,   # [heads]
    addr_mean: torch.Tensor,  # [heads]
    addr_std: torch.Tensor,   # [heads]
    num_bins: int = 64,
    addr_clip: float = 3.0,
    max_batches: int = 999999,
):
    """
    Run calibration set through frozen model, collect (bin_idx, target_delta).

    Returns:
        bin_indices: [N, heads] long
        targets:     [N, group_size]
        bucket_init: Tensor[num_bins^heads, group_size]  (bucket average for init)
    """
    model.eval()
    device = next(model.parameters()).device
    target_mod = get_hook_target(model, layer_id, candidate_type)

    cap_in = CaptureInputHook()
    cap_out = CaptureOutputHook()
    handle_pre = target_mod.register_forward_pre_hook(cap_in)
    handle_post = target_mod.register_forward_hook(cap_out)

    all_bin_idx = []
    all_targets = []

    try:
        for bi, batch in enumerate(tqdm(calib_loader, desc="Collect targets", leave=False)):
            if bi >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            x = cap_in.input   # [B, seq, dim]
            y = cap_out.output  # [B, seq, dim]
            if x is None or y is None:
                continue

            B, seq, D = y.shape
            num_groups = D // group_size

            # Target: delta for group_id
            if candidate_type == "mlp_delta":
                delta = y - x
            else:
                delta = y  # for down_proj/attn_out, target is output itself

            delta_g = delta.view(B, seq, num_groups, group_size)[:, :, group_id, :]  # [B, seq, gs]

            # Address activations
            addr_flat = addr_idx.to(x.device).view(-1)  # [heads]
            addr_acts = x.index_select(-1, addr_flat)   # [B, seq, heads]

            mean = addr_mean.to(x.device, x.dtype).view(1, 1, -1)
            std = addr_std.to(x.device, x.dtype).view(1, 1, -1).clamp_min(1e-6)
            z = (addr_acts - mean) / std
            z = z.clamp(-addr_clip, addr_clip)
            qf = (z + addr_clip) / (2.0 * addr_clip) * (num_bins - 1)
            bin_idx = torch.round(qf).long().clamp(0, num_bins - 1)  # [B, seq, heads]

            # Mask padding
            if attention_mask is not None:
                mask = attention_mask.bool()  # [B, seq]
                bin_idx = bin_idx[mask]       # [N, heads]
                delta_g = delta_g[mask]       # [N, group_size]
            else:
                bin_idx = bin_idx.view(-1, bin_idx.shape[-1])
                delta_g = delta_g.view(-1, group_size)

            all_bin_idx.append(bin_idx.cpu())
            all_targets.append(delta_g.cpu())
    finally:
        handle_pre.remove()
        handle_post.remove()

    if len(all_bin_idx) == 0:
        raise RuntimeError("No teacher targets collected!")

    bin_indices = torch.cat(all_bin_idx, dim=0)  # [N, heads]
    targets = torch.cat(all_targets, dim=0)      # [N, group_size]

    # Build bucket average init from collected data
    heads = bin_indices.shape[1]
    if heads == 1:
        bucket_init = torch.zeros(num_bins, group_size)
        counts = torch.zeros(num_bins)
        for b in range(num_bins):
            mask = (bin_indices[:, 0] == b)
            cnt = mask.sum().item()
            counts[b] = cnt
            if cnt > 0:
                bucket_init[b] = targets[mask].mean(dim=0)
    else:
        bucket_init = torch.zeros(num_bins, num_bins, group_size)
        counts = torch.zeros(num_bins, num_bins)
        for b0 in range(num_bins):
            for b1 in range(num_bins):
                mask = (bin_indices[:, 0] == b0) & (bin_indices[:, 1] == b1)
                cnt = mask.sum().item()
                counts[b0, b1] = cnt
                if cnt > 0:
                    bucket_init[b0, b1] = targets[mask].mean(dim=0)

    coverage = (counts > 0).sum().item() / counts.numel()
    print(f"  [TARGETS] N={len(bin_indices)}, heads={heads}, coverage={coverage:.2%}")
    return bin_indices, targets, bucket_init


def train_lut_table(
    lut_table,
    bin_indices: torch.Tensor,
    targets: torch.Tensor,
    num_epochs: int = 40,
    lr: float = 1e-3,
    alpha_cosine: float = 0.1,
    batch_size: int = 256,
    device: str = "cuda:0",
):
    """
    Local prefit: train LUT table on pre-collected (bin_idx, target) pairs.

    Returns:
        history: list of dicts with epoch, loss, mse, cos_loss
    """
    lut_table = lut_table.to(device)
    bin_indices = bin_indices.to(device)
    targets = targets.to(device)

    optimizer = torch.optim.AdamW([lut_table.table], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    dataset = TensorDataset(bin_indices, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0
        n_batches = 0

        for b_idx, b_tgt in loader:
            pred = lut_table(b_idx)  # [B, group_size]

            mse = F.mse_loss(pred, b_tgt)
            # Cosine distance: 1 - cos_sim
            cos_sim = F.cosine_similarity(pred, b_tgt, dim=-1).mean()
            cos_loss = 1.0 - cos_sim

            loss = mse + alpha_cosine * cos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)
        avg_cos = total_cos / max(n_batches, 1)
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "mse": avg_mse,
            "cos_loss": avg_cos,
        })
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [TRAIN] Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}, mse={avg_mse:.6f}, cos={avg_cos:.6f}")

    return history


def evaluate_lut(
    model,
    eval_loader,
    reference_probs,
    lut_table,
    layer_id: int,
    candidate_type: str,
    group_id: int,
    group_size: int,
    addr_idx: torch.Tensor,
    addr_mean: torch.Tensor,
    addr_std: torch.Tensor,
    num_bins: int = 64,
    addr_clip: float = 3.0,
):
    """Evaluate model with trainable LUT hook installed."""
    from lut_hook import TrainableLUTHook

    target_mod = get_hook_target(model, layer_id, candidate_type)
    hook = TrainableLUTHook(
        lut_table=lut_table,
        candidate_type=candidate_type,
        group_size=group_size,
        group_id=group_id,
        addr_idx=addr_idx,
        addr_mean=addr_mean,
        addr_std=addr_std,
        num_bins=num_bins,
        addr_clip=addr_clip,
    )
    handle = target_mod.register_forward_hook(hook)
    try:
        metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
    finally:
        handle.remove()
    return metrics


def evaluate_baseline_modes(
    model,
    eval_loader,
    reference_probs,
    layer_id: int,
    candidate_type: str,
    group_id: int,
    group_size: int,
    addr_idx: torch.Tensor,
    addr_mean: torch.Tensor,
    addr_std: torch.Tensor,
    num_bins: int = 64,
    addr_clip: float = 3.0,
    bucket_table: torch.Tensor = None,
    mean_vec: torch.Tensor = None,
):
    """Evaluate zero, mean, bucket baselines."""
    from hooks import PerturbationHook
    from config import get_hook_target

    target_mod = get_hook_target(model, layer_id, candidate_type)
    results = {}

    for mode in ["zero", "mean", "bucket"]:
        if mode == "mean" and mean_vec is None:
            continue
        if mode == "bucket" and bucket_table is None:
            continue

        hook = PerturbationHook(
            candidate_type=candidate_type,
            group_size=group_size,
            group_id=group_id,
            mode=mode,
            mean_vec=mean_vec,
            bucket_table=bucket_table,
            addr_idx=addr_idx,
            addr_mean=addr_mean,
            addr_std=addr_std,
            num_bins=num_bins,
            addr_clip=addr_clip,
        )
        handle = target_mod.register_forward_hook(hook)
        try:
            metrics = compute_model_metrics(model, eval_loader, reference_probs_list=reference_probs)
            results[mode] = metrics
        finally:
            handle.remove()

    return results
