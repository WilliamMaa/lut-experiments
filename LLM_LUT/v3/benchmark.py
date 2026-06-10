"""V3 Benchmark: Latency comparison between baseline, functional hook, and partial skip.

Measures:
1. Per-layer down_proj latency (micro-benchmark)
2. End-to-end token generation latency (macro-benchmark)

Usage:
    from v3.benchmark import benchmark_down_proj, benchmark_generation
    benchmark_down_proj(model, layer_id, engine, num_iters=100)
    benchmark_generation(model, tokenizer, prompt, engine, max_new_tokens=128)
"""

import torch
import time


def benchmark_down_proj(model, layer_id, engine=None, num_iters: int = 100):
    """Micro-benchmark: measure down_proj forward latency.

    Args:
        engine: V3PartialEngine instance (if None, measures baseline)
        num_iters: number of iterations for averaging

    Returns:
        dict with median/avg latency in ms
    """
    device = next(model.parameters()).device
    mlp = model.model.layers[layer_id].mlp
    down_proj = mlp.down_proj

    # Infer shapes from down_proj weight
    hidden_size, intermediate_size = down_proj.weight.shape
    batch, seq = 1, 128  # typical decode batch

    # Warmup
    dummy_hidden = torch.randn(batch, seq, intermediate_size, device=device, dtype=down_proj.weight.dtype)
    for _ in range(10):
        _ = down_proj(dummy_hidden)
    torch.cuda.synchronize()

    # Install engine if provided
    if engine is not None:
        engine.install()
        # Need to provide a dummy normed_x in cache for the engine to work
        dummy_normed_x = torch.randn(batch, seq, hidden_size, device=device, dtype=down_proj.weight.dtype)
        engine._cached_normed_x = dummy_normed_x
        # Compute bins for all groups
        for gid, (addr_idx, addr_mean, addr_std, _table) in engine.group_configs.items():
            bin_idx = engine._compute_bin_indices(dummy_normed_x, addr_idx.to(device), addr_mean.to(device), addr_std.to(device))
            engine._cached_bin_idx[gid] = bin_idx

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    try:
        for _ in range(num_iters):
            # Refresh cache if engine is installed
            if engine is not None:
                engine._cached_normed_x = dummy_normed_x
                for gid, (addr_idx, addr_mean, addr_std, _table) in engine.group_configs.items():
                    bin_idx = engine._compute_bin_indices(dummy_normed_x, addr_idx.to(device), addr_mean.to(device), addr_std.to(device))
                    engine._cached_bin_idx[gid] = bin_idx

            start_event.record()
            _ = down_proj(dummy_hidden)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    finally:
        if engine is not None:
            engine.uninstall()

    times = torch.tensor(times)
    return {
        "median_ms": torch.median(times).item(),
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
    }


def benchmark_generation(model, tokenizer, prompt: str, engine=None, max_new_tokens: int = 128, num_iters: int = 5):
    """Macro-benchmark: measure end-to-end token generation latency.

    Args:
        engine: V3PartialEngine instance (if None, measures baseline)
        max_new_tokens: number of new tokens to generate
        num_iters: number of iterations for averaging

    Returns:
        dict with tokens/sec and ms/token
    """
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    if engine is not None:
        engine.install()

    try:
        # Warmup
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    finally:
        if engine is not None:
            engine.uninstall()

    times = torch.tensor(times)
    total_tokens = output_ids.shape[1] - input_len

    return {
        "median_ms": torch.median(times).item(),
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / (times.mean().item() / 1000),
        "ms_per_token": times.mean().item() / total_tokens,
    }


def print_comparison(baseline: dict, partial: dict, label: str = "Partial Skip"):
    """Pretty-print benchmark comparison."""
    print(f"\n{'='*60}")
    print(f"Benchmark Comparison: {label}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Partial':>12} {'Delta':>12}")
    print("-" * 60)

    for key in ["median_ms", "mean_ms", "std_ms"]:
        if key in baseline and key in partial:
            delta = partial[key] - baseline[key]
            delta_pct = (delta / baseline[key]) * 100 if baseline[key] > 0 else 0
            print(f"{key:<25} {baseline[key]:>12.3f} {partial[key]:>12.3f} {delta_pct:>+11.2f}%")

    # Speedup
    if "mean_ms" in baseline and "mean_ms" in partial:
        speedup = baseline["mean_ms"] / partial["mean_ms"]
        print(f"{'Speedup':<25} {'1.00x':>12} {speedup:>11.2f}x")
