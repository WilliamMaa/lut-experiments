"""Ranking and report generation for LLM-LUT v0.5.

Replaces the unreliable Final Score with Recovery + Bucket Advantage.
"""

import os
import json
from datetime import datetime


def rank_candidates(results, baseline_metrics, save_path: str = "results/rank_report_v0_5.md"):
    """
    Compute new metrics and generate markdown report.
    
    Key metrics:
        Recovery = (KL_Zero - KL_Bucket) / KL_Zero
        BucketAdvantage = KL_Mean - KL_Bucket
        Coverage = bucket coverage
        Entropy = occupancy entropy
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    ranked = []
    for r in results:
        kl_zero = r.get("kl_zero") or 0.0
        kl_mean = r.get("kl_mean") or 0.0
        kl_bucket = r.get("kl_bucket") or 0.0
        coverage = r.get("bucket_coverage", 0.0)
        entropy = r.get("bucket_entropy", 0.0)
        
        recovery = (kl_zero - kl_bucket) / max(kl_zero, 1e-8) if kl_zero > 0 else 0.0
        bucket_advantage = kl_mean - kl_bucket
        
        ranked.append({
            **r,
            "recovery": recovery,
            "bucket_advantage": bucket_advantage,
        })
    
    # Sort by recovery descending
    ranked.sort(key=lambda x: x["recovery"], reverse=True)
    
    lines = []
    lines.append("# LLM-LUT v0.5 Sensitivity Scan Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if baseline_metrics:
        lines.append("## Baseline Metrics")
        lines.append(f"- Next-token accuracy: {baseline_metrics.get('next_token_acc', 'N/A')}")
        lines.append(f"- Perplexity: {baseline_metrics.get('ppl', 'N/A'):.2f}")
        lines.append("")
    
    lines.append("## Top Candidates (by Recovery)")
    lines.append("")
    lines.append("| Rank | Layer | Type | Group | Binning | Bins | KL Zero | KL Mean | KL Bucket | Recovery | BucketAdv | Coverage | Entropy |")
    lines.append("|------|-------|------|-------|---------|------|---------|---------|-----------|----------|-----------|----------|---------|")
    
    for i, r in enumerate(ranked[:50]):
        lines.append(
            f"| {i+1} | {r['layer']} | {r['candidate_type']} | {r['group']} | "
            f"{r.get('binning_mode', 'uniform')} | {r.get('num_bins', 64)} | "
            f"{r.get('kl_zero', 0):.6f} | {r.get('kl_mean', 0):.6f} | {r.get('kl_bucket', 0):.6f} | "
            f"{r['recovery']:.4f} | {r['bucket_advantage']:.6f} | "
            f"{r.get('bucket_coverage', 0):.2%} | {r.get('bucket_entropy', 0):.4f} |"
        )
    
    lines.append("")
    lines.append("## Full Results")
    lines.append("")
    lines.append("| Layer | Type | Group | Binning | Bins | KL Zero | KL Mean | KL Bucket | Recovery | BucketAdv | Coverage | Entropy |")
    lines.append("|-------|------|-------|---------|------|---------|---------|-----------|----------|-----------|----------|---------|")
    for r in ranked:
        lines.append(
            f"| {r['layer']} | {r['candidate_type']} | {r['group']} | "
            f"{r.get('binning_mode', 'uniform')} | {r.get('num_bins', 64)} | "
            f"{r.get('kl_zero', 0):.6f} | {r.get('kl_mean', 0):.6f} | {r.get('kl_bucket', 0):.6f} | "
            f"{r['recovery']:.4f} | {r['bucket_advantage']:.6f} | "
            f"{r.get('bucket_coverage', 0):.2%} | {r.get('bucket_entropy', 0):.4f} |"
        )
    
    md = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    json_path = save_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline_metrics,
            "ranked": ranked,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[RANK] Report saved to {save_path} and {json_path}")
    if ranked:
        top = ranked[0]
        print(f"[RANK] Top: L{top['layer']} {top['candidate_type']} g{top['group']} "
              f"{top.get('binning_mode', 'uniform')} bins={top.get('num_bins', 64)} "
              f"recovery={top['recovery']:.4f}")
    return ranked
