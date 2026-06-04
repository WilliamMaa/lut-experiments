"""Ranking and report generation for LLM-LUT v0."""

import os
import json
from datetime import datetime


def rank_candidates(results, baseline_metrics, save_path: str = "results/rank_report.md"):
    """
    Compute scores and generate a markdown report.
    
    Scoring:
        sensitivity_penalty = kl_bucket (smaller is better)
        addressability = (kl_mean - kl_bucket) / max(kl_mean - kl_zero, 1e-8)
        compute_saving = 1.0 / num_groups  (normalized per-group saving)
        final_score = compute_saving + addressability - sensitivity_penalty
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    ranked = []
    for r in results:
        kl_zero = r.get("kl_zero") or 0.0
        kl_mean = r.get("kl_mean") or 0.0
        kl_bucket = r.get("kl_bucket") or 0.0
        coverage = r.get("bucket_coverage", 0.0)
        
        sensitivity_penalty = kl_bucket
        addressability = (kl_mean - kl_bucket) / max(kl_mean - kl_zero, 1e-8) if kl_mean > kl_zero else 0.0
        compute_saving = 1.0  # per-group unit; can be scaled by actual MAC counts later
        
        # Coverage penalty: if too many empty bins, reduce score
        coverage_factor = min(coverage / 0.7, 1.0) if coverage > 0 else 0.0
        
        final_score = compute_saving * coverage_factor + addressability - sensitivity_penalty
        
        ranked.append({
            **r,
            "sensitivity_penalty": sensitivity_penalty,
            "addressability": addressability,
            "compute_saving": compute_saving,
            "coverage_factor": coverage_factor,
            "final_score": final_score,
        })
    
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Write markdown report
    lines = []
    lines.append("# LLM-LUT v0 Sensitivity Scan Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if baseline_metrics:
        lines.append("## Baseline Metrics")
        lines.append(f"- Next-token accuracy: {baseline_metrics.get('next_token_acc', 'N/A')}")
        lines.append(f"- Perplexity: {baseline_metrics.get('ppl', 'N/A'):.2f}")
        lines.append("")
    
    lines.append("## Top Candidates")
    lines.append("")
    lines.append("| Rank | Layer | Type | Group | KL Zero | KL Mean | KL Bucket | Coverage | Addressability | Final Score |")
    lines.append("|------|-------|------|-------|---------|---------|-----------|----------|----------------|-------------|")
    
    for i, r in enumerate(ranked[:50]):
        lines.append(
            f"| {i+1} | {r['layer']} | {r['candidate_type']} | {r['group']} | "
            f"{r.get('kl_zero', 0):.6f} | {r.get('kl_mean', 0):.6f} | {r.get('kl_bucket', 0):.6f} | "
            f"{r.get('bucket_coverage', 0):.2%} | {r['addressability']:.4f} | {r['final_score']:.4f} |"
        )
    
    lines.append("")
    lines.append("## Full Results (All Candidates)")
    lines.append("")
    lines.append("| Layer | Type | Group | KL Zero | KL Mean | KL Bucket | PPL Bucket | ACC Bucket | Coverage | Score |")
    lines.append("|-------|------|-------|---------|---------|-----------|------------|------------|----------|-------|")
    for r in ranked:
        lines.append(
            f"| {r['layer']} | {r['candidate_type']} | {r['group']} | "
            f"{r.get('kl_zero', 0):.6f} | {r.get('kl_mean', 0):.6f} | {r.get('kl_bucket', 0):.6f} | "
            f"{r.get('ppl_bucket', 0):.2f} | {r.get('acc_bucket', 0):.4f} | "
            f"{r.get('bucket_coverage', 0):.2%} | {r['final_score']:.4f} |"
        )
    
    md = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    # Also save JSON
    json_path = save_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": baseline_metrics,
            "ranked": ranked,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[RANK] Report saved to {save_path} and {json_path}")
    print(f"[RANK] Top candidate: L{ranked[0]['layer']} {ranked[0]['candidate_type']} group {ranked[0]['group']} score={ranked[0]['final_score']:.4f}")
    return ranked
