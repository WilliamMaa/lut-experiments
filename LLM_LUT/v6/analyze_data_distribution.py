#!/usr/bin/env python3
"""
检查 LUT calibration 数据的分布特征
"""

import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyze_data_distribution(input_dir, output_dir, sample_limit=100):
    """
    分析输入/输出数据的分布特征
    """
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))[:sample_limit]
    
    if not input_files:
        print(f"No files found in {input_dir}")
        return
    
    stats = {
        'input_mean': [],
        'input_std': [],
        'input_norm': [],
        'output_mean': [],
        'output_std': [],
        'output_norm': [],
        'sample_shapes': [],
    }
    
    output_files_map = {}
    if output_dir:
        output_files_map = {os.path.basename(p): p for p in glob.glob(os.path.join(output_dir, "*.pt"))}
    
    print(f"Analyzing {len(input_files)} files...")
    
    for in_path in tqdm(input_files):
        try:
            # 加载输入
            x = torch.load(in_path, map_location="cpu")
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            stats['sample_shapes'].append(x.shape[0])
            stats['input_mean'].append(x.mean().item())
            stats['input_std'].append(x.std().item())
            stats['input_norm'].append(torch.norm(x, dim=-1).mean().item())
            
            # 加载输出
            if output_dir:
                out_path = output_files_map.get(os.path.basename(in_path))
                if out_path:
                    y = torch.load(out_path, map_location="cpu")
                    if y.dim() == 1:
                        y = y.unsqueeze(0)
                    stats['output_mean'].append(y.mean().item())
                    stats['output_std'].append(y.std().item())
                    stats['output_norm'].append(torch.norm(y, dim=-1).mean().item())
            
        except Exception as e:
            print(f"Error loading {in_path}: {e}")
            continue
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据分布统计")
    print("="*60)
    
    print(f"\n样本形状（每文件样本数）：")
    shapes = stats['sample_shapes']
    print(f"  均值: {np.mean(shapes):.1f}")
    print(f"  中位数: {np.median(shapes):.1f}")
    print(f"  最小: {np.min(shapes)}")
    print(f"  最大: {np.max(shapes)}")
    print(f"  总样本数: {np.sum(shapes)}")
    
    print(f"\n输入统计（Layer 39 输入）：")
    print(f"  均值范围: [{np.min(stats['input_mean']):.4f}, {np.max(stats['input_mean']):.4f}]")
    print(f"  标准差范围: [{np.min(stats['input_std']):.4f}, {np.max(stats['input_std']):.4f}]")
    print(f"  L2范数范围: [{np.min(stats['input_norm']):.4f}, {np.max(stats['input_norm']):.4f}]")
    
    if stats['output_mean']:
        print(f"\n输出统计（Layer 39 FFN 输出）：")
        print(f"  均值范围: [{np.min(stats['output_mean']):.4f}, {np.max(stats['output_mean']):.4f}]")
        print(f"  标准差范围: [{np.min(stats['output_std']):.4f}, {np.max(stats['output_std']):.4f}]")
        print(f"  L2范数范围: [{np.min(stats['output_norm']):.4f}, {np.max(stats['output_norm']):.4f}]")
    
    # 分析是否存在明显的分布分层（可能是不同序列长度导致）
    print("\n" + "="*60)
    print("分布分层分析（用于推断序列长度多样性）")
    print("="*60)
    
    # 按输入范数聚类分析
    norms = np.array(stats['input_norm'])
    q25, q50, q75 = np.percentile(norms, [25, 50, 75])
    print(f"\n输入范数四分位数：")
    print(f"  Q1 (25%): {q25:.4f}")
    print(f"  Q2 (50%): {q50:.4f}")
    print(f"  Q3 (75%): {q75:.4f}")
    
    # 计算 IQR 和异常值
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = np.sum((norms < lower_bound) | (norms > upper_bound))
    print(f"\n异常值检测：")
    print(f"  异常值数量: {outliers}/{len(norms)} ({100*outliers/len(norms):.1f}%)")
    
    # 简单的分层分析
    low_norm = np.sum(norms < q25)
    mid_norm = np.sum((norms >= q25) & (norms <= q75))
    high_norm = np.sum(norms > q75)
    
    print(f"\n按输入范数分层：")
    print(f"  低范数 (< Q1): {low_norm} 文件 ({100*low_norm/len(norms):.1f}%)")
    print(f"  中范数 (Q1-Q3): {mid_norm} 文件 ({100*mid_norm/len(norms):.1f}%)")
    print(f"  高范数 (> Q3): {high_norm} 文件 ({100*high_norm/len(norms):.1f}%)")
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    
    # 判断是否有足够的多样性
    cv = np.std(norms) / np.mean(norms)  # 变异系数
    print(f"\n范数变异系数 (CV): {cv:.4f}")
    
    if cv < 0.1:
        print("⚠️  CV < 0.1: 数据分布较为集中，可能主要来自相似长度的序列")
        print("    建议：增加更多长序列（>1024 tokens）的数据")
    elif cv < 0.3:
        print("✓ CV 0.1-0.3: 数据有一定多样性，但可能仍需补充极端长度样本")
    else:
        print("✓ CV > 0.3: 数据分布较广，应该覆盖了不同长度的序列")
    
    if outliers > len(norms) * 0.1:
        print(f"\n⚠️  检测到 {outliers} 个异常值，可能存在：")
        print("    - 不同长度的序列（短序列 vs 长序列）")
        print("    - 或者数据质量问题")
    
    return stats


def check_sequence_length_correlation(input_dir, output_dir):
    """
    尝试推断序列长度与 activation 范数的相关性
    """
    print("\n" + "="*60)
    print("序列长度推断分析")
    print("="*60)
    print("\n注意：由于 .pt 文件只包含 layer 39 的 hidden states，")
    print("      没有直接的序列长度信息。")
    print("      但我们可以通过 activation 的统计特征推断：")
    print("      - 通常长序列的深层 activation 会有更大的方差")
    print("      - 短序列的 activation 往往更集中\n")
    
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    output_files_map = {}
    if output_dir:
        output_files_map = {os.path.basename(p): p for p in glob.glob(os.path.join(output_dir, "*.pt"))}
    
    # 收集更多统计信息
    all_input_norms = []
    all_output_norms = []
    
    print("收集详细统计信息...")
    for in_path in tqdm(input_files[:50]):  # 检查前50个文件
        try:
            x = torch.load(in_path, map_location="cpu")
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # 计算每个样本的范数（每个token一个范数）
            sample_norms = torch.norm(x, dim=-1)
            all_input_norms.extend(sample_norms.numpy())
            
            if output_dir:
                out_path = output_files_map.get(os.path.basename(in_path))
                if out_path:
                    y = torch.load(out_path, map_location="cpu")
                    if y.dim() == 1:
                        y = y.unsqueeze(0)
                    sample_output_norms = torch.norm(y, dim=-1)
                    all_output_norms.extend(sample_output_norms.numpy())
        except Exception as e:
            continue
    
    if not all_input_norms:
        print("No data collected")
        return
    
    norms = np.array(all_input_norms)
    
    print(f"\n总共分析了 {len(norms)} 个 token 的 activation")
    print(f"\n范数分布：")
    print(f"  均值: {np.mean(norms):.4f}")
    print(f"  标准差: {np.std(norms):.4f}")
    print(f"  最小: {np.min(norms):.4f}")
    print(f"  最大: {np.max(norms):.4f}")
    print(f"  中位数: {np.median(norms):.4f}")
    print(f"  95%分位数: {np.percentile(norms, 95):.4f}")
    print(f"  99%分位数: {np.percentile(norms, 99):.4f}")
    
    # 检查是否有明显的多峰分布（可能对应不同长度）
    p10, p90 = np.percentile(norms, [10, 90])
    print(f"\n10%-90% 范围: [{p10:.4f}, {p90:.4f}]")
    
    if np.max(norms) > 2 * np.median(norms):
        print("\n⚠️  检测到部分样本范数远高于中位数")
        print("    这可能对应长序列的深层位置")
        print(f"    （最大范数 {np.max(norms):.4f} vs 中位数 {np.median(norms):.4f}）")
    
    # 建议
    print("\n" + "="*60)
    print("数据多样性建议")
    print("="*60)
    
    unique_norms = len(np.unique(norms))
    print(f"\n唯一范数值数量: {unique_norms}/{len(norms)}")
    
    if unique_norms < len(norms) * 0.5:
        print("⚠️  大量重复值，可能存在：")
        print("    - 数据来自相同/相似的输入序列")
        print("    - 需要更多多样化的训练数据")
    else:
        print("✓ 数据分布较多样化")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LUT calibration data distribution")
    parser.add_argument("--input_dir", required=True, help="Input data directory")
    parser.add_argument("--output_dir", default=None, help="Output data directory (optional)")
    parser.add_argument("--samples", type=int, default=100, help="Number of files to analyze")
    args = parser.parse_args()
    
    # 主要分析
    stats = analyze_data_distribution(args.input_dir, args.output_dir, args.samples)
    
    # 序列长度推断
    check_sequence_length_correlation(args.input_dir, args.output_dir)
