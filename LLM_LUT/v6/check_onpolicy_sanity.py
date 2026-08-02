#!/usr/bin/env python3
"""
check_onpolicy_sanity.py

一键检查 collect_onpolicy_data.py 的 sanity run 输出。

用法：
  python -u check_onpolicy_sanity.py --output_root ./onpolicy_sanity_10x16

输出：
  - 每个 prompt 的 T/kept/cos_mean/cos_p10
  - 形状检查
  - 生成文本异常检查（重复、模板泄漏、乱码）
  - 汇总统计
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch


def check_anomalies(text: str) -> dict:
    """检查生成文本中的异常模式。"""
    anomalies = {
        "has_human_label": any(k in text for k in ("Human:", "user:", "User:")),
        "has_assistant_label": any(k in text for k in ("Assistant:", "assistant:")),
        "repetition_10gram": 0.0,
        "unusual_chars": 0.0,
    }

    n = 10
    if len(text) >= n:
        ngrams = set()
        total = 0
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])
            total += 1
        anomalies["repetition_10gram"] = 1.0 - len(ngrams) / max(total, 1)

    if text:
        unusual = sum(1 for c in text if ord(c) < 32 and c not in "\n\t")
        anomalies["unusual_chars"] = unusual / len(text)

    return anomalies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True, help="collect_onpolicy_data.py output root")
    parser.add_argument("--expected_max_new_tokens", type=int, default=None,
                        help="Expected max_new_tokens; if not set, use max T as reference")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    long_dir = out_root / "long_rollout"
    if not long_dir.exists():
        print(f"ERROR: {long_dir} not found")
        sys.exit(1)

    metrics_files = sorted(long_dir.glob("prompt_*/metrics.json"))
    if not metrics_files:
        print(f"ERROR: no metrics.json found in {long_dir}")
        sys.exit(1)

    print(f"Found {len(metrics_files)} prompt outputs\n")
    print("=" * 80)

    all_T = []
    all_kept = []
    all_cos_mean = []
    all_cos_p10 = []
    errors = []

    for mfile in metrics_files:
        subdir = mfile.parent
        pidx = subdir.name

        with open(mfile, "r", encoding="utf-8") as f:
            m = json.load(f)

        T = m.get("T", -1)
        kept = m.get("kept", -1)
        cos_mean = m.get("cos_mean", float("nan"))
        cos_p10 = m.get("cos_p10", float("nan"))
        prompt = m.get("prompt", "")
        gen_text = m.get("generated_text", "")

        all_T.append(T)
        all_kept.append(kept)
        all_cos_mean.append(cos_mean)
        all_cos_p10.append(cos_p10)

        # 检查 tensor 文件
        shape_ok = True
        shapes = {}
        for name in ["ffn_input.pt", "teacher_ffn_out.pt", "lut_ffn_out.pt"]:
            fpath = subdir / name
            if not fpath.exists():
                errors.append(f"{pidx}: missing {name}")
                shape_ok = False
                continue
            try:
                t = torch.load(fpath, map_location="cpu", weights_only=False)
                shapes[name] = tuple(t.shape)
                if t.dim() != 2:
                    errors.append(f"{pidx}: {name} dim != 2, got {t.dim()}")
                    shape_ok = False
            except Exception as e:
                errors.append(f"{pidx}: failed to load {name}: {e}")
                shape_ok = False

        # 检查形状一致性
        if shape_ok:
            shape_list = list(shapes.values())
            if len(set(shape_list)) != 1:
                errors.append(f"{pidx}: shape mismatch {shapes}")

        # 异常检查
        anomalies = check_anomalies(gen_text)
        anomaly_flags = []
        if anomalies["has_human_label"]:
            anomaly_flags.append("human_label")
        if anomalies["has_assistant_label"]:
            anomaly_flags.append("assistant_label")
        if anomalies["repetition_10gram"] > 0.9:
            anomaly_flags.append(f"high_rep({anomalies['repetition_10gram']:.2f})")
        if anomalies["unusual_chars"] > 0.05:
            anomaly_flags.append(f"unusual_chars({anomalies['unusual_chars']:.2f})")

        status = "OK" if shape_ok and not anomaly_flags else "WARN"
        print(f"[{status}] {pidx}: T={T:3d} kept={kept:3d} cos_mean={cos_mean:6.4f} cos_p10={cos_p10:6.4f}")
        print(f"       prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"       gen: {gen_text[:80]}{'...' if len(gen_text) > 80 else ''}")
        if not shape_ok:
            print(f"       shapes: {shapes}")
        if anomaly_flags:
            print(f"       anomalies: {', '.join(anomaly_flags)}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    expected_T = args.expected_max_new_tokens if args.expected_max_new_tokens is not None else max(all_T) if all_T else 0

    print(f"Prompts: {len(metrics_files)}")
    print(f"T: min={min(all_T) if all_T else 0}, max={max(all_T) if all_T else 0}, mean={sum(all_T)/len(all_T) if all_T else 0:.1f}")
    print(f"Expected T: {expected_T}")
    off_T = [T for T in all_T if T != expected_T]
    if off_T:
        errors.append(f"{len(off_T)} prompts have T != {expected_T}: {off_T}")
    print(f"T mismatch count: {len(off_T)}")

    print(f"kept: min={min(all_kept) if all_kept else 0}, max={max(all_kept) if all_kept else 0}, mean={sum(all_kept)/len(all_kept) if all_kept else 0:.1f}")

    cos_mean_vals = [c for c in all_cos_mean if c == c]  # filter nan
    cos_p10_vals = [c for c in all_cos_p10 if c == c]
    if cos_mean_vals:
        print(f"cos_mean: min={min(cos_mean_vals):.4f}, max={max(cos_mean_vals):.4f}, mean={sum(cos_mean_vals)/len(cos_mean_vals):.4f}")
    if cos_p10_vals:
        print(f"cos_p10:  min={min(cos_p10_vals):.4f}, max={max(cos_p10_vals):.4f}, mean={sum(cos_p10_vals)/len(cos_p10_vals):.4f}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
