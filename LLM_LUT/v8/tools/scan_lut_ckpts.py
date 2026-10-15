#!/usr/bin/env python3
"""scan_lut_ckpts.py — 列出 /data/mamingyu 下所有 LUT 训练输出目录，
打印最新 checkpoint step 和 summary/eval JSON 里的 cos/ppl 指标。

用法（lut_py310，服务器上）：
    python tools/scan_lut_ckpts.py --root /data/mamingyu
"""
import argparse
import glob
import json
import os
import re


def newest_step(ckpt_dir):
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "*")):
        m = re.findall(r"(\d+)", os.path.basename(f))
        if m:
            steps.append(int(m[-1]))
    return max(steps) if steps else None


def find_metrics(obj, path="", out=None, depth=0):
    if out is None:
        out = {}
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if isinstance(v, (int, float)) and any(
                    t in kl for t in ("cos", "ppl", "rel_l2", "acc")):
                out[f"{path}/{k}"] = v
            else:
                find_metrics(v, f"{path}/{k}", out, depth + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/mamingyu")
    ap.add_argument("--pattern", default="outputs_ffn_lut_*")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not dirs:
        print(f"no dirs matching {args.pattern} under {args.root}")
        return
    for d in dirs:
        name = os.path.basename(d)
        line = [name]
        ck = os.path.join(d, "checkpoints")
        if os.path.isdir(ck):
            line.append(f"step={newest_step(ck)}")
        else:
            line.append("no-checkpoints")
        found = False
        for jf in sorted(glob.glob(os.path.join(d, "*.json"))) + \
                  sorted(glob.glob(os.path.join(d, "**/*.json"),
                                   recursive=True))[:8]:
            if "checkpoint" in jf:
                continue
            try:
                with open(jf) as f:
                    metrics = find_metrics(json.load(f))
            except Exception:
                continue
            if metrics:
                found = True
                rel = os.path.relpath(jf, d)
                for k, v in list(metrics.items())[:12]:
                    line.append(f"{rel}:{k}={v}")
        print(" | ".join(line) if found or os.path.isdir(ck)
              else f"{name} (empty)")


if __name__ == "__main__":
    main()
