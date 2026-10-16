"""
v9 等价性小实验：对比 v6 旧法"从头重跑"与 v9 新法"逐层推进"（docs/00-setup.md 方向 1 小实验）。

同一小批完整序列（固定文本，teacher-forced）：
  A 组（旧法）：挂上已训好的浅层 LUT（如 L37）hook，从 tokens 重新完整 forward，
                捕获更深层（L38/L39）的 shared_expert 输入。
  B 组（新法）：原始 pass（无 hook）+ 逐层推进（注入与 A 组相同的浅层 LUT hook），
                捕获同样的边界。A/B 差异应仅为 bf16 数值噪声。

对比指标：
  - 边界 ffn_in 的逐 token cosine / rel-L2（A vs B）
  - （可选 --leaf_probe_checkpoint_dir）用该层 LUT 的 coarse address 在 A/B 输入上
    计算叶节点 id，比较直方图（TV 距离 + top-leaf 重合度）
  - 两组端到端 wall time

判定：cosine 差异 ~1e-3 量级（bf16 数值噪声）且叶分布一致 -> 推进与重跑语义等价。

用法：
  python compare_rerun_vs_propagation.py \
    --model_path /path/Qwen3.6-35B-A3B \
    --layers 37,38,39 --lut_checkpoint 37=/path/outputs_l37/checkpoints \
    --prompt_file ../../v6/example_custom_prompts.jsonl \
    --n_seqs 8 --max_len 1024 --output_json compare_result.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lw_common as LWC
from lw_common import BoundaryCapture, parse_layers, shared_expert_path  # noqa: E402

from v6_replacement_engine import V6ReplacementEngine  # noqa: E402


def _tokenize_seqs(tokenizer, texts, max_len, n_seqs):
    return [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]
        for t in texts[:n_seqs]
    ]


def group_a_rerun(model, seqs, layers, ckpt_map):
    """挂浅层 LUT hook，从 tokens 完整 forward，捕获 ffn_in。"""
    engines = []
    for idx in layers:
        if idx in ckpt_map:
            hook_mod = eval(shared_expert_path(idx), {"model": model})
            eng = V6ReplacementEngine(
                model=model, layer_idx=idx, checkpoint_dir=ckpt_map[idx],
                device=next(hook_mod.parameters()).device,
                hook_path=shared_expert_path(idx))
            eng.install()
            engines.append(eng)
    capture = BoundaryCapture(model, layers)
    outs = {idx: [] for idx in layers}
    t0 = time.time()
    try:
        for input_ids in seqs:
            capture.reset()
            with torch.no_grad():
                model(input_ids.to(model.device), use_cache=False)
            for idx in layers:
                outs[idx].append(capture.get(idx, "ffn_in"))
    finally:
        capture.close()
        for e in engines:
            e.uninstall()
    return outs, time.time() - t0


def group_b_propagation(model, seqs, layers, ckpt_map):
    """原始 pass 捕获首层边界，然后逐层推进捕获后续边界。

    推进段注入与 A 组完全相同的浅层 LUT engine——这样 A/B 比较的是
    "推进 vs 重跑"的执行语义差异（应为 bf16 数值噪声），而不是分布偏移。
    """
    capture = BoundaryCapture(model, layers)
    ffn = {idx: [] for idx in layers}
    residual = {idx: [] for idx in layers}
    t_pass0 = time.time()
    try:
        for input_ids in seqs:
            capture.reset()
            with torch.no_grad():
                model(input_ids.to(model.device), use_cache=False)
            for idx in layers:
                ffn[idx].append(capture.get(idx, "ffn_in"))
                if idx != layers[-1]:
                    residual[idx].append(capture.get(idx, "residual_in"))
    finally:
        capture.close()
    t_pass = time.time() - t_pass0

    engines = []
    t_prop_total = 0.0
    try:
        for li in range(len(layers) - 1):
            j, k = layers[li], layers[li + 1]
            if j in ckpt_map and all(e.layer_idx != j for e in engines):
                hook_mod = eval(shared_expert_path(j), {"model": model})
                eng = V6ReplacementEngine(
                    model=model, layer_idx=j, checkpoint_dir=ckpt_map[j],
                    device=next(hook_mod.parameters()).device,
                    hook_path=shared_expert_path(j))
                eng.install()
                engines.append(eng)
            cap = BoundaryCapture(model, [k])
            t0 = time.time()
            try:
                for n, h_cpu in enumerate(residual[j]):
                    h = h_cpu  # BoundaryCapture 存的是 [1, S, H]，直接用
                    cap.reset()
                    with torch.no_grad():
                        LWC.run_decoder_slice(model, h, j, k + 1)
                    ffn[k][n] = cap.get(k, "ffn_in")
                    if k != layers[-1]:
                        residual[k].append(cap.get(k, "residual_in"))
            finally:
                cap.close()
            t_prop_total += time.time() - t0
    finally:
        for e in engines:
            e.uninstall()
    return ffn, t_pass, t_prop_total


def compare_boundary(a_tensors, b_tensors):
    a = torch.cat(a_tensors, dim=0)
    b = torch.cat(b_tensors, dim=0)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    a, b = a.float(), b.float()
    cos = F.cosine_similarity(a, b, dim=-1)
    rel_l2 = (a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-12)
    return {
        "n_tokens": a.shape[0],
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "rel_l2_mean": rel_l2.mean().item(),
        "rel_l2_max": rel_l2.max().item(),
        "abs_max": (a - b).abs().max().item(),
    }


def leaf_histogram(indices):
    cnt = Counter(indices.tolist())
    total = sum(cnt.values())
    return {k: v / total for k, v in cnt.items()}


def compare_leaf_distributions(engine, a_tensors, b_tensors, device):
    """用 engine 的 shared coarse address 在 A/B 边界输入上算叶 id，比较直方图。"""
    addr = engine.coarse_address
    hists = {}
    for name, tensors in (("A", a_tensors), ("B", b_tensors)):
        idxs = []
        for t in tensors:
            x = t.to(device)  # ffn_in 已是 [1, S, H]
            with torch.no_grad():
                ind = addr.compute_indices(x).view(-1, addr.num_tables)[:, 0]
            idxs.append(ind.cpu())
        hists[name] = leaf_histogram(torch.cat(idxs))
    keys = set(hists["A"]) | set(hists["B"])
    tv = 0.5 * sum(abs(hists["A"].get(k, 0.0) - hists["B"].get(k, 0.0)) for k in keys)
    top_a = sorted(hists["A"], key=hists["A"].get, reverse=True)[:64]
    top_b = sorted(hists["B"], key=hists["B"].get, reverse=True)[:64]
    overlap = len(set(top_a) & set(top_b)) / 64.0
    return {"tv_distance": tv, "top64_overlap": overlap, "n_leaves_A": len(hists["A"])}


def main():
    ap = argparse.ArgumentParser(description="rerun-vs-propagation equivalence experiment")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--layers", type=str, required=True, help="如 37,38,39（浅->深）")
    ap.add_argument("--lut_checkpoint", type=str, nargs="*", default=None,
                    help="已训好的浅层 LUT：37=/path/checkpoints ...（缺省的层按原始模块推进）")
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--n_seqs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--device_map", default="balanced_low_0")
    ap.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--leaf_probe_checkpoint_dir", default=None,
                    help="可选：某层 LUT checkpoint，用其 coarse address 对比 A/B 叶分布")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    if layers != sorted(layers):
        raise ValueError("--layers 必须按从浅到深排列")
    ckpt_map = {}
    for p in args.lut_checkpoint or []:
        k, v = p.split("=", 1)
        ckpt_map[int(k)] = v

    model, tokenizer = LWC.load_model(args.model_path, args.device_map, args.torch_dtype)
    texts = LWC.load_prompt_texts(args.prompt_file, args.n_seqs)
    seqs = _tokenize_seqs(tokenizer, texts, args.max_len, args.n_seqs)
    print(f"[compare] {len(seqs)} seqs, layers={layers}")

    # A 组：旧法重跑（带浅层 LUT hook 的完整 forward）
    a_ffn, a_time = group_a_rerun(model, seqs, layers, ckpt_map)
    print(f"[compare] A (rerun) done in {a_time:.1f}s")

    # B 组：原始 pass + 逐层推进（注入与 A 组相同的浅层 LUT）
    b_ffn, b_pass, b_prop = group_b_propagation(model, seqs, layers, ckpt_map)
    b_time = b_pass + b_prop
    print(f"[compare] B (propagation) done: pass {b_pass:.1f}s + prop {b_prop:.1f}s = {b_time:.1f}s")

    result = {"args": vars(args), "layers": layers, "timings": {
        "A_rerun_s": a_time, "B_original_pass_s": b_pass,
        "B_propagate_s": b_prop, "B_total_s": b_time,
        "speedup_B_over_A": a_time / max(b_time, 1e-9)}, "boundaries": {}}

    for idx in layers[1:]:
        m = compare_boundary(a_ffn[idx], b_ffn[idx])
        result["boundaries"][f"layer{idx}"] = m
        print(f"[compare] layer{idx} boundary: {json.dumps(m)}")

    if args.leaf_probe_checkpoint_dir:
        probe_layer = layers[-1]
        hook_mod = eval(shared_expert_path(probe_layer), {"model": model})
        eng = V6ReplacementEngine(
            model=model, layer_idx=probe_layer,
            checkpoint_dir=args.leaf_probe_checkpoint_dir,
            device=next(hook_mod.parameters()).device,
            hook_path=shared_expert_path(probe_layer))
        for idx in layers[1:]:
            ld = compare_leaf_distributions(eng, a_ffn[idx], b_ffn[idx],
                                            next(hook_mod.parameters()).device)
            result["boundaries"][f"layer{idx}"].update({"leaf": ld})
            print(f"[compare] layer{idx} leaf: {json.dumps(ld)}")

    verdict = all(result["boundaries"][f"layer{i}"]["cos_mean"] > 0.999
                  for i in layers[1:])
    result["equivalent"] = verdict
    print(f"[compare] equivalent={verdict} (cos_mean > 0.999 on all deeper boundaries)")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[compare] result -> {args.output_json}")


if __name__ == "__main__":
    main()
