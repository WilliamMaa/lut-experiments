#!/usr/bin/env python3
"""verify_lut_bundle.py — 排查 serve 端 LUT 输出爆炸是否是权重问题。

两个独立检查：

  pack   打包保真度（纯 CPU，秒级，不需要模型）：
         bundle layer{N}.pt 里的表/树 是否与原 v4 checkpoint
         （replacement_g*.pt）完全一致。排查 prep_lut_bundle 是否打包错。

  qual   数值质量（需要加载 70GB 模型，分钟级）：
         真实激活 x 下，bundle 的 LUT 输出 vs 原 dense shared_expert 输出，
         报 cosine / 相对 L2。参照系：v6 侧 eval L39 单层 cos≈0.84
         （PPL 8→11.68）。cos 明显低于参照系 → 权重/训练问题；
         cos 正常而 serve 仍炸 → 运行时（kernel/替换路径）问题。

用法（remote，vllm_py310，from LLM_LUT/v8）：

  # 只查打包（快，先跑这个）
  python tools/verify_lut_bundle.py pack \
      --bundle_dir /data/mamingyu/lut_bundles \
      --ckpt_root /data/mamingyu \
      --layers "37 38 39"

  # 再查数值质量（慢，加载模型）
  python tools/verify_lut_bundle.py qual \
      --bundle_dir /data/mamingyu/lut_bundles \
      --model_path /home/u/downloads/models \
      --layers "37 38 39" --device cuda:0
"""
import argparse
import json
import sys
import types
from pathlib import Path

import torch


# v4 checkpoint 里的树对象是 pickle，类引用指向训练脚本模块名。
# 与 v6/scripts/conversion/prep_lut_bundle.py 同款方案：注入按需造类的
# stub 模块，只读属性、不 import 真实训练脚本。
class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls


for _mod_name in ("build_lut_ffn_output_v3_shared_coarse",
                  "build_lut_ffn_output",
                  "build_lut_ffn_output_v3_lowrank",
                  "build_pairwise_correction_v3"):
    sys.modules.setdefault(_mod_name, _StubModule(_mod_name))


def check_pack(bundle_dir: str, ckpt_root: str, layers) -> dict:
    report = {}
    for L in layers:
        bundle = torch.load(f"{bundle_dir}/layer{L}.pt", map_location="cpu",
                            weights_only=False)
        ckpt_dir = Path(ckpt_root)
        cands = list(ckpt_dir.glob(
            f"outputs_ffn_lut_layer{L}_*/checkpoints/replacement_g0.pt"))
        # prep_lut_bundle 打的是 _as_v4/checkpoints（v3 目录里也有同名文件，不能混）
        cands = [p for p in cands if "_as_v4" in str(p)] or cands
        if not cands:
            report[L] = {"error": f"no v4 checkpoint found for layer {L}"}
            continue
        ck_dir = cands[0].parent
        G = bundle["num_groups"]
        E_c, E_r = (1 << bundle["coarse_num_bits"],
                    1 << bundle["residual_num_bits"])
        gs = bundle["group_size"]
        max_tbl, max_tree = 0.0, 0
        n_groups = 0
        for gid in range(G):
            ck = torch.load(ck_dir / f"replacement_g{gid}.pt",
                            map_location="cpu", weights_only=False)
            tables = ck["lut_tables"]
            coarse = tables[0].reshape(E_c, gs).half()
            resid = tables[1].reshape(E_r, gs).half()
            max_tbl = max(max_tbl,
                          (coarse - bundle["coarse_table"][:, gid*gs:(gid+1)*gs]).abs().max().item(),
                          (resid - bundle["resid_table"][gid]).abs().max().item())
            t_b = bundle["resid"]
            n = ck["addresses"][1].node_channel_idx.shape[0]
            for key, ck_key in (("ch", "node_channel_idx"),
                                ("thr", "node_threshold")):
                a = getattr(ck["addresses"][1], ck_key)
                b = t_b[key][gid, :n]
                if key == "thr":
                    ok = torch.isclose(a.float(), b.float(),
                                       rtol=1e-4, atol=1e-4,
                                       equal_nan=True) | (a == b)
                    max_tree = max(max_tree, int((~ok).sum()))
                else:
                    max_tree = max(max_tree, int((a != b).sum()))
            n_groups += 1
        report[L] = {"checkpoint": str(ck_dir), "groups": n_groups,
                     "max_table_diff": max_tbl, "tree_row_mismatches": max_tree}
    return report


def check_qual(bundle_dir, model_path, layers, device, prompt=None,
               max_new=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vllm_plugin.lut_ffn import LutFfnReplacer

    prompt = prompt or "请你简述明朝灭亡的原因，并分析其经济、政治、军事背景。"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="balanced_low_0")
    model.eval()

    reps = {L: LutFfnReplacer(f"{bundle_dir}/layer{L}.pt", "cpu")
            for L in layers}
    captured = {}

    def mk_hook(L):
        def hook(mod, inp, out):
            captured[L] = (inp[0].detach().float().cpu(),
                           out.detach().float().cpu())
        return hook

    handles = []
    for L in layers:
        m = model.model.layers[L].mlp.shared_expert
        handles.append(m.register_forward_hook(mk_hook(L)))

    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    for h in handles:
        h.remove()

    report = {}
    for L in layers:
        if L not in captured:
            report[L] = {"error": "hook not fired"}
            continue
        x, y_true = captured[L]
        y_lut = reps[L].forward_reference(x).float()
        cos = torch.nn.functional.cosine_similarity(
            y_true.flatten(), y_lut.flatten(), dim=0).item()
        rel = ((y_true - y_lut).norm() / y_true.norm().clamp_min(1e-9)).item()
        per_tok = torch.nn.functional.cosine_similarity(
            y_true, y_lut, dim=-1)
        report[L] = {"tokens": int(x.shape[0]),
                     "cos": round(cos, 4), "rel_l2": round(rel, 4),
                     "cos_p10": round(per_tok.quantile(0.1).item(), 4),
                     "cos_p50": round(per_tok.median().item(), 4)}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["pack", "qual"])
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--ckpt_root", default="/data/mamingyu")
    ap.add_argument("--model_path", default="/home/u/downloads/models")
    ap.add_argument("--layers", default="37 38 39")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    layers = [int(t) for t in args.layers.replace(",", " ").split()]

    if args.mode == "pack":
        rep = check_pack(args.bundle_dir, args.ckpt_root, layers)
    else:
        rep = check_qual(args.bundle_dir, args.model_path, layers,
                         args.device)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
