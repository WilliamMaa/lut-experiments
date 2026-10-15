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

  # 生成对比（dense vs 全装 LUT，短+长 prompt，人眼看输出）
  python tools/verify_lut_bundle.py gen \
      --bundle_dir /data/mamingyu/lut_bundles \
      --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
      --layers "37 38 39" \
      --data data/longctx_multi_turn_65536_64docs.jsonl \
      --max_new 128 --out logs/gen_compare.txt
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch


# v4 checkpoint 里的树对象是 pickle，类引用指向训练脚本模块名。
# 与 v6/scripts/conversion/prep_lut_bundle.py 同款方案：注入按需造类的
# stub 模块，只读属性、不 import 真实训练脚本。
class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        # 双下划线属性（如 __file__）必须正常缺失，否则 inspect 等遍历
        # sys.modules 的标准库会把假类当文件名用（2026-10-14 实机炸过）。
        if name.startswith("__"):
            raise AttributeError(name)
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
        G = bundle["num_groups"]
        E_c, E_r = (1 << bundle["coarse_num_bits"],
                    1 << bundle["residual_num_bits"])
        gs = bundle["group_size"]
        t_b = bundle["resid"]
        entries = []
        for c0 in cands:
            ck_dir = c0.parent
            max_tbl, max_tree = 0.0, 0
            for gid in range(G):
                ck = torch.load(ck_dir / f"replacement_g{gid}.pt",
                                map_location="cpu", weights_only=False)
                tables = ck["lut_tables"]
                coarse = tables[0].reshape(E_c, gs).half()
                resid = tables[1].reshape(E_r, gs).half()
                max_tbl = max(max_tbl,
                              (coarse - bundle["coarse_table"][:, gid*gs:(gid+1)*gs]).abs().max().item(),
                              (resid - bundle["resid_table"][gid]).abs().max().item())
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
            entries.append({"checkpoint": str(ck_dir), "groups": G,
                            "max_table_diff": max_tbl,
                            "tree_row_mismatches": max_tree})
        exact = [e for e in entries
                 if e["max_table_diff"] == 0.0
                 and e["tree_row_mismatches"] == 0]
        report[L] = {"PASS": bool(exact), "candidates": entries}
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
    devs: dict = {}
    for p in model.parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + 1
    print("[qual] param devices:", devs, flush=True)
    if any("cpu" in k for k in devs):
        print("[qual] WARNING: 部分参数在 CPU（共享机显存不够，accelerate "
              "自动 offload）。LUT kernel 只吃 CUDA 张量，请用 "
              "CUDA_VISIBLE_DEVICES 指定更多空卡后重跑。", flush=True)

    reps = {L: LutFfnReplacer(f"{bundle_dir}/layer{L}.pt", "cpu")
            for L in layers}
    mods = {L: model.model.layers[L].mlp.shared_expert for L in layers}
    dense_fwd = {L: mods[L].forward for L in layers}
    captured: dict = {}

    def mk_hook(L):
        def hook(mod, inp, out):
            # 累积 prefill+decode 直到采够样本（prefill 一次只有
            # 十几 token，统计不稳）
            if captured.get("n", 0) >= 4096:
                return
            x = inp[0].detach().float().cpu()
            y = out.detach().float().cpu()
            captured.setdefault("xs", []).append(x.reshape(-1, x.shape[-1]))
            captured.setdefault("ys", []).append(y.reshape(-1, y.shape[-1]))
            captured["n"] = captured.get("n", 0) + x.reshape(-1, x.shape[-1]).shape[0]
        return hook

    report = {}
    # 公平模式（on-policy 链是逐层累积的）：测第 L 层时，只有比它浅的层
    # 装 LUT（与训练该层时的上游分布一致），L 自身和更深的层保持 dense。
    installed = []
    for L in sorted(layers):
        for L2 in layers:
            mods[L2].forward = reps[L2] if L2 in installed else dense_fwd[L2]
        captured.clear()
        h = mods[L].register_forward_hook(mk_hook(L))
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        h.remove()
        installed.append(L)
        if captured.get("n", 0) == 0:
            report[L] = {"error": "hook not fired"}
            continue
        x = torch.cat(captured["xs"])
        y_true = torch.cat(captured["ys"])
        x2d = x.reshape(-1, x.shape[-1])
        y2d = y_true.reshape(-1, y_true.shape[-1])
        y_lut = reps[L].forward_reference(x2d).float()
        cos = torch.nn.functional.cosine_similarity(
            y2d.flatten(), y_lut.flatten(), dim=0).item()
        rel = ((y2d - y_lut).norm() / y2d.norm().clamp_min(1e-9)).item()
        per_tok = torch.nn.functional.cosine_similarity(y2d, y_lut, dim=-1)
        report[L] = {"tokens": int(x2d.shape[0]),
                     "cos": round(cos, 4), "rel_l2": round(rel, 4),
                     "cos_p10": round(per_tok.quantile(0.1).item(), 4),
                     "cos_p50": round(per_tok.median().item(), 4)}
    for L in layers:  # 还原 dense，避免影响调用方后续使用
        mods[L].forward = dense_fwd[L]
    return report


def check_gen(bundle_dir, model_path, layers, data=None, max_new=128,
              max_prompt_tokens=8192, out_path="logs/gen_compare.txt"):
    """同一批 prompt（短→长）各生成两遍：dense 一遍、全装 LUT 一遍，
    输出写文件人工对比。这才是判据（cos 只是代理指标）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vllm_plugin.lut_ffn import LutFfnReplacer

    short_prompts = [
        "请用一句话解释什么是存算一体芯片。",
        "明朝灭亡的主要原因是什么？请简要回答。",
    ]
    long_prompts = []
    if data and os.path.exists(data):
        with open(data) as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                rec = json.loads(line)
                long_prompts.append(
                    rec["document"] + "\n\n问题：" + rec["questions"][0])
    prompts = [("short", p) for p in short_prompts] + \
              [("long", p) for p in long_prompts]

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="balanced_low_0")
    model.eval()

    reps = {L: LutFfnReplacer(f"{bundle_dir}/layer{L}.pt", "cpu")
            for L in layers}
    mods = {L: model.model.layers[L].mlp.shared_expert for L in layers}
    dense_fwd = {L: mods[L].forward for L in layers}

    def gen(prompt):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        ids = ids["input_ids"][:, :max_prompt_tokens]
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for tag, prompt in prompts:
            plen = len(tok(prompt).input_ids)
            fo.write(f"\n{'='*72}\n[{tag}] prompt_tokens={plen}\n"
                     f"PROMPT 末尾: ...{prompt[-200:]}\n{'-'*72}\n")
            print(f"[gen] {tag} prompt_tokens={plen} dense...", flush=True)
            fo.write(f"DENSE:\n{gen(prompt)}\n{'-'*72}\n")
            for L in layers:
                mods[L].forward = reps[L]
            print(f"[gen] {tag} prompt_tokens={plen} LUT{layers}...",
                  flush=True)
            fo.write(f"LUT {layers}:\n{gen(prompt)}\n")
            for L in layers:
                mods[L].forward = dense_fwd[L]
    return {"out": out_path, "prompts": len(prompts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["pack", "qual", "gen"])
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--ckpt_root", default="/data/mamingyu")
    ap.add_argument("--model_path", default="/home/u/downloads/models")
    ap.add_argument("--layers", default="37 38 39")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default=None,
                    help="gen: 长 prompt 来源 jsonl（document+questions）")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--out", default="logs/gen_compare.txt")
    args = ap.parse_args()
    layers = [int(t) for t in args.layers.replace(",", " ").split()]

    if args.mode == "pack":
        rep = check_pack(args.bundle_dir, args.ckpt_root, layers)
    elif args.mode == "qual":
        rep = check_qual(args.bundle_dir, args.model_path, layers,
                         args.device)
    else:
        rep = check_gen(args.bundle_dir, args.model_path, layers,
                        data=args.data, max_new=args.max_new,
                        out_path=args.out)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
