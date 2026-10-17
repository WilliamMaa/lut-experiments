"""
v9 逐层推进多层 LUT 构建（对应 docs/00-setup.md 方向 1）。

流程（校准文本固定、batch=1、无 padding，teacher-forced 非生成）：

  Phase 0  原始 pass：原始模型对每个校准序列完整 forward 一次，
           捕获每个替换层的 ffn_in / residual_in 边界，存盘。
  Phase j  对每层 j（从浅到深）：
    1) 用原始 shared_expert 给 ffn_in 打标签 y（不重跑 LLM，TeacherLabeler 同款思路），
       写出 v6 训练脚本格式的 {dataset}/input + output .pt 分片；
    2) 调 v6 训练脚本训出该层 LUT checkpoint（或 --existing_ckpt 复用已训好的）；
    3) 注入该层 LUT（engine.install，此后不再改动），
       在 residual_in[j] 上跑 decoder slice [j, k]（k=下一替换层），
       捕获经过真实 LUT 的 ffn_in[k] / residual_in[k] 并缓存——
       训练第 k 层时不再需要重跑 0..j 前缀。

成本对比 v6 旧法（每层重新 generate）：前缀只算一次 + 各层仅若干 block 的推进。

红线遵守：显式 device_map（禁 auto）；LUT 查表替代 FFN，无 MLP/HyperNetwork 生成器。

用法示例：
  python collect_layerwise_propagation.py \
    --model_path /path/Qwen3.6-35B-A3B \
    --prompt_file ../../v6/candidate_prompts.jsonl \
    --layers 37,38,39 \
    --teacher_weights 37=/path/teacher_l37.pt 38=/path/teacher_l38.pt 39=/path/teacher_l39.pt \
    --output_root /data/outputs/v9_layerwise_l37_39 \
    --device_map balanced_low_0 --max_len 2048
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lw_common as LWC
from lw_common import BoundaryCapture, parse_layers, shared_expert_path  # noqa: E402

from v6_replacement_engine import V6ReplacementEngine  # noqa: E402


def _norm_key_map(pairs, name):
    m = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        m[int(k)] = v
    return m


def phase0_original_pass(model, tokenizer, texts, layers, max_len, out_root):
    """原始模型完整 forward，缓存各替换层边界。返回 {layer: n_tokens}。"""
    orig_dir = Path(out_root) / "boundary" / "original"
    stats = {idx: 0 for idx in layers}
    capture = BoundaryCapture(model, layers)
    t0 = time.time()
    try:
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = inputs["input_ids"].to(model.device)
            capture.reset()
            with torch.no_grad():
                model(input_ids, use_cache=False)
            for idx in layers:
                ffn = capture.get(idx, "ffn_in")
                LWC.save_shard(str(orig_dir / f"layer{idx}_ffn_seq{i:05d}.pt"), ffn)
                stats[idx] += ffn.shape[0]
                if idx != layers[-1]:  # 最后一层的 residual_in 不会被后续推进使用
                    res = capture.get(idx, "residual_in")
                    LWC.save_shard(str(orig_dir / f"layer{idx}_residual_seq{i:05d}.pt"), res)
            if (i + 1) % 25 == 0:
                print(f"[phase0] {i + 1}/{len(texts)} seqs, {time.time() - t0:.1f}s", flush=True)
    finally:
        capture.close()
    dt = time.time() - t0
    print(f"[phase0] original pass done: {len(texts)} seqs in {dt:.1f}s; tokens/层: {stats}")
    meta = {"n_seqs": len(texts), "max_len": max_len,
            "tokens_per_layer": stats, "time_s": dt}
    (Path(orig_dir) / "_phase0_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return stats, dt


def _phase0_cache(out_root, layers, n_seqs, max_len):
    """phase0 断点：分片齐全且配置一致则跳过最贵的原始 pass，直接复用缓存。"""
    orig_dir = Path(out_root) / "boundary" / "original"
    meta_p = orig_dir / "_phase0_meta.json"
    if not meta_p.is_file():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("n_seqs", 0) < n_seqs or meta.get("max_len") != max_len:
        return None
    for idx in layers:
        if len(list(orig_dir.glob(f"layer{idx}_ffn_*.pt"))) < n_seqs:
            return None
    print(f"[phase0] 缓存命中（{meta['n_seqs']} seqs），跳过原始 pass")
    return meta["tokens_per_layer"], meta["time_s"]


def _shard_source_dirs(out_root, idx):
    """第 idx 层的边界分片目录：若已有推进缓存（prop）则优先，否则用原始 pass。"""
    base = Path(out_root) / "boundary"
    prop = base / "prop" / f"layer{idx}"
    orig = base / "original"
    if prop.is_dir() and any(prop.glob("*_ffn_*.pt")):
        ffn = LWC.shard_paths(str(prop))
        res = sorted(prop.glob("*_residual_*.pt"))
        res = [str(p) for p in res]
        return ffn, res
    ffn = sorted(orig.glob(f"layer{idx}_ffn_*.pt"))
    res = sorted(orig.glob(f"layer{idx}_residual_*.pt"))
    return [str(p) for p in ffn], [str(p) for p in res]


def label_and_write_dataset(model, idx, ffn_shards, out_root, dtype):
    """TeacherLabeler 同款：只对边界 x 用原始 shared_expert 打标签，不重跑 LLM。"""
    se = eval(shared_expert_path(idx), {"model": model})
    dev = next(se.parameters()).device
    dt = next(se.parameters()).dtype
    ds_in = Path(out_root) / "dataset" / f"layer{idx}" / "input"
    ds_out = Path(out_root) / "dataset" / f"layer{idx}" / "output"
    n_tokens = 0
    t0 = time.time()
    for n, sp in enumerate(ffn_shards):
        x = torch.load(sp, map_location="cpu", weights_only=False).to(dev).to(dt)
        with torch.no_grad():
            y = se(x)
        LWC.save_shard(str(ds_in / f"shard_{n:05d}.pt"), x.to(dtype).cpu())
        LWC.save_shard(str(ds_out / f"shard_{n:05d}.pt"), y.to(dtype).cpu())
        n_tokens += x.shape[0]
    dt_sec = time.time() - t0
    print(f"[layer{idx}] labeled {n_tokens} tokens in {dt_sec:.1f}s -> {ds_in.parent}")
    return n_tokens, dt_sec, str(ds_in), str(ds_out)


def train_layer(idx, ds_in, ds_out, teacher_path, out_root, args):
    out_dir = Path(out_root) / f"train_layer{idx}"
    cmd = [
        sys.executable, os.path.join(LWC.V6_TRAINING, "build_lut_ffn_output_v3_shared_coarse.py"),
        "--teacher_weight_path", teacher_path,
        "--dataset_dir", ds_in,
        "--output_dataset_dir", ds_out,
        "--output_root", str(out_dir),
        "--device", args.train_device,
        "--calib_size", str(args.calib_size),
        "--eval_size", str(args.eval_size),
        "--finetune_epochs", str(args.finetune_epochs),
        "--seed", str(args.seed),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.extra_train_args:
        cmd += args.extra_train_args.split()
    print(f"[layer{idx}] training: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # 已知场景：训练干完后子进程 CUDA 析构挂死，人工杀进程后走到这里。
        # 产物齐全就说明训练完成，不当作失败（共享机上的常规操作）。
        if _v3_train_complete(out_dir / "checkpoints"):
            print(f"[layer{idx}] [warn] 训练子进程退出码 {e.returncode}，"
                  f"但产物齐全（析构挂死被杀场景），继续", flush=True)
        else:
            raise
    dt = time.time() - t0
    ckpt = out_dir / "checkpoints"
    if not _v3_train_complete(ckpt):
        raise RuntimeError(f"Training finished but v3 checkpoints incomplete: {ckpt}")
    print(f"[layer{idx}] training done in {dt:.1f}s, checkpoints at {ckpt}")
    return str(ckpt), dt


def _v3_train_complete(ckpt_dir: Path) -> bool:
    """v3 训练产物齐全判定：shared_coarse.pt + 连续编号的 residual_g*.pt。"""
    if not (Path(ckpt_dir) / "shared_coarse.pt").is_file():
        return False
    gid = 0
    while (Path(ckpt_dir) / f"residual_g{gid}.pt").is_file():
        gid += 1
    return gid > 0


def ensure_v4_checkpoint(ckpt_dir, out_root, idx, device):
    """V6ReplacementEngine 只认 replacement_g*.pt（v4 格式）；v3 目录先转换。"""
    ckpt_dir = Path(ckpt_dir)
    if any(ckpt_dir.glob("replacement_g*.pt")):
        return str(ckpt_dir)
    conv_dir = Path(out_root) / f"converted_layer{idx}"
    conv_ckpt = conv_dir / "checkpoints"
    if any(conv_ckpt.glob("replacement_g*.pt")):
        print(f"[layer{idx}] v4 转换已存在，跳过: {conv_ckpt}")
        return str(conv_ckpt)
    cmd = [
        sys.executable, os.path.join(LWC.V6_ROOT, "scripts", "conversion",
                                     "convert_v3_to_v4_checkpoints.py"),
        "--v3_checkpoint_dir", str(ckpt_dir),
        "--output_root", str(conv_dir),
        "--device", device,
    ]
    print(f"[layer{idx}] v3->v4: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return str(conv_ckpt)


def propagate(model, idx_from, idx_to, res_shards, out_root):
    """
    在 residual_in[idx_from] 上跑 decoder slice [idx_from, idx_to]，
    捕获经过真实 LUT 的 ffn_in[idx_to] / residual_in[idx_to]。
    （多跑 idx_to 这一个 block 是为了让它的 pre-hook 自然触发；输出弃用。）
    """
    capture = BoundaryCapture(model, [idx_to])
    prop_dir = Path(out_root) / "boundary" / "prop" / f"layer{idx_to}"
    t0 = time.time()
    try:
        for n, sp in enumerate(res_shards):
            h = torch.load(sp, map_location="cpu", weights_only=False)  # residual 分片已是 [1,S,H]
            capture.reset()
            with torch.no_grad():
                LWC.run_decoder_slice(model, h, idx_from, idx_to + 1)
            ffn = capture.get(idx_to, "ffn_in")
            res = capture.get(idx_to, "residual_in")
            LWC.save_shard(str(prop_dir / f"layer{idx_to}_ffn_seq{n:05d}.pt"), ffn)
            LWC.save_shard(str(prop_dir / f"layer{idx_to}_residual_seq{n:05d}.pt"), res)
            if (n + 1) % 50 == 0:
                print(f"[propagate {idx_from}->{idx_to}] {n + 1}/{len(res_shards)} "
                      f"{time.time() - t0:.1f}s", flush=True)
    finally:
        capture.close()
    dt = time.time() - t0
    print(f"[propagate {idx_from}->{idx_to}] {len(res_shards)} seqs in {dt:.1f}s")
    return dt


def main():
    ap = argparse.ArgumentParser(description="v9 layer-wise LUT construction (propagation)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt_file", required=True, help=".jsonl(prompt/text) 或 .txt(每行一条)")
    ap.add_argument("--layers", type=str, required=True, help="替换层，浅->深，逗号分隔，如 37,38,39")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--teacher_weights", type=str, nargs="*", default=None,
                    help="每层 teacher expert .pt：37=/path/teacher_l37.pt ...（训练阶段必填，可用 extract_shared_expert.py 生成）")
    ap.add_argument("--device_map", default="balanced_low_0",
                    help="显式 device_map；禁止 auto")
    ap.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--shard_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
                    help="训练数据分片存储 dtype（边界缓存始终 fp32）")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_prompts", type=int, default=None,
                    help="校准序列条数；默认用满整个 prompt 文件")
    # 训练
    ap.add_argument("--no_train", action="store_true", help="只采集+推进，不训练（需 --existing_ckpt）")
    ap.add_argument("--existing_ckpt", type=str, nargs="*", default=None,
                    help="复用已训好的 checkpoint：37=/path/to/checkpoints ...")
    ap.add_argument("--train_device", default="cuda:0")
    ap.add_argument("--calib_size", type=int, default=400000,
                    help="训练用校准样本数；超过采集量时按比例自动下调")
    ap.add_argument("--eval_size", type=int, default=100000)
    ap.add_argument("--finetune_epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra_train_args", default=None, help="透传给 v6 训练脚本的额外参数")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    if layers != sorted(layers):
        raise ValueError("--layers 必须按从浅到深排列")
    teacher_map = _norm_key_map(args.teacher_weights, "teacher_weights")
    existing_map = _norm_key_map(args.existing_ckpt, "existing_ckpt")

    model, tokenizer = LWC.load_model(args.model_path, args.device_map, args.torch_dtype)
    n_layers = len(model.model.layers)
    for idx in layers:
        if not (0 <= idx < n_layers):
            raise ValueError(f"layer {idx} out of range ({n_layers} layers)")
    if layers[-1] == n_layers - 1:
        print(f"[warn] 最后一层 {layers[-1]} 的 shared_expert 输入即 lm_head 输入，确认替换目标无误")

    texts = LWC.load_prompt_texts(args.prompt_file, args.max_prompts)
    out_root = args.output_root
    Path(out_root).mkdir(parents=True, exist_ok=True)

    log = {"args": vars(args), "layers": layers, "phases": {}}

    cached = _phase0_cache(out_root, layers, len(texts), args.max_len)
    if cached is not None:
        stats, dt = cached
    else:
        stats, dt = phase0_original_pass(model, tokenizer, texts, layers, args.max_len, out_root)
    log["phases"]["original_pass"] = {"time_s": dt, "tokens_per_layer": stats}

    engines = []
    dtype = getattr(torch, args.shard_dtype)
    try:
        for li, idx in enumerate(layers):
            ffn_shards, res_shards = _shard_source_dirs(out_root, idx)
            if not ffn_shards:
                raise RuntimeError(f"layer {idx}: no ffn_in shards found")

            n_tokens, label_dt, ds_in, ds_out = label_and_write_dataset(
                model, idx, ffn_shards, out_root, dtype)

            need = args.calib_size + args.eval_size
            if not args.no_train and n_tokens < need:
                scale = n_tokens * 0.95 / need
                args.calib_size = max(int(args.calib_size * scale), 1000)
                args.eval_size = max(int(args.eval_size * scale), 200)
                print(f"[warn] 采集 {n_tokens} token < calib+eval {need}，"
                      f"自动下调为 calib={args.calib_size} eval={args.eval_size}")

            if args.no_train:
                if idx not in existing_map:
                    raise ValueError(f"--no_train 需要提供 --existing_ckpt {idx}=...")
                train_dt = 0.0
                v3_ckpt_dir = existing_map[idx]
            else:
                if idx not in teacher_map:
                    raise ValueError(f"layer {idx} 缺少 teacher 权重：--teacher_weights {idx}=...")
                trained_ckpt = Path(out_root) / f"train_layer{idx}" / "checkpoints"
                if _v3_train_complete(trained_ckpt):
                    print(f"[layer{idx}] 训练产物已存在，跳过训练: {trained_ckpt}")
                    v3_ckpt_dir, train_dt = str(trained_ckpt), 0.0
                else:
                    v3_ckpt_dir, train_dt = train_layer(
                        idx, ds_in, ds_out, teacher_map[idx], out_root, args)
            ckpt_dir = ensure_v4_checkpoint(v3_ckpt_dir, out_root, idx, args.train_device)

            engine = V6ReplacementEngine(
                model=model, layer_idx=idx, checkpoint_dir=ckpt_dir,
                device=None, hook_path=shared_expert_path(idx),
            )
            engine.install()
            engines.append(engine)

            entry = {"tokens": n_tokens, "label_time_s": label_dt,
                     "train_time_s": train_dt, "ckpt_dir": ckpt_dir}
            if li + 1 < len(layers):
                nxt = layers[li + 1]
                if not res_shards:
                    raise RuntimeError(f"layer {idx}: propagation 需要 residual_in 分片")
                entry["propagate_time_s"] = propagate(model, idx, nxt, res_shards, out_root)
            log["phases"][f"layer{idx}"] = entry
            with open(Path(out_root) / "build_log.json", "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
    finally:
        for e in engines:
            e.uninstall()

    total_train = sum(v.get("train_time_s", 0) for k, v in log["phases"].items()
                      if k.startswith("layer"))
    total_prop = sum(v.get("propagate_time_s", 0) for k, v in log["phases"].items()
                     if k.startswith("layer"))
    log["summary"] = {
        "original_pass_s": dt,
        "total_label_s": sum(v["label_time_s"] for k, v in log["phases"].items() if k.startswith("layer")),
        "total_train_s": total_train,
        "total_propagate_s": total_prop,
        "n_layers": len(layers),
    }
    with open(Path(out_root) / "build_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(json.dumps(log["summary"], indent=2))
    print("下一步评估（v6 评估脚本）：")
    cli = " ".join(
        f"--layer_idx {i} --checkpoint_dir {log['phases'][f'layer{i}']['ckpt_dir']}"
        for i in layers)
    print(f"python {os.path.join(LWC.V6_EVAL, 'run_multilayer_model_eval.py')} "
          f"--model_path {args.model_path} {cli} --device_map {args.device_map} "
          f"--torch_dtype {args.torch_dtype} --output_json {out_root}/eval_result.json")


if __name__ == "__main__":
    main()
