#!/usr/bin/env python3
"""
run_hf_factacc_eval.py

V6 LUT 模型级评估：在 HF transformers 离线流程上跑 v8 协议的长上下文
多轮 fact_acc 评测 + 可选 PPL，用于 v6 LUT 替换（40 层 FFN 边界扫描）的
质量评估。

fact_acc 协议照抄 v8/tools/eval_longctx_server.py：
  每个 doc 一个 session，先 anchor turn（整个 document），固定回复
  "好的，我已读完。"，然后 questions 逐条追加进同一 message 历史。
  打分 = 答案包含该题全部 ground-truth 字符串（子串 AND）。

KV 增量复用是硬要求（每轮全量重 forward 单 doc 要 22 小时）：
  turn t 只对"本轮新增 token"做 forward，传入上轮 generate 留下的
  past_key_values；若渲染后的 prompt 不再是上轮 full_ids 的前缀
  （render mismatch），退化全量 forward 并计数。

用法：
  python -u run_hf_factacc_eval.py \
    --model_path /path/to/Qwen3.6-35B-A3B \
    --data_file ../../v8/data/longctx_multi_turn_32768.jsonl \
    --layer_idx 17 --checkpoint_dir outputs_l17_as_v4/checkpoints \
    --layer_idx 18 --checkpoint_dir outputs_l18_as_v4/checkpoints \
    --ppl_file eval_texts.txt \
    --output_json factacc_l17_18.json
"""

import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v6_replacement_engine import V6ReplacementEngine

ANCHOR_REPLY = "好的，我已读完。"


def load_docs(data_file: str, max_docs: int):
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"data_file not found: {data_file}")
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    if max_docs > 0:
        docs = docs[:max_docs]
    return docs


def load_eval_texts(eval_file: str, max_samples: int):
    texts = []
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"eval_file not found: {eval_file}")
    if path.suffix in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", obj.get("sentence", "")))
                if text:
                    texts.append(text)
                if len(texts) >= max_samples:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                if len(texts) >= max_samples:
                    break
    return texts


def compute_ppl(model, tokenizer, texts, device, max_length=512, batch_size=1):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= 1:
            continue
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        n_tokens = input_ids.shape[1]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def render_prompt_ids(tokenizer, messages, enable_thinking: Optional[bool]):
    """Render messages through the chat template.

    Qwen3-style templates accept enable_thinking=False (skip reasoning);
    fall back gracefully if the template does not accept the kwarg.
    """
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def template_supports_kwarg(tokenizer, kwarg: str) -> Optional[bool]:
    """Return True/False if we can tell, None if unknown (do not pass it)."""
    try:
        import inspect
        template = tokenizer.get_chat_template()
        if template is None:
            return None
        if isinstance(template, str):
            # Jinja string: best-effort text check
            return f"{kwarg}" in template
        try:
            params = inspect.signature(template).parameters
            return kwarg in params
        except (TypeError, ValueError):
            return None
    except Exception:
        return None


def generate_reply(model, tokenizer, prompt_ids, device,
                   max_new_tokens, past_key_values=None, past_attn_len=0):
    """Feed (possibly incremental) tokens and return (answer_text, new_ids, past_key_values).

    If past_key_values is given, prompt_ids is only the NEW tokens for this
    turn and past_attn_len is the number of tokens already in the cache
    (previous prompt + previous generations), used to build the attention
    mask. The returned new_ids are just the generated tokens.
    """
    prompt_ids = prompt_ids.to(device)
    if past_key_values is None:
        inputs = {"input_ids": prompt_ids}
    else:
        attn = torch.ones(1, past_attn_len + prompt_ids.shape[1], dtype=torch.long, device=device)
        inputs = {
            "input_ids": prompt_ids,
            "attention_mask": attn,
            "past_key_values": past_key_values,
        }
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = outputs.sequences[0, prompt_ids.shape[1]:]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    return answer, new_ids, outputs.past_key_values


def run_factacc(model, tokenizer, docs, device, max_new_tokens, enable_thinking):
    model.eval()
    n_correct = 0
    n_total = 0
    per_qtype = {}
    results = []
    n_fallback = 0  # render mismatch -> full forward

    for di, doc in enumerate(docs):
        doc_t0 = time.time()
        doc_correct = 0
        messages = [{"role": "user", "content": doc["document"]}]

        prev_full_ids = None  # token ids covering prompt+gen of last turn
        past_key_values = None
        past_attn_len = 0

        # anchor turn
        try:
            prompt_ids = render_prompt_ids(tokenizer, messages, enable_thinking)
            _, _, past_key_values = generate_reply(
                model, tokenizer, prompt_ids, device, 8,
                past_key_values=None,
            )
            prev_full_ids = prompt_ids[0].tolist()
            past_attn_len = len(prev_full_ids)
            messages.append({"role": "assistant", "content": ANCHOR_REPLY})
        except Exception as e:
            print(f"[doc {di}] ANCHOR FAIL: {e!r}")
            results.append({"doc_index": di, "error": repr(e)})
            continue

        questions = doc["questions"]
        answers = doc["answers"]
        qtypes = doc["qtype"]
        for qi, (q, gts, qt) in enumerate(zip(questions, answers, qtypes)):
            messages.append({"role": "user", "content": q})
            t0 = time.time()
            try:
                prompt_ids = render_prompt_ids(tokenizer, messages, enable_thinking)
                prompt_list = prompt_ids[0].tolist()
                if (prev_full_ids is not None
                        and len(prompt_list) >= len(prev_full_ids)
                        and prompt_list[:len(prev_full_ids)] == prev_full_ids):
                    # prefix matches: feed only the incremental tokens
                    delta_ids = prompt_ids[:, len(prev_full_ids):]
                    ans, new_ids, past_key_values = generate_reply(
                        model, tokenizer, delta_ids, device, max_new_tokens,
                        past_key_values=past_key_values,
                        past_attn_len=past_attn_len,
                    )
                    prev_full_ids = prompt_list + new_ids.tolist()
                else:
                    n_fallback += 1
                    print(f"[doc {di} q {qi}] render mismatch, full forward fallback")
                    ans, new_ids, past_key_values = generate_reply(
                        model, tokenizer, prompt_ids, device, max_new_tokens,
                        past_key_values=None,
                    )
                    prev_full_ids = prompt_list + new_ids.tolist()
                past_attn_len = len(prev_full_ids)
                dt = time.time() - t0
            except Exception as e:
                ans, dt = f"<ERROR {e!r}>", time.time() - t0
                # cache is now unreliable; force full forward next turn
                prev_full_ids = None
                past_key_values = None
                past_attn_len = 0

            ok = all(gt in ans for gt in gts)
            n_total += 1
            doc_correct += ok
            n_correct += ok
            per_qtype.setdefault(qt, [0, 0])
            per_qtype[qt][0] += ok
            per_qtype[qt][1] += 1
            results.append({"doc_index": di, "q_index": qi, "qtype": qt,
                            "question": q, "gt": gts, "answer": ans,
                            "correct": ok, "seconds": round(dt, 2)})
            print(f"[doc {di} q {qi}] correct={ok} {dt:.1f}s")
            messages.append({"role": "assistant", "content": ans})

        nq = len(questions)
        actual_tokens = doc.get("meta", {}).get("actual_tokens", -1)
        print(f"[doc {di}] acc={doc_correct}/{nq} "
              f"({time.time() - doc_t0:.0f}s, ~{actual_tokens:.0f} tokens)")

    return n_correct, n_total, per_qtype, results, n_fallback


def main():
    parser = argparse.ArgumentParser(description="V6 LUT long-context multi-turn fact_acc + PPL eval")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_file", required=True, help="v8 longctx_multi_turn jsonl")
    parser.add_argument("--max_docs", type=int, default=0, help="0 = all docs")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--layer_idx", nargs="+", type=int, default=None,
                        help="Layer indices to replace. Omit for full baseline.")
    parser.add_argument("--checkpoint_dir", nargs="+", type=str, default=None,
                        help="LUT checkpoint dirs (_as_v4/checkpoints), one per --layer_idx.")
    parser.add_argument("--ppl_file", default=None, help="optional corpus for PPL (txt or jsonl)")
    parser.add_argument("--ppl_max_seqs", type=int, default=32)
    parser.add_argument("--ppl_seq_len", type=int, default=512)
    parser.add_argument("--device_map", default="balanced_low_0",
                        help="HuggingFace device_map, e.g. balanced_low_0. Do NOT use 'auto'.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    layer_idx = args.layer_idx or []
    checkpoint_dir = args.checkpoint_dir or []
    if len(layer_idx) != len(checkpoint_dir):
        raise ValueError("Number of --layer_idx and --checkpoint_dir must match")

    if args.device_map == "auto":
        raise ValueError("device_map='auto' is forbidden. Use an explicit map like 'balanced_low_0'.")

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading model: {args.model_path}")
    if args.device_map is not None:
        print(f"  device_map={args.device_map}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=args.device_map,
        )
        device = next(model.parameters()).device
        print(f"  first-layer device is {device}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Qwen3 reasoning models must answer directly; only pass the kwarg if
    # the template seems to accept it.
    enable_thinking = False if template_supports_kwarg(tokenizer, "enable_thinking") else None

    # Install engines for all replacement layers
    engines = []
    for idx, ckpt_dir in zip(layer_idx, checkpoint_dir):
        hook_path = f"model.model.layers[{idx}].mlp.shared_expert"
        hook_module = eval(hook_path, {"model": model})
        engine_device = next(hook_module.parameters()).device
        print(f"Installing engine for layer {idx} from {ckpt_dir} on {engine_device}")
        engine = V6ReplacementEngine(
            model=model,
            layer_idx=idx,
            checkpoint_dir=ckpt_dir,
            device=engine_device,
            hook_path=hook_path,
        )
        engine.install()
        engines.append(engine)

    docs = load_docs(args.data_file, args.max_docs)
    print(f"Evaluating fact_acc on {len(docs)} docs "
          f"(layers={layer_idx or 'baseline'})")

    n_correct, n_total, per_qtype, results, n_fallback = run_factacc(
        model, tokenizer, docs, device, args.max_new_tokens, enable_thinking,
    )
    acc = n_correct / max(n_total, 1)
    print()
    print(f"OVERALL fact_acc = {acc:.4f}  ({n_correct}/{n_total})")
    for qt, (c, t) in sorted(per_qtype.items()):
        print(f"  {qt:>18s}: {c}/{t} = {c / max(t, 1):.4f}")
    print(f"render mismatch fallbacks: {n_fallback}")

    ppl = None
    if args.ppl_file:
        texts = load_eval_texts(args.ppl_file, args.ppl_max_seqs)
        print(f"Evaluating PPL on {len(texts)} sequences (seq_len={args.ppl_seq_len})")
        ppl = compute_ppl(model, tokenizer, texts, device, max_length=args.ppl_seq_len)
        print(f"PPL: {ppl:.4f}")

    for engine in engines:
        engine.uninstall()

    summary = {
        "fact_acc": acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "per_qtype": {k: {"correct": v[0], "total": v[1]}
                      for k, v in per_qtype.items()},
        "results": results,
        "n_render_fallback": n_fallback,
        "layers": layer_idx,
        "ppl": ppl,
        "config": {
            "model_path": args.model_path,
            "data_file": args.data_file,
            "max_docs": args.max_docs,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint_dirs": checkpoint_dir,
            "ppl_file": args.ppl_file,
            "ppl_max_seqs": args.ppl_max_seqs,
            "ppl_seq_len": args.ppl_seq_len,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"Summary written to {args.output_json}")


if __name__ == "__main__":
    main()
