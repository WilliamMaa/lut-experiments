#!/usr/bin/env python3
"""
Multi-turn dialogue evaluation for multi-layer V6 LUT replacement.

Tests realistic chat scenarios with system prompts, multiple user/assistant turns,
and structured output requirements.

Usage:
  python -u run_multilayer_dialogue_eval.py \
    --model_path /home/u/downloads/models/Qwen3.6-35B-A3B \
    --layer_idx 37 --checkpoint_dir outputs_l37_as_v4/checkpoints \
    --layer_idx 38 --checkpoint_dir outputs_l38_as_v4/checkpoints \
    --layer_idx 39 --checkpoint_dir outputs_l39_as_v4/checkpoints \
    --device_map balanced_low_0 --torch_dtype bfloat16 \
    --max_new_tokens 2048 \
    --output_json dialogue_eval.json
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v6_replacement_engine import V6ReplacementEngine


DEFAULT_SCENARIOS = [
    {
        "name": "customer_support",
        "system": "You are a helpful customer support agent for an online bookstore. Be polite, concise, and ask clarifying questions when needed.",
        "turns": [
            "Hi, I ordered a book last week but it hasn't arrived yet. My order number is #12345.",
            "I ordered 'The Structure of Scientific Revolutions'. Can you check the tracking?",
            "Actually, I think I entered the wrong shipping address. Can I still change it?",
            "No, I'll keep the current address then. When should I expect delivery?",
        ],
    },
    {
        "name": "coding_debugging",
        "system": "You are an expert Python programmer. Help the user debug and improve their code. Provide clear explanations.",
        "turns": [
            "I'm getting a KeyError when accessing a dictionary in my loop. What could be wrong?",
            "Here's the code:\nfor item in data:\n    print(item['name'])\nSome items don't have 'name'.",
            "Can you rewrite the loop to handle missing keys gracefully and also count how many items are missing the name?",
            "Now I want to save the cleaned data to a JSON file. How should I do that?",
        ],
    },
    {
        "name": "socratic_tutor",
        "system": "You are a Socratic tutor in economics. Do not give direct answers. Guide the student to discover the answer through questions.",
        "turns": [
            "Why does inflation happen?",
            "I think it's because the government prints too much money.",
            "What happens to prices when more people want to buy the same amount of goods?",
            "So inflation can also happen from the demand side? Can you give me an example?",
        ],
    },
    {
        "name": "structured_output",
        "system": "You are a structured data assistant. Always respond with valid JSON objects. Do not include explanatory text outside the JSON.",
        "turns": [
            "Create a JSON profile for a software engineer named Alice, age 30, skills Python and Rust.",
            "Add a 'projects' field with two projects, each having name and status.",
            "Now add a 'contact' field with email and phone, but mark the phone as private: true.",
            "Convert the entire profile to a JSON schema description with required fields listed.",
        ],
    },
    {
        "name": "creative_writing",
        "system": "You are a creative writing coach. Help the user develop a short story with vivid characters and coherent plot.",
        "turns": [
            "I want to write a short story about a detective who can hear objects' memories.",
            "Give me a name and backstory for the detective.",
            "Now introduce the main mystery: a stolen object with a dark history.",
            "Write the opening paragraph of the story.",
        ],
    },
]


def clean_response(text):
    """Strip thinking-process preamble and markdown fences from generated text."""
    # Native thinking tags (Qwen3 / DeepSeek-R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[思考开始\].*?\[思考结束\]", "", text, flags=re.DOTALL)

    # Free-form "Here's a thinking process..." preamble: drop everything up to
    # the last code fence or up to a clear answer boundary.
    if re.search(r"(?i)here'?s a thinking process|thinking process:|思考过程", text):
        parts = re.split(r"\n```\w*\n", text)
        if len(parts) >= 2:
            text = "```\n" + parts[-1]
        else:
            text = re.sub(r"(?is)^.*?\n\n(?=[A-Z\u4e00-\u9fff]|\d+[\.\)\s])", "", text, count=1)

    # Remove markdown fences for JSON/structured-output scenarios
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def generate_turn(model, tokenizer, messages, device, max_new_tokens=1024, do_sample=False):
    """Generate one assistant turn given the conversation history."""
    model.eval()

    # Inject a no-think instruction into the system prompt.  We copy messages so
    # the caller's conversation history stays clean.
    messages = [dict(m) for m in messages]
    no_think_text = (
        "直接给出最终回复，不要输出思考过程、分析步骤、'Here's a thinking process' "
        "或任何元说明。"
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + no_think_text
    else:
        messages.insert(0, {"role": "system", "content": no_think_text})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # Qwen3+ supports enable_thinking=False to suppress native CoT mode.
        tmpl_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        try:
            input_ids = tokenizer.apply_chat_template(messages, **tmpl_kwargs, enable_thinking=False)
        except TypeError:
            input_ids = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
        # apply_chat_template may return a BatchEncoding or a tensor depending on
        # the tokenizer/transformers version; normalize to a 2-D LongTensor.
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
    else:
        # Fallback: concatenate manually
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt_text += "\nassistant:"
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_response(text)

def run_dialogue_scenario(model, tokenizer, engines, scenario, device, max_new_tokens, baseline=False):
    """Run one multi-turn scenario with verbose logging."""
    mode = "BASELINE" if baseline else "LUT"
    print(f"[{mode}] Starting scenario: {scenario['name']} ({len(scenario['turns'])} turns)")
    messages = [{"role": "system", "content": scenario["system"]}]
    conversation = []

    for i, user_text in enumerate(scenario["turns"]):
        print(f"[{mode}][{scenario['name']}] Turn {i + 1}/{len(scenario['turns'])}: user input len={len(user_text)}")
        messages.append({"role": "user", "content": user_text})

        t0 = time.time()
        response = generate_turn(model, tokenizer, messages, device, max_new_tokens=max_new_tokens)
        t1 = time.time()

        print(f"[{mode}][{scenario['name']}] Turn {i + 1} generated: {len(response)} chars, time={t1 - t0:.2f}s")
        messages.append({"role": "assistant", "content": response})
        conversation.append({
            "turn": i + 1,
            "user": user_text,
            "assistant": response,
        })

    print(f"[{mode}] Finished scenario: {scenario['name']}")
    return {
        "name": scenario["name"],
        "system": scenario["system"],
        "conversation": conversation,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-turn dialogue evaluation with V6 LUT")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--layer_idx", action="append", type=int, required=True)
    parser.add_argument("--checkpoint_dir", action="append", type=str, required=True)
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--scenarios_json", default=None,
                        help="Path to JSON file with custom scenarios. If not provided, uses defaults.")
    parser.add_argument("--scenario", default=None,
                        help="Run only one scenario by name (e.g. 'customer_support').")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip baseline run, only run with LUT.")
    parser.add_argument("--no_verify_replacement", action="store_true")
    args = parser.parse_args()

    if len(args.layer_idx) != len(args.checkpoint_dir):
        raise ValueError("Number of --layer_idx and --checkpoint_dir must match")

    dtype = getattr(torch, args.torch_dtype)
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )
    device = next(model.parameters()).device
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Install LUT engines
    engines = []
    for idx, ckpt_dir in zip(args.layer_idx, args.checkpoint_dir):
        hook_path = f"model.model.layers[{idx}].mlp.shared_expert"
        engine = V6ReplacementEngine(model, idx, ckpt_dir, device=device, hook_path=hook_path)
        engine.install()
        engines.append(engine)
        print(f"[V6Engine] Installed layer {idx} LUT from {ckpt_dir}")

    if not args.no_verify_replacement:
        for engine in engines:
            ok = engine.verify_replacement(model.config.hidden_size)
            if not ok:
                print(f"[Warning] Replacement verification failed for layer {engine.layer_idx}")

    # Load scenarios
    if args.scenarios_json:
        with open(args.scenarios_json, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
    else:
        scenarios = DEFAULT_SCENARIOS

    if args.scenario:
        scenarios = [s for s in scenarios if s["name"] == args.scenario]
        if not scenarios:
            raise ValueError(f"Scenario '{args.scenario}' not found. Available: {[s['name'] for s in (DEFAULT_SCENARIOS if not args.scenarios_json else json.load(open(args.scenarios_json)))]}")
        print(f"Running single scenario: {args.scenario}")

    # Run baseline (no LUT)
    baseline_results = []
    if not args.no_baseline:
        print("\n===== Baseline (no LUT) =====")
        for engine in engines:
            engine.uninstall()
        baseline_results = [run_dialogue_scenario(model, tokenizer, engines, s, device, args.max_new_tokens, baseline=True) for s in scenarios]

    # Run with LUT
    print("\n===== With V6 Multi-Layer LUT =====")
    for engine in engines:
        engine.install()
    lut_results = [run_dialogue_scenario(model, tokenizer, engines, s, device, args.max_new_tokens, baseline=False) for s in scenarios]

    for engine in engines:
        engine.uninstall()

    summary = {
        "model_path": args.model_path,
        "layers": args.layer_idx,
        "checkpoint_dirs": args.checkpoint_dir,
        "device": str(device),
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
        "scenarios": [s["name"] for s in scenarios],
    }
    if baseline_results:
        summary["baseline"] = baseline_results
    summary["lut"] = lut_results

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved dialogue evaluation to {args.output_json}")

    print("\n===== Summary =====")
    for s, l in zip(scenarios, lut_results):
        print(f"Scenario: {s['name']}")
        if baseline_results:
            b = next((x for x in baseline_results if x["name"] == s["name"]), None)
            if b:
                print(f"  Baseline turns: {len(b['conversation'])}")
        print(f"  LUT turns:      {len(l['conversation'])}")


if __name__ == "__main__":
    main()
