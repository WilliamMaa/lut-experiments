#!/usr/bin/env python3
"""
build_candidate_pool.py

为方案 1 构造 1000 条混合候选 prompt 池。

配比（默认）：
  - 300 条 LIFEBench 长文/长度约束
  - 150 条 LongGenBench 多约束长生成
  - 200 条 Infinity-Instruct / 综合真实 instruction
  - 100 条数学与推理
  - 100 条代码与技术（LiveCodeBench）
  - 100 条 Aya 多语言
  - 50  条用户自定义复杂 prompt

流程：
  1. 从各数据源按比例抽取原始真实 prompt
  2. 用固定模板把短问题改写成 500–2048 token 的长输出任务
  3. 混合、去重、添加元数据
  4. 输出 JSONL 供 collect_onpolicy_data.py 使用

JSONL 字段：
  {
    "prompt": "改写后的长输出 prompt",
    "source": "lifebench",
    "language": "zh",
    "task": "long_analysis",
    "format": "sectioned_essay",
    "target_length": 2048,
    "original": "原始 prompt（可选）"
  }

依赖：
  pip install datasets

python build_candidate_pool.py \
  --output_file ./candidate_prompts.jsonl \
  --hf_token 
"""

import os
import json
import gc
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

# datasets 是可选依赖；如果某个数据集加载失败，该来源会被跳过
DATASETS_AVAILABLE = False
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    pass

_HF_TOKEN: Optional[str] = None  # set in main() from --hf_token or HF_TOKEN env


# =============================================================================
# 1. 模板
# =============================================================================

TEMPLATES = {
    "multi_dim_analysis": """请从至少五个不同维度深入分析下列问题，重点说明各因素之间的因果关系和反馈机制。要求分章节展开论述，最后给出综合判断。

目标长度：约{length}字。

原始问题：{prompt}""",

    "tech_report": """围绕下列任务撰写一份完整技术报告，内容应包括：背景、需求、核心方案、实现步骤、复杂度或成本分析、潜在风险、测试方式以及替代方案。不要只给概览。

目标长度：约{length}字。

任务：{prompt}""",

    "reasoning_selfcheck": """解决下列问题。请先拆解已知条件与目标，再给出至少两种可能的解决思路，选择其中一种逐步证明，检查边界条件，并指出最容易出错的步骤。不要只给最终答案。

目标长度：约{length}字。

问题：{prompt}""",

    "long_format": """请完成下列任务，输出必须包含不少于8个有实质内容的章节，每章承担不同作用，避免内容重复。最后增加总结和自我检查。

目标长度：约{length}字。

任务：{prompt}""",

    "cross_language_ja": """请阅读下面的中文任务，并用日文给出完整答案。关键术语首次出现时请附中文或英文原词。答案应为结构化长文。

目标长度：约{length}字。

任务：{prompt}""",

    "cross_language_en": """请阅读下面的中文任务，并用英文给出完整答案。答案应为结构化长文，包含分析、论证和结论。

目标长度：约{length} words.

任务：{prompt}""",

    "code_full": """请完成下列编程任务。要求包括：需求分析、算法思路与复杂度、完整可运行代码、关键函数解释、测试用例、边界情况讨论，以及一个可能的替代实现。

目标长度：约{length}字。

任务：{prompt}""",

    "math_full": """请完整推导下列问题的答案。要求包括：已知条件整理、目标明确、至少两种思路、选择一种逐步证明、边界条件检查、常见错误说明。

目标长度：约{length}字。

问题：{prompt}""",
}


def apply_template(template_name: str, prompt: str, length: int) -> str:
    template = TEMPLATES.get(template_name, TEMPLATES["multi_dim_analysis"])
    return template.format(prompt=prompt.strip(), length=length)


def choose_template(task: str, language: str, format_hint: Optional[str] = None) -> str:
    """根据任务类型和语言选择默认模板。"""
    if format_hint and format_hint in TEMPLATES:
        return format_hint
    if task in ("coding", "code"):
        return "code_full"
    if task in ("math", "reasoning"):
        return "reasoning_selfcheck"
    if language in ("ja", "en"):
        return f"cross_language_{language}"
    if task in ("analysis", "long_analysis", "essay"):
        return "multi_dim_analysis"
    if task in ("tech_report", "design"):
        return "tech_report"
    return "long_format"


# =============================================================================
# 2. 通用工具
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def deduplicate_prompts(items: List[Dict], key: str = "prompt") -> List[Dict]:
    """按 prompt 文本去重。"""
    seen = set()
    out = []
    for it in items:
        text = it[key].strip()
        if text not in seen:
            seen.add(text)
            out.append(it)
    return out


def downsample(items: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    rng = np.random.RandomState(seed)
    if len(items) <= n:
        return items
    indices = rng.choice(len(items), size=n, replace=False)
    return [items[int(i)] for i in indices]


def length_to_tokens(length_chars: int) -> int:
    """中文字符 ≈ 1 token；英文 ≈ 0.3 token per char；粗略估计。"""
    return int(length_chars * 0.8)


# =============================================================================
# 3. 各数据源加载器
# =============================================================================

def warn_missing(msg: str):
    print(f"[WARNING] {msg}")


def load_hf_dataset(name: str, config: Optional[str] = None, split: str = "train", streaming: bool = False, token: Optional[str] = None):
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library not installed")
    kwargs = {"split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    try:
        return load_dataset(name, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load {name}: {e}")


def _extract_field(example: Dict, candidates: List[str], default: str = "") -> str:
    for c in candidates:
        if c in example and example[c]:
            return str(example[c])
    return default


def _peek_first_example(ds, label: str):
    """打印第一个 example 的 keys 和 values，帮助诊断字段名。"""
    try:
        for example in ds:
            print(f"[{label}] first example keys: {list(example.keys())}")
            for k, v in example.items():
                preview = str(v)[:200].replace("\n", " ")
                print(f"  {k}: {preview}")
            break
    except Exception as e:
        print(f"[{label}] cannot peek: {e}")


def _collect_texts_from_dataset(ds, text_fields: List[str], label: str, n: int) -> List[str]:
    """通用文本抽取，失败时打印第一个 example 帮助调试。"""
    texts = []
    first = True
    for example in ds:
        if first:
            print(f"[{label}] first example keys: {list(example.keys())}")
            first = False
        text = _extract_field(example, text_fields)
        if text:
            texts.append(text)
        if len(texts) >= n:
            break
    return texts


# -----------------------------------------------------------------------------
# 3.1 LIFEBench
# -----------------------------------------------------------------------------

def load_lifebench(n_total: int = 300, seed: int = 42) -> List[Dict]:
    """
    从 LIFEBench 加载长文/长度约束任务。
    默认三档长度：100条 800–1200字；100条 1500–2500字；100条 分6–10节 ~2048 token。
    """
    ds = None
    for split in ["main", "lite", "label", "refactor"]:
        try:
            ds = load_hf_dataset("LIFEBench/LIFEBench", split=split, streaming=True, token=_HF_TOKEN)
            print(f"LIFEBench loaded split={split}")
            break
        except Exception as e:
            if split == "refactor":
                warn_missing(f"LIFEBench load failed: {e}")
                return []
            continue

    texts = _collect_texts_from_dataset(ds, ["raw_data", "prompt", "instruction", "question", "input", "query"], "LIFEBench", n_total * 3)

    print(f"  [LIFEBench] collected {len(texts)} raw texts")
    if len(texts) == 0:
        warn_missing("LIFEBench loaded but extracted 0 valid prompts; check raw_data field format")
        return []
    if len(texts) < n_total:
        warn_missing(f"LIFEBench only collected {len(texts)} items, expected {n_total}")

    collected = [{"original": t, "source": "lifebench", "language": "zh", "task": "long_analysis", "format": "sectioned_essay"} for t in texts]

    # 三等分长度
    n = n_total // 3
    chunks = [collected[i * n:(i + 1) * n] for i in range(3)]
    if n_total % 3:
        chunks[2].extend(collected[n * 3:n_total])

    lengths = [1000, 2000, 2400]
    out = []
    for chunk, length in zip(chunks, lengths):
        for it in chunk:
            it["prompt"] = apply_template("multi_dim_analysis", it["original"], length)
            it["target_length"] = length_to_tokens(length)
            out.append(it)
    return out[:n_total]


# -----------------------------------------------------------------------------
# 3.2 LongGenBench
# -----------------------------------------------------------------------------

def load_longgenbench(n_total: int = 150, seed: int = 42) -> List[Dict]:
    """
    从 LongGenBench 加载多约束长生成任务。
    压缩到 ~2048 token，保留多约束结构。
    """
    ds = None
    for ds_name in ["mozhu/LongGenBench", "mozhu621/LongGenBench"]:
        try:
            ds = load_hf_dataset(ds_name, split="train", streaming=True, token=_HF_TOKEN)
            print(f"LongGenBench loaded from {ds_name}")
            break
        except Exception as e:
            if ds_name == "mozhu621/LongGenBench":
                warn_missing(f"LongGenBench load failed: {e}")
                return []
            continue

    texts = _collect_texts_from_dataset(
        ds,
        ["prompt", "instruction", "question", "input", "query", "content", "text"],
        "LongGenBench",
        n_total * 2,
    )
    print(f"  [LongGenBench] collected {len(texts)} raw texts")
    if len(texts) == 0:
        warn_missing("LongGenBench loaded but extracted 0 valid prompts; check prompt field format")
        return []
    if len(texts) < n_total:
        warn_missing(f"LongGenBench only collected {len(texts)} items, expected {n_total}")

    rng = np.random.RandomState(seed)
    selected = rng.choice(len(texts), size=min(n_total, len(texts)), replace=False)
    out = []
    for idx in selected:
        prompt = texts[int(idx)].strip()
        if "目标长度" not in prompt and "length" not in prompt.lower():
            prompt += "\n\n请确保输出为结构化长文，总长度约 1500–2000 字。"
        out.append({
            "original": texts[int(idx)],
            "prompt": prompt,
            "source": "longgenbench",
            "language": "en",
            "task": "long_generation",
            "format": "constrained_essay",
            "target_length": 2048,
        })
    return out


# -----------------------------------------------------------------------------
# 3.3 Infinity-Instruct
# -----------------------------------------------------------------------------

def load_infinity_instruct(n_total: int = 200, seed: int = 42) -> List[Dict]:
    """
    从 Infinity-Instruct 抽综合真实 instruction。
    过滤可扩展的问题，并用模板改写成长输出。
    """
    # Infinity-Instruct 多为 gated；尝试多个公开 instruction 数据集作为替代
    candidates = [
        "BAAI/Infinity-Instruct-7M",
        "BAAI/Infinity-Instruct",
        "Open-Orca/OpenOrca",
        "HuggingFaceH4/no_robots",
        "tatsu-lab/alpaca",
        "allenai/WildChat-1M",
    ]
    ds = None
    for name in candidates:
        try:
            ds = load_hf_dataset(name, split="train", streaming=True, token=_HF_TOKEN)
            break
        except Exception as e:
            if name == candidates[-1]:
                warn_missing(f"Infinity-Instruct load failed: {e}")
                return []
            continue

    texts = _collect_texts_from_dataset(
        ds,
        ["instruction", "prompt", "input", "question", "query", "content", "text"],
        "Infinity-Instruct",
        n_total * 4,
    )
    print(f"  [Infinity-Instruct] collected {len(texts)} raw texts")
    if len(texts) == 0:
        warn_missing("Infinity-Instruct loaded but extracted 0 valid prompts")
        return []
    if len(texts) < n_total:
        warn_missing(f"Infinity-Instruct only collected {len(texts)} raw texts, expected at least {n_total}")

    # 关键词：更容易自然扩展
    extend_keywords = [
        "分析", "比较", "解释", "讨论", "原因", "影响", "方案", "步骤",
        "报告", "文章", "教程", "设计", "实现", "评估", "总结",
        "analyze", "compare", "explain", "discuss", "evaluate", "design",
        "implement", "report", "article", "steps", "solution",
    ]

    filtered = []
    for prompt in texts:
        text = prompt.lower()
        if any(kw in text for kw in [k.lower() for k in extend_keywords]):
            filtered.append(prompt)
        if len(filtered) >= n_total * 2:
            break

    if len(filtered) < n_total:
        warn_missing(f"Infinity-Instruct only {len(filtered)} items after keyword filter")
        return []

    collected = downsample(filtered, n_total, seed)
    out = []
    for prompt in collected:
        lang = _detect_language(prompt)
        template = choose_template("instruction", lang)
        out.append({
            "original": prompt,
            "prompt": apply_template(template, prompt, 1500),
            "source": "infinity_instruct",
            "language": lang,
            "task": "instruction",
            "format": "essay",
            "target_length": 1500,
        })
    return out


# -----------------------------------------------------------------------------
# 3.4 数学与推理
# -----------------------------------------------------------------------------

def load_math_reasoning(n_total: int = 100, seed: int = 42) -> List[Dict]:
    """
    从数学/推理数据集加载中等复杂度问题，改写成“长推导版本”。
    默认尝试 MATH、GSM8K、NuminaMath 等。
    """
    # 数学数据集；hf 上存在多个镜像，优先用用户确认的 qwedsacf/competition_math
    # EleutherAI/hendrycks_math 需要按数学分支传 config
    math_configs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    sources = [
        ("qwedsacf/competition_math", None, "train"),
    ]
    for cfg in math_configs:
        sources.append(("EleutherAI/hendrycks_math", cfg, "test"))
    sources += [
        ("openai/gsm8k", "main", "test"),
        ("AI-MO/NuminaMath-CoT", None, "train"),
    ]

    collected = []
    for name, config, split in sources:
        if len(collected) >= n_total * 3:
            break
        print(f"  [Math] loading {name} (config={config}, split={split}) ...")
        try:
            ds = load_hf_dataset(name, config=config, split=split, streaming=True, token=_HF_TOKEN)
            print(f"  [Math] {name} loaded, iterating ...")
            n_iter = 0
            for example in ds:
                prompt = _extract_field(example, ["problem", "question", "input"])
                if not prompt:
                    continue
                collected.append({
                    "original": prompt,
                    "source": "math_reasoning",
                    "language": _detect_language(prompt),
                    "task": "math",
                    "format": "derivation",
                })
                n_iter += 1
                if len(collected) >= n_total * 3:
                    break
            print(f"  [Math] {name}: iterated {n_iter} examples, collected {len(collected)}")
        except Exception as e:
            warn_missing(f"{name} load failed: {e}")

    if len(collected) < n_total:
        warn_missing(f"Math/Reasoning only collected {len(collected)} items")

    collected = downsample(collected, n_total, seed)
    out = []
    for it in collected:
        it["prompt"] = apply_template("math_full", it["original"], 1200)
        it["target_length"] = 1200
        out.append(it)
    return out


# -----------------------------------------------------------------------------
# 3.5 代码与技术
# -----------------------------------------------------------------------------

def load_code_tech(n_total: int = 100, seed: int = 42) -> List[Dict]:
    """
    从 LiveCodeBench 等加载代码任务，改写成“分析 + 代码 + 解释”的长输出。
    """
    # 代码数据集的 HF 路径和 split/config 需要精确匹配
    """只用一个 code repo：openai/openai_humaneval，非 streaming 直接加载。"""
    name = "openai/openai_humaneval"
    print(f"  [Code/Tech] loading {name} ...")
    try:
        ds = load_hf_dataset(name, config=None, split="test", streaming=False, token=_HF_TOKEN)
        print(f"  [Code/Tech] {name} loaded: {len(ds)} examples")
    except Exception as e:
        warn_missing(f"{name} load failed: {e}")
        return []

    collected = []
    for example in ds:
        prompt = _extract_field(example, ["prompt", "question", "instruction", "text", "task_id"])
        if not prompt:
            continue
        collected.append({
            "original": prompt,
            "source": "code_tech",
            "language": _detect_language(prompt),
            "task": "coding",
            "format": "code_report",
        })
        if len(collected) >= n_total * 3:
            break

    print(f"  [Code/Tech] collected {len(collected)} prompts")

    collected = downsample(collected, n_total, seed)
    out = []
    for it in collected:
        it["prompt"] = apply_template("code_full", it["original"], 1500)
        it["target_length"] = 1500
        out.append(it)
    return out


# -----------------------------------------------------------------------------
# 3.6 Aya 多语言
# -----------------------------------------------------------------------------

def load_aya(n_total: int = 100, seed: int = 42) -> List[Dict]:
    """
    从 Aya Dataset 加载多语言 instruction。
    配额：中文30，英文30，日文20，中英混合8，中日混合6，其他6。
    """
    try:
        ds = load_hf_dataset("CohereForAI/aya_dataset", split="train", streaming=True, token=_HF_TOKEN)
    except Exception as e:
        warn_missing(f"Aya dataset load failed: {e}")
        return []

    by_lang = defaultdict(list)
    for example in ds:
        prompt = _extract_field(example, ["inputs", "prompt", "instruction"])
        lang = _extract_field(example, ["language", "lang"], "unknown")
        if not prompt:
            continue
        by_lang[lang].append({
            "original": prompt,
            "source": "aya",
            "language": lang,
            "task": "multilingual",
            "format": "essay",
        })
        # 为了效率，每个语言收集到一定数量就停止
        if sum(len(v) for v in by_lang.values()) >= n_total * 5:
            break

    quotas = {
        "Chinese": 30, "zh": 30,
        "English": 30, "eng_Latn": 30, "en": 30,
        "Japanese": 20, "jpn_Jpan": 20, "ja": 20,
        "mixed_zh_en": 8,
        "mixed_zh_ja": 6,
        "other": 6,
    }

    # 映射到统一语言标签
    lang_map = {
        "Chinese": "zh", "zh": "zh",
        "English": "en", "eng_Latn": "en", "en": "en",
        "Japanese": "ja", "jpn_Jpan": "ja", "ja": "ja",
    }

    selected = []
    for raw_lang, items in by_lang.items():
        lang = lang_map.get(raw_lang, raw_lang)
        target = quotas.get(lang, quotas["other"])
        for it in items[:target]:
            it["language"] = lang
            selected.append(it)

    # 如果某语言不够，从 other 补
    selected = downsample(selected, n_total, seed)

    # 生成中英/中日混合
    mixed = []
    zh_items = [it for it in selected if it["language"] == "zh"]
    en_items = [it for it in selected if it["language"] == "en"]
    ja_items = [it for it in selected if it["language"] == "ja"]

    for i in range(8):
        if i < len(zh_items) and i < len(en_items):
            mixed.append({
                "original": zh_items[i]["original"],
                "source": "aya_mixed",
                "language": "mixed_zh_en",
                "task": "multilingual",
                "format": "essay",
                "prompt": apply_template("cross_language_en", zh_items[i]["original"], 1200),
                "target_length": 1500,
            })
    for i in range(6):
        if i < len(zh_items) and i < len(ja_items):
            mixed.append({
                "original": zh_items[i]["original"],
                "source": "aya_mixed",
                "language": "mixed_zh_ja",
                "task": "multilingual",
                "format": "essay",
                "prompt": apply_template("cross_language_ja", zh_items[i]["original"], 1200),
                "target_length": 1500,
            })

    # 给单语 prompt 加长
    single = []
    for it in selected:
        template = choose_template("multilingual", it["language"])
        it["prompt"] = apply_template(template, it["original"], 1200)
        it["target_length"] = 1500
        single.append(it)

    result = single[:n_total - len(mixed)] + mixed
    print(f"  [Aya] returning {len(result)} prompts")
    return result


def load_aya_flat(n_total: int, seed: int = 42) -> List[Dict]:
    """
    Aya 简单平铺版本：不严格保证语言配额，用于补齐缺失数量。
    """
    try:
        ds = load_hf_dataset("CohereForAI/aya_dataset", split="train", streaming=True, token=_HF_TOKEN)
    except Exception as e:
        warn_missing(f"Aya fallback load failed: {e}")
        return []

    out = []
    for example in ds:
        prompt = _extract_field(example, ["inputs", "prompt", "instruction"])
        if not prompt:
            continue
        lang = _detect_language(prompt)
        template = choose_template("multilingual", lang)
        out.append({
            "original": prompt,
            "prompt": apply_template(template, prompt, 1200),
            "source": "aya_fallback",
            "language": lang,
            "task": "multilingual",
            "format": "essay",
            "target_length": 1500,
        })
        if len(out) >= n_total:
            break

    rng = np.random.RandomState(seed)
    indices = np.arange(len(out))
    rng.shuffle(indices)
    return [out[int(i)] for i in indices]


# -----------------------------------------------------------------------------
# 3.7 用户自定义复杂 prompt
# -----------------------------------------------------------------------------

def load_custom(custom_file: str, n_total: int = 50, seed: int = 42) -> List[Dict]:
    if not Path(custom_file).exists():
        warn_missing(f"Custom prompt file not found: {custom_file}")
        return []

    items = []
    with open(custom_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({
                "original": obj.get("prompt", ""),
                "source": "custom",
                "language": obj.get("language", _detect_language(obj.get("prompt", ""))),
                "task": obj.get("task", "custom"),
                "format": obj.get("format", "essay"),
                "prompt": obj.get("prompt", ""),
                "target_length": obj.get("target_length", 2048),
            })

    items = downsample(items, n_total, seed)
    # 如果用户没给长输出 prompt，用模板改写
    for it in items:
        if not it["prompt"] or len(it["prompt"]) < 50:
            template = choose_template(it["task"], it["language"])
            it["prompt"] = apply_template(template, it["original"], it["target_length"])
    return items


# =============================================================================
# 4. 语言检测
# =============================================================================

def _detect_language(text: str) -> str:
    """Very rough language detection based on character ranges."""
    if not text:
        return "unknown"
    # CJK Unified Ideographs
    zh_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # Hiragana/Katakana
    ja_chars = sum(1 for c in text if ("\u3040" <= c <= "\u309f") or ("\u30a0" <= c <= "\u30ff"))
    # Basic Latin
    en_chars = sum(1 for c in text if ("a" <= c.lower() <= "z"))

    total = max(len(text), 1)
    if zh_chars / total > 0.1:
        return "zh"
    if ja_chars / total > 0.05:
        return "ja"
    if en_chars / total > 0.3:
        return "en"
    return "mixed"


# =============================================================================
# 5. 主流程
# =============================================================================

def build_pool(args) -> List[Dict]:
    set_seed(args.seed)

    pool = []

    if args.use_lifebench:
        print("Loading LIFEBench ...")
        pool.extend(load_lifebench(args.n_lifebench, args.seed))

    if args.use_longgenbench:
        print("Loading LongGenBench ...")
        pool.extend(load_longgenbench(args.n_longgenbench, args.seed))

    if args.use_infinity:
        print("Loading Infinity-Instruct ...")
        pool.extend(load_infinity_instruct(args.n_infinity, args.seed))

    if args.use_math:
        print("Loading Math/Reasoning ...")
        pool.extend(load_math_reasoning(args.n_math, args.seed))

    if args.use_code:
        print("Loading Code/Tech ...")
        pool.extend(load_code_tech(args.n_code, args.seed))

    if args.use_aya:
        print("Loading Aya multilingual ...")
        pool.extend(load_aya(args.n_aya, args.seed))

    if args.custom_file:
        print("Loading custom prompts ...")
        pool.extend(load_custom(args.custom_file, args.n_custom, args.seed))

    # 去重
    print(f"Before dedup: {len(pool)}")
    pool = deduplicate_prompts(pool)
    print(f"After dedup: {len(pool)}")

    # 如果总数不足，用 Aya 补齐；去重后若仍不足，循环再补
    while len(pool) < args.n_total and not args.no_fill_missing:
        n_fill = args.n_total - len(pool)
        print(f"Filling {n_fill} missing prompts from Aya fallback ...")
        extra = load_aya_flat(n_fill + 50, args.seed)  # 多要一些，留出去重冗余
        before = len(pool)
        pool.extend(extra)
        pool = deduplicate_prompts(pool)
        if len(pool) == before:
            print("No new prompts from Aya fallback, stopping fill.")
            break

    # 如果提供了本地 fallback 文件，也可以补
    if args.fallback_file and len(pool) < args.n_total:
        n_fill = args.n_total - len(pool)
        print(f"Filling {n_fill} missing prompts from {args.fallback_file} ...")
        pool.extend(load_custom(args.fallback_file, n_fill + 50, args.seed))
        pool = deduplicate_prompts(pool)

    # 如果总数超过目标，随机下采样
    if len(pool) > args.n_total:
        pool = downsample(pool, args.n_total, args.seed)
        print(f"Downsampled to {len(pool)}")

    # 打乱顺序
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(pool))
    rng.shuffle(indices)
    pool = [pool[int(i)] for i in indices]

    return pool


def main():
    parser = argparse.ArgumentParser(description="Build candidate prompt pool for Scheme 1")
    parser.add_argument("--output_file", required=True, help="Output JSONL path")
    parser.add_argument("--n_total", type=int, default=1000, help="Total number of candidate prompts")
    parser.add_argument("--custom_file", type=str, default=None, help="User custom prompts JSONL")
    parser.add_argument("--fallback_file", type=str, default=None, help="Extra local prompts JSONL to fill missing slots")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated datasets")
    parser.add_argument("--no_fill_missing", action="store_true", help="Do not fill missing prompts with Aya")
    parser.add_argument("--seed", type=int, default=42)

    # source toggles: default True, can disable with --no-use-xxx
    parser.add_argument("--use_lifebench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_longgenbench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_infinity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_math", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_aya", action=argparse.BooleanOptionalAction, default=True)

    # source counts
    parser.add_argument("--n_lifebench", type=int, default=300)
    parser.add_argument("--n_longgenbench", type=int, default=150)
    parser.add_argument("--n_infinity", type=int, default=200)
    parser.add_argument("--n_math", type=int, default=100)
    parser.add_argument("--n_code", type=int, default=100)
    parser.add_argument("--n_aya", type=int, default=100)
    parser.add_argument("--n_custom", type=int, default=50)

    args = parser.parse_args()

    global _HF_TOKEN
    _HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if _HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", _HF_TOKEN)

    pool = build_pool(args)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for it in pool:
            f.write(json.dumps({
                "prompt": it["prompt"],
                "source": it.get("source", "unknown"),
                "language": it.get("language", "unknown"),
                "task": it.get("task", "unknown"),
                "format": it.get("format", "unknown"),
                "target_length": it.get("target_length", 2048),
                "original": it.get("original", ""),
            }, ensure_ascii=False) + "\n")

    print(f"Saved {len(pool)} candidate prompts to {output_path}")

    # 显式清理 streaming dataset 引用，减少 datasets 库退出时的 core dump 风险
    gc.collect()


if __name__ == "__main__":
    main()
