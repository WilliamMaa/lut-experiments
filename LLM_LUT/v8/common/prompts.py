#!/usr/bin/env python3
"""Default prompts and eval text loaders for v8 evaluation."""

from pathlib import Path
import json


DEFAULT_PROMPTS = [
    "What is the capital of Japan?",
    "Explain the concept of overfitting in machine learning.",
    "If a train travels at 60 km/h for 2 hours, how far does it go?",
    "Write a Python function to reverse a string.",
    "请介绍一下长城的历史。",
    "What are the main differences between TCP and UDP?",
    "第一次世界大战爆发的根本原因是什么？",
    "Explain the Transformer attention mechanism in simple terms.",
]


def load_eval_texts(eval_file: str, max_samples: int):
    """Load evaluation texts from JSONL / JSON / plain text."""
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
