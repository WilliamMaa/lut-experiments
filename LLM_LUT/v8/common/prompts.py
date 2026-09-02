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


def load_eval_texts(eval_file: str, max_samples: int, min_length: int = 0, sort_by_length: bool = False):
    """Load evaluation texts from JSONL / JSON / plain text.

    These texts are used to compute PPL.  Prefer longer passages over short
    prompts because PPL on very short sequences is noisy.

    Args:
        min_length: skip texts shorter than this.
        sort_by_length: if True, return the longest texts up to max_samples.
    """
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
                if text and len(text) >= min_length:
                    texts.append(text)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and len(line) >= min_length:
                    texts.append(line)
    if sort_by_length:
        texts.sort(key=len, reverse=True)
    return texts[:max_samples]


def load_prompts(prompt_file: str, max_samples: int, min_length: int = 0, sort_by_length: bool = False):
    """Load generation prompts from a JSONL / JSON / plain text file.

    Each line should contain a prompt; accepted JSON fields are:
        prompt / text / content / sentence

    Args:
        min_length: skip prompts shorter than this.
        sort_by_length: if True, return the longest prompts up to max_samples.
    """
    prompts = []
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")
    if path.suffix in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", obj.get("text", obj.get("content", obj.get("sentence", ""))))
                if prompt and len(prompt) >= min_length:
                    prompts.append(prompt)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and len(line) >= min_length:
                    prompts.append(line)
    if sort_by_length:
        prompts.sort(key=len, reverse=True)
    return prompts[:max_samples]


def load_multi_turn_prompts(prompt_file: str, max_samples: int):
    """Load multi-turn conversation prompts from a JSONL file.

    Each line must be a JSON object with:
        document: long context text
        questions: list of user questions

    Returns a list of dicts: [{"document": str, "questions": list[str]}]
    """
    samples = []
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"multi_turn prompt_file not found: {prompt_file}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            document = obj.get("document", "")
            questions = obj.get("questions", [])
            if document and questions:
                samples.append({"document": document, "questions": questions})
    return samples[:max_samples]
