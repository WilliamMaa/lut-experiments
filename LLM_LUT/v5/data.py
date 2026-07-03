"""Data preparation for LLM-LUT v0.

Builds small calib/eval JSONL files from lightweight public sources + manual prompts.
"""

import json
import os
import random
from pathlib import Path

from transformers import AutoTokenizer


# Small built-in prompt corpus to guarantee immediate execution without large downloads.
_CALIB_PROMPTS_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "In 1969, humans first landed on the Moon during the Apollo 11 mission.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Photosynthesis allows plants to convert sunlight into chemical energy.",
    "The Great Wall of China is one of the most famous landmarks in the world.",
    "Python is a high-level programming language known for readability.",
    "Quantum mechanics describes the behavior of matter at very small scales.",
    "Shakespeare wrote Romeo and Juliet around 1594.",
    "DNA stands for deoxyribonucleic acid and carries genetic information.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "Electric cars use batteries to store energy and power electric motors.",
    "The Amazon rainforest is the largest tropical rainforest on Earth.",
    "Einstein's theory of relativity revolutionized modern physics.",
    "Computer vision enables machines to interpret and understand visual information.",
    "The periodic table organizes chemical elements by atomic number.",
    "A black hole is a region of spacetime where gravity is so strong nothing escapes.",
    "The Internet was originally developed as a research project called ARPANET.",
]

_CALIB_PROMPTS_ZH = [
    "北京是中国的首都，有着悠久的历史。",
    "人工智能正在改变我们的生活方式。",
    "深度学习是一种基于神经网络的机器学习方法。",
    "长城是中国古代最伟大的建筑工程之一。",
    "春节是中国最重要的传统节日。",
    "量子计算有望在某些问题上远超传统计算机。",
    "大熊猫是中国的国宝，主要生活在四川。",
    "丝绸之路是古代连接东西方的重要贸易通道。",
    "中医理论强调阴阳平衡和气血调和。",
    "唐诗宋词是中国文学史上的瑰宝。",
    "长江是中国最长的河流，也是世界第三长河。",
    "青铜器是中国商周时期的重要文物。",
    "太极拳是一种结合武术和健身的中国传统运动。",
    "印刷术是中国古代四大发明之一。",
    "中秋节人们通常会吃月饼、赏月。",
]

_INSTRUCTION_PROMPTS = [
    {"role": "user", "content": "Explain the concept of overfitting in machine learning."},
    {"role": "user", "content": "Write a short poem about autumn."},
    {"role": "user", "content": "What are the benefits of regular exercise?"},
    {"role": "user", "content": "Summarize the theory of evolution in three sentences."},
    {"role": "user", "content": "How does a transformer model work?"},
    {"role": "user", "content": "请介绍一下中国的茶文化。"},
    {"role": "user", "content": "如何学习一门新的编程语言？"},
    {"role": "user", "content": "解释什么是区块链。"},
    {"role": "user", "content": "请写一个Python函数计算斐波那契数列。"},
    {"role": "user", "content": "What is the difference between TCP and UDP?"},
]

_REASONING_PROMPTS = [
    {"role": "user", "content": "If a train travels at 60 km/h for 2 hours and then 80 km/h for 1 hour, what is the average speed?"},
    {"role": "user", "content": "There are 5 apples. You take away 3. How many do you have?"},
    {"role": "user", "content": "一个水池有两个进水管，A管单独注满需6小时，B管单独注满需4小时，两管同时开需几小时注满？"},
    {"role": "user", "content": "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?"},
]


def _build_text_pool(tokenizer, max_seq_len: int = 512) -> list:
    """Build a pool of tokenized-ready text strings."""
    pool = []

    # Plain text
    for text in _CALIB_PROMPTS_EN + _CALIB_PROMPTS_ZH:
        pool.append(text)

    # Instruction prompts with chat template
    for msgs in _INSTRUCTION_PROMPTS + _REASONING_PROMPTS:
        try:
            prompt = tokenizer.apply_chat_template([msgs], tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = msgs["content"]
        pool.append(prompt)

    return pool


def prepare_data(tokenizer, calib_path: str, eval_path: str,
                 calib_size: int = 512, eval_size: int = 256,
                 max_seq_len: int = 512, seed: int = 42):
    """Generate calib.jsonl and eval.jsonl if they don't exist."""
    os.makedirs(os.path.dirname(calib_path) if os.path.dirname(calib_path) else ".", exist_ok=True)

    if os.path.exists(calib_path) and os.path.exists(eval_path):
        print(f"[DATA] Reusing existing {calib_path} and {eval_path}")
        return

    random.seed(seed)
    pool = _build_text_pool(tokenizer, max_seq_len)

    # Repeat pool until we have enough samples
    while len(pool) < calib_size + eval_size:
        pool.extend(pool)

    random.shuffle(pool)
    calib_texts = pool[:calib_size]
    eval_texts = pool[calib_size:calib_size + eval_size]

    def write_jsonl(path, texts):
        with open(path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    write_jsonl(calib_path, calib_texts)
    write_jsonl(eval_path, eval_texts)
    print(f"[DATA] Written {calib_path} ({calib_size} lines) and {eval_path} ({eval_size} lines)")


def load_jsonl(path: str):
    """Load a jsonl file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class TextDataset:
    """Simple wrapper over list of texts with tokenization."""
    def __init__(self, texts, tokenizer, max_seq_len: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def collate(self, batch):
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        return encoded

    def make_loader(self, batch_size: int = 4, shuffle: bool = False):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.texts,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            num_workers=0,
        )
