"""GPU sanity check: verify model loads and forward passes on a SINGLE GPU.

Run this BEFORE run_v0.py. If this hangs or crashes, do NOT proceed.
"""

import os
os.environ["ACCELERATE_USE_CPU"] = "False"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 32
BATCH_SIZE = 2


def check():
    print("[CHECK] Locking to single GPU:", DEVICE)
    torch.cuda.set_device(DEVICE)

    print("[CHECK] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[CHECK] Loading model to", DEVICE, "...")
    # CRITICAL: never use device_map="auto" or accelerate here.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[CHECK] Model loaded. VRAM used:", torch.cuda.memory_allocated(DEVICE) / 1024**2, "MB")

    # Prepare dummy input
    texts = ["Hello world", "The quick brown fox jumps over the lazy dog"]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    print("[CHECK] Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    print("[CHECK] Forward done. Logits shape:", outputs.logits.shape)
    print("[CHECK] VRAM after forward:", torch.cuda.memory_allocated(DEVICE) / 1024**2, "MB")

    # Verify only DEVICE is used
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i)
        if i == int(DEVICE.split(":")[-1]):
            if mem == 0:
                print(f"[FAIL] GPU {i} should have memory allocated but doesn't.")
                sys.exit(1)
        else:
            if mem > 0:
                print(f"[FAIL] GPU {i} has {mem/1024**2:.1f} MB allocated! Model leaked to other GPUs!")
                sys.exit(1)

    print("[CHECK] SUCCESS. Only", DEVICE, "is used. Safe to run run_v0.py.")


if __name__ == "__main__":
    try:
        check()
    except Exception as e:
        print("[FAIL]", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
