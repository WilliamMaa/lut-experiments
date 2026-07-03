"""Data loading helpers for v5."""

import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import prepare_data, load_jsonl, TextDataset


def load_model_and_data(model_name, eval_size, max_seq_len, batch_size, device_str="cuda:0", calib_size=0):
    device = torch.device(device_str)
    print(f"[load_model_and_data] device: {device_str}")
    print(f"[load_model_and_data] torch.version.cuda = {torch.version.cuda}")
    print(f"[load_model_and_data] torch.cuda.is_available() = {torch.cuda.is_available()}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    print(f"[load_model_and_data] model device: {model.device}")
    print(f"[load_model_and_data] GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(base_dir, "..", "v0", "data", "calib.jsonl")
    eval_path = os.path.join(base_dir, "..", "v0", "data", "eval.jsonl")
    prepare_data(tokenizer, calib_path, eval_path, calib_size=calib_size, eval_size=eval_size, max_seq_len=max_seq_len)
    calib_texts = load_jsonl(calib_path)
    eval_texts = load_jsonl(eval_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len=max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=batch_size, shuffle=False)
    eval_loader = eval_dataset.make_loader(batch_size=batch_size, shuffle=False)

    return model, tokenizer, calib_loader, eval_loader


def collect_baseline_logits(model, data_loader):
    """Pre-compute baseline logits on calibration set, per batch."""
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collect baseline logits", leave=False):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
    return all_logits
