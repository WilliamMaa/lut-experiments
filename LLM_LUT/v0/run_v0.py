"""One-click entry for LLM-LUT v0 sensitivity scan.

MANDATORY: Run gpu_sanity_check.py FIRST and confirm it prints SUCCESS.

Usage:
    cd LLM_LUT/v0
    python gpu_sanity_check.py   # MUST PASS
    python run_v0.py --calib_size 8 --eval_size 4 --max_seq_len 64 --batch_size 2 --layer_ids 6
"""

import os
import sys

# Single-GPU safety: no auto device map, but GPU is selectable via --device
os.environ["ACCELERATE_USE_CPU"] = "False"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import V0Config
from data import prepare_data, load_jsonl, TextDataset
from calibrate import calibrate_llm_address
from scan import run_sensitivity_scan
from rank import rank_candidates


def main():
    parser = argparse.ArgumentParser(description="LLM-LUT v0 Sensitivity Scan")
    parser.add_argument("--model_name", type=str, default=V0Config.model_name)
    parser.add_argument("--calib_size", type=int, default=128)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--layer_ids", type=int, nargs="+", default=None)
    parser.add_argument("--skip_calib", action="store_true")
    parser.add_argument("--skip_scan", action="store_true")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use (e.g. cuda:0, cuda:3)")
    args = parser.parse_args()

    config = V0Config()
    config.model_name = args.model_name
    config.calib_size = args.calib_size
    config.eval_size = args.eval_size
    config.max_seq_len = args.max_seq_len
    config.calib_batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    if args.layer_ids is not None:
        config.layer_ids = tuple(args.layer_ids)
    config.result_dir = args.result_dir

    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # CRITICAL: single GPU only. Never use device_map="auto".
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Device: {device} (locked to single GPU)")

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading model...")
    dtype = getattr(torch, config.torch_dtype, torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as e:
        print(f"[WARN] Failed to load with {dtype}: {e}. Falling back to float32.")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=config.trust_remote_code,
        )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[MODEL] Loaded to {device}. VRAM: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")

    # Verify no other GPUs are touched
    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            print(f"[FATAL] GPU {i} has {torch.cuda.memory_allocated(i)/1024**2:.1f} MB allocated!")
            print("[FATAL] Model leaked to other GPUs. Aborting.")
            sys.exit(1)

    print("\n[3/5] Preparing calibration and evaluation data...")
    prepare_data(tokenizer, config.calib_path, config.eval_path,
                 calib_size=config.calib_size, eval_size=config.eval_size,
                 max_seq_len=config.max_seq_len)

    calib_texts = load_jsonl(config.calib_path)
    eval_texts = load_jsonl(config.eval_path)
    calib_dataset = TextDataset(calib_texts, tokenizer, max_seq_len=config.max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len=config.max_seq_len)
    calib_loader = calib_dataset.make_loader(batch_size=config.calib_batch_size, shuffle=False)
    eval_loader = eval_dataset.make_loader(batch_size=config.eval_batch_size, shuffle=False)

    stats_path = os.path.join(config.result_dir, "addr_stats.pt")
    if not args.skip_calib and not args.skip_scan:
        print("\n[4/5] Running address calibration...")
        addr_stats = calibrate_llm_address(
            model=model,
            tokenizer=tokenizer,
            calib_loader=calib_loader,
            layer_ids=config.layer_ids,
            candidate_types=config.candidate_types,
            hidden_group_size=config.hidden_group_size,
            heads=config.heads,
        )
        torch.save(addr_stats, stats_path)
        print(f"[CALIB] Saved addr_stats to {stats_path}")
    else:
        if os.path.exists(stats_path):
            print(f"\n[4/5] Loading cached addr_stats from {stats_path}...")
            addr_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        else:
            print("[ERROR] No cached addr_stats found. Run without --skip_calib first.")
            sys.exit(1)

    scan_path = os.path.join(config.result_dir, "scan_results.json")
    if not args.skip_scan:
        print("\n[5/5] Running sensitivity scan...")
        results, baseline = run_sensitivity_scan(
            model=model,
            tokenizer=tokenizer,
            calib_loader=calib_loader,
            eval_loader=eval_loader,
            addr_stats=addr_stats,
            config=config,
            save_path=scan_path,
        )
    else:
        print("\n[5/5] Loading existing scan results...")
        with open(scan_path, "r", encoding="utf-8") as f:
            scan_data = f.read()
        import json
        scan_data = json.loads(scan_data)
        results = scan_data["results"]
        baseline = scan_data.get("baseline")

    print("\n[REPORT] Generating ranking report...")
    rank_path = os.path.join(config.result_dir, "rank_report.md")
    rank_candidates(results, baseline, save_path=rank_path)

    print("\n" + "=" * 70)
    print("LLM-LUT v0 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
