"""
Trainable engine and data loading helpers for v4 multi-layer fine-tuning.

This module is self-contained within v4 and does not import from v3 or v0.
"""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import prepare_data, load_jsonl, TextDataset
from partial_linear import V3PartialEngine
from triton_kernels import lut_fill


class TrainableV3PartialEngine(V3PartialEngine):
    """V3PartialEngine variant that reads active_weight directly from down_proj.weight
    on every forward, so gradients flow back to the original weight during fine-tuning.
    """

    def _patched_down_proj_forward(self, hidden):
        B, S, intermediate_size = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        normed_x = self._cached_normed_x
        if normed_x is None:
            raise RuntimeError("normed_x not cached!")

        hidden_size = normed_x.shape[-1]
        replaced_groups = sorted(self.group_configs.keys())

        # Compute active matmul and LUT fill in fp32 for numerical stability,
        # then cast the assembled output back to the model dtype.
        # Disable autocast so fp32 inputs are not silently cast back to fp16.
        compute_dtype = torch.float32
        with torch.autocast(device_type=device.type, enabled=False):
            hidden_f32 = hidden.to(compute_dtype)
            normed_x_f32 = normed_x.to(compute_dtype)

            # --- Always slice from original weight (no cache) for trainability ---
            active_weight = self.down_proj.weight[self._active_indices, :].to(compute_dtype)
            active_bias = None
            if self.down_proj.bias is not None:
                active_bias = self.down_proj.bias[self._active_indices].to(compute_dtype)
            active_out = F.linear(hidden_f32, active_weight, active_bias)
            if not torch.isfinite(active_out).all():
                raise RuntimeError(
                    f"active_out contains NaN/Inf: "
                    f"hidden finite={torch.isfinite(hidden).all().item()}, "
                    f"weight finite={torch.isfinite(self.down_proj.weight).all().item()}, "
                    f"active abs max={torch.nan_to_num(active_out.detach()).abs().max().item():.2e}"
                )

            # --- LUT fill in fp32 ---
            if len(replaced_groups) == 0:
                lut_outputs = torch.empty(B, S, 0, device=device, dtype=compute_dtype)
            elif self._batched_tables is not None and self._cached_bin_idx_tensor is not None:
                M = B * S
                normed_x_flat = normed_x_f32.view(M, hidden_size)
                try:
                    tables_f32 = self._batched_tables.to(compute_dtype)
                    lut_outputs_flat = lut_fill(
                        self._cached_bin_idx_tensor,
                        tables_f32,
                        normed_x_flat,
                        self._group_starts,
                    )
                    lut_outputs = lut_outputs_flat.view(B, S, -1)
                except Exception as e:
                    print(f"[V4] LUT fill error ({e}), falling back to per-group loop")
                    lut_outputs = self._lut_fill_loop(B, S, normed_x_f32, device, compute_dtype)
            else:
                lut_outputs = self._lut_fill_loop(B, S, normed_x_f32, device, compute_dtype)

            if not torch.isfinite(lut_outputs).all():
                raise RuntimeError(
                    f"lut_outputs contains NaN/Inf: "
                    f"normed_x finite={torch.isfinite(normed_x).all().item()}, "
                    f"lut abs max={torch.nan_to_num(lut_outputs.detach()).abs().max().item():.2e}"
                )

            # --- Assemble in fp32 and cast back ---
            full_out = torch.zeros(B, S, hidden_size, device=device, dtype=compute_dtype)
            full_out = full_out.index_copy_(2, self._active_indices, active_out)
            if lut_outputs.shape[-1] > 0:
                full_out = full_out.index_copy_(2, self._replaced_indices, lut_outputs)

        return full_out.to(dtype)


def load_model_and_data(model_name, eval_size, max_seq_len, batch_size, device_str="cuda:0", calib_size=0):
    device = torch.device(device_str)
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device_str, low_cpu_mem_usage=True,
    )
    model.eval()

    for i in range(torch.cuda.device_count()):
        if i != device.index and torch.cuda.memory_allocated(i) > 0:
            print(f"[WARN] GPU {i} has allocated memory; proceeding because device_map={device_str} is explicit single-GPU.")

    # Reuse the same calib/eval data location as v0/v3.
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
    """Pre-compute baseline logits on calibration set, per batch.

    Returns a list of [B, S, vocab] tensors so batches with different sequence
    lengths (due to dynamic padding) do not need to be concatenated.
    """
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
