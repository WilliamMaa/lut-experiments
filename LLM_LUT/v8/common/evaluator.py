#!/usr/bin/env python3
"""Unified v8 model-level evaluation framework.

Supports both VQK and KV Cache Compression experiments by accepting a
user-provided "patch" object that modifies the model in-place.

Example patch interface:

    class MyPatch:
        def install(self, model): ...
        def uninstall(self, model): ...

Usage:

    from common.evaluator import Evaluator
    from vqk.vqk_patch import VQKPatch  # example

    ev = Evaluator(
        model_path="/path/to/Qwen3.6-35B-A3B",
        device_map="balanced_low_0",
        torch_dtype="bfloat16",
        logit_metrics=True,
    )
    patch = VQKPatch(layer_idx=39, module_path="self_attn.o_proj", bits=4, block_size=64)
    result = ev.evaluate(
        patch=patch,
        texts=texts,
        prompts=prompts,
        max_length=512,
        max_new_tokens=256,
        output_json="vqk_l39_o_proj.json",
    )
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from common.utils import load_model_and_tokenizer
from common.metrics import (
    compute_ppl,
    compute_logit_metrics,
    run_generation,
    compute_generation_metrics,
    run_multi_turn_generation,
    compute_multi_turn_metrics,
    compute_decode_divergence_metrics,
    measure_peak_memory_mb,
    reset_peak_memory_stats,
)
from common.prompts import DEFAULT_PROMPTS, load_eval_texts, load_prompts, load_multi_turn_prompts


class EvalPatch:
    """Base class for eval patches. Subclasses must implement install/uninstall."""

    def install(self, model: torch.nn.Module) -> None:
        """Apply the patch to a model instance."""
        raise NotImplementedError

    def uninstall(self, model: torch.nn.Module) -> None:
        """Remove the patch from a model instance."""
        raise NotImplementedError

    def name(self) -> str:
        """Human-readable name for logging and JSON output."""
        return self.__class__.__name__

    def config(self) -> Dict[str, Any]:
        """Configuration dict to include in the output JSON."""
        return {}

    def storage_stats(self) -> Dict[str, Any]:
        """Optional storage / arithmetic statistics from the patch."""
        return {}


class NullPatch(EvalPatch):
    """No-op patch; useful for running a pure baseline eval."""

    def install(self, model: torch.nn.Module) -> None:
        pass

    def uninstall(self, model: torch.nn.Module) -> None:
        pass

    def name(self) -> str:
        return "baseline"


class Evaluator:
    """Unified evaluator for v8 experiments."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        device_map: Optional[str] = "balanced_low_0",
        torch_dtype: str = "bfloat16",
        logit_metrics: bool = False,
    ):
        self.model_path = model_path
        self.device_arg = device
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.logit_metrics = logit_metrics

        # Teacher model: always loaded first, never patched.
        self.teacher, self.tokenizer, self.teacher_device = load_model_and_tokenizer(
            model_path, torch_dtype, device, device_map
        )

        # Student model: a second copy if we need logit KL / top-k agreement.
        # Otherwise the same instance is reused (patch toggled on/off).
        if logit_metrics:
            print("\n[Evaluator] logit_metrics=True, loading second model copy as student")
            self.student, _, self.student_device = load_model_and_tokenizer(
                model_path, torch_dtype, device, device_map
            )
        else:
            print("\n[Evaluator] logit_metrics=False, using single model (teacher == student)")
            self.student = self.teacher
            self.student_device = self.teacher_device

    def _to_device(self, tensor_or_dict):
        """Move inputs to the effective device used by the active model."""
        device = self.student_device
        if isinstance(tensor_or_dict, dict):
            return {k: v.to(device) for k, v in tensor_or_dict.items()}
        return tensor_or_dict.to(device)

    def evaluate(
        self,
        patch: EvalPatch,
        texts: List[str],
        prompts: Optional[List[str]] = None,
        multi_turn_samples: Optional[List[Dict]] = None,
        max_length: int = 512,
        max_new_tokens: int = 128,
        output_json: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run baseline and patched evaluation, return structured results."""
        prompts = prompts if prompts is not None else DEFAULT_PROMPTS

        # GDN replacement patches intercept the recurrent decode path and therefore
        # need independent teacher/student instances to compare unpatched vs patched
        # decode distributions. This requires logit_metrics=True.
        if hasattr(patch, "verify_decode_calls") and self.student is self.teacher:
            raise RuntimeError(
                "GDN replacement patches require logit_metrics=True so that teacher and student "
                "are independent model instances. Re-run with --logit_metrics."
            )

        print(f"\n{'='*60}")
        print(f"Evaluating patch: {patch.name()}")
        print(f"  eval samples: {len(texts)}")
        print(f"  generation prompts: {len(prompts)}")
        print(f"  max_length={max_length}, max_new_tokens={max_new_tokens}")
        print(f"  logit_metrics={self.logit_metrics}")
        print(f"{'='*60}")

        # ---- Baseline (teacher) ----
        if verbose:
            print("\n[Baseline] computing PPL ...")
        reset_peak_memory_stats()
        t0 = time.time()
        baseline_ppl = compute_ppl(self.teacher, self.tokenizer, texts, self.teacher_device, max_length)
        baseline_ppl_time = time.time() - t0
        baseline_peak_mem = measure_peak_memory_mb()

        use_multi_turn = multi_turn_samples is not None and len(multi_turn_samples) > 0

        if verbose:
            print(f"  Baseline PPL: {baseline_ppl:.4f}  ({baseline_ppl_time:.1f}s)")
            if use_multi_turn:
                print("\n[Baseline] running multi-turn generation ...")
            else:
                print("\n[Baseline] running generation ...")
        baseline_cache_factory = getattr(patch, "get_baseline_cache", None)
        baseline_cache_kwargs = {}
        if baseline_cache_factory is not None and hasattr(self.teacher, "config"):
            baseline_cache_kwargs["config"] = self.teacher.config
        if use_multi_turn:
            baseline_gen = run_multi_turn_generation(
                self.teacher, self.tokenizer, multi_turn_samples, self.teacher_device, max_new_tokens,
                cache_factory=baseline_cache_factory,
                cache_kwargs=baseline_cache_kwargs,
            )
            baseline_gen_metrics = compute_multi_turn_metrics(baseline_gen)
        else:
            baseline_gen = run_generation(
                self.teacher, self.tokenizer, prompts, self.teacher_device, max_new_tokens,
                cache_factory=baseline_cache_factory,
                cache_kwargs=baseline_cache_kwargs,
            )
            baseline_gen_metrics = compute_generation_metrics(baseline_gen)
        if verbose:
            print(f"  EOS success rate: {baseline_gen_metrics['eos_success_rate']:.2%}")
            print(f"  Avg output length: {baseline_gen_metrics['avg_output_length']:.1f}")
            print(f"  Repetition rate: {baseline_gen_metrics['repetition_rate']:.2%}")

        # ---- Apply patch to student ----
        patch.install(self.student)
        if verbose:
            print("\n[Patched] patch installed")
        storage_stats = patch.storage_stats()
        if storage_stats and verbose:
            print(f"  Storage stats: {storage_stats}")

        # ---- Patched evaluation ----
        if verbose:
            print("\n[Patched] computing PPL ...")
        reset_peak_memory_stats()
        t0 = time.time()
        patched_ppl = compute_ppl(self.student, self.tokenizer, texts, self.student_device, max_length)
        patched_ppl_time = time.time() - t0
        patched_peak_mem = measure_peak_memory_mb()

        if verbose:
            print(f"  Patched PPL: {patched_ppl:.4f}  ({patched_ppl_time:.1f}s)")
            if use_multi_turn:
                print("\n[Patched] running multi-turn generation ...")
            else:
                print("\n[Patched] running generation ...")
        patched_cache_factory = getattr(patch, "get_cache", None)
        patched_cache_kwargs = {}
        if patched_cache_factory is not None and hasattr(self.student, "config"):
            patched_cache_kwargs["config"] = self.student.config
        if use_multi_turn:
            patched_gen = run_multi_turn_generation(
                self.student, self.tokenizer, multi_turn_samples, self.student_device, max_new_tokens,
                cache_factory=patched_cache_factory,
                cache_kwargs=patched_cache_kwargs,
            )
            patched_gen_metrics = compute_multi_turn_metrics(patched_gen)
        else:
            patched_gen = run_generation(
                self.student, self.tokenizer, prompts, self.student_device, max_new_tokens,
                cache_factory=patched_cache_factory,
                cache_kwargs=patched_cache_kwargs,
            )
            patched_gen_metrics = compute_generation_metrics(patched_gen)
        if verbose:
            print(f"  EOS success rate: {patched_gen_metrics['eos_success_rate']:.2%}")
            print(f"  Avg output length: {patched_gen_metrics['avg_output_length']:.1f}")
            print(f"  Repetition rate: {patched_gen_metrics['repetition_rate']:.2%}")

        # ---- Runtime assertion: decode replacement path was hit ----
        # The first generated token for each prompt comes from the prefill logits,
        # so the number of decode forward calls is total_output_length - num_prompts.
        if use_multi_turn:
            total_turns = sum(len(sample["turns"]) for sample in patched_gen)
            expected_decode_calls = (
                sum(turn["output_length"] for sample in patched_gen for turn in sample["turns"]) - total_turns
            )
        else:
            expected_decode_calls = sum(g["output_length"] for g in patched_gen) - len(patched_gen)
        if hasattr(patch, "verify_decode_calls"):
            patch.verify_decode_calls(expected_decode_calls)
            if verbose:
                print(f"  Decode replacement calls verified: {expected_decode_calls}")

        # ---- Fixed-trajectory decode divergence metrics ----
        if use_multi_turn:
            decode_prompts = [
                f"{s['document']}\n\n{s['questions'][0]}" for s in multi_turn_samples
            ]
        else:
            decode_prompts = prompts
        if verbose:
            print("\n[Decode divergence] fixed-trajectory decode metrics ...")
        decode_metrics = compute_decode_divergence_metrics(
            self.teacher,
            self.student,
            self.tokenizer,
            decode_prompts,
            self.teacher_device,
            max_new_tokens=max_new_tokens,
            student_cache_factory=patched_cache_factory,
            student_cache_kwargs=patched_cache_kwargs,
        )
        if verbose:
            print(f"  Avg decode KL: {decode_metrics['avg_decode_kl']:.6f}")
            print(f"  Decode top-1 agreement: {decode_metrics['decode_top1_agreement']:.2%}")
            print(f"  Decode top-5 agreement: {decode_metrics['decode_top5_agreement']:.2%}")
            print(f"  Teacher greedy token prob under student: {decode_metrics['avg_teacher_greedy_token_prob_under_student']:.4f}")

        # ---- Logit-level functional metrics ----
        logit_metrics = {}
        if self.logit_metrics:
            if verbose:
                print("\n[Logit metrics] computing KL / top-k agreement ...")
            t0 = time.time()
            logit_metrics = compute_logit_metrics(
                self.teacher,
                self.student,
                self.tokenizer,
                texts,
                self.teacher_device,
                max_length=max_length,
            )
            logit_metrics["compute_time_s"] = round(time.time() - t0, 2)
            if verbose:
                print(f"  Avg KL: {logit_metrics['avg_kl']:.4f}")
                print(f"  Top-1 agreement: {logit_metrics['top1_agreement']:.2%}")
                print(f"  Top-5 agreement: {logit_metrics['top5_agreement']:.2%}")

        # ---- Cleanup ----
        patch.uninstall(self.student)
        if verbose:
            print("\n[Patched] patch uninstalled")

        # ---- Build result ----
        result = {
            "model_path": self.model_path,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "logit_metrics_enabled": self.logit_metrics,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "patch": {
                "name": patch.name(),
                "config": patch.config(),
            },
            "baseline": {
                "ppl": baseline_ppl,
                "ppl_compute_time_s": round(baseline_ppl_time, 2),
                "peak_memory_mb": baseline_peak_mem,
                "generation": baseline_gen,
                "generation_metrics": baseline_gen_metrics,
            },
            "patched": {
                "ppl": patched_ppl,
                "ppl_compute_time_s": round(patched_ppl_time, 2),
                "peak_memory_mb": patched_peak_mem,
                "generation": patched_gen,
                "generation_metrics": patched_gen_metrics,
            },
            "delta": {
                "ppl": round(patched_ppl - baseline_ppl, 6),
                "ppl_relative": round(
                    (patched_ppl - baseline_ppl) / baseline_ppl if baseline_ppl else float("inf"), 6
                ),
                "eos_success_rate": round(
                    patched_gen_metrics["eos_success_rate"] - baseline_gen_metrics["eos_success_rate"], 6
                ),
                "avg_output_length": round(
                    patched_gen_metrics["avg_output_length"] - baseline_gen_metrics["avg_output_length"], 6
                ),
                "repetition_rate": round(
                    patched_gen_metrics["repetition_rate"] - baseline_gen_metrics["repetition_rate"], 6
                ),
            },
            "decode_metrics": decode_metrics,
            "logit_metrics": logit_metrics,
            "storage_stats": storage_stats,
        }

        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"\n[Output] summary written to {output_json}")

        return result


def main():
    """CLI entry point for running a pure baseline eval (no patch)."""
    import argparse

    parser = argparse.ArgumentParser(description="v8 unified model-level baseline evaluation")
    parser.add_argument("--model_path", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--eval_file", default=None, help="JSONL/JSON/text file with LONG texts for PPL evaluation")
    parser.add_argument("--prompt_file", default=None, help="JSONL/JSON/text file with SHORT prompts for generation evaluation")
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_map", default="balanced_low_0", help="Use 'balanced_low_0' for multi-GPU. Do NOT use 'auto'.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--logit_metrics", action="store_true", help="Compute logit KL / top-k agreement (loads second model copy)")
    parser.add_argument("--output_json", default=None, help="Path to write summary JSON")
    parser.add_argument("--prompt", action="append", default=None, help="Custom generation prompt; repeat for multiple")
    args = parser.parse_args()

    if args.prompt_file:
        prompts = load_prompts(args.prompt_file, args.max_eval_samples)
    elif args.prompt:
        prompts = args.prompt
    else:
        prompts = DEFAULT_PROMPTS

    if args.eval_file:
        texts = load_eval_texts(args.eval_file, args.max_eval_samples)
    else:
        print("No --eval_file provided, using prompts for PPL (noisy on short prompts)")
        texts = prompts[: args.max_eval_samples]

    evaluator = Evaluator(
        model_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        logit_metrics=args.logit_metrics,
    )

    result = evaluator.evaluate(
        patch=NullPatch(),
        texts=texts,
        prompts=prompts,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        output_json=args.output_json,
    )
    return result


if __name__ == "__main__":
    main()
