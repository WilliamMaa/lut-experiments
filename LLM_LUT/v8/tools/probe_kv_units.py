#!/usr/bin/env python3
"""Gate 1 probe: print vLLM 0.19.1's OWN block-unit view for the target
model, config-only (no weights, no GPU tensors). Docs/31 §3 Gate 1.

This runs the same config-time code path the server runs
(HybridAttentionMambaModelConfig.verify_and_update_config is what forces
the attention block size up for hybrid GDN models), then prints every
unit the contract (docs/31 §1) needs pinned:

  T   max_model_len (scheduler token space)
  B_g attention group block size (cache_config.block_size after override)
  mamba block size / page sizes used for the override math

Usage (remote, vllm_py310 env, CPU only):
    python tools/probe_kv_units.py --model-path /home/u/downloads/models/Qwen3.6-35B-A3B
"""
import argparse

from vllm.config import VllmConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()

    vllm_config = VllmConfig(
        model=args.model_path,
        tokenizer=args.model_path,
        enforce_eager=True,
        max_model_len=131072,
    )
    cc = vllm_config.cache_config
    mc = vllm_config.model_config

    # Trigger the same hybrid override the server hits at startup.
    mc._try_verify_and_update_model_config()

    print("=== v8 Gate-1 probe: vLLM's own unit view ===")
    print(f"architecture        : {mc.architecture}")
    print(f"max_model_len (T)   : {vllm_config.model_config.max_model_len}")
    print(f"cache_config.block_size (B_g, attention group) : {cc.block_size}")
    print(f"cache_config.mamba_block_size                  : "
          f"{getattr(cc, 'mamba_block_size', None)}")
    print(f"cache_config.mamba_page_size_padded            : "
          f"{getattr(cc, 'mamba_page_size_padded', None)}")
    print(f"cache_config.mamba_cache_mode                : "
          f"{getattr(cc, 'mamba_cache_mode', None)}")
    for name in ("num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "hidden_size", "head_dim",
                 "linear_attn_config", "full_attn_layers"):
        val = getattr(mc.hf_config, name, None)
        if val is not None:
            print(f"hf_config.{name}: {val}")

    # Recompute the override math from the source (F2, docs/31 §1) so the
    # log is self-explanatory without the source tree at hand.
    try:
        from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.model_executor.models.config import MambaModelConfig
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
        from vllm.utils.math_utils import cdiv

        kv_dtype = STR_DTYPE_TO_TORCH_DTYPE[
            "bfloat16" if cc.cache_dtype == "auto" else cc.cache_dtype]
        attn_1tok = FullAttentionSpec(
            block_size=1,
            num_kv_heads=mc.get_num_kv_heads(vllm_config.parallel_config),
            head_size=mc.get_head_size(),
            dtype=kv_dtype,
        ).page_size_bytes
        model_cls, _ = ModelRegistry.resolve_model_cls(
            mc.architecture, model_config=mc)
        mamba_page = MambaSpec(
            shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
            dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
            block_size=-1,
        ).page_size_bytes
        print("--- override math (model_executor/models/config.py) ---")
        print(f"attn page bytes per 1 token : {attn_1tok}")
        print(f"mamba page bytes (per state): {mamba_page}")
        print(f"kernel alignment            : 16 (non-MLA)")
        print(f"derived attn_block_size     : "
              f"{16 * cdiv(mamba_page, 16 * attn_1tok)} tokens")
        print("NOTE: pool page P is read at runtime from "
              "kv_cache.shape[2]; this probe pins B_g. P vs B_g equality "
              "must also hold -- served by the layer-cfg print in the "
              "integration (docs/31 I2).")
    except Exception as e:  # probe math is informational only
        print(f"(override recompute skipped: {type(e).__name__}: {e})")
    print("=== end probe ===")


if __name__ == "__main__":
    main()
