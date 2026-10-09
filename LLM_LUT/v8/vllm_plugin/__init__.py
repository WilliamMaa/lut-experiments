"""v8 compressed-KV vLLM plugin (vLLM 0.19.1 target).

patch() performs the monkeypatches, in order:
1. Attention.__init__: inject CompressedKVBackend into every full-attention
   layer of the target model. GDN/linear layers never construct vllm
   Attention modules, so gating on the model arch is sufficient.
2. Attention.get_kv_cache_spec: convert the layer's FullAttentionSpec into
   CompressedKVSpec (0.19.1 has no customize_spec hook). Without this the
   allocator group keeps the stock max_model_len-sized spec.
3-6. backend.py: KV cache manager enum/spec manager registration,
   per-request block cap (get_num_blocks_to_allocate), step-context patch.
7. lut_ffn (P1.3, optional): shared_expert.forward -> triton LUT lookup on
   V8_LUT_LAYERS. Skipped when the env switch is empty.

v1 constraints (enforced in serve.py, not optional):
- --enforce-eager (per-request Python state, no CUDA graphs)
- prefix caching disabled (block reuse would alias compact regions; 0.19.1
  has no per-spec prefix_cacheable flag, so it is forced off on the CLI)
"""
from . import config  # noqa: F401  (env constants, imported by all modules)

_patched = False


def patch() -> None:
    global _patched
    if _patched:
        return
    _patch_attention_backend()
    _patch_get_kv_cache_spec()
    from .backend import (patch_allocator, patch_step_context,
                          register_backend_enum,
                          register_spec_manager)
    register_backend_enum()
    register_spec_manager()
    patch_allocator()
    patch_step_context()
    # 7. P1.3 shared_expert LUT replacement (docs/43): swaps
    #    shared_expert.forward for a triton LUT lookup on V8_LUT_LAYERS.
    #    Runs only when the env switch is set — default is off.
    from . import lut_ffn
    lut_layers = lut_ffn.parse_layers(config.V8_LUT_LAYERS)
    if lut_layers:
        lut_ffn.patch_shared_expert_lut(lut_layers)
    _patched = True
    # Runs in the API-server process: this line in the log proves patch()
    # executed and which code version is live. Absent => stale files or
    # serve.py never reached patch().
    import os
    print(f"[v8_plugin] patch() installed v{config.PLUGIN_VERSION}, "
          f"TARGET_ARCH={config.TARGET_ARCH}, "
          f"slots={config.V8_COMPRESS_SLOTS}, "
          f"lut={config.V8_LUT_LAYERS or 'off'}, "
          f"pid={os.getpid()}", flush=True)


def _patch_attention_backend() -> None:
    from vllm.model_executor.layers.attention.attention import Attention

    from .backend import CompressedKVBackend

    orig_init = Attention.__init__

    def patched_init(self, *args, **kwargs):
        ok, reason = _target_model()
        if _first_attn_init[0] is None:
            _first_attn_init[0] = (ok, reason)
            print(f"[v8_plugin] first Attention init: target={ok} ({reason})",
                  flush=True)
        if kwargs.get("attn_backend") is None and ok:
            kwargs["attn_backend"] = CompressedKVBackend
            _log_inject_once()
        orig_init(self, *args, **kwargs)

    Attention.__init__ = patched_init


_first_attn_init = [None]


def _patch_get_kv_cache_spec() -> None:
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from .backend import spec_from_full

    orig_get_spec = Attention.get_kv_cache_spec

    def patched_get_spec(self, vllm_config):
        spec = orig_get_spec(self, vllm_config)
        # No _target_model() gate here: this runs during engine init and
        # may sit outside set_current_vllm_config (get_current_vllm_config
        # would raise and silently skip conversion). The backend-name check
        # is sufficient — V8_COMPRESSED implies our injection already gated
        # on arch at layer construction time.
        if (type(spec) is FullAttentionSpec  # not SlidingWindowSpec
                and self.attn_backend is not None
                and self.attn_backend.get_name() == "V8_COMPRESSED"):
            if not _spec_converted[0]:
                _spec_converted[0] = True
                print("[v8_plugin] FullAttentionSpec -> CompressedKVSpec "
                      f"(v{config.PLUGIN_VERSION})", flush=True)
            return spec_from_full(spec)
        return spec

    Attention.get_kv_cache_spec = patched_get_spec


_spec_converted = [False]


def _target_model():
    """Returns (is_target, reason). Never raises — failures are reported."""
    try:
        from vllm.config import get_current_vllm_config
        archs = get_current_vllm_config().model_config.architectures
    except Exception as e:
        return False, f"get_current_vllm_config failed: {type(e).__name__}: {e}"
    if not archs:
        return False, "architectures is empty"
    hit = any(t in a for t in config.TARGET_ARCH.split(",") for a in archs)
    if hit:
        return True, f"arch matched: {archs}"
    return False, f"arch mismatch: {archs} vs TARGET_ARCH={config.TARGET_ARCH}"


_inject_logged = False


def _log_inject_once() -> None:
    global _inject_logged
    if not _inject_logged:
        _inject_logged = True
        print("[v8_plugin] injected CompressedKVBackend into full-attn "
              f"layers v{config.PLUGIN_VERSION} "
              f"(slots={config.V8_COMPRESS_SLOTS})", flush=True)
