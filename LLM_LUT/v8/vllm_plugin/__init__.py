"""v8 compressed-KV vLLM plugin (vLLM 0.19.1 target).

patch() performs three monkeypatches:
1. Attention.__init__: inject CompressedKVBackend into every full-attention
   layer of the target model. GDN/linear layers never construct vllm
   Attention modules, so gating on the model arch is sufficient.
2. Attention.get_kv_cache_spec: convert the layer's FullAttentionSpec into
   CompressedKVSpec (0.19.1 has no customize_spec hook). Without this the
   allocator group keeps the stock max_model_len-sized spec.
3. SingleTypeKVCacheManager.get_num_blocks_to_allocate: cap per-request
   blocks at spec.blocks_per_request (see backend.patch_allocator).

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
    from .backend import patch_allocator
    patch_allocator()
    _patched = True


def _patch_attention_backend() -> None:
    from vllm.model_executor.layers.attention.attention import Attention

    from .backend import CompressedKVBackend

    orig_init = Attention.__init__

    def patched_init(self, *args, **kwargs):
        if kwargs.get("attn_backend") is None and _target_model():
            kwargs["attn_backend"] = CompressedKVBackend
        orig_init(self, *args, **kwargs)

    Attention.__init__ = patched_init


def _patch_get_kv_cache_spec() -> None:
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from .backend import spec_from_full

    orig_get_spec = Attention.get_kv_cache_spec

    def patched_get_spec(self, vllm_config):
        spec = orig_get_spec(self, vllm_config)
        if (_target_model()
                and type(spec) is FullAttentionSpec  # not SlidingWindowSpec
                and self.attn_backend is not None
                and self.attn_backend.get_name() == "V8_COMPRESSED"):
            return spec_from_full(spec)
        return spec

    Attention.get_kv_cache_spec = patched_get_spec


def _target_model() -> bool:
    try:
        from vllm.config import get_current_vllm_config
        archs = get_current_vllm_config().model_config.architectures
    except Exception:
        return False
    return any(config.TARGET_ARCH in a for a in (archs or []))
