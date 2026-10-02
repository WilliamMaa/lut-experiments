"""v8 compressed-KV vLLM plugin.

Importing this package and calling patch() (or using `python -m
vllm_plugin.serve`) injects CompressedKVBackend into every full-attention
layer of the target model (default Qwen3_5MoeForCausalLM). GDN/linear
layers never construct vllm Attention modules, so gating on the model arch
is sufficient — every Attention built during Qwen3.6 model init is a
full-attn layer.

v1 constraints (enforced/documented, not optional):
- --enforce-eager (per-request Python state, no CUDA graphs)
- no prefix caching for the compressed group (spec reports
  prefix_cacheable=False)
"""
from . import config  # noqa: F401  (env constants, imported by all modules)

_patched = False


def patch() -> None:
    global _patched
    if _patched:
        return
    from vllm.model_executor.layers.attention.attention import Attention

    from .backend import CompressedKVBackend

    orig_init = Attention.__init__

    def patched_init(self, *args, **kwargs):
        if kwargs.get("attn_backend") is None and _target_model():
            kwargs["attn_backend"] = CompressedKVBackend
        orig_init(self, *args, **kwargs)

    Attention.__init__ = patched_init
    _patched = True


def _target_model() -> bool:
    try:
        from vllm.config import get_current_vllm_config
        archs = get_current_vllm_config().model_config.architectures
    except Exception:
        return False
    return any(config.TARGET_ARCH in a for a in (archs or []))
