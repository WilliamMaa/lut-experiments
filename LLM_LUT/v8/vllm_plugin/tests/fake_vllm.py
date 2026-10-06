"""Minimal vLLM 0.19.1 API surface so vllm_plugin.backend/impl/spec import
on a machine without vllm installed (docs/32 harness). Only the symbols the
plugin touches are provided; behavior matches the real 0.19.1 contracts the
plugin relies on (see backend.py header notes).

Usage: call install() BEFORE importing vllm_plugin.backend / .impl.
"""
import sys
import types


class FlashAttentionMetadata:
    """Plain stand-in. The real one is a dataclass built by the builder's
    super().build(); the plugin only attaches extra fields (subclass class
    attributes), which works identically on a plain base."""


class FlashAttentionMetadataBuilder:
    """Stand-in matching the 0.19.1 signature
    (kv_cache_spec, layer_names, vllm_config, device) and build() return."""

    def __init__(self, *args, **kwargs):
        self.spec = args[0] if args else kwargs.get("kv_cache_spec")
        self.layer_names = (args[1] if len(args) > 1
                            else kwargs.get("layer_names", []))

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build: bool = False):
        return FlashAttentionMetadata()


class FlashAttentionBackend:
    @staticmethod
    def get_name():
        return "FAKE"


class FlashAttentionImpl:
    """Stand-in providing the attributes the v8 impl reads from the base
    (num_kv_heads, scale)."""

    def __init__(self, *args, **kwargs):
        self.num_kv_heads = kwargs.get("num_kv_heads", 1)
        self.scale = kwargs.get("scale", 1.0)


def _full_attention_spec_dataclass():
    """Frozen-dataclass shim with the fields CompressedKVSpec declares on
    top of (block_size, num_kv_heads, head_size, head_size_v, dtype) and
    the members spec.py touches (__post_init__, page_size_bytes)."""
    from dataclasses import dataclass

    @dataclass(frozen=True, kw_only=True)
    class FullAttentionSpec:
        block_size: int = 16
        num_kv_heads: int = 1
        head_size: int = 64
        head_size_v: int = None
        dtype: object = None
        num_blocks: int = 0
        attention_chunk_size: object = None

        def __post_init__(self):
            if self.head_size_v is None:
                object.__setattr__(self, "head_size_v", self.head_size)

        @property
        def page_size_bytes(self) -> int:
            return self.block_size * self.num_kv_heads * self.head_size * 2

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return self.page_size_bytes

        @classmethod
        def merge(cls, specs):
            raise NotImplementedError("fake spec merge not needed")

    return FullAttentionSpec


def install() -> None:
    if "vllm.v1.attention.backends.flash_attn" in sys.modules:
        return

    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    attn = types.ModuleType("vllm.v1.attention")
    backends = types.ModuleType("vllm.v1.attention.backends")
    flash = types.ModuleType("vllm.v1.attention.backends.flash_attn")
    flash.FlashAttentionMetadata = FlashAttentionMetadata
    flash.FlashAttentionMetadataBuilder = FlashAttentionMetadataBuilder
    flash.FlashAttentionBackend = FlashAttentionBackend
    flash.FlashAttentionImpl = FlashAttentionImpl
    iface = types.ModuleType("vllm.v1.kv_cache_interface")
    iface.FullAttentionSpec = _full_attention_spec_dataclass()

    for name, mod in [
        ("vllm", vllm), ("vllm.v1", v1), ("vllm.v1.attention", attn),
        ("vllm.v1.attention.backends", backends),
        ("vllm.v1.attention.backends.flash_attn", flash),
        ("vllm.v1.kv_cache_interface", iface),
    ]:
        sys.modules[name] = mod
