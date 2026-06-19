"""
支持量化 LUT (FP16/INT8) 的 V4PartialEngine。

当前实现：INT8 table 在 install 时 dequantize 到 compute dtype，然后复用 v3 的
Triton/PyTorch LUT fill 路径。这样保证数值正确性，同时 checkpoint 和主机侧存储是 INT8。

未来可扩展：在 Triton kernel 内部直接读 INT8 table 并 dequantize，以进一步减少
LUT SRAM 读取带宽。

用法:
    from partial_linear_quantized import V4PartialEngine
    engine = V4PartialEngine(model, layer_id=21)
    engine.add_group(
        group_id=5,
        addr_idx=ckpt["addr_idx"],
        addr_mean=ckpt["addr_mean"],
        addr_std=ckpt["addr_std"],
        table=ckpt["table"],          # can be fp32, fp16, or int8
        scale=ckpt.get("scale"),       # required if int8
        zero_point=ckpt.get("zero_point", 0.0),
        quantization=ckpt.get("quantization", "fp32"),
    )
    engine.install()
"""

import torch

from partial_linear import V3PartialEngine


def dequantize_table(table: torch.Tensor, scale: float, zero_point: float,
                     quantization: str, target_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a quantized table to target dtype."""
    if quantization in ("fp32", "fp16") or table.dtype.is_floating_point:
        return table.to(target_dtype)
    if quantization == "symmetric_int8":
        return (table.float() * scale + zero_point).to(target_dtype)
    if quantization == "int8" or table.dtype == torch.int8 or table.dtype == torch.uint8:
        return (table.float() - zero_point) * scale
    raise ValueError(f"Unsupported quantization: {quantization}")


class V4PartialEngine(V3PartialEngine):
    """V3PartialEngine extended to accept INT8/FP16 LUT tables."""

    def add_group(self, group_id: int, addr_idx: torch.Tensor, addr_mean: torch.Tensor,
                  addr_std: torch.Tensor, table: torch.Tensor,
                  scale: float = None, zero_point: float = 0.0,
                  quantization: str = "fp32"):
        """Add a group to be replaced; table may be quantized.

        Args:
            group_id: target group within layer output
            addr_idx: [2] address channel indices
            addr_mean: [2] per-channel mean
            addr_std: [2] per-channel std
            table: [num_bins, num_bins, group_size] LUT table (fp32/fp16/int8)
            scale: quantization scale (required for int8)
            zero_point: quantization zero point
            quantization: 'fp32', 'fp16', 'symmetric_int8', or 'int8'
        """
        self.group_configs[group_id] = (
            addr_idx.cpu(),
            addr_mean.cpu(),
            addr_std.cpu(),
            table.cpu(),
            scale,
            zero_point,
            quantization,
        )

    def install(self):
        """Install partial skip; dequantize INT8 tables before building batched tensors."""
        # Build a temporary dict of dequantized tables so the rest of the logic
        # in V3PartialEngine.install can use self.group_configs normally.
        target_dtype = self.down_proj.weight.dtype if self.down_proj is not None else torch.float16

        dequantized_group_configs = {}
        for gid, cfg in self.group_configs.items():
            if len(cfg) == 4:
                # Legacy v3 format: (addr_idx, addr_mean, addr_std, table)
                addr_idx, addr_mean, addr_std, table = cfg
                dequantized_group_configs[gid] = (addr_idx, addr_mean, addr_std, table)
            else:
                # v4 format: (addr_idx, addr_mean, addr_std, table, scale, zero_point, quantization)
                addr_idx, addr_mean, addr_std, table, scale, zero_point, quantization = cfg
                table_f = dequantize_table(table, scale, zero_point, quantization, target_dtype)
                dequantized_group_configs[gid] = (addr_idx, addr_mean, addr_std, table_f)

        # Temporarily swap so parent install() builds batched tensors from float tables.
        original_group_configs = self.group_configs
        self.group_configs = dequantized_group_configs
        try:
            super().install()
        finally:
            # Restore original quantized configs for save()/inspect.
            self.group_configs = original_group_configs

    def save(self, path: str):
        """Save engine state, preserving quantization info."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        serialized = {}
        for gid, cfg in self.group_configs.items():
            if len(cfg) == 4:
                addr_idx, addr_mean, addr_std, table = cfg
                serialized[gid] = {
                    "addr_idx": addr_idx,
                    "addr_mean": addr_mean,
                    "addr_std": addr_std,
                    "table": table,
                    "quantization": "fp32" if table.dtype == torch.float32 else "fp16",
                }
            else:
                addr_idx, addr_mean, addr_std, table, scale, zero_point, quantization = cfg
                serialized[gid] = {
                    "addr_idx": addr_idx,
                    "addr_mean": addr_mean,
                    "addr_std": addr_std,
                    "table": table,
                    "scale": scale,
                    "zero_point": zero_point,
                    "quantization": quantization,
                }
        torch.save({
            "layer_id": self.layer_id,
            "group_size": self.group_size,
            "num_bins": self.num_bins,
            "addr_clip": self.addr_clip,
            "group_configs": serialized,
        }, path)
        print(f"[V4] Saved to {path}")

    @classmethod
    def load(cls, model, path: str):
        """Load engine state and attach to model."""
        ckpt = torch.load(path, map_location="cpu")
        engine = cls(
            model=model,
            layer_id=ckpt["layer_id"],
            group_size=ckpt["group_size"],
            num_bins=ckpt["num_bins"],
            addr_clip=ckpt["addr_clip"],
        )
        for gid, cfg in ckpt["group_configs"].items():
            engine.add_group(
                group_id=gid,
                addr_idx=cfg["addr_idx"],
                addr_mean=cfg["addr_mean"],
                addr_std=cfg["addr_std"],
                table=cfg["table"],
                scale=cfg.get("scale"),
                zero_point=cfg.get("zero_point", 0.0),
                quantization=cfg.get("quantization", "fp32"),
            )
        print(f"[V4] Loaded from {path}")
        return engine
