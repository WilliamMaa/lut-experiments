"""LLM-LUT v0 Configuration."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class V0Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    # CRITICAL: never use device_map="auto" here. Always manual single-GPU placement.
    trust_remote_code: bool = True

    # Architecture dims (Qwen2.5-0.5B-Instruct)
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2

    # Scan scope
    layer_ids: Tuple[int, ...] = (6, 12, 18)
    candidate_types: Tuple[str, ...] = ("down_proj", "mlp_delta", "attn_out")

    # Grouping
    hidden_group_size: int = 64
    intermediate_group_size: int = 128
    heads: int = 2

    # Bucket
    num_bins: int = 64
    addr_clip: float = 3.0

    # Data
    max_seq_len: int = 512
    calib_size: int = 512
    eval_size: int = 256
    calib_batch_size: int = 4
    eval_batch_size: int = 4

    # Perturbation
    noise_sigmas: Tuple[float, ...] = (0.05, 0.10, 0.20)

    # Paths
    calib_path: str = "data/calib.jsonl"
    eval_path: str = "data/eval.jsonl"
    result_dir: str = "results"

    # Misc
    seed: int = 42

    @property
    def hidden_num_groups(self) -> int:
        return self.hidden_size // self.hidden_group_size

    @property
    def intermediate_num_groups(self) -> int:
        return self.intermediate_size // self.intermediate_group_size


# Hook target accessors
def get_hook_target(model, layer_id: int, candidate_type: str):
    """Return the nn.Module to hook for a given layer and candidate type."""
    layer = model.model.layers[layer_id]
    if candidate_type == "down_proj":
        return layer.mlp.down_proj
    elif candidate_type == "mlp_delta":
        return layer.mlp
    elif candidate_type == "attn_out":
        # Use o_proj instead of full self_attn because self_attn receives kwargs,
        # making input capture unreliable via forward hooks.
        return layer.self_attn.o_proj
    elif candidate_type == "intermediate":
        return layer.mlp
    else:
        raise ValueError(f"Unknown candidate_type: {candidate_type}")
