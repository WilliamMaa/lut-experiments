"""LLM-LUT v1 Configuration."""

import sys
import os

# Add v0 to path for shared modules
V0_DIR = os.path.join(os.path.dirname(__file__), "..", "v0")
sys.path.insert(0, V0_DIR)

from dataclasses import dataclass
from typing import Tuple

# Re-export v0 config for compatibility
from config import V0Config, get_hook_target  # noqa: F401


@dataclass
class V1Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True

    # Architecture dims (Qwen2.5-0.5B-Instruct)
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24

    # Scan scope (narrowed from v0.5 analysis)
    layer_id: int = 6
    candidate_type: str = "mlp_delta"
    target_group: int = 4

    # Grouping
    hidden_group_size: int = 64
    heads: int = 2  # main experiment; 1-head is control ablation

    # LUT
    num_bins: int = 64
    addr_clip: float = 3.0
    binning_mode: str = "uniform"

    # Training (local prefit)
    lr: float = 1e-3
    num_epochs: int = 40
    alpha_cosine: float = 0.1  # weight for cosine loss
    scheduler: str = "cosine"

    # Data
    max_seq_len: int = 512
    calib_size: int = 1024
    eval_size: int = 512
    calib_batch_size: int = 4
    eval_batch_size: int = 4

    # Paths
    calib_path: str = "../v0/data/calib.jsonl"
    eval_path: str = "../v0/data/eval.jsonl"
    result_dir: str = "results"

    # Misc
    seed: int = 42

    @property
    def group_size(self) -> int:
        return self.hidden_group_size

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.hidden_group_size

    @property
    def hidden_num_groups(self) -> int:
        return self.hidden_size // self.hidden_group_size
