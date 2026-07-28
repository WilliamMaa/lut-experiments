"""
LLM_LUT v6 - Core library for LUT-based FFN replacement.
"""

__version__ = "6.0.0"

from .models.qwen_moe import QwenMoEExpert, load_real_teacher
from .address.tree import AddressGreedyTree, _TreeNode
from .address.two_d import Address2D
from .address.high_order import AddressHighOrderRandom
from .lut.core import LUTGroup, table_storage_bytes_for_group, count_leaves_for_group
from .engine.replacement import V6ReplacementEngine

__all__ = [
    "QwenMoEExpert",
    "load_real_teacher",
    "AddressGreedyTree",
    "_TreeNode",
    "Address2D",
    "AddressHighOrderRandom",
    "LUTGroup",
    "table_storage_bytes_for_group",
    "count_leaves_for_group",
    "V6ReplacementEngine",
]
