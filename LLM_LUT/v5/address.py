"""
Address generators for v5 hybrid LUT.

All address generation is fixed/random (no trainable neural networks), so it stays
within the O(1) LUT-lookup regime.  Table values themselves are trainable.
"""

from typing import Tuple, Optional
import torch


class Address2D:
    """Original v3/v4 style: 2 selected channels -> 2D bins -> flattened index."""

    def __init__(self, addr_idx: torch.Tensor, addr_mean: torch.Tensor,
                 addr_std: torch.Tensor, num_bins: int = 64, addr_clip: float = 3.0):
        """
        Args:
            addr_idx: [2] channel indices in the residual/input tensor
            addr_mean: [2] calibration mean
            addr_std: [2] calibration std
        """
        self.addr_idx = addr_idx.cpu().long()
        self.addr_mean = addr_mean.cpu()
        self.addr_std = addr_std.cpu()
        self.num_bins = num_bins
        self.addr_clip = addr_clip
        self.num_entries = num_bins * num_bins
        self.num_tables = 1

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_size]
        Returns:
            indices: [B, S, 1] flattened 2D bin index
        """
        B, S, _ = x.shape
        device = x.device
        idx = self.addr_idx.to(device)
        addr = x.index_select(-1, idx)  # [B, S, 2]

        mean = self.addr_mean.to(device, x.dtype).view(1, 1, -1)
        std = self.addr_std.to(device, x.dtype).view(1, 1, -1).clamp_min(1e-6)

        z = (addr - mean) / std
        z = z.clamp(-self.addr_clip, self.addr_clip)
        qf = (z + self.addr_clip) / (2.0 * self.addr_clip) * (self.num_bins - 1)
        b = torch.round(qf).long().clamp(0, self.num_bins - 1)
        flat = b[:, :, 0] * self.num_bins + b[:, :, 1]
        return flat.unsqueeze(-1)  # [B, S, 1]


class AddressHighOrderRandom:
    """
    Fixed random high-order address.

    For each of M tables and B bits, randomly select K input channels and random
    signs, project x, standardize, and threshold to get a binary bit.
    The B bits form an integer index in [0, 2^B).

    This is inspired by the high-order comparisons in new_lut.py, but kept as
    fixed random LSH with no trainable index generator.
    """

    def __init__(self, input_dim: int, num_tables: int, num_bits: int,
                 channels_per_bit: int = 4, seed: int = 0,
                 addr_mean: Optional[torch.Tensor] = None,
                 addr_std: Optional[torch.Tensor] = None):
        """
        Args:
            input_dim: dimension of x (e.g. hidden_size)
            num_tables: number of independent address tables (ensemble)
            num_bits: bits per address -> 2^num_bits entries per table
            channels_per_bit: how many input channels are combined per bit
            seed: random seed for reproducibility
            addr_mean: [num_tables, num_bits] calibration mean of projection
            addr_std: [num_tables, num_bits] calibration std of projection
        """
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.channels_per_bit = channels_per_bit
        self.num_entries = 2 ** num_bits

        gen = torch.Generator().manual_seed(seed)
        # channel_idx: [num_tables, num_bits, channels_per_bit]
        self.channel_idx = torch.randint(
            0, input_dim, (num_tables, num_bits, channels_per_bit), generator=gen
        ).long()
        # signs: [num_tables, num_bits, channels_per_bit] random +/- 1
        self.signs = torch.randint(0, 2, (num_tables, num_bits, channels_per_bit), generator=gen).float() * 2 - 1

        if addr_mean is None:
            addr_mean = torch.zeros(num_tables, num_bits)
        if addr_std is None:
            addr_std = torch.ones(num_tables, num_bits)
        self.addr_mean = addr_mean.cpu()
        self.addr_std = addr_std.cpu().clamp_min(1e-6)

        # powers of two for bit -> integer conversion
        self.register_buffer("powers", (2 ** torch.arange(num_bits)).long())

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, input_dim]
        Returns:
            indices: [B, S, num_tables] integer index per table
        """
        B, S, _ = x.shape
        device = x.device
        N = B * S
        x_flat = x.view(N, self.input_dim)

        # gather selected channels: [N, num_tables, num_bits, channels_per_bit]
        idx = self.channel_idx.to(device)
        selected = x_flat[:, idx]  # advanced indexing -> [N, num_tables, num_bits, channels_per_bit]
        signs = self.signs.to(device, x.dtype)
        proj = (selected * signs).sum(dim=-1)  # [N, num_tables, num_bits]

        mean = self.addr_mean.to(device, x.dtype).view(1, self.num_tables, self.num_bits)
        std = self.addr_std.to(device, x.dtype).view(1, self.num_tables, self.num_bits)
        z = (proj - mean) / std
        bits = (z > 0).long()  # [N, num_tables, num_bits]

        powers = self.powers.to(device).view(1, 1, self.num_bits)
        indices = (bits * powers).sum(dim=-1)  # [N, num_tables]
        return indices.view(B, S, self.num_tables)

    def fit_calibration(self, x: torch.Tensor):
        """Re-compute addr_mean/std from calibration data."""
        with torch.no_grad():
            B, S, _ = x.shape
            N = B * S
            x_flat = x.view(N, self.input_dim)
            idx = self.channel_idx.to(x_flat.device)
            selected = x_flat[:, idx]
            signs = self.signs.to(x_flat.device, x_flat.dtype)
            proj = (selected * signs).sum(dim=-1)  # [N, num_tables, num_bits]
            self.addr_mean = proj.mean(dim=0).cpu()
            self.addr_std = proj.std(dim=0).cpu().clamp_min(1e-6)


class _TreeNode:
    __slots__ = ("channel_idx", "signs", "threshold", "left", "right", "is_leaf", "leaf_index")

    def __init__(self, channel_idx=None, signs=None, threshold=None,
                 left=None, right=None, is_leaf=False, leaf_index=None):
        self.channel_idx = channel_idx
        self.signs = signs
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.leaf_index = leaf_index


class AddressGreedyTree:
    """
    Data-dependent decision-tree address.

    Instead of random hyperplanes, each split is chosen from many random
    candidates to maximize the reduction in target residual variance.
    The tree is built offline on calibration data and fixed afterwards, so
    inference is still O(1) lookup without any trainable index generator.

    Depth = num_bits gives 2^num_bits leaf entries.
    """

    def __init__(self, input_dim: int, num_bits: int, channels_per_bit: int = 4,
                 seed: int = 0, tree_state: Optional[dict] = None):
        self.input_dim = input_dim
        self.num_bits = num_bits
        self.channels_per_bit = channels_per_bit
        self.num_entries = 2 ** num_bits
        self.num_tables = 1
        self.gen = torch.Generator().manual_seed(seed)
        self.root = None
        self._leaf_counter = 0
        if tree_state is not None:
            self.root = self._deserialize(tree_state)

    def build(self, x: torch.Tensor, target: torch.Tensor,
              num_candidates: int = 64, min_samples: int = 32,
              max_samples: int = 65536):
        """
        Build the tree from calibration data.

        Args:
            x: [N, input_dim]
            target: [N, group_size] residual target
            num_candidates: how many random projections to try per split
            min_samples: do not split a node with fewer than 2*min_samples samples
            max_samples: subsample calibration data to this size if larger
        """
        N = x.shape[0]
        if N > max_samples:
            perm = torch.randperm(N, device=x.device)[:max_samples]
            x = x[perm]
            target = target[perm]
        self._leaf_counter = 0
        self.root = self._build_node(x, target, depth=0,
                                     num_candidates=num_candidates,
                                     min_samples=min_samples)

    def _build_node(self, x, target, depth, num_candidates, min_samples):
        N = x.shape[0]
        if depth >= self.num_bits or N < 2 * min_samples:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        parent_var = target.var(dim=0).sum().item()
        if parent_var < 1e-12:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        best_reduction = -1.0
        best_ch = None
        best_signs = None
        best_threshold = None
        best_left_mask = None

        for _ in range(num_candidates):
            ch = torch.randint(0, self.input_dim, (self.channels_per_bit,), generator=self.gen).to(x.device)
            signs = (torch.randint(0, 2, (self.channels_per_bit,), generator=self.gen).float() * 2 - 1).to(x.device)
            proj = (x[:, ch] * signs.to(x.dtype)).sum(dim=-1)
            threshold = proj.median().item()
            left_mask = proj <= threshold
            right_mask = ~left_mask
            n_l = int(left_mask.sum().item())
            n_r = N - n_l
            if n_l < min_samples or n_r < min_samples:
                continue
            var_l = target[left_mask].var(dim=0).sum().item()
            var_r = target[right_mask].var(dim=0).sum().item()
            reduction = parent_var - (n_l * var_l + n_r * var_r) / N
            if reduction > best_reduction:
                best_reduction = reduction
                best_ch = ch
                best_signs = signs
                best_threshold = threshold
                best_left_mask = left_mask

        # If no candidate helps, make a leaf
        if best_reduction <= 0 or best_left_mask is None:
            leaf = _TreeNode(is_leaf=True, leaf_index=self._leaf_counter)
            self._leaf_counter += 1
            return leaf

        left = self._build_node(x[best_left_mask], target[best_left_mask],
                                depth + 1, num_candidates, min_samples)
        right = self._build_node(x[~best_left_mask], target[~best_left_mask],
                                 depth + 1, num_candidates, min_samples)
        return _TreeNode(
            channel_idx=best_ch,
            signs=best_signs,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def compute_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, input_dim]
        Returns:
            indices: [B, S, 1] leaf index
        """
        B, S, _ = x.shape
        N = B * S
        x_flat = x.view(N, self.input_dim)
        out = torch.empty(N, dtype=torch.long, device=x.device)
        all_idx = torch.arange(N, device=x.device)
        self._traverse(self.root, x_flat, all_idx, out)
        return out.view(B, S, 1)

    def _traverse(self, node: _TreeNode, x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor):
        if node.is_leaf:
            out[idx] = node.leaf_index
            return
        ch = node.channel_idx.to(x.device)
        signs = node.signs.to(x.device, x.dtype)
        proj = (x[idx][:, ch] * signs).sum(dim=-1)
        left_mask = proj <= node.threshold
        self._traverse(node.left, x, idx[left_mask], out)
        self._traverse(node.right, x, idx[~left_mask], out)

    def serialize(self) -> dict:
        """Serialize tree to nested dict."""
        return {"tree": self._serialize_node(self.root)}

    def _serialize_node(self, node: _TreeNode):
        if node.is_leaf:
            return {"leaf_index": node.leaf_index}
        return {
            "channel_idx": node.channel_idx.cpu().tolist(),
            "signs": node.signs.cpu().tolist(),
            "threshold": float(node.threshold),
            "left": self._serialize_node(node.left),
            "right": self._serialize_node(node.right),
        }

    def _deserialize(self, state: dict):
        return self._deserialize_node(state["tree"])

    def _deserialize_node(self, d: dict):
        if "leaf_index" in d:
            return _TreeNode(is_leaf=True, leaf_index=d["leaf_index"])
        return _TreeNode(
            channel_idx=torch.tensor(d["channel_idx"], dtype=torch.long),
            signs=torch.tensor(d["signs"], dtype=torch.float32),
            threshold=d["threshold"],
            left=self._deserialize_node(d["left"]),
            right=self._deserialize_node(d["right"]),
        )
