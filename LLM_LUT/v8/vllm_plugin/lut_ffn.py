"""lut_ffn.py — P1.3 shared_expert LUT replacement (v8/docs/43).

Replaces `layer.mlp.shared_expert.forward` (Qwen2MoeMLP/Qwen3NextMLP) on
the configured layers with a triton LUT lookup so the dense FFN GEMMs
never execute. A post-hoc hook that overwrites the output would leave
MAC savings at zero (docs/40) — this swaps the module forward instead.

Data flow per token x [N, hidden]:
  1. _k_leaf (x2 launches): fixed-depth walk of the padded oblique
     trees. Coarse tree (T=1, coarse_num_bits deep, shared across
     groups) + per-group residual trees (T=num_groups, residual
     num_bits deep). Trees may stop early: leaf/pad rows self-loop, so
     a fixed-depth loop is exact. Produces leaf_out [N, 1+G] int32.
  2. _k_gather_add: out[n, :] = coarse_table[leaf_c] + sum over groups
     of resid_table[g, leaf_g], accumulated in fp32 (the v6 engine
     adds in fp16; fp32 can only be closer to the teacher), stored bf16.

TP: SharedFusedMoE all-reduces the shared output across ranks
(shared_fused_moe.py:32-37), so rank 0 returns the LUT result and the
other ranks return zeros — the sum equals the single-rank output.

Tables are registered as non-persistent buffers on the shared_expert
instance at model-build time, so .to(device) moves them and profile_run
counts them in non-KV memory. Every TP rank holds the full table
(simple v1, ~320 MiB/layer fp16).

vLLM serves with --enforce-eager (serve.py), no CUDA-graph burden.
"""
import os
import re

import torch

from . import config


def _detect_triton():
    """True when triton imports (remote vllm_py310); False on this box.
    A def + top-level assign (not a try/except or `if` block) so the
    test_ast_names scanner sees HAS_TRITON from every scope."""
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except Exception:  # local box has no GPU stack
        return False


HAS_TRITON = _detect_triton()


def _define_kernels():
    # Kernels live inside a top-level def (not a module-level `if` block)
    # so the test_ast_names top-level-scope scanner can place them; the
    # triton import is local for the same reason. Called (via _KERNELS)
    # only when HAS_TRITON.
    try:
        import triton
        import triton.language as tl
    except Exception:
        return None, None

    @triton.jit
    def _k_leaf(x_ptr, ch_ptr, sg_ptr, thr_ptr, leaf_ptr, left_ptr,
                right_ptr, out_ptr, D, n_max, T_out, OUT_OFF,
                DEPTH: tl.constexpr):
        """One program walks ONE tree for ONE token. grid = (N, T).

        Tree params are stacked [T, n_max, ...]; program_id(1) = t.
        out_ptr is leaf_out [N, 1+G] int32; this tree's slot is
        t + OUT_OFF (coarse launches with T=1, OUT_OFF=0; residual
        launches with T=G, OUT_OFF=1).
        """
        n = tl.program_id(0)
        t = tl.program_id(1)
        offs = tl.arange(0, 4)
        base = t.to(tl.int64) * n_max
        node = tl.full((), 0, tl.int64)
        for _ in range(DEPTH):
            row = (base + node) * 4
            ch = tl.load(ch_ptr + row + offs)
            sg = tl.load(sg_ptr + row + offs)
            xs = tl.load(x_ptr + n * D + ch)
            proj = tl.sum(xs.to(tl.float32) * sg, axis=0)
            thr = tl.load(thr_ptr + base + node)
            go_left = proj <= thr
            lft = tl.load(left_ptr + base + node)
            rgt = tl.load(right_ptr + base + node)
            node = tl.where(go_left, lft, rgt)
        lf = tl.load(leaf_ptr + base + node)
        tl.store(out_ptr + n * T_out + (t + OUT_OFF), lf.to(tl.int32))

    @triton.jit
    def _k_gather_add(leaf_out_ptr, coarse_ptr, resid_ptr, out_ptr,
                      D, CH, E_R, T_out,
                      BLOCK: tl.constexpr):
        """out[n, :] = coarse[leaf_c] + resid gather-add, fp32 acc.

        grid = (N,). resid_table layout [G, E_R, CH] (contiguous):
        group g starts at g * E_R * CH; row leaf_g at leaf_g * CH.
        """
        n = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        leaf_c = tl.load(leaf_out_ptr + n * T_out).to(tl.int64)
        base_c = coarse_ptr + leaf_c * D
        acc = tl.load(base_c + cols, mask=mask, other=0.0).to(tl.float32)
        g = cols // CH
        ch_off = cols % CH
        leaf_g = tl.load(leaf_out_ptr + n * T_out + 1 + g,
                         mask=mask, other=0).to(tl.int64)
        g64 = g.to(tl.int64)
        addr = g64 * (E_R * CH) + leaf_g * CH + ch_off
        acc += tl.load(resid_ptr + addr, mask=mask,
                       other=0.0).to(tl.float32)
        tl.store(out_ptr + n * D + cols, acc.to(tl.bfloat16), mask=mask)

    return _k_leaf, _k_gather_add


# Bound via a top-level assign (scanner visibility), not an `if` block.
_KERNELS = _define_kernels() if HAS_TRITON else (None, None)
_k_leaf, _k_gather_add = _KERNELS


def _load_bundle_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        return torch.load(path, map_location="cpu")


class LutFfnReplacer:
    """Loads one layer's bundle and runs the LUT forward.

    `bundle` is a bundle.pt path or an already-loaded dict (tests).
    The reference path (forward_reference) is pure torch — used for the
    CPU test comparison and for on-machine debugging.
    """

    def __init__(self, bundle, device):
        if isinstance(bundle, (str, os.PathLike)):
            bundle = _load_bundle_file(bundle)
        self.hidden = int(bundle["hidden"])
        self.group_size = int(bundle["group_size"])
        self.num_groups = int(bundle["num_groups"])
        self.coarse_num_bits = int(bundle["coarse_num_bits"])
        self.residual_num_bits = int(bundle["residual_num_bits"])
        if bundle.get("has_lowrank") or bundle.get("has_pairwise"):
            print("[v8_plugin] WARNING: bundle carries lowrank/pairwise "
                  "corrections that P1.3 v1 does not implement; they are "
                  "IGNORED (quality will be lower than the v6 engine)",
                  flush=True)
        self.coarse_table = bundle["coarse_table"].to(torch.float16)
        self.resid_table = bundle["resid_table"].to(torch.float16)
        # coarse tensors get a leading T=1 dim so the kernel/reference
        # treat {coarse, resid} uniformly (stacked [T, n_max, ...]).
        self._c = {k: v.unsqueeze(0).contiguous()
                   for k, v in bundle["coarse"].items()}
        self._r = {k: v.contiguous() for k, v in bundle["resid"].items()}
        self._c_n = self._c["thr"].shape[1]
        self._r_n = self._r["thr"].shape[1]
        self._blocks = 1 << (self.hidden - 1).bit_length()
        self._to(device)

    # -- device management --
    def _to(self, device):
        device = torch.device(device)
        self.coarse_table = self.coarse_table.to(device)
        self.resid_table = self.resid_table.to(device)
        for d in (self._c, self._r):
            for k in d:
                d[k] = d[k].to(device)
        self._device = device
        return self

    def to(self, device):
        return self._to(device)

    @property
    def device(self):
        return self._device

    # -- forward --
    def __call__(self, x):
        if x.dim() == 3:
            b, s, d = x.shape
            return self._forward2d(x.reshape(b * s, d)).view(b, s, -1)
        return self._forward2d(x)

    def _forward2d(self, x):
        if x.device != self._device:
            self._to(x.device)
        if not HAS_TRITON:
            return self.forward_reference(x)
        n, d = x.shape
        t_out = 1 + self.num_groups
        leaf_out = torch.empty((n, t_out), dtype=torch.int32,
                               device=x.device)
        xc = x.contiguous()
        _k_leaf[(n, 1)](
            xc, self._c["ch"], self._c["signs"], self._c["thr"],
            self._c["leaf"], self._c["left"], self._c["right"],
            leaf_out, d, self._c_n, t_out, 0,
            DEPTH=self.coarse_num_bits)
        _k_leaf[(n, self.num_groups)](
            xc, self._r["ch"], self._r["signs"], self._r["thr"],
            self._r["leaf"], self._r["left"], self._r["right"],
            leaf_out, d, self._r_n, t_out, 1,
            DEPTH=self.residual_num_bits)
        out = torch.empty((n, d), dtype=torch.bfloat16, device=x.device)
        _k_gather_add[(n,)](
            leaf_out, self.coarse_table, self.resid_table, out,
            d, self.group_size, self.resid_table.shape[1], t_out,
            BLOCK=self._blocks)
        return out

    # -- pure-torch reference (same padded-tree semantics) --
    def forward_reference(self, x):
        if x.dim() == 3:
            b, s, d = x.shape
            return self.forward_reference(
                x.reshape(b * s, d)).view(b, s, -1).to(x.dtype)
        xf = x.float()
        leaf_c = self._walk_ref(xf, self._c, self.coarse_num_bits)[0]  # [N]
        leaf_r = self._walk_ref(xf, self._r, self.residual_num_bits)  # [G,N]
        n = x.shape[0]
        g, ch = self.num_groups, self.group_size
        gid = torch.arange(g, device=x.device).view(g, 1)
        out = self.coarse_table[leaf_c].float().view(n, g, ch)
        out += self.resid_table[gid, leaf_r].permute(1, 0, 2)
        return out.view(n, self.hidden).to(x.dtype)

    @staticmethod
    def _walk_ref(x, tree, depth):
        """Fixed-depth walk of stacked padded trees. tree tensors are
        [T, n_max, ...]; x is [N, D] fp32. Returns leaf rows [T, N]."""
        ch_t, sg_t, thr_t = tree["ch"], tree["signs"], tree["thr"]
        left_t, right_t, leaf_t = tree["left"], tree["right"], tree["leaf"]
        t_count, n_max = thr_t.shape
        n = x.shape[0]
        dev = x.device
        tid = torch.arange(t_count, device=dev).view(t_count, 1)
        node = torch.zeros((t_count, n), dtype=torch.long, device=dev)
        xg = x.unsqueeze(0).expand(t_count, n, x.shape[1])
        for _ in range(depth):
            c = ch_t[tid, node]                    # [T, N, 4]
            s = sg_t[tid, node]
            xs = torch.take_along_dim(xg, c, dim=2)
            proj = (xs * s).sum(dim=-1)
            go_left = proj <= thr_t[tid, node]
            node = torch.where(go_left, left_t[tid, node],
                               right_t[tid, node])
        return leaf_t[tid, node]


def load_lut_bundles(bundle_dir, layers, device="cpu"):
    """{layer_idx: LutFfnReplacer} for layer{N}.pt under bundle_dir."""
    out = {}
    for layer_idx in layers:
        path = os.path.join(bundle_dir, f"layer{layer_idx}.pt")
        if not os.path.exists(path):
            print(f"[v8_plugin] WARNING: LUT bundle not found: {path}",
                  flush=True)
            continue
        out[layer_idx] = LutFfnReplacer(path, device)
    return out


def is_tp_rank0():
    """True on TP rank 0 (or when vllm/TP is unavailable)."""
    try:
        from vllm.distributed import parallel_state
        fn = getattr(parallel_state, "get_tensor_model_parallel_rank", None)
        if fn is None:
            return True
        return int(fn()) == 0
    except Exception:
        return True


def parse_layers(spec):
    """\"39\" / \"37,38,39\" / \"37 38 39\" -> [37, 38, 39]."""
    out = []
    for tok in re.split(r"[,\s]+", str(spec).strip()):
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            print(f"[v8_plugin] WARNING: bad V8_LUT_LAYERS token {tok!r} "
                  f"ignored", flush=True)
    return out


_bundle_cache = {}


def _get_replacer(layer_idx, bundle_dir):
    if layer_idx in _bundle_cache:
        return _bundle_cache[layer_idx]
    replacer = None
    path = os.path.join(bundle_dir, f"layer{layer_idx}.pt")
    try:
        if not os.path.exists(path):
            print(f"[v8_plugin] WARNING: LUT bundle not found: {path} "
                  f"(layer {layer_idx} stays dense)", flush=True)
        else:
            replacer = LutFfnReplacer(path, "cpu")
    except Exception as e:
        print(f"[v8_plugin] WARNING: LUT bundle load failed for layer "
              f"{layer_idx}: {type(e).__name__}: {e} "
              f"(layer stays dense)", flush=True)
    _bundle_cache[layer_idx] = replacer
    return replacer


def _find_layer_index(module, args, kwargs):
    """Layer number from the constructor prefix (vllm passes
    '...layers.N....' as prefix kwarg). None if not found."""
    hay = [kwargs.get("prefix", ""), getattr(module, "prefix", "")]
    hay += [a for a in args if isinstance(a, str)]
    hay += [v for v in kwargs.values() if isinstance(v, str)]
    for s in hay:
        m = re.search(r"\blayers\.(\d+)\.", str(s))
        if m:
            return int(m.group(1))
    return None


def _install_lut(module, layer_idx, replacer):
    shared = module.shared_expert
    # Non-persistent buffers: ride .to(device), stay out of state_dict
    # (load_weights never touches them), counted in non-KV memory.
    shared.register_buffer("v8_lut_coarse_table", replacer.coarse_table,
                           persistent=False)
    shared.register_buffer("v8_lut_resid_table", replacer.resid_table,
                           persistent=False)
    rank0 = is_tp_rank0()

    def lut_forward(x, _rep=replacer, _rank0=rank0):
        # SharedFusedMoE all-reduces the shared output across TP ranks:
        # only rank 0 contributes the LUT result, the rest contribute
        # zeros so the post-reduce sum equals the single-rank output.
        if not _rank0:
            return torch.zeros_like(x)
        return _rep(x)

    shared.forward = lut_forward
    print(f"[v8_plugin] LUT shared_expert layer {layer_idx} installed "
          f"(bundle layer{layer_idx}.pt)", flush=True)


def patch_shared_expert_lut(layers):
    """Wrap Qwen3NextSparseMoeBlock.__init__ to swap shared_expert.forward
    for the LUT lookup on the given layers. Silent no-op on vllm builds
    without that class. Called from patch() only when V8_LUT_LAYERS is
    non-empty — default behavior is unchanged."""
    if not HAS_TRITON:
        print("[v8_plugin] WARNING: triton not available, shared_expert "
              "LUT replacement disabled", flush=True)
        return
    if not config.V8_LUT_BUNDLE_DIR:
        print("[v8_plugin] WARNING: V8_LUT_LAYERS set but "
              "V8_LUT_BUNDLE_DIR is empty; LUT replacement disabled",
              flush=True)
        return
    try:
        from vllm.model_executor.models.qwen3_next import (
            Qwen3NextSparseMoeBlock)
    except Exception as e:
        print("[v8_plugin] WARNING: Qwen3NextSparseMoeBlock import failed "
              f"({type(e).__name__}: {e}); LUT replacement skipped",
              flush=True)
        return
    layers = frozenset(layers)
    bundle_dir = config.V8_LUT_BUNDLE_DIR
    orig_init = Qwen3NextSparseMoeBlock.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            layer_idx = _find_layer_index(self, args, kwargs)
        except Exception:
            layer_idx = None
        if layer_idx is None or layer_idx not in layers:
            return
        replacer = _get_replacer(layer_idx, bundle_dir)
        if replacer is None:
            return
        try:
            _install_lut(self, layer_idx, replacer)
        except Exception as e:
            print(f"[v8_plugin] WARNING: LUT install failed on layer "
                  f"{layer_idx}: {type(e).__name__}: {e}", flush=True)

    Qwen3NextSparseMoeBlock.__init__ = patched_init
    print(f"[v8_plugin] Qwen3NextSparseMoeBlock.__init__ wrapped for LUT "
          f"layers {sorted(layers)} (v{config.PLUGIN_VERSION})", flush=True)
