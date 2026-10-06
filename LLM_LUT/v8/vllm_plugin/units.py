"""Unit-tagged KV addressing (docs/31 §2 I1-I3). Pure python, no torch/vllm.

Every length/offset in the v8<->vLLM adapter must carry its unit; the only
way to cross units is through the conversion functions in this module.
Units (docs/31 §1 + §1.1 probe + block_table.py read):

  T   scheduler token (EngineCore accounting: num_scheduled/computed_tokens)
  BG  group manager block size (kv_cache_groups[g].kv_cache_spec.block_size;
      the SCHEDULER allocates cdiv(tokens, BG) blocks per group).
      Measured for Qwen3.6-35B-A3B: BG = 1056 (hybrid GDN override, NOT
      user-settable, NOT 16/32).
  H   hash block (BlockPool.hash_block_size; group block is a multiple of H)
  P   kernel/pool page (runtime kv_cache.shape[2]; the attention kernel's
      page and the granule of block_table_tensor rows). Measured: P = 32.
      block_table.py:64-68 expands manager blocks to kernel blocks when
      BG != P (blocks_per_kv_block = BG // P, kernel_id = mgr_id * q + r),
      so the block table the metadata builder sees is ALREADY in P units.
  S   v8 compact slot (logical token slot in [sink|HH|recent]; 1 S = 1 T
      position, the mapping st.orig[s] -> T)

Conversion chain (docs/31 I2), the ONLY legal addressing path:

  slot s (S, == T position)
    -> j = s // P, r = s % P          (S -> KERNEL block index + inner offset;
                                       the divisor is P, NOT BG, because the
                                       block table rows are kernel units)
    -> b_j = block_row[j]             (bounds-checked against the step's own
                                       T frontier, see blockplan.py — rows
                                       have NO in-band sentinel, padding is
                                       stale/0, block_table.py:124-171)
    -> pool index (b_j, r)

Manager units (BG) appear ONLY in scheduler-facing accounting
(spec.blocks_per_request, the allocator cap): blocks = cdiv(tokens, BG).
Mixing these up is exactly the v2026-10-04t crash: scheduler counted in
1056-token blocks, the plugin sliced and indexed in 32-token units.
"""
from dataclasses import dataclass
from enum import Enum
from math import ceil


class Unit(Enum):
    T = "T"      # scheduler token
    BG = "BG"    # group manager block size (tokens per manager block)
    H = "H"      # hash block (tokens per hash block)
    P = "P"      # kernel/pool page (tokens per pool page)
    S = "S"      # v8 compact slot


class UnitError(ValueError):
    """Mixed-unit arithmetic, or an impossible state (docs/31 I6 fields)."""


@dataclass(frozen=True)
class Qty:
    """A quantity WITH its unit. Arithmetic across units is a hard error
    (docs/31 I1): you cannot divide S by P, you must name the conversion."""
    value: int
    unit: Unit

    def _check(self, other: "Qty", op: str) -> None:
        if self.unit is not other.unit:
            raise UnitError(
                f"unit mismatch in {op}: {self.value}{self.unit.value} vs "
                f"{other.value}{other.unit.value}")

    def __add__(self, other: "Qty") -> "Qty":
        self._check(other, "+")
        return Qty(self.value + other.value, self.unit)

    def __sub__(self, other: "Qty") -> "Qty":
        self._check(other, "-")
        return Qty(self.value - other.value, self.unit)

    def __floordiv__(self, other: "Qty") -> "Qty":
        self._check(other, "//")
        if other.value <= 0:
            raise UnitError(f"division by non-positive {other.value}")
        return Qty(self.value // other.value, Unit.S)  # count of granules

    def __mod__(self, other: "Qty") -> "Qty":
        self._check(other, "%")
        if other.value <= 0:
            raise UnitError(f"modulo by non-positive {other.value}")
        return Qty(self.value % other.value, self.unit)


def as_(q: Qty, unit: Unit) -> int:
    """Assert the unit and return the bare int. Static-grep friendly:
    every bare int that leaves this module passed through as_()."""
    if q.unit is not unit:
        raise UnitError(f"expected {unit.value}, got {q.value}{q.unit.value}")
    return q.value


def cdiv(a: int, b: int) -> int:
    if b <= 0:
        raise UnitError(f"cdiv by non-positive {b}")
    return -(-a // b)


def required_mgr_blocks(compact_len: Qty, block_size: Qty) -> int:
    """Blocks the write-back/gather span needs (docs/31 I3 numerator).
    Both args must be in S/T-compatible granules: compact_len in S, block
    size in the group's BG tokens-per-block (plain int count)."""
    n = as_(compact_len, Unit.S)
    bs = as_(block_size, Unit.BG)
    if n < 0:
        raise UnitError(f"negative compact_len {n}")
    return cdiv(n, bs)


@dataclass(frozen=True)
class AddrCtx:
    """Minimum failure context (docs/31 I6). Every raise carries one."""
    request_id: str
    group_id: int
    token_start_t: int
    token_end_t: int
    token_computed_t: int
    token_scheduled_t: int
    mgr_block_size: int
    pool_page_size: int

    def describe(self) -> str:
        return (f"req={self.request_id} grp={self.group_id} "
                f"T=[{self.token_start_t}..{self.token_end_t}) "
                f"computed={self.token_computed_t} "
                f"scheduled={self.token_scheduled_t} "
                f"B_g={self.mgr_block_size} P={self.pool_page_size}")


def slot_address(slot: Qty, block_row, pool_page: Qty, ctx: AddrCtx):
    """The I2 chain: compact slot -> (physical block id, inner offset).

    The divisor is the POOL PAGE P (kernel units): block_table_tensor rows
    are already kernel-unit ids (block_table.py expands manager blocks), so
    BG must NOT appear here. Bounds are checked against the caller-visible
    row (docs/31 I3: fail-closed, NEVER clamp); the authoritative frontier
    check lives in blockplan.build_block_plan and runs BEFORE any tensor
    indexing. Returns (b_j: int, r: int) with 0 <= r < P.
    """
    s = as_(slot, Unit.S)
    pp = as_(pool_page, Unit.P)
    if s < 0:
        raise UnitError(f"negative slot {s}; {ctx.describe()}")
    j, r = s // pp, s % pp
    need = j + 1
    have = len(block_row)
    if need > have:
        raise UnitError(
            f"block row shorter than write span: need {need} pool blocks "
            f"(slot {s}, P {pp}), have {have}; "
            f"row[:16]={list(block_row[:16])}; {ctx.describe()}")
    return int(block_row[j]), r


def mgr_block_to_pool_pages(j: int, mgr_block_size: Qty,
                            pool_page_size: Qty) -> range:
    """Map one manager block to its pool pages. Only legal if B_g % P == 0
    (probed in Gate 1; if vLLM ever reshapes the other way this function is
    where the new rule gets written, nowhere else)."""
    bg = as_(mgr_block_size, Unit.BG)
    pp = as_(pool_page_size, Unit.P)
    if bg % pp != 0:
        raise UnitError(f"B_g {bg} is not a multiple of P {pp}")
    pages_per_block = bg // pp
    return range(j * pages_per_block, (j + 1) * pages_per_block)
