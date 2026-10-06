"""Unit-tagged KV addressing (docs/31 §2 I1-I3). Pure python, no torch/vllm.

Every length/offset in the v8<->vLLM adapter must carry its unit; the only
way to cross units is through the conversion functions in this module.
Units (docs/31 §1):

  T   scheduler token (EngineCore accounting: num_scheduled/computed_tokens)
  BG  group manager block size (kv_cache_groups[g].kv_cache_spec.block_size;
      the scheduler allocates cdiv(tokens, BG) blocks per group)
  H   hash block (BlockPool.hash_block_size; group block is a multiple of H)
  P   kernel/pool page (runtime kv_cache.shape[2], the tensor index granule)
  S   v8 compact slot (logical token slot in [sink|HH|recent]; 1 S = 1 T
      position, the mapping st.orig[s] -> T)

Conversion chain (docs/31 I2), the ONLY legal addressing path:

  slot s (S)
    -> j = s // BG, r = s % BG          (S -> manager block index + inner offset)
    -> b_j = block_ids[j]               (block-table lookup, bounds-checked)
    -> pool token offset = j * BG + r   (== s; identity, kept for clarity)

If the Gate-1 probe finds P != BG for any group (hybrid unified-page reshape),
mgr_block_to_pool_pages() is the single place where that ratio is applied;
nothing outside this module may do unit arithmetic.
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


def slot_address(slot: Qty, block_ids, block_size: Qty, ctx: AddrCtx):
    """The I2 chain: compact slot -> (physical block id, inner offset).

    Raises UnitError with the full I6 context if the block table is shorter
    than the span needs (docs/31 I3: fail-closed, NEVER clamp).
    Returns (b_j: int, r: int) with 0 <= r < B_g.
    """
    s = as_(slot, Unit.S)
    bs = as_(block_size, Unit.BG)
    if s < 0:
        raise UnitError(f"negative slot {s}; {ctx.describe()}")
    j, r = s // bs, s % bs
    need = j + 1
    have = len(block_ids)
    if need > have:
        raise UnitError(
            f"block table shorter than write span: need {need} blocks "
            f"(slot {s}, B_g {bs}), have {have}; "
            f"block_ids[:16]={list(block_ids[:16])}; {ctx.describe()}")
    return int(block_ids[j]), r


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
