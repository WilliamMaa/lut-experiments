"""Request identity registry (docs/31 I4/I5). Pure python, no vllm import.

Identity rule (the ONLY one): a v8 per-request state is keyed by vLLM's
request_id, published each step by the worker as input_batch.req_ids
(F4, gpu_model_runner.py). Blocks, block tables, seq_lens, snap_len and
every other heuristic field are BANNED from identity decisions.

Integration contract (implemented at integration time, not here):
- The runner patch stashes `tuple(input_batch.req_ids[:num_reqs])` into
  StepContext once per executed step (eager mode is single-threaded per
  step, so a module-global current context is exact, not approximate).
- The metadata builder calls sync(step_req_ids) BEFORE serving any state;
  any request_id not yet in the registry gets a fresh state.
- Finished requests are dropped via drop_finished(finished_req_ids), which
  the worker learns from the scheduler output every step.

Why this is safe under async scheduling: req_ids and batch order are the
worker's own view of the step it is ABOUT to execute; they cannot lag the
metadata the way seq_lens can, because both come from the same scheduler
output the worker just consumed.
"""
from .units import UnitError


class IdentityError(UnitError):
    """Identity contract violation (I4/I5) with full context."""


class RequestStateStub:
    """Placeholder carried by the registry in tests; the integration
    substitutes the real per-layer state container. Identity only ever
    compares object identity of what the registry handed out — it never
    looks inside."""

    def __init__(self, request_id):
        self.request_id = request_id
        # arbitrary mutable payload the algorithm attaches; the registry
        # treats it as opaque.
        self.payload = {}


class StepContext:
    """Per-executed-step request identity, set by the runner patch."""

    def __init__(self, req_ids):
        self.req_ids = tuple(req_ids)

    def __eq__(self, other):
        return isinstance(other, StepContext) and self.req_ids == other.req_ids


_current = None


def set_step_context(ctx: StepContext | None) -> None:
    global _current
    _current = ctx


def get_step_context() -> StepContext | None:
    return _current


class IdentityRegistry:
    """request_id -> state. One per metadata builder (per KV cache group);
    states are cheap control planes, the algorithm state they hold is what
    matters."""

    def __init__(self, state_cls=RequestStateStub):
        self.states: dict = {}
        self.state_cls = state_cls
        self.created = 0      # diagnostics: new states handed out
        self.dropped = 0      # diagnostics: finished requests removed

    def sync(self, req_ids) -> list:
        """Call once per step BEFORE any state is used. Returns the list of
        state objects, one per batch slot, in batch order (I4: batch slot i
        <-> req_ids[i] <-> states[i], nothing else participates)."""
        states = []
        for rid in req_ids:
            st = self.states.get(rid)
            if st is None:
                st = self.state_cls(rid)
                self.states[rid] = st
                self.created += 1
            states.append(st)
        return states

    def drop_finished(self, finished_req_ids) -> None:
        for rid in finished_req_ids:
            if self.states.pop(rid, None) is not None:
                self.dropped += 1

    def require_live(self, req_id) -> None:
        """I5 guard: using a state after drop is a contract violation."""
        if req_id not in self.states:
            raise IdentityError(
                f"request_id {req_id!r} used after finish/drop; "
                f"live={sorted(self.states)[:8]}")
