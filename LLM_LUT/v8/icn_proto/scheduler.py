#!/usr/bin/env python3
"""Scheduler: fast-path dispatch over the prefix block chain
(docs/icn-defined-addressing/05-request-lifecycle.md v3 §2).

Per turn:
  Derive  block-name chain from the full prefix token ids (pure function)
  Match   per worker: E_local = longest resume boundary (a checkpoint tip
          position whose whole block chain [0, E) is resident locally)
  Schedule  C_j = fresh_tokens/R_prefill + fetch_bytes/R_xfer
          (fetch = contiguous single-holder extension past E_local;
           multi-source plans are treated as not-fetchable in this v1)
  Dispatch  fetch blocks (bundle) -> deliver -> assign
  Publish worker extracts chain blocks it does not yet hold; the last
          block of every batch carries the GDN linear checkpoint
  Slow    placement controller (05 v3 §3, 'ours' only): proactive
          segment replication under the G_rep economic trigger, plus
          coldest-segment eviction under a per-worker memory budget

Run (spawned by run_cluster.py, which also starts the workers):
    python -m icn_proto.run_cluster --sessions 16 ...
"""

import json
import os
import random
import sys
import time
import heapq
from dataclasses import dataclass, field
from datetime import datetime

import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icn_proto import msg
from icn_proto.blkchain import GENESIS, BlockName, chain_through, derive_chain

STALL_S = 300.0   # watchdog hard-fail: max seconds a worker may stall

TRACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "multi_turn_prompts_v3.jsonl")


@dataclass
class Turn:
    """One dispatchable unit of a session's chain.

    turn == -1 is the synthetic doc-turn (prefills the shared document,
    publishes its chain). Question turns t >= 0 append their question
    tokens onto the session prefix.

    prefill_ids: tokens the PREVIOUS design prefilled; retained for
    workload building. The block protocol computes its own resume
    boundary, so only prefix_ids matters at dispatch time."""

    session: str
    turn: int
    prefill_ids: list
    prefix_ids: list
    t_ready: float = 0.0      # entered the ready queue (queue_s baseline)
    t_arrive: float = 0.0     # client-observed arrival (TTFT baseline);
                              # == t_ready in closed-loop mode
    _chain_cache: dict = field(default_factory=dict, repr=False,
                               compare=False)

    @property
    def cum_tokens(self) -> int:
        return len(self.prefix_ids)

    def chain(self, repr_name: str, block_tokens: int,
              model_tag: str = "qwen35b") -> list:
        key = ("chain", repr_name, block_tokens)
        if key not in self._chain_cache:
            # COMPLETE blocks only: the trailing partial block [kB, n) is
            # not yet a publishable unit — its name would change once more
            # tokens arrive (the same positions become a full block with
            # different contents). It becomes a block when the prefix
            # grows past the next multiple of b.
            self._chain_cache[key] = [
                b for b in derive_chain(self.prefix_ids, repr_name,
                                        block_tokens, model_tag)
                if b.span_tokens == block_tokens]
        return self._chain_cache[key]

    def tip_name_at(self, n_tokens: int, repr_name: str, block_tokens: int,
                    model_tag: str = "qwen35b") -> str:
        """Block name of the chain tip at exactly n_tokens (the partial
        block [floor, n) if n is not block-aligned, else the last complete
        block). This is where the GDN checkpoint for a prefix of length
        n_tokens lives."""
        key = ("tipat", n_tokens, repr_name, block_tokens)
        if key not in self._chain_cache:
            sub = derive_chain(self.prefix_ids[:n_tokens], repr_name,
                               block_tokens, model_tag)
            self._chain_cache[key] = str(sub[-1])
        return self._chain_cache[key]

    def prefix_fingerprint(self, repr_name: str, block_tokens: int) -> str:
        """Content ID of the whole prefix: the exact tip name at cum_tokens
        — its block_hash chains over every block INCLUDING the partial
        tail, so any token or length difference changes it. Used by
        check_consistency to pair turns that must produce identical
        decodes."""
        return self.tip_name_at(self.cum_tokens, repr_name, block_tokens)

    def publish_names(self, repr_name: str, block_tokens: int,
                      model_tag: str = "qwen35b") -> list:
        """All blocks this turn produces: complete blocks + the tip block
        (partial unless the turn end is block-aligned). The tip carries
        the GDN checkpoint and is always last."""
        names = [str(b) for b in self.chain(repr_name, block_tokens,
                                            model_tag)]
        tip = self.tip_name_at(self.cum_tokens, repr_name,
                               block_tokens, model_tag)
        if tip not in names:
            names.append(tip)
        return names


@dataclass
class WorkerState:
    ident: bytes
    busy: bool = False
    resident: set = field(default_factory=set)
    tips: set = field(default_factory=set)   # blocks carrying GDN checkpoints
    current: str | None = None
    t_assign: float = 0.0
    cur_plan: dict | None = None    # {prefill_tokens, decode_steps} of
                                    # the in-flight turn (scheduler knows
                                    # the plan it assigned; no worker
                                    # round-trip needed)
    resident_bytes: int = 0
    prefill_rate: float = 0.0       # per-worker EWMA, tok/s (0 = unknown)
    via_repl: set = field(default_factory=set)
    # names that reached this worker through a controller REPLICATION
    # (not a demand fetch, not a local publish) — the E1 accounting of
    # "remote resume served local because we placed it here proactively"
    spilled: set = field(default_factory=set)
    # E2: names in the host-DRAM backing tier (evicted but not lost).
    # Resume feasibility and holder checks are resident ∪ spilled;
    # resident_bytes / budget accounting stay HBM-only.
    spill_bytes: int = 0
    spill_stat: dict = field(default_factory=dict)


class Scheduler:
    # measured 2026-09-19 (workload-fixed runs): heavy-hitter prefill
    # ~2.5K tok/s, block fetch ~100MB/s. EWMA-refined online.
    PREFILL_RATE0 = 2_500.0
    XFER_RATE0 = 100e6

    def __init__(self, args, worker_ids):
        self.args = args
        self.workers = {wid.encode(): WorkerState(ident=wid.encode())
                        for wid in worker_ids}
        self.ready = []
        self.pending_next = {}
        self.turns_of = {}
        # Directory (05 v3 §2): block name -> metadata. Identity is
        # content-derived (blkchain), so entries are created on publish;
        # zero-replica entries may simply be dropped — identity survives
        # via re-derivation. lambda = EWMA of observed access rate,
        # counted at assign time on the resume set; a tip block's lambda
        # is the arrival rate of requests resuming at that position, i.e.
        # the demand signal for the Step-4 placement trigger.
        self.dir = {}
        self.tips = set()          # all known checkpoint tip blocks
        self.records = []
        self.transfer_bytes = 0
        self.transfers = 0
        self.t_start = time.time()
        self.done_sessions = 0
        self._xfer = {}
        # Placement controller (slow path, 05 v3 §3): proactive segment
        # replication + eviction. Replication transfers ride the same
        # fetch/deliver data plane as demand fetches but live in their
        # own table (no turn, no busy flag).
        self._repl = {}
        self._repl_seq = 0
        self._repl_cool = {}        # (tip, target_ident) -> cooldown expiry
        self.replications = 0
        self.replicated_bytes = 0
        self.repl_planned = 0       # actions the controller committed to
                                    # (vs replications = completed acks)
        self.evictions = 0
        self.evicted_blocks = 0
        self.evicted_bytes = 0
        self.prefill_rate = self.PREFILL_RATE0
        self.xfer_rate = self.XFER_RATE0
        # Open-arrival machinery (step 6): sessions arrive on a Poisson
        # clock and a session's next turn only becomes ready after a
        # think-time delay. Until then turns sit in the arrivals min-heap
        # and the poll loop sleeps until the earliest one is due —
        # closed-loop mode leaves the heap empty and behaves as before.
        self.arrivals = []            # heap of (t_arrive, seq, Turn)
        self._arr_seq = 0
        self._think_rng = random.Random(getattr(args, "seed", 0) + 1)
        # E1 (07-regime-study §4) residency-opportunity accounting:
        # session -> worker id of its last ASSIGNED turn (== publisher of
        # the session's latest state for successful turns). A question
        # turn assigned to a different worker is a remote-resume
        # opportunity — replication can serve it locally, recompute can
        # only eat the cost.
        self._last_worker = {}
        self.remote_resume_opportunities = 0
        self.repl_served_local = 0
        # why the placement controller passed on a (tip, target) — the
        # binding gate is a measured histogram, not a guess (E1 smoke
        # 20260923: 0 replications with two different candidate causes)
        self._repl_reject = {"min_lambda": 0, "hot": 0, "no_chain": 0,
                             "unknown_block": 0, "no_missing": 0,
                             "inflight": 0, "no_holder": 0,
                             "cooldown": 0, "g_nonpos": 0,
                             # closest-to-zero rejected G: if deeply
                             # negative the gate is decisively closed,
                             # if within ~0.01s of firing the negative
                             # verdict is a hair's breadth (E1 writeup)
                             "g_best": None}
        # tokens re-prefilled because a planned fetch degraded to the
        # local boundary (holder lost blocks / empty delivery / stalled
        # xfer) — absolute churn account, grows with prefix length
        self.degrade_rederiv_tokens = 0
        # E2 (10-e2-backing-tier §3) spill-tier accounting. Recall
        # counters are scheduler-side (assign-time); the worker's own
        # counters (evicted/dropped/recall bytes) arrive via status and
        # are aggregated into summary's "spill" block.
        self.spill_rate = getattr(args, "spill_rate", 20e9)
        self.spill_recall_turns = 0
        self.spill_recall_blocks = 0
        self.spill_recall_bytes = 0
        self.spill_fetch_avoided = 0

    def dir_add(self, name, nbytes, t=None):
        e = self.dir.get(name)
        if e is None:
            self.dir[name] = e = {"bytes": int(nbytes), "first": t or time.time(),
                                  "last": 0.0, "count": 0, "lambda": 0.0,
                                  "loc": {}}
        else:
            e["bytes"] = int(nbytes)
        return e

    def note_access(self, names, wid=None, t=None):
        t = t or time.time()
        for n in names:
            e = self.dir.get(n)
            if e is None:
                continue
            if e["count"] >= 1 and e["last"] > 0:
                inst = 1.0 / max(t - e["last"], 1e-3)
                e["lambda"] = (inst if e["count"] == 1
                               else 0.7 * e["lambda"] + 0.3 * inst)
                if wid:
                    loc = e["loc"].setdefault(wid, 0.0)
                    e["loc"][wid] = (inst if loc == 0.0
                                     else 0.7 * loc + 0.3 * inst)
            e["last"] = t
            e["count"] += 1

    # ---- workload -------------------------------------------------------

    def build_workload(self, tokenizer):
        with open(TRACE, encoding="utf-8") as f:
            docs = [json.loads(l) for l in f]
        # --doc-share K: sessions reuse the first K samples (i % K), so
        # identical chains are forced onto different workers and the
        # scheduler MUST fetch instead of recompute. NOTE: rep also
        # depends on i % 2, so with K=2 the pairs (even sessions) and
        # (odd sessions) are exactly identical.
        n_samples = self.args.doc_share or len(docs)
        poisson = self.args.arrival == "poisson"
        rng = random.Random(self.args.seed)
        # zipf catalog: doc popularity across --zipf-n catalog slots.
        # Weights 1/k^s; the hottest slot is shared by a large fraction
        # of all sessions, the tail is seen once or twice — this is what
        # keeps "hot docs living on few workers" a STEADY state instead
        # of the closed-loop one-shot warm-up.
        if poisson and self.args.zipf_n > 0:
            s = self.args.zipf_s
            weights = [1.0 / (k + 1) ** s for k in range(self.args.zipf_n)]
        else:
            weights = None
        now = time.time()
        t_arr = 0.0
        for i in range(self.args.sessions):
            if weights:
                idx = rng.choices(range(len(weights)), weights=weights)[0]
            else:
                idx = i % n_samples
            if poisson:
                t_arr += rng.expovariate(self.args.arrival_rate)
            session = f"doc{i}"
            # chain identity follows the DOC index (not session parity):
            # every session drawing the same catalog slot must produce
            # the IDENTICAL chain, otherwise content sharing is silently
            # destroyed by the rep alternation
            rep = (self.args.doc_repeat if idx % 2 == 0
                   else (self.args.doc_repeat_alt
                         if self.args.doc_repeat_alt is not None
                         else self.args.doc_repeat))
            sample = docs[idx % len(docs)]
            doc_text = (sample["document"] * rep)[: self.args.doc_chars]
            doc_ids = tokenizer(doc_text,
                                return_tensors="pt").input_ids[0].tolist()
            turns = [Turn(session, -1, list(doc_ids), list(doc_ids))]
            prefix = list(doc_ids)
            # E1 (07 §4): question turns CYCLE the sample's question list
            # so --turns-per-session can exceed the ~7 questions in the
            # trace; --q-tokens N pins every question turn's token delta
            # to exactly N (tile short questions, truncate long ones) so
            # the prefix grows linearly and the horizon is controlled.
            qt = getattr(self.args, "q_tokens", None)
            qs = sample["questions"]
            for t in range(self.args.turns_per_session):
                q_ids = tokenizer("\n\n" + qs[t % len(qs)],
                                  return_tensors="pt").input_ids[0].tolist()
                if qt:
                    if len(q_ids) >= qt:
                        q_ids = q_ids[:qt]
                    else:
                        q_ids = (q_ids * (qt // max(1, len(q_ids)) + 1))[:qt]
                prefix = prefix + q_ids
                turns.append(Turn(session, t, list(q_ids), list(prefix)))
            self.turns_of[session] = turns
            self.pending_next[session] = 0
            turns[0].t_arrive = now + t_arr
            if poisson:
                heapq.heappush(self.arrivals,
                               (turns[0].t_arrive, self._arr_seq, turns[0]))
                self._arr_seq += 1
            else:
                turns[0].t_ready = now
                self.ready.append(turns[0])

    def _release_arrivals(self, now):
        """Move due arrivals into the ready queue. t_ready is pinned to
        the ARRIVAL time (not release time) so queue_s stays a true
        arrival→assign queueing measure."""
        while self.arrivals and self.arrivals[0][0] <= now:
            _, _, turn = heapq.heappop(self.arrivals)
            turn.t_ready = turn.t_arrive
            self.ready.append(turn)

    # ---- Match + Schedule (05 v3 §2) -------------------------------------

    def resume_names(self, turn, t_pos):
        """Block names needed to resume at exactly t_pos: all complete
        blocks below floor(t_pos) plus the tip block at t_pos (which
        carries the GDN checkpoint captured at that position)."""
        if t_pos <= 0:
            return []
        bt = self.args.block_tokens
        floor_t = t_pos - (t_pos % bt)
        names = [str(b) for b in chain_through(
            turn.chain("bf16", bt), floor_t)]
        tip = turn.tip_name_at(t_pos, "bf16", bt)
        if tip not in names:
            names.append(tip)
        return names

    def match_local(self, turn, w, tip_bound):
        """Longest resume position T on worker w: a tip position whose
        full resume-name set (complete blocks + tip block) is available
        in the fast tier or the spill tier (E2: resident ∪ spilled —
        eviction no longer destroys the resume path)."""
        bt = self.args.block_tokens
        best = 0
        for tip_name in w.tips:
            t_pos = self._tip_end(tip_name)
            if t_pos == 0 or t_pos > tip_bound:
                continue
            if turn.tip_name_at(t_pos, "bf16", bt) != tip_name:
                continue                      # not a tip of THIS turn's chain
            if not all(n in w.resident or n in w.spilled
                       for n in self.resume_names(turn, t_pos)):
                continue
            best = max(best, t_pos)
        return best

    DECODE_STEP0 = 0.08   # s per decode token; refined once decode timing
                          # is EWMA-tracked (step 3 pipelining)

    def eta(self, w):
        """Expected seconds until worker w finishes its in-flight turn.
        0 for idle workers — the queue-wait term of the scheduling cost,
        inert today (one turn per worker) and activated by pipelining."""
        if not w.busy or not w.cur_plan:
            return 0.0
        plan = w.cur_plan
        elapsed = time.time() - w.t_assign
        total = (plan["prefill_tokens"]
                 / (w.prefill_rate or self.prefill_rate)
                 + plan["decode_steps"] * self.DECODE_STEP0)
        return max(0.0, total - elapsed)

    def _match_cap(self, turn):
        """Resume/fetch target ceiling. Question turns must not target
        past the PREVIOUS turn's end: this turn's new tokens are always
        prefilled locally — a target at cum_tokens would leave zero
        prefill while decode_steps > 0 (worker rejects that). Doc turns
        are publish-only (decode_steps=0), so they may target cum."""
        if turn.turn == -1:
            return turn.cum_tokens
        idx = turn.turn + 1
        return self.turns_of[turn.session][idx - 1].cum_tokens

    def choose(self, turn):
        """Returns (ident, E, fetch, decision) or None.

        E: resume position (tokens, may be non-block-aligned — it is a
        tip position). fetch: None or (holder_ident, [block names]) — the
        missing blocks for a single-holder extension. decision: cost log."""
        # Match bound = the session's previous turn end (see _match_cap),
        # NOT this turn's own tip: cross-session sharing means blocks
        # beyond this session's own history can still exist (published by
        # an earlier identical session). Blocks are named by content, so
        # any published block anywhere is a candidate. Question turns
        # never target their own tail — those tokens are prefilled
        # locally (zero-prefill + decode is illegal on the worker).
        tip_bound = self._match_cap(turn)
        pol = self.args.policy
        if pol == "p2":            # legacy alias
            pol = "ours"
        idle = [w for w in self.workers.values() if not w.busy]
        # B2 planned affinity (CacheRoute-style): if a worker holding this
        # chain locally is busy but nearly done, and no idle worker is
        # warm, HOLD the turn — it stays in self.ready and is retried on
        # the next result. Deadlock-free: once eta exceeds the threshold
        # the next choose falls through to a cold start.
        if pol == "b2" and not any(
                self.match_local(turn, w, tip_bound) > 0 for w in idle):
            warm_busy = [w for w in self.workers.values()
                         if w.busy
                         and self.eta(w) < self.args.wait_threshold
                         and self.match_local(turn, w, tip_bound) > 0]
            if warm_busy:
                return None
        if not idle:
            return None
        tips_global = set().union(*(w.tips for w in self.workers.values()))
        best = None
        costs = {}
        for w in idle:
            # B0 (load-only): no content awareness at all — never resume,
            # never fetch; every turn prefills its full prefix on the
            # least-loaded idle worker.
            e_loc = 0 if pol == "b0" else \
                self.match_local(turn, w, tip_bound)
            print(f"[match] {turn.session}:{turn.turn} w={w.ident.decode()} "
                  f"tips={len(w.tips)} resident={len(w.resident)} "
                  f"e_loc={e_loc}", flush=True)
            # candidate extension targets: tip positions above e_loc whose
            # missing blocks are all on ONE other worker. Fetch is a B3+
            # capability.
            fetch = None
            e_max = e_loc
            if pol in ("b3", "ours"):
                cand = sorted(self._tip_end(tn) for tn in tips_global
                              if e_loc < self._tip_end(tn) <= tip_bound)
                for t_pos in reversed(cand):
                    need = [n for n in self.resume_names(turn, t_pos)
                            if n not in w.resident
                            and n not in w.spilled]
                    if not need:
                        # the whole resume set (tip block included) is
                        # already available locally — a FREE extension,
                        # not a fetch. An empty `need` would also make
                        # the all(...) holder check vacuous and ship a
                        # zero-block fetch that crashes the deliver
                        # path (share=8/ours/budget=48 crash, 20260922)
                        e_max = t_pos
                        break
                    holders = [x for x in self.workers.values()
                               if x is not w and all(
                                   n in x.resident or n in x.spilled
                                   for n in need)]
                    if holders:
                        fetch = (holders[0].ident, need)
                        e_max = t_pos
                        break
            fresh = turn.cum_tokens - e_max
            fetch_bytes = sum(self.dir.get(bn, {}).get("bytes", 0)
                              for bn in (fetch[1] if fetch else []))
            # E2 recall cost: the spilled part of the resume set is
            # pulled host->device at spill_rate (PCIe, ~20GB/s), far
            # below the cross-worker fetch price — local recall is
            # therefore attributed ahead of a demand fetch.
            recall_bytes = 0
            if e_max > 0:
                recall_bytes = sum(
                    self.dir.get(bn, {}).get("bytes", 0)
                    for bn in self.resume_names(turn, e_max)
                    if bn in w.spilled)
            # queue wait: 0 for idle workers by construction (one in-flight
            # turn per worker); the term lands with pipelining in step 3
            rate = w.prefill_rate or self.prefill_rate
            cost = fresh / rate
            if fetch:
                cost += fetch_bytes / self.xfer_rate
            cost += recall_bytes / self.spill_rate
            eta = self.eta(w)   # expected finish of the in-flight turn
            cost += eta
            if e_max == e_loc and e_loc > 0:
                mode = "local"
            elif fetch:
                mode = "fetch"
            elif e_max > 0:
                mode = "local"      # free extension over resident blocks
            else:
                mode = "fresh"
            costs[w.ident.decode()] = {
                "mode": mode, "E": e_max, "fresh": fresh,
                "eta_s": round(eta, 4), "rate": round(rate, 1),
                "recall_bytes": recall_bytes,
                "cost_s": round(cost, 4)}
            # argmin cost; ties: prefer local resume, then larger E
            rank = (round(cost, 6), 0 if mode == "local" else 1, -e_max)
            if best is None or rank < best[0]:
                best = (rank, w.ident, e_max, fetch,
                        {"mode": mode, "E": e_max, "E_loc": e_loc,
                         "cost_s": round(cost, 4), "policy": pol,
                         "fresh": fresh, "fetch_bytes": fetch_bytes,
                         "recall_bytes": recall_bytes,
                         "options": costs})
        if best is None:
            return None
        _, ident, e, fetch, decision = best
        return ident, e, fetch, decision

    # ---- Placement controller (slow path, 05 v3 §3) ------------------------

    def controller(self, sock):
        """Event-driven slow path, runs after every result; 'ours' only
        (b3 stays pure reactive — that contrast IS the step-4 claim).
        Plans are pure (_plan_*); _apply_* mutates scheduler state and,
        when sock is given, sends the corresponding messages."""
        if self.args.policy not in ("ours", "p2"):
            return
        self._apply_evict(sock, self._plan_evict())
        # replication plans see post-eviction residency
        self._apply_repl(sock, self._plan_repl())

    @staticmethod
    def _chain_names(tip_name, pool):
        """Ancestors of a tip block derived from NAMES ALONE by walking
        parent_hash links (blkchain §1). `pool` maps block_hash -> name
        over every known block (directory + all workers' residency).
        Returns None when the chain is broken below the tip — a partial
        segment cannot be replicated or resumed, so callers skip it."""
        names = []
        cur = tip_name
        for _ in range(1_000_000):
            try:
                bn = BlockName.parse(cur)
            except ValueError:
                return None
            if bn.parent_hash == GENESIS or bn.span_start == 0:
                break
            nxt = pool.get(bn.parent_hash)
            if nxt is None:
                return None
            names.append(nxt)         # tip -> genesis order while walking
            cur = nxt
        else:
            return None               # parent walk never terminated
        names.reverse()               # genesis-first, matching resume_names
        return names

    def _block_pool(self):
        pool = {}
        for n in list(self.dir) + [n for w in self.workers.values()
                                   for n in w.resident]:
            try:
                pool.setdefault(BlockName.parse(n).block_hash, n)
            except ValueError:
                continue
        return pool

    def _inflight(self):
        """Names currently riding the wire (demand + replication)."""
        out = []
        for x in list(self._xfer.values()) + list(self._repl.values()):
            out.append((x.get("holder"), x.get("target"), x["names"]))
        return out

    def _plan_evict(self):
        """Symmetric criterion of G_rep (05 v3 §3): drop the coldest
        resident segments until a worker is back under its memory
        budget. Ancestors shared with any hotter resident tip are
        protected (segments are the placement unit, not blocks)."""
        budget = self.args.worker_mem_budget_mb * 1e6
        if budget <= 0:
            return []
        # Eviction is a SHARED substrate, not our novelty: under a
        # memory budget every policy must respect, all of them drop
        # coldest segments (b0-b3 degrade; ours additionally
        # replicates, gated by G_rep). The policy gate lives in
        # _plan_repl only.
        inflight = self._inflight()
        actions = []
        for w in self.workers.values():
            if w.busy or w.resident_bytes <= budget:
                continue
            if any(h == w.ident or t == w.ident
                   for h, t, _ in inflight):
                continue
            excess = w.resident_bytes - budget
            pool = {}
            for n in w.resident | w.spilled:
                # E2: spilled names join the walk — a tip living in the
                # tier keeps its chain derivable (pre-E2 a resident tip
                # implied a resident chain; eviction to the tier breaks
                # that invariant). Non-resident hits are filtered from
                # the action below, so accounting stays resident-only.
                try:
                    pool.setdefault(BlockName.parse(n).block_hash, n)
                except ValueError:
                    continue
            # Protection is counted per BLOCK over every resident tip's
            # FULL resume set (ancestors + tip): a tip block that is also
            # an interior chain block of a hotter tip (block-aligned
            # tips) is infrastructure and must survive.
            fulls, counts = {}, {}
            broken = False
            for tip in w.tips:
                c = self._chain_names(tip, pool)
                if c is None:
                    broken = True
                    break
                fulls[tip] = c + [tip]
                for n in fulls[tip]:
                    counts[n] = counts.get(n, 0) + 1
            if broken:
                continue          # residency state we cannot reason about
            names, freed = [], 0
            wid = w.ident.decode()

            def _local_cold(tip):
                e = self.dir.get(tip, {})
                return e.get("loc", {}).get(wid, e.get("lambda", 0.0))

            cold_first = sorted(fulls, key=_local_cold)
            for tip in cold_first:
                if freed >= excess:
                    break
                for n in fulls[tip]:
                    counts[n] -= 1
                    if counts[n] == 0:
                        names.append(n)
                freed += sum(self.dir.get(n, {}).get("bytes", 0)
                             for n in names)
            names = list(dict.fromkeys(names))
            # E2: only RESIDENT blocks can be evicted. A tip living in
            # the spill tier keeps its chain protected above, but its
            # own (non-resident) name must not ride the evict message —
            # the worker would no-op it while evicted_bytes double-counts.
            names = [n for n in names if n in w.resident]
            if names:
                actions.append({"worker": w.ident, "names": names})
        return actions

    def _apply_evict(self, sock, actions):
        for a in actions:
            w = self.workers[a["worker"]]
            if sock is not None:
                msg.send(sock, {"type": "evict", "names": a["names"]},
                         ident=w.ident)
            nbytes = sum(self.dir.get(n, {}).get("bytes", 0)
                         for n in a["names"])
            for n in a["names"]:
                w.resident.discard(n)
                w.tips.discard(n)
                w.via_repl.discard(n)
            # Optimistic accounting: the worker's status ack lags one
            # cycle, so without this the controller re-plans eviction
            # against a stale (still-full) byte count and the next
            # cycle evicts the next-coldest segment needlessly.
            w.resident_bytes = max(0, w.resident_bytes - nbytes)
            self.evictions += 1
            self.evicted_blocks += len(a["names"])
            self.evicted_bytes += nbytes
            print(f"[ctl  ] evict {len(a['names'])} blocks from "
                  f"{w.ident.decode()}", flush=True)

    def _plan_repl(self):
        """G_rep(p, j) = λ̂_p · ΔC_future − C_copy − C_memory > 0 (05 v3 §1).
        ΔC_future (per-hit critical-path saving) and C_copy are both the
        transfer time nbytes/xfer_rate in this prototype; C_memory is a
        per-byte price (--repl-mem-price, 0 by default); there is no
        hard residency-budget skip — under pressure _plan_evict frees
        space for the copy. Spatial demand: any worker NOT holding the
        segment is a candidate (requests land on whichever worker a
        turn is dispatched to), and the least loaded holder is the
        copy source.

        Copy granularity (E1 fix, 2026-09-23): the copy set is the
        MISSING SUFFIX of the resume set relative to each target
        (resume_set minus target_resident) — the same granularity the
        demand fetch path uses. The full-segment copy of the 3-turn
        era cannot fire at long horizons: a 40-turn session's whole
        chain (~70-100MB) exceeds any realistic per-worker budget, so
        no complete holder exists and the controller degenerates to
        b3 without ever evaluating G_rep. When the target holds
        nothing the suffix IS the full chain (3-turn behaviour is
        unchanged; 06 results stand).

        ΔC_future calibration (E1, pre-registered in 07 §4
        "C_recompute(L) 校准不等式"): the step-4 placeholder priced a
        hit's saving as the transfer time itself, making the
        break-even λ̂* = 1 hit/s — unreachable for migratable state
        (per-target demand ~0.1/s), so the feasible set was empty by
        construction (smoke 20260923: 17,710 g_nonpos rejections vs 0
        actions). A hit's actual worth is the re-derivation it
        avoids, measured as missing_tokens / prefill_rate (the
        c_recompute calibration), while C_copy stays nbytes/xfer_rate.
        Both are the scheduler's own measured EWMAs; nothing is
        assumed about demand (λ̂ stays observed-only)."""
        if len(self._repl) >= self.args.max_repl_inflight:
            return []
        if self.args.policy not in ("ours", "p2"):
            return []
        pool = self._block_pool()
        inflight = self._inflight()
        actions = []
        rej = self._repl_reject
        hot_first = sorted(
            self.tips,
            key=lambda n: -self.dir.get(n, {}).get("lambda", 0.0))
        for tip in hot_first:
            e = self.dir.get(tip)
            if not e or e["lambda"] < self.args.repl_min_lambda:
                rej["min_lambda"] += 1
                continue
            rej["hot"] += 1
            chain = self._chain_names(tip, pool)
            if chain is None:
                rej["no_chain"] += 1
                continue
            names = chain + [tip]
            if any(n not in self.dir for n in names):
                rej["unknown_block"] += 1
                continue          # cannot price an unknown block
            for w in self.workers.values():
                if len(self._repl) + len(actions) \
                        >= self.args.max_repl_inflight:
                    return actions
                # copy only what this target lacks; a target holding
                # blocks in the spill tier (E2) has them addressable
                # already — spill residency counts, nothing re-copied
                missing = [n for n in names if n not in w.resident
                           and n not in w.spilled]
                if not missing:
                    rej["no_missing"] += 1
                    continue
                if w.ident in {h for h, _, ns in inflight
                               if set(missing) <= set(ns)}:
                    rej["inflight"] += 1
                    continue      # demand fetch already delivering it
                holders = [x for x in self.workers.values()
                           if all(n in x.resident or n in x.spilled
                                  for n in missing)]
                if not holders:
                    rej["no_holder"] += 1
                    continue
                holders.sort(key=lambda x: x.resident_bytes)
                nbytes = sum(self.dir[n]["bytes"] for n in missing)
                try:
                    missing_tokens = sum(BlockName.parse(n).span_tokens
                                         for n in missing)
                except ValueError:
                    rej["unknown_block"] += 1
                    continue
                # ΔC_future: avoided RE-DERIVATION of the missing
                # tokens (measured c_recompute calibration, 07 §4);
                # C_copy: the off-path transfer time
                delta_c = missing_tokens / (self.prefill_rate
                                            or self.PREFILL_RATE0)
                c_copy = nbytes / self.xfer_rate
                c_mem = nbytes * self.args.repl_mem_price
                # No hard budget guard here: under pressure _plan_evict
                # frees space every cycle, so replication is how a hot
                # segment survives eviction on the holder's worker.
                if (tip, w.ident) in self._repl_cool:
                    rej["cooldown"] += 1
                    continue
                # G_rep evaluated with the demand OBSERVED AT THIS
                # TARGET (spatial demand); unseen locators get a
                # uniform-routing prior share of the global rate
                lam = e["loc"].get(w.ident.decode())
                if lam is None:
                    lam = e["lambda"] / max(1, len(self.workers))
                g = lam * delta_c - c_copy - c_mem
                if g <= 0:
                    rej["g_nonpos"] += 1
                    rg = round(g, 4)
                    if rej["g_best"] is None or rg > rej["g_best"]:
                        rej["g_best"] = rg
                    continue
                actions.append({"tip": tip, "holder": holders[0].ident,
                                "target": w.ident, "names": missing,
                                "bytes": nbytes, "g": round(g, 4)})
        return actions

    def _apply_repl(self, sock, actions):
        for a in actions:
            rid = f"repl:{self._repl_seq}"
            self._repl_seq += 1
            self.repl_planned += 1
            self._repl[rid] = {"stage": "fetch", "holder": a["holder"],
                               "target": a["target"], "names": a["names"],
                               "tip": a["tip"], "t": time.time()}
            self._repl_cool[(a["tip"], a["target"])] = \
                time.time() + self.args.repl_cooldown
            if sock is not None:
                msg.send(sock, {"type": "fetch", "names": a["names"],
                                "repl": rid}, ident=a["holder"])
            print(f"[ctl  ] replicate {len(a['names'])} blocks "
                  f"{a['holder'].decode()} -> {a['target'].decode()} "
                  f"tip=...{a['tip'][-40:]} G={a['g']}", flush=True)

    # ---- event loop -------------------------------------------------------

    def run(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.args.model_path,
                                            trust_remote_code=True)
        self.build_workload(tok)

        ctx = zmq.Context()
        sock = msg.router(ctx, self.args.bind)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        hellos = set()
        poller0 = zmq.Poller()
        poller0.register(sock, zmq.POLLIN)
        hello_deadline = time.time() + 180.0
        while len(hellos) < len(self.workers):
            # a worker that never comes up (GPU OOM, orphan holding the
            # port, crashed at load) must fail FAST and loudly — the
            # pre-timeout version blocked here forever and burned the
            # whole cell budget on a startup deadlock
            left = hello_deadline - time.time()
            if left <= 0 or not dict(poller0.poll(timeout=int(min(30.0, left) * 1000))):
                missing = [wid.decode() for wid in self.workers
                           if wid.decode() not in hellos]
                raise RuntimeError(
                    f"workers never said hello within 180s: {missing} — "
                    "check for orphaned icn_proto processes holding GPU "
                    "memory or the zmq port (pkill -f icn_proto)")
            ident, hdr, _ = msg.recv(sock)
            if hdr.get("type") == "hello":
                hellos.add(hdr["worker_id"])
                w = self.workers.get(ident)
                if w is not None:
                    w.resident = set(hdr.get("resident", []))
                    w.tips = set(hdr.get("tips", []))
                    w.spilled = set(hdr.get("spilled", []))
                    w.spill_bytes = hdr.get("spill_bytes", 0)
                    w.spill_stat = hdr.get("spill", {})

        try:
            while not self.finished():
                self._release_arrivals(time.time())
                self.dispatch(sock)
                # sleep until the next interesting event: an arriving
                # turn (open loop) or the 1s housekeeping tick, whichever
                # is sooner. Without this the open-loop scheduler would
                # idle-spin a second past every arrival.
                if self.arrivals:
                    wait_ms = max(0.0, (self.arrivals[0][0] - time.time()))
                    wait_ms = min(1000.0, wait_ms * 1000.0)
                else:
                    wait_ms = 1000.0
                evts = dict(poller.poll(timeout=int(wait_ms)))
                now = time.time()
                for w in self.workers.values():
                    if w.busy and now - w.t_assign > 180:
                        print(f"[watchdog] {w.ident.decode()} busy on "
                              f"{w.current} for {int(now - w.t_assign)}s "
                              f"(no result)", flush=True)
                for rid, x in self._xfer.items():
                    if now - x["t_fetch"] > 120:
                        print(f"[watchdog] xfer {rid} stuck in stage "
                              f"{x['stage']} for {int(now - x['t_fetch'])}s",
                              flush=True)
                for rid, x in self._repl.items():
                    if now - x["t"] > 120:
                        print(f"[watchdog] repl {rid} stuck in stage "
                              f"{x['stage']} for {int(now - x['t'])}s",
                              flush=True)
                self._watchdog_fail(sock, now)
                if sock not in evts:
                    continue
                ident, hdr, payload = msg.recv(sock)
                self.on_message(sock, ident, hdr, payload)
                self.controller(sock)
        finally:
            for w in self.workers.values():
                msg.send(sock, {"type": "shutdown"}, ident=w.ident)
            time.sleep(0.5)
            sock.close()
            ctx.term()
        return self.summary()

    def _watchdog_fail(self, sock, now):
        """Last-resort liveness (the 120/180s loops above only print).
        After STALL_S seconds a wedged worker or a lost ack is FAILED:
        the turn records ok=False, the session advances, and the cell
        finishes with failed>0 — the matrix marks it BAD and moves on
        instead of hanging until the cell timeout. Late results for an
        already-failed turn are discarded by the wd_failed guard."""
        for w in self.workers.values():
            if not w.busy or now - w.t_assign <= STALL_S:
                continue
            rid = w.current
            if rid is None:
                # busy in a demand fetch/deliver (no turn assigned yet):
                # degrade to the pre-fetch local boundary, exactly like
                # a fetch-failed ack does
                for xrid, x in list(self._xfer.items()):
                    if x["target"] == w.ident:
                        print(f"[watchdog] xfer {xrid} stalled, degrade "
                              f"to E_loc={x['E_loc']}", flush=True)
                        self.degrade_rederiv_tokens += max(
                            0, len(x["turn"].prefix_ids) - x["E_loc"])
                        self._xfer.pop(xrid, None)
                        self.send_assign(sock, x["target"], x["turn"],
                                         x["E_loc"])
                continue
            print(f"[watchdog] turn {rid} stuck on {w.ident.decode()} "
                  f"for {int(now - w.t_assign)}s — failing", flush=True)
            w.busy = False
            w.current = None
            w.cur_plan = None
            rec = next((r for r in reversed(self.records)
                        if r["request_id"] == rid), None)
            if rec and "ok" not in rec:
                rec.update({"ok": False,
                            "error": "watchdog: worker stall",
                            "prefill_s": None, "published": []})
                rec["latency_s"] = round(now - rec.pop("t_assigned"), 4)
                rec["wd_failed"] = True
                self.advance(rid)
        # replications are best-effort: a stuck copy just aborts
        for rid, x in list(self._repl.items()):
            if now - x["t"] > STALL_S:
                print(f"[watchdog] repl {rid} abandoned after "
                      f"{int(now - x['t'])}s", flush=True)
                self._repl.pop(rid, None)

    def dispatch(self, sock):
        for turn in list(self.ready):
            chosen = self.choose(turn)
            if chosen is None:
                break
            ident, e_resume, fetch, decision = chosen
            rid = f"{turn.session}:{turn.turn}"
            w = self.workers[ident]
            if fetch is None and e_resume > 0:
                # E2: local recall that replaces a cross-worker fetch.
                # Without the tier this resume set was not resident on w
                # and another worker could have served it — the demand
                # fetch would have fired. Counted only when a holder
                # actually exists (a fetch was the real alternative).
                rn = self.resume_names(turn, e_resume)
                need_ro = [n for n in rn if n not in w.resident]
                if any(n in w.spilled for n in need_ro):
                    holders = [x for x in self.workers.values()
                               if x is not w and all(
                                   n in x.resident or n in x.spilled
                                   for n in need_ro)]
                    if holders:
                        self.spill_fetch_avoided += 1
            if fetch is not None:
                holder_ident, names = fetch
                self._xfer[rid] = {"stage": "fetch", "target": ident,
                                   "holder": holder_ident,
                                   "names": names, "E_loc": decision["E_loc"],
                                   "turn": turn, "decision": decision,
                                   "t_fetch": time.time()}
                print(f"[xfer ] {rid} fetch {len(names)} blocks from "
                      f"{holder_ident.decode()} -> {ident.decode()} "
                      f"(E target {decision['E']})", flush=True)
                msg.send(sock, {"type": "fetch", "names": names},
                         ident=holder_ident)
                w.busy = True
                w.t_assign = time.time()
                self.ready.remove(turn)
                continue
            self.send_assign(sock, ident, turn, e_resume, decision=decision)
            self.ready.remove(turn)

    def send_assign(self, sock, ident, turn, e_resume, xfer_names=None,
                    xfer_bytes=0, xfer_s=0.0, decision=None):
        rid = f"{turn.session}:{turn.turn}"
        w = self.workers[ident]
        w.busy = True
        w.current = rid
        w.t_assign = time.time()
        decode_steps = (0 if turn.turn == -1 else self.args.decode_steps)
        w.cur_plan = {"prefill_tokens": len(turn.prefix_ids) - e_resume,
                      "decode_steps": decode_steps}
        self._xfer.pop(rid, None)
        resume_names = self.resume_names(turn, e_resume)
        new_names = turn.publish_names("bf16", self.args.block_tokens)
        # demand accounting: every resumed block contributed to this turn;
        # the accessing worker id feeds the per-locator demand signal
        # (05 v3 §3: placement follows SPATIAL demand, not global popularity)
        self.note_access(resume_names, w.ident.decode())
        print(f"[assign] {rid} -> {w.ident.decode()} "
              f"E={e_resume} resume_blocks={len(resume_names)} "
              f"publish_blocks={len(new_names)} "
              f"prefill_tokens={len(turn.prefix_ids) - e_resume}", flush=True)
        msg.send(sock, {
            "type": "assign", "request_id": rid,
            "session": turn.session, "turn": turn.turn,
            "resume_names": resume_names,
            "new_block_names": new_names,
            "prefill_ids": turn.prefix_ids[e_resume:],
            "decode_steps": decode_steps,
            "repr": "bf16",
        }, ident=ident)
        self.records.append({"request_id": rid, "worker": w.ident.decode(),
                             "t_assigned": time.time(),
                             "t_arrive": turn.t_arrive,
                             "queue_s": round(time.time() - turn.t_ready, 4),
                             "E": e_resume,
                             "fp": turn.prefix_fingerprint(
                                 "bf16", self.args.block_tokens),
                             "xfer_blocks": xfer_names or [],
                             "transfer_bytes": xfer_bytes,
                             "xfer_s": round(xfer_s, 4),
                             "decision": decision})
        rec = self.records[-1]
        # ---- E1 residency-opportunity accounting (07 §4) ----
        wid = w.ident.decode()
        prev_worker = self._last_worker.get(turn.session)
        rec["migrated"] = bool(prev_worker and prev_worker != wid)
        self._last_worker[turn.session] = wid
        # ---- E2 spill accounting (10 §3.2): blocks this turn pulls
        # back from the backing tier into residency ----
        recall = [n for n in resume_names if n in w.spilled]
        if recall:
            self.spill_recall_turns += 1
            self.spill_recall_blocks += len(recall)
            self.spill_recall_bytes += sum(
                self.dir.get(n, {}).get("bytes", 0) for n in recall)
            rec["spill_recall_blocks"] = len(recall)
        if turn.turn >= 0 and prev_worker and prev_worker != wid:
            # the session's latest state was published on another worker:
            # a remote-resume opportunity. It is "served local due to
            # replication" when the exact tip is resident HERE only
            # because the controller placed it (via_repl) and the resume
            # reaches it in full (a b3 demand fetch does NOT count — the
            # tip arrives reactively, that is B3's own path)
            self.remote_resume_opportunities += 1
            rec["remote_opp"] = True
            prev_turn = self.turns_of[turn.session][turn.turn]
            prev_cum = prev_turn.cum_tokens
            tip = prev_turn.tip_name_at(prev_cum, "bf16",
                                        self.args.block_tokens)
            if e_resume == prev_cum and tip in w.via_repl:
                self.repl_served_local += 1
                rec["repl_served"] = True

    def on_message(self, sock, ident, hdr, payload):
        w = self.workers.get(ident)
        if w is None:
            return
        mtype = hdr.get("type")
        if mtype == "status":
            # NOTE: busy is NOT taken from status. The worker's status is
            # stale by the time it arrives (it reports the moment between
            # turns, but the scheduler may already have assigned/fetched
            # a new turn to this worker). Scheduler-side busy lifecycle:
            # set True on assign / fetch dispatch, cleared on result.
            w.resident = set(hdr.get("resident", []))
            w.tips = set(hdr.get("tips", []))
            w.resident_bytes = hdr.get("resident_bytes", w.resident_bytes)
            w.spilled = set(hdr.get("spilled", []))
            w.spill_bytes = hdr.get("spill_bytes", w.spill_bytes)
            w.spill_stat = hdr.get("spill", w.spill_stat)
            return
        if mtype == "result":
            rid = hdr["request_id"]
            print(f"[result] {rid} from {w.ident.decode()} "
                  f"ok={hdr.get('ok')} "
                  f"prefill_s={hdr.get('prefill_s')} "
                  f"published={len(hdr.get('published') or [])} "
                  f"err={hdr.get('error')}", flush=True)
            rec = next((r for r in reversed(self.records)
                        if r["request_id"] == rid), None)
            if rec and rec.get("wd_failed"):
                # the watchdog already failed and advanced this turn; a
                # late result from a wedged worker only frees the worker
                w.busy = False
                w.current = None
                w.cur_plan = None
                return
            if rec:
                rec.update({k: hdr.get(k) for k in
                            ("ok", "resumed", "prefill_s", "decode_s",
                             "prefill_tokens", "error", "published",
                             "resident_bytes", "decoded_ids")})
                rec["latency_s"] = round(time.time() - rec.pop("t_assigned"), 4)
            w.busy = False
            w.current = None
            w.cur_plan = None
            w.resident_bytes = hdr.get("resident_bytes", w.resident_bytes)
            # per-worker prefill-rate EWMA (resumed turns carry tiny
            # prefills and would pollute the estimate)
            if hdr.get("ok") and hdr.get("prefill_s") \
                    and hdr.get("prefill_tokens", 0) > 64:
                rate = hdr["prefill_tokens"] / hdr["prefill_s"]
                w.prefill_rate = (rate if w.prefill_rate == 0
                                  else 0.7 * w.prefill_rate + 0.3 * rate)
                self.prefill_rate = (0.7 * self.prefill_rate
                                     + 0.3 * rate)
            if hdr.get("ok"):
                session, t = rid.split(":")
                chain_idx = 0 if t == "-1" else int(t) + 1
                if hdr.get("published"):
                    for p in hdr["published"]:
                        w.resident.add(p["name"])
                        self.dir_add(p["name"], p["bytes"])
                    # this turn's chain tip (last of publish_names) carries
                    # the GDN checkpoint (worker.extract_blocks contract)
                    turn_done = self.turns_of[session][chain_idx]
                    tip = turn_done.publish_names("bf16",
                                                  self.args.block_tokens)[-1]
                    w.tips.add(tip)
                    self.tips.add(tip)
            self.advance(rid)
            return
        if mtype == "fetched":
            # pair by holder + names: two sessions may fetch the SAME
            # blocks to different targets concurrently. Demand fetches
            # (a waiting turn) take priority over controller replications.
            rid, xfer, kind = self._pair_fetch(ident, hdr.get("names"))
            if not xfer:
                print(f"[xfer ] WARN fetched with no matching xfer from "
                      f"{ident.decode()}: ok={hdr.get('ok')} "
                      f"names={len(hdr.get('names') or [])}", flush=True)
                return
            if hdr.get("ok"):
                xfer["stage"] = "deliver"
                xfer["bytes"] = len(payload)
                if kind == "repl":
                    xfer_s = time.time() - xfer["t"]
                    print(f"[ctl  ] {rid} fetched {xfer['bytes']/1e6:.1f}MB "
                          f"from {ident.decode()} in {xfer_s:.2f}s",
                          flush=True)
                    msg.send(sock, {"type": "deliver", "repl": rid},
                             payload=payload, ident=xfer["target"])
                else:
                    print(f"[xfer ] {rid} fetched {xfer['bytes']/1e6:.1f}MB "
                          f"from {ident.decode()} in "
                          f"{time.time() - xfer['t_fetch']:.2f}s", flush=True)
                    msg.send(sock, {"type": "deliver"}, payload=payload,
                             ident=xfer["target"])
            else:
                # holder lost blocks: demand degrades to the local
                # boundary; a failed replication just aborts (zero-replica
                # is legal — identity survives via re-derivation)
                if kind == "repl":
                    print(f"[ctl  ] {rid} replication FAILED (missing "
                          f"{len(hdr.get('missing') or [])}), abort",
                          flush=True)
                    self._repl.pop(rid, None)
                    return
                print(f"[xfer ] {rid} fetch FAILED (missing "
                      f"{len(hdr.get('missing') or [])}), degrade to "
                      f"E_loc={xfer['E_loc']}", flush=True)
                turn = xfer["turn"]
                self.degrade_rederiv_tokens += max(
                    0, len(turn.prefix_ids) - xfer["E_loc"])
                self._xfer.pop(rid, None)
                self.send_assign(sock, xfer["target"], turn, xfer["E_loc"])
            return
        if mtype == "delivered":
            names = hdr.get("names") or []
            if hdr.get("repl"):
                # controller replication ack: paired by the echoed rid —
                # a demand fetch may be delivering the SAME names to the
                # same worker concurrently and must not swallow this ack
                rid = hdr["repl"]
                xfer = self._repl.get(rid)
                if not xfer or xfer["stage"] != "deliver" \
                        or xfer["target"] != ident:
                    return
                self._repl.pop(rid, None)
                self.replications += 1
                # fetched normally recorded wire bytes; the worker's
                # delivered ack carries NO payload, so never call
                # len(payload) here (the default would be evaluated
                # eagerly and crash on None)
                if xfer.get("bytes") is None:
                    xfer["bytes"] = len(payload) if payload is not None else 0
                self.replicated_bytes += xfer["bytes"]
                xfer_s = time.time() - xfer["t"]
                if xfer_s > 0:
                    self.xfer_rate = (0.7 * self.xfer_rate
                                      + 0.3 * xfer["bytes"] / xfer_s)
                w = self.workers[xfer["target"]]
                w.resident |= set(xfer["names"])
                w.via_repl |= set(xfer["names"])
                w.tips.add(xfer["tip"])
                print(f"[ctl  ] {rid} delivered to {ident.decode()} "
                      f"({xfer['bytes']/1e6:.1f}MB in {xfer_s:.2f}s), "
                      f"tip now resident", flush=True)
                return
            # demand fetch ack: pair by (worker, names)
            rid, xfer = next(
                ((r, x) for r, x in self._xfer.items()
                 if x["stage"] == "deliver"
                 and x["target"] == ident and x["names"] == names),
                (None, None))
            if xfer:
                self._xfer.pop(rid, None)
                self.transfer_bytes += xfer["bytes"]
                self.transfers += 1
                if not xfer["names"]:
                    # zero-block fetch (only possible via a stale plan):
                    # nothing was delivered — degrade to the local
                    # boundary instead of indexing into an empty list
                    print(f"[xfer ] {rid} empty delivery, degrade to "
                          f"E_loc={xfer['E_loc']}", flush=True)
                    self.degrade_rederiv_tokens += max(
                        0, len(xfer["turn"].prefix_ids) - xfer["E_loc"])
                    self.send_assign(sock, xfer["target"], xfer["turn"],
                                     xfer["E_loc"])
                    return
                # the worker stored these blocks the moment deliver
                # landed — reflect it NOW instead of waiting for the
                # trailing status; a stale view makes the controller
                # plan redundant copies of just-delivered segments
                w = self.workers[xfer["target"]]
                w.resident |= set(xfer["names"])
                tip = xfer["names"][-1]
                if tip in self.tips:
                    w.tips.add(tip)
                xfer_s = time.time() - xfer["t_fetch"]
                if xfer_s > 0:
                    self.xfer_rate = (0.7 * self.xfer_rate
                                      + 0.3 * xfer["bytes"] / xfer_s)
                print(f"[xfer ] {rid} delivered to {ident.decode()}, "
                      f"assign E={self._tip_end(xfer['names'][-1])}",
                      flush=True)
                e = self._tip_end(xfer["names"][-1])
                self.send_assign(sock, xfer["target"], xfer["turn"], e,
                                 xfer_names=xfer["names"],
                                 xfer_bytes=xfer.get("bytes", 0),
                                 xfer_s=xfer_s,
                                 decision=xfer.get("decision"))
            return

    def _pair_fetch(self, holder, names):
        """A holder's fetched reply can serve a demand fetch or a
        controller replication; demand wins (a turn is waiting)."""
        for table, kind in ((self._xfer, "demand"), (self._repl, "repl")):
            hit = next(
                ((r, x) for r, x in table.items()
                 if x["stage"] == "fetch" and x["holder"] == holder
                 and x["names"] == names), (None, None))
            if hit[0] is not None:
                return hit[0], hit[1], kind
        return None, None, None

    @staticmethod
    def _tip_end(name):
        # block name .../span/<start>-<end>/... — parse span_end
        parts = name.split("/")
        return int(parts[parts.index("span") + 1].split("-")[1])

    def advance(self, request_id):
        session, t = request_id.split(":")
        idx = 0 if t == "-1" else int(t) + 1
        nxt = idx + 1
        self.pending_next[session] = nxt
        turns = self.turns_of[session]
        if nxt < len(turns):
            nxt_turn = turns[nxt]
            if self.args.arrival == "poisson":
                # a real user thinks before asking the next question:
                # the turn enters the arrivals stream, not the ready queue
                delay = self._think_rng.expovariate(1.0 / self.args.think_s)
                nxt_turn.t_arrive = time.time() + delay
                heapq.heappush(self.arrivals,
                               (nxt_turn.t_arrive, self._arr_seq, nxt_turn))
                self._arr_seq += 1
            else:
                nxt_turn.t_ready = time.time()
                self.ready.append(nxt_turn)
        else:
            self.done_sessions += 1

    def finished(self):
        return self.done_sessions >= len(self.turns_of)

    # ---- reporting --------------------------------------------------------

    @staticmethod
    def _pctl(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 4)

    def summary(self):
        ok = [r for r in self.records if r.get("ok")]
        failed = [r for r in self.records if not r.get("ok")]
        wall = time.time() - self.t_start
        # open loop: count throughput over the busy span (first arrival →
        # last result), not from scheduler init — otherwise a long idle
        # head before the first arrival dilutes QPS
        if self.args.arrival == "poisson" and ok:
            done = [(r.get("t_arrive") or 0) + r.get("queue_s", 0)
                    + r["latency_s"] for r in ok]
            span = max(done) - min(r.get("t_arrive") or 0 for r in ok)
        else:
            span = wall
        total_new = sum(r.get("prefill_tokens", 0) for r in ok)
        resumed = [r for r in ok if r.get("resumed")]
        # ---- E1 (07 §4) summary metrics ----
        q_recs = []
        for r in ok:
            session, t = r["request_id"].split(":")
            if t == "-1":
                continue
            q_recs.append((session, int(t), r))
        # rederivation: question-turn prefill beyond the turn's genuinely
        # new tokens — the absolute churn account of "evicted/lost prefix
        # re-computed" (per-token cost grows with prefix length)
        rederiv = 0
        for session, t, r in q_recs:
            turns = self.turns_of.get(session) or []
            if t + 1 >= len(turns):
                continue
            growth = turns[t + 1].cum_tokens - turns[t].cum_tokens
            rederiv += max(0, r.get("prefill_tokens", 0) - growth)
        migrated_q = [1 for _, _, r in q_recs if r.get("migrated")]
        # C_recompute(L): measured prefill rate by prefix-length bucket —
        # calibrates the re-derivation cost curve instead of assuming a
        # growth order (hybrid arch + real kernels, 07 §4)
        edges = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        buckets = {}
        for r in ok:
            pt, ps = r.get("prefill_tokens") or 0, r.get("prefill_s") or 0
            if pt <= 0 or ps <= 0:
                continue
            b = next((e for e in edges if pt <= e), edges[-1] * 2)
            agg = buckets.setdefault(b, [0, 0, 0])
            agg[0] += 1
            agg[1] += pt
            agg[2] += ps
        c_recompute = [
            {"bucket_max": b, "n": agg[0],
             "tok_per_s": round(agg[1] / agg[2], 1)}
            for b, agg in sorted(buckets.items())]
        # state lifetime: publish (dir first) -> last access, over blocks
        # that were ever accessed
        lifetimes = [m["last"] - m["first"] for m in self.dir.values()
                     if m["count"] > 0 and m["last"] > 0 and m["first"] > 0]
        state_lifetime = {
            "n": len(lifetimes),
            "p10": self._pctl(lifetimes, 0.1),
            "p50": self._pctl(lifetimes, 0.5),
            "p90": self._pctl(lifetimes, 0.9),
            "max": round(max(lifetimes), 1) if lifetimes else None,
        }
        published_blocks = sum(len(r.get("published") or []) for r in ok)
        published_bytes = sum(p["bytes"] for r in ok
                              for p in (r.get("published") or []))
        top = sorted(self.dir.items(), key=lambda kv: kv[1]["lambda"],
                     reverse=True)[:10]
        workers = {w.ident.decode(): {
            "resident_blocks": len(w.resident),
            "tips": len(w.tips),
            "resident_bytes": w.resident_bytes,
            "prefill_rate": round(w.prefill_rate, 1),
        } for w in self.workers.values()}
        # E2 spill-tier verdict block: scheduler-side recall accounting
        # (assign-time) + worker-side tier counters (from status)
        spill = {
            "recall_turns": self.spill_recall_turns,
            "recall_blocks": self.spill_recall_blocks,
            "recall_bytes": self.spill_recall_bytes,
            "fetch_avoided_by_recall": self.spill_fetch_avoided,
            "workers": {w.ident.decode(): dict(w.spill_stat,
                                               spill_bytes=w.spill_bytes)
                        for w in self.workers.values()},
        }
        out = {
            "config": {k: v for k, v in vars(self.args).items()},
            "prefill_rate": round(self.prefill_rate, 1),
            "xfer_rate": round(self.xfer_rate, 1),
            "wall_s": round(wall, 2),
            "requests": len(self.records),
            "failed": len(failed),
            "throughput_rps": round(len(ok) / max(span, 1e-6), 4),
            "resumed": len(resumed),
            "hit_rate": round(len(resumed) / max(1, len(ok)), 4),
            "new_tokens_processed": total_new,
            "published_blocks": published_blocks,
            "published_bytes": published_bytes,
            "transfers": self.transfers,
            "transfer_bytes": self.transfer_bytes,
            "replications": self.replications,
            "replicated_bytes": self.replicated_bytes,
            "repl_planned": self.repl_planned,
            "evictions": self.evictions,
            "evicted_blocks": self.evicted_blocks,
            "evicted_bytes": self.evicted_bytes,
            "session_turn_migration_rate": round(
                len(migrated_q) / max(1, len(q_recs)), 4),
            "remote_resume_opportunities": self.remote_resume_opportunities,
            "remote_resume_served_local_due_to_replication":
                self.repl_served_local,
            "rederivation_tokens": rederiv,
            "degrade_rederiv_tokens": self.degrade_rederiv_tokens,
            "repl_reject": dict(self._repl_reject),
            "c_recompute": c_recompute,
            "state_lifetime": state_lifetime,
            "spill": spill,
            "avg_latency_s": round(sum(r["latency_s"] for r in ok)
                                   / max(1, len(ok)), 4),
            "p50_resp_s": self._pctl([r.get("queue_s", 0) + r["latency_s"]
                                      for r in ok], 0.5),
            "p95_resp_s": self._pctl([r.get("queue_s", 0) + r["latency_s"]
                                      for r in ok], 0.95),
            "workers": workers,
            "directory": {
                "entries": len(self.dir),
                "total_bytes": sum(m["bytes"] for m in self.dir.values()),
                "ever_accessed": sum(1 for m in self.dir.values()
                                     if m["count"] > 0),
                "top_lambda": [
                    {"name": n[-60:], "lambda": round(m["lambda"], 4),
                     "count": m["count"], "bytes": m["bytes"],
                     "is_tip": n in self.tips}
                    for n, m in top],
            },
            "records": self.records,
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.args.out,
            f"blkcluster_s{self.args.sessions}t{self.args.turns_per_session}_"
            f"{stamp}.json")
        os.makedirs(self.args.out, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"summary -> {path}")
        for k in ("wall_s", "throughput_rps", "hit_rate", "resumed",
                  "new_tokens_processed", "published_blocks",
                  "transfers", "transfer_bytes",
                  "replications", "replicated_bytes",
                  "evictions", "evicted_blocks",
                  "avg_latency_s", "failed", "prefill_rate", "xfer_rate",
                  "p50_resp_s", "p95_resp_s",
                  "session_turn_migration_rate",
                  "remote_resume_opportunities",
                  "remote_resume_served_local_due_to_replication",
                  "rederivation_tokens", "degrade_rederiv_tokens"):
            print(f"  {k}: {out[k]}")
        for wid, ws in workers.items():
            print(f"  worker {wid}: {ws}")
        print(f"  spill: {json.dumps(out['spill'])}")
        d = out["directory"]
        print(f"  directory: {d['entries']} entries, "
              f"{d['total_bytes']/1e6:.1f}MB, {d['ever_accessed']} accessed")
        for e in d["top_lambda"][:5]:
            print(f"    lambda={e['lambda']:<8} count={e['count']:<3} "
                  f"tip={e['is_tip']} {e['name'][-45:]}")
        return out


def add_args(ap):
    ap.add_argument("--bind", default="tcp://127.0.0.1:5570")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--policy", default="ours",
                    choices=["b0", "b1", "b2", "b3", "ours", "p2"],
                    help="fast-path capability subset (05 v3 §6 step 3): "
                         "b0 = load-only, no content awareness; "
                         "b1 = reactive locality+load, no fetch; "
                         "b2 = planned affinity (waits for a nearly-done "
                         "warm worker up to --wait-threshold); "
                         "b3 = b1 + cross-worker KV fetch; "
                         "ours = b3 (+ placement controller in step 4). "
                         "p2 is a legacy alias of ours.")
    ap.add_argument("--wait-threshold", type=float, default=2.0,
                    help="b2: hold a turn for a busy warm worker while its "
                         "ETA is below this many seconds")
    ap.add_argument("--worker-mem-budget-mb", type=float, default=0.0,
                    help="placement controller: per-worker residency budget "
                         "in MB; 0 (default) disables eviction and the "
                         "replication memory guard")
    ap.add_argument("--repl-min-lambda", type=float, default=0.05,
                    help="placement controller: ignore tips with demand "
                         "EWMA below this many hits/s")
    ap.add_argument("--repl-cooldown", type=float, default=15.0,
                    help="placement controller: seconds before re-attempting "
                         "the same (tip, target) replication")
    ap.add_argument("--max-repl-inflight", type=int, default=2,
                    help="placement controller: cap on concurrent "
                         "replication transfers")
    ap.add_argument("--repl-mem-price", type=float, default=0.0,
                    help="placement controller: C_memory per byte-second in "
                         "G_rep; 0 = memory is free at prototype scale")
    ap.add_argument("--spill-mb", type=float, default=0.0,
                    help="E2 backing tier: per-worker host-DRAM spill "
                         "capacity in MB, passed through to the workers; "
                         "0 (default) = off (evict drops), -1 = unlimited, "
                         "a full tier drops LRU-oldest")
    ap.add_argument("--spill-rate", type=float, default=20e9,
                    help="E2 cost model: spill-recall bandwidth host->device "
                         "in B/s (PCIe gen4 ~20GB/s, ~200x the cross-worker "
                         "xfer EWMA — local recall is priced well below a "
                         "demand fetch)")
    ap.add_argument("--repr", default="bf16", choices=["bf16"])
    ap.add_argument("--block-tokens", type=int, default=16,
                    help="prefix block length b (05 v3 §1)")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns-per-session", type=int, default=4)
    ap.add_argument("--q-tokens", type=int, default=None,
                    help="E1 (07 §4): pin every question turn's token delta "
                         "to exactly N (tile/truncate the cycled question "
                         "text) so the session prefix grows linearly; "
                         "default None = keep each question's natural "
                         "length")
    ap.add_argument("--doc-chars", type=int, default=4000)
    ap.add_argument("--doc-repeat", type=int, default=1)
    ap.add_argument("--doc-repeat-alt", type=int, default=None)
    ap.add_argument("--doc-share", type=int, default=None,
                    help="reuse the first K doc samples across sessions "
                         "(i %% K) to force identical chains / cross-worker "
                         "fetch. K=2 pairs even/odd sessions exactly.")
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--arrival", choices=["none", "poisson"], default="none",
                    help="workload shape: none (default) = closed loop, all "
                         "sessions start at once; poisson = open arrival "
                         "stream — sessions arrive at --arrival-rate, doc "
                         "popularity follows a zipf catalog, and a "
                         "session's next turn arrives after --think-s")
    ap.add_argument("--arrival-rate", type=float, default=1.0,
                    help="poisson session arrival rate (sessions/s)")
    ap.add_argument("--zipf-n", type=int, default=16,
                    help="poisson: catalog size for zipf doc popularity "
                         "(0 = fall back to --doc-share cycling)")
    ap.add_argument("--zipf-s", type=float, default=1.2,
                    help="poisson: zipf exponent (higher = hotter hotspot)")
    ap.add_argument("--think-s", type=float, default=2.0,
                    help="poisson: mean user think time between turns (s)")
    ap.add_argument("--seed", type=int, default=0,
                    help="workload RNG seed (arrival times + zipf draws)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "icn_proto"))
    return ap
