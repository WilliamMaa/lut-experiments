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

Run (spawned by run_cluster.py, which also starts the workers):
    python -m icn_proto.run_cluster --sessions 16 ...
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icn_proto import msg
from icn_proto.blkchain import chain_through, derive_chain

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
    t_ready: float = 0.0
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
        self.prefill_rate = self.PREFILL_RATE0
        self.xfer_rate = self.XFER_RATE0

    def dir_add(self, name, nbytes, t=None):
        e = self.dir.get(name)
        if e is None:
            self.dir[name] = e = {"bytes": int(nbytes), "first": t or time.time(),
                                  "last": 0.0, "count": 0, "lambda": 0.0}
        else:
            e["bytes"] = int(nbytes)
        return e

    def note_access(self, names, t=None):
        t = t or time.time()
        for n in names:
            e = self.dir.get(n)
            if e is None:
                continue
            if e["count"] >= 1 and e["last"] > 0:
                inst = 1.0 / max(t - e["last"], 1e-3)
                e["lambda"] = (inst if e["count"] == 1
                               else 0.7 * e["lambda"] + 0.3 * inst)
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
        for i in range(self.args.sessions):
            sample = docs[i % n_samples]
            session = f"doc{i}"
            rep = (self.args.doc_repeat if i % 2 == 0
                   else (self.args.doc_repeat_alt
                         if self.args.doc_repeat_alt is not None
                         else self.args.doc_repeat))
            doc_text = (sample["document"] * rep)[: self.args.doc_chars]
            doc_ids = tokenizer(doc_text,
                                return_tensors="pt").input_ids[0].tolist()
            turns = [Turn(session, -1, list(doc_ids), list(doc_ids))]
            prefix = list(doc_ids)
            for t, q in enumerate(sample["questions"][: self.args.turns_per_session]):
                q_ids = tokenizer("\n\n" + q,
                                  return_tensors="pt").input_ids[0].tolist()
                prefix = prefix + q_ids
                turns.append(Turn(session, t, list(q_ids), list(prefix)))
            self.turns_of[session] = turns
            self.pending_next[session] = 0
            turns[0].t_ready = time.time()
            self.ready.append(turns[0])

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
        full resume-name set (complete blocks + tip block) is resident."""
        bt = self.args.block_tokens
        best = 0
        for tip_name in w.tips:
            t_pos = self._tip_end(tip_name)
            if t_pos == 0 or t_pos > tip_bound:
                continue
            if turn.tip_name_at(t_pos, "bf16", bt) != tip_name:
                continue                      # not a tip of THIS turn's chain
            if not all(n in w.resident
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
        idle = [w for w in self.workers.values() if not w.busy]
        if not idle:
            return None
        tips_global = set().union(*(w.tips for w in self.workers.values()))
        best = None
        costs = {}
        for w in idle:
            e_loc = self.match_local(turn, w, tip_bound)
            print(f"[match] {turn.session}:{turn.turn} w={w.ident.decode()} "
                  f"tips={len(w.tips)} resident={len(w.resident)} "
                  f"e_loc={e_loc}", flush=True)
            # candidate extension targets: tip positions above e_loc whose
            # missing blocks are all on ONE other worker.
            fetch = None
            e_max = e_loc
            cand = sorted(self._tip_end(tn) for tn in tips_global
                          if e_loc < self._tip_end(tn) <= tip_bound)
            for t_pos in reversed(cand):
                need = [n for n in self.resume_names(turn, t_pos)
                        if n not in w.resident]
                holders = [x for x in self.workers.values()
                           if x is not w and all(n in x.resident
                                                 for n in need)]
                if holders:
                    fetch = (holders[0].ident, need)
                    e_max = t_pos
                    break
            fresh = turn.cum_tokens - e_max
            fetch_bytes = sum(self.dir.get(bn, {}).get("bytes", 0)
                              for bn in (fetch[1] if fetch else []))
            # queue wait: 0 for idle workers by construction (one in-flight
            # turn per worker); the term lands with pipelining in step 3
            rate = w.prefill_rate or self.prefill_rate
            cost = fresh / rate
            if fetch:
                cost += fetch_bytes / self.xfer_rate
            eta = self.eta(w)   # expected finish of the in-flight turn
            cost += eta
            mode = ("local" if e_max == e_loc and e_loc > 0
                    else "fetch" if fetch else "fresh")
            costs[w.ident.decode()] = {
                "mode": mode, "E": e_max, "fresh": fresh,
                "eta_s": round(eta, 4), "rate": round(rate, 1),
                "cost_s": round(cost, 4)}
            # argmin cost; ties: prefer local resume, then larger E
            rank = (round(cost, 6), 0 if mode == "local" else 1, -e_max)
            if best is None or rank < best[0]:
                best = (rank, w.ident, e_max, fetch,
                        {"mode": mode, "E": e_max, "E_loc": e_loc,
                         "cost_s": round(cost, 4),
                         "fresh": fresh, "fetch_bytes": fetch_bytes,
                         "options": costs})
        if best is None:
            return None
        _, ident, e, fetch, decision = best
        return ident, e, fetch, decision

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
        while len(hellos) < len(self.workers):
            ident, hdr, _ = msg.recv(sock)
            if hdr.get("type") == "hello":
                hellos.add(hdr["worker_id"])
                w = self.workers.get(ident)
                if w is not None:
                    w.resident = set(hdr.get("resident", []))
                    w.tips = set(hdr.get("tips", []))

        try:
            while not self.finished():
                self.dispatch(sock)
                evts = dict(poller.poll(timeout=1000))
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
                if sock not in evts:
                    continue
                ident, hdr, payload = msg.recv(sock)
                self.on_message(sock, ident, hdr, payload)
        finally:
            for w in self.workers.values():
                msg.send(sock, {"type": "shutdown"}, ident=w.ident)
            time.sleep(0.5)
            sock.close()
            ctx.term()
        return self.summary()

    def dispatch(self, sock):
        for turn in list(self.ready):
            chosen = self.choose(turn)
            if chosen is None:
                break
            ident, e_resume, fetch, decision = chosen
            rid = f"{turn.session}:{turn.turn}"
            w = self.workers[ident]
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
        # demand accounting: every resumed block contributed to this turn
        self.note_access(resume_names)
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
                             "queue_s": round(time.time() - turn.t_ready, 4),
                             "E": e_resume,
                             "fp": turn.prefix_fingerprint(
                                 "bf16", self.args.block_tokens),
                             "xfer_blocks": xfer_names or [],
                             "transfer_bytes": xfer_bytes,
                             "xfer_s": round(xfer_s, 4),
                             "decision": decision})

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
            # blocks to different targets concurrently
            rid, xfer = next(
                ((r, x) for r, x in self._xfer.items()
                 if x["stage"] == "fetch"
                 and x["holder"] == ident
                 and x["names"] == hdr.get("names")), (None, None))
            if not xfer:
                print(f"[xfer ] WARN fetched with no matching xfer from "
                      f"{ident.decode()}: ok={hdr.get('ok')} "
                      f"names={len(hdr.get('names') or [])}", flush=True)
                return
            if hdr.get("ok"):
                xfer["stage"] = "deliver"
                xfer["bytes"] = len(payload)
                print(f"[xfer ] {rid} fetched {xfer['bytes']/1e6:.1f}MB "
                      f"from {ident.decode()} in "
                      f"{time.time() - xfer['t_fetch']:.2f}s", flush=True)
                msg.send(sock, {"type": "deliver"}, payload=payload,
                         ident=xfer["target"])
            else:
                # holder lost blocks: degrade to the local boundary
                print(f"[xfer ] {rid} fetch FAILED (missing "
                      f"{len(hdr.get('missing') or [])}), degrade to "
                      f"E_loc={xfer['E_loc']}", flush=True)
                turn = xfer["turn"]
                self._xfer.pop(rid, None)
                self.send_assign(sock, xfer["target"], turn, xfer["E_loc"])
            return
        if mtype == "delivered":
            # pair by target worker: concurrent transfers to different
            # workers must not cross
            rid, xfer = next(
                ((r, x) for r, x in self._xfer.items()
                 if x["stage"] == "deliver"
                 and x["target"] == ident), (None, None))
            if xfer:
                self._xfer.pop(rid, None)
                self.transfer_bytes += xfer["bytes"]
                self.transfers += 1
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
            turns[nxt].t_ready = time.time()
            self.ready.append(turns[nxt])
        else:
            self.done_sessions += 1

    def finished(self):
        return self.done_sessions >= len(self.turns_of)

    # ---- reporting --------------------------------------------------------

    def summary(self):
        ok = [r for r in self.records if r.get("ok")]
        failed = [r for r in self.records if not r.get("ok")]
        wall = time.time() - self.t_start
        total_new = sum(r.get("prefill_tokens", 0) for r in ok)
        resumed = [r for r in ok if r.get("resumed")]
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
        out = {
            "config": {k: v for k, v in vars(self.args).items()},
            "prefill_rate": round(self.prefill_rate, 1),
            "xfer_rate": round(self.xfer_rate, 1),
            "wall_s": round(wall, 2),
            "requests": len(self.records),
            "failed": len(failed),
            "throughput_rps": round(len(ok) / wall, 4),
            "resumed": len(resumed),
            "hit_rate": round(len(resumed) / max(1, len(ok)), 4),
            "new_tokens_processed": total_new,
            "published_blocks": published_blocks,
            "published_bytes": published_bytes,
            "transfers": self.transfers,
            "transfer_bytes": self.transfer_bytes,
            "avg_latency_s": round(sum(r["latency_s"] for r in ok)
                                   / max(1, len(ok)), 4),
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
                  "avg_latency_s", "failed", "prefill_rate", "xfer_rate"):
            print(f"  {k}: {out[k]}")
        for wid, ws in workers.items():
            print(f"  worker {wid}: {ws}")
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
    ap.add_argument("--policy", default="p2", choices=["p2"],
                    help="v1 of the block protocol implements the single "
                         "fast-path chooser; B0-B3 policy switches land in "
                         "step 3 (05 v3 §6).")
    ap.add_argument("--repr", default="bf16", choices=["bf16"])
    ap.add_argument("--block-tokens", type=int, default=16,
                    help="prefix block length b (05 v3 §1)")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns-per-session", type=int, default=4)
    ap.add_argument("--doc-chars", type=int, default=4000)
    ap.add_argument("--doc-repeat", type=int, default=1)
    ap.add_argument("--doc-repeat-alt", type=int, default=None)
    ap.add_argument("--doc-share", type=int, default=None,
                    help="reuse the first K doc samples across sessions "
                         "(i %% K) to force identical chains / cross-worker "
                         "fetch. K=2 pairs even/odd sessions exactly.")
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "icn_proto"))
    return ap
