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
        key = (repr_name, block_tokens)
        if key not in self._chain_cache:
            self._chain_cache[key] = derive_chain(
                self.prefix_ids, repr_name, block_tokens, model_tag)
        return self._chain_cache[key]


@dataclass
class WorkerState:
    ident: bytes
    busy: bool = False
    resident: set = field(default_factory=set)
    tips: set = field(default_factory=set)   # blocks carrying GDN checkpoints
    current: str | None = None


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
        self.obj_bytes = {}        # block name -> nbytes (published)
        self.tips = set()          # all known checkpoint tip blocks
        self.records = []
        self.transfer_bytes = 0
        self.transfers = 0
        self.t_start = time.time()
        self.done_sessions = 0
        self._xfer = {}
        self.prefill_rate = self.PREFILL_RATE0
        self.xfer_rate = self.XFER_RATE0

    # ---- workload -------------------------------------------------------

    def build_workload(self, tokenizer):
        with open(TRACE, encoding="utf-8") as f:
            docs = [json.loads(l) for l in f]
        for i in range(self.args.sessions):
            sample = docs[i % len(docs)]
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

    def match_local(self, turn, w, chain, prev_len):
        """Longest resume boundary on worker w: a tip position E <= prev_len
        whose entire block chain [0, E) is resident on w."""
        best = 0
        pos = sorted(b.span_end for b in chain
                     if str(b) in w.tips and b.span_end <= prev_len)
        for e in pos:
            if all(str(b) in w.resident
                   for b in chain_through(chain, e)):
                best = e          # sorted ascending: keep the largest valid
        return best

    def choose(self, turn):
        """Returns (ident, E, fetch, decision) or None.

        E: resume boundary (tokens). fetch: None or (holder_ident,
        [block names]) — a contiguous single-holder extension of the
        local chain. decision: cost log for records."""
        repr_name = "bf16"
        chain = turn.chain(repr_name, self.args.block_tokens)
        # Match bound = this turn's chain tip, NOT the session's previous
        # tip: cross-session sharing means blocks beyond this session's
        # own history can still exist (published by an earlier identical
        # session). Blocks are named by content, so any published block
        # anywhere is a candidate.
        tip_bound = turn.cum_tokens
        idle = [w for w in self.workers.values() if not w.busy]
        if not idle:
            return None
        tips_global = set().union(*(w.tips for w in self.workers.values()))
        best = None
        costs = {}
        for w in idle:
            e_loc = self.match_local(turn, w, chain, tip_bound)
            # candidate extension targets: tip positions above e_loc,
            # whose intermediate blocks all exist somewhere, single-holder.
            fetch = None
            e_max = e_loc
            cand = sorted(b.span_end for b in chain
                          if str(b) in tips_global
                          and e_loc < b.span_end <= tip_bound)
            for e in reversed(cand):
                ext = [b for b in chain_through(chain, e)
                       if b.span_end > e_loc]
                holders = [x for x in self.workers.values()
                           if x is not w and all(
                               str(b) in x.resident for b in ext)]
                if holders:
                    fetch = (holders[0].ident, [str(b) for b in ext])
                    e_max = e
                    break
            fresh = turn.cum_tokens - e_max
            fetch_bytes = sum(self.obj_bytes.get(bn, 0)
                              for bn in (fetch[1] if fetch else []))
            cost = fresh / self.prefill_rate
            if fetch:
                cost += fetch_bytes / self.xfer_rate
            mode = ("local" if e_max == e_loc and e_loc > 0
                    else "fetch" if fetch else "fresh")
            costs[w.ident.decode()] = {
                "mode": mode, "E": e_max, "fresh": fresh,
                "cost_s": round(cost, 4)}
            # argmin cost; ties: prefer local resume, then larger E
            rank = (round(cost, 6), 0 if mode == "local" else 1, -e_max)
            if best is None or rank < best[0]:
                best = (rank, w.ident, e_max, fetch,
                        {"mode": mode, "E": e_max, "cost_s": round(cost, 4),
                         "fresh": fresh, "fetch_bytes": fetch_bytes,
                         "options": costs})
        if best is None:
            return None
        _, ident, e, fetch, decision = best
        if fetch is not None:
            # the extension's tip becomes the resume boundary
            e = self._tip_end(fetch[1][-1])
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
                                   "names": names, "E_loc": decision["E"],
                                   "turn": turn, "decision": decision,
                                   "t_fetch": time.time()}
                msg.send(sock, {"type": "fetch", "names": names},
                         ident=holder_ident)
                w.busy = True
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
        self._xfer.pop(rid, None)
        chain = turn.chain("bf16", self.args.block_tokens)
        resume_names = [str(b) for b in chain_through(chain, e_resume)]
        new_names = [str(b) for b in chain if b.span_start >= e_resume]
        msg.send(sock, {
            "type": "assign", "request_id": rid,
            "session": turn.session, "turn": turn.turn,
            "resume_names": resume_names,
            "new_block_names": new_names,
            "prefill_ids": turn.prefix_ids[e_resume:],
            "decode_steps": (0 if turn.turn == -1
                             else self.args.decode_steps),
            "repr": "bf16",
        }, ident=ident)
        self.records.append({"request_id": rid, "worker": w.ident.decode(),
                             "t_assigned": time.time(),
                             "queue_s": round(time.time() - turn.t_ready, 4),
                             "E": e_resume, "xfer_blocks": xfer_names or [],
                             "transfer_bytes": xfer_bytes,
                             "xfer_s": round(xfer_s, 4),
                             "decision": decision})

    def on_message(self, sock, ident, hdr, payload):
        w = self.workers.get(ident)
        if w is None:
            return
        mtype = hdr.get("type")
        if mtype == "status":
            w.resident = set(hdr.get("resident", []))
            w.tips = set(hdr.get("tips", []))
            w.busy = hdr.get("busy", False)
            return
        if mtype == "result":
            rid = hdr["request_id"]
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
            if hdr.get("ok"):
                session, t = rid.split(":")
                chain_idx = 0 if t == "-1" else int(t) + 1
                chain = self.turns_of[session][chain_idx].chain(
                    "bf16", self.args.block_tokens)
                if hdr.get("published"):
                    for p in hdr["published"]:
                        w.resident.add(p["name"])
                        self.obj_bytes[p["name"]] = p["bytes"]
                    # the last new block of the batch carries the GDN
                    # checkpoint (worker.extract_blocks contract)
                    new_names = [str(b) for b in chain
                                 if b.span_start >= rec.get("E", 0)]
                    if new_names:
                        w.tips.add(new_names[-1])
                        self.tips.add(new_names[-1])
                if not rec.get("resumed") and hdr.get("prefill_s"):
                    rate = hdr["prefill_tokens"] / hdr["prefill_s"]
                    self.prefill_rate = 0.7 * self.prefill_rate + 0.3 * rate
            self.advance(rid)
            return
        if mtype == "fetched":
            rid, xfer = next(
                ((r, x) for r, x in self._xfer.items()
                 if x["stage"] == "fetch"
                 and x["names"] == hdr.get("names")), (None, None))
            if not xfer:
                return
            if hdr.get("ok"):
                xfer["stage"] = "deliver"
                xfer["bytes"] = len(payload)
                msg.send(sock, {"type": "deliver"}, payload=payload,
                         ident=xfer["target"])
            else:
                # holder lost blocks: degrade to the local boundary
                turn = xfer["turn"]
                self._xfer.pop(rid, None)
                self.send_assign(sock, xfer["target"], turn, xfer["E_loc"])
            return
        if mtype == "delivered":
            rid, xfer = next(
                ((r, x) for r, x in self._xfer.items()
                 if x["stage"] == "deliver"), (None, None))
            if xfer:
                self._xfer.pop(rid, None)
                self.transfer_bytes += xfer["bytes"]
                self.transfers += 1
                xfer_s = time.time() - xfer["t_fetch"]
                if xfer_s > 0:
                    self.xfer_rate = (0.7 * self.xfer_rate
                                      + 0.3 * xfer["bytes"] / xfer_s)
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
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "icn_proto"))
    return ap
