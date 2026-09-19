#!/usr/bin/env python3
"""Scheduler: trace replay, name-aware placement, metrics aggregation.

Object identity is content-addressed (docs/icn-defined-addressing/03
§4.2): a turn's KV object is named H(model, encoding, prefix token ids),
so identical prefixes share objects across sessions through the NRS
(scheduler-side resident index + reuse_count). Every turn — including each
session's synthetic doc-turn (turn == -1, the shared document prefix) —
does the NRS lookup and picks one of: resume local / fetch->deliver /
recompute. Policies differ only in how they choose.

Run (spawned by run_cluster.py, which also starts the workers):
    python -m icn_proto.run_cluster --policy p2 ...
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
from icn_proto.kvname import KVName, Repr

TRACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "multi_turn_prompts_v3.jsonl")


@dataclass
class Turn:
    """One dispatchable unit of a session's chain.

    turn == -1 is the synthetic doc-turn: it prefills the shared document
    prefix and publishes it as its own content-addressed object (decode 0).
    Question turns t >= 0 prefill only their question tokens onto the
    predecessor's object.

    prefill_ids: tokens the worker prefills when RESUMING (q tokens, or []
    for a doc-turn reuse hit). When RECOMPUTING, the worker prefills
    prefix_ids instead (see send_assign).
    prefix_ids: the full token prefix through this turn — the hash input
    for this turn's content-addressed name."""

    session: str
    turn: int
    prefill_ids: list
    prefix_ids: list
    t_ready: float = 0.0
    _name_cache: dict = field(default_factory=dict, repr=False,
                              compare=False)

    @property
    def cum_tokens(self) -> int:
        return len(self.prefix_ids)

    def name(self, repr_name: str, model_tag: str = "qwen35b") -> str:
        if repr_name not in self._name_cache:
            import hashlib
            from array import array
            h = hashlib.sha256()
            h.update(model_tag.encode())
            h.update(b"|" + repr_name.encode() + b"|")
            h.update(array("q", self.prefix_ids).tobytes())
            self._name_cache[repr_name] = str(
                KVName(h.hexdigest()[:16], len(self.prefix_ids),
                       Repr(repr_name)))
        return self._name_cache[repr_name]


@dataclass
class WorkerState:
    ident: bytes
    busy: bool = False
    resident: set = field(default_factory=set)
    current: str | None = None   # request_id while busy


class Scheduler:
    # online-calibrated cost-model coefficients (Step 2.5 rotate/P0 runs:
    # prefill 55-62k tok/s, m_sp4 object xfer 65.9MB @ ~98MB/s effective).
    PREFILL_RATE0 = 50_000.0        # tokens/s, EWMA-seeded
    XFER_RATE0 = 100e6              # bytes/s, EWMA-seeded
    COMPRESSED_OBJ_BYTES = 66 * 2**20   # m_sp4/k8v8 latest-per-session object

    def __init__(self, args, worker_ids):
        self.args = args
        self.workers = {wid.encode(): WorkerState(ident=wid.encode())
                        for wid in worker_ids}
        self.ready = []            # Turn objects whose predecessor completed
        self.pending_next = {}     # session -> next turn index
        self.turns_of = {}         # session -> [Turn]
        self.repr_of = {}          # session -> repr name (chain-homogeneous)
        self.obj_bytes = {}        # object name -> nbytes (reported by workers)
        self.records = []
        self.transfer_bytes = 0
        self.transfers = 0
        self.t_start = time.time()
        self.done_sessions = 0
        self._xfer = {}            # request_id -> transfer state
        self.prefill_rate = self.PREFILL_RATE0
        self.xfer_rate = self.XFER_RATE0
        self.nrs_reuse = {}        # content name -> cross-session reuse count

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
        self.assign_representations()

    def assign_representations(self):
        """P2: representation-aware admission. Candidates per session: bf16,
        or the workers' compressed repr (--repr) when it is not bf16. Each
        worker keeps only a session's latest object, so the steady-state
        footprint of one session ~= one object of its final turn. Greedy:
        longest sessions get the compressed repr first when the system-wide
        resident budget (mem_budget_gb per worker) would be exceeded."""
        compressed = None if self.args.repr == "bf16" else self.args.repr
        for s in self.turns_of:
            self.repr_of[s] = "bf16"
        if self.args.policy != "p2" or not compressed:
            return
        total_budget = self.args.mem_budget_gb * (2 ** 30) * len(self.workers)
        per_session = total_budget / max(1, len(self.turns_of))
        order = sorted(self.turns_of,
                       key=lambda s: self.turns_of[s][-1].cum_tokens,
                       reverse=True)
        for s in order:
            final_cum = self.turns_of[s][-1].cum_tokens
            if final_cum * self.args.kv_bytes_per_token <= per_session:
                self.repr_of[s] = "bf16"     # fits: keep full quality
            else:
                self.repr_of[s] = compressed # budget: take the quality hit
        n_c = sum(1 for r in self.repr_of.values() if r == compressed)
        print(f"[p2] repr assignment: {len(self.repr_of) - n_c} bf16, "
              f"{n_c} {compressed} (per-session allowance "
              f"{per_session / 2**20:.0f} MiB)")

    # ---- placement policies ----------------------------------------------

    def chain_prev(self, turn):
        """Predecessor turn in the session chain (None for the doc-turn)."""
        if turn.turn == -1:
            return None
        return self.turns_of[turn.session][turn.turn].name(
            self.repr_of[turn.session])

    def choose(self, turn):
        """NRS lookup + policy choice.

        Returns (ident, resume_from|None, fetch_from|None, decision|None)
        or None when no worker is available.

        Lookup order (03-icn-kv-principle §3):
          1. reuse   — doc-turn whose own content-addressed name is already
                       resident (another session produced the identical doc
                       prefix): zero-prefill resume, the ICN sharing case.
          2. resume  — predecessor object resident somewhere: local, or
                       fetch->deliver from its holder.
          3. recompute — nothing to resume from.

        P0 always recomputes; P1 takes the first option found (idle worker,
        transferring when the holder is busy); P2 argmins the calibrated
        cost model; rotate pins turns to workers for coefficient probing.
        """
        repr_name = self.repr_of[turn.session]
        prev = self.chain_prev(turn)
        own = turn.name(repr_name)
        reuse_holders = ([w for w in self.workers.values()
                          if own in w.resident]
                         if turn.turn == -1 else [])
        if self.args.policy == "rotate":
            idx = int("".join(c for c in turn.session if c.isdigit()))
            k = 0 if turn.turn == -1 else turn.turn + 1
            target = list(self.workers.values())[
                (idx + k) % len(self.workers)]
            if target.busy:
                return None
            if reuse_holders:
                h = reuse_holders[0]
                return (target.ident, own,
                        None if h is target else h.ident, None)
            if not prev or prev in target.resident:
                return target.ident, prev, None, None
            holders = [w for w in self.workers.values()
                       if prev in w.resident]
            if holders:
                return target.ident, prev, holders[0].ident, None
            return target.ident, None, None, None
        idle = [w for w in self.workers.values() if not w.busy]
        if not idle:
            return None
        if self.args.policy == "p2":
            return self.choose_p2(turn, prev, own, reuse_holders, idle)
        if self.args.policy == "p0":
            return idle[0].ident, None, None, None
        # P1: reuse hit (route to holder or fetch), else local resume,
        # else fetch, else recompute.
        if reuse_holders:
            h = reuse_holders[0]
            for w in idle:
                if w is h:
                    self.nrs_reuse[own] = self.nrs_reuse.get(own, 0) + 1
                    return w.ident, own, None, None
            self.nrs_reuse[own] = self.nrs_reuse.get(own, 0) + 1
            return idle[0].ident, own, h.ident, None
        if not prev:
            return idle[0].ident, None, None, None
        for w in idle:
            if prev in w.resident:
                return w.ident, prev, None, None
        holders = [w for w in self.workers.values() if prev in w.resident]
        if holders:
            return idle[0].ident, prev, holders[0].ident, None
        return idle[0].ident, None, None, None

    def choose_p2(self, turn, prev, own, reuse_holders, idle):
        """ICN cost model over the three honest options:

          reuse    : 0-prefill resume of own name (doc-turn sharing);
                     on another worker it costs obj_bytes / xfer_rate
          local    : len(prefill_ids) / prefill_rate
          transfer : obj_bytes / xfer_rate + len(prefill_ids) / prefill_rate
          recompute: len(prefix_ids) / prefill_rate   (the FULL prefix,
                     doc included — this is what makes recompute honest)

        The quality penalty (EOS_DELTA[repr]) is constant across workers
        for a chain-homogeneous session; it shaped repr assignment at
        admission, not per-turn placement. Ties break reuse < local <
        transfer < recompute."""
        prefill_n = max(1, len(turn.prefill_ids))
        best = None
        costs = {}
        for w in idle:
            if turn.turn == -1 and any(h is w for h in reuse_holders):
                cost, resume_from, fetch_from, mode = 0.0, own, None, "reuse"
            elif turn.turn == -1 and reuse_holders:
                nbytes = self.obj_bytes.get(own, self.estimate_obj_bytes(own))
                cost, resume_from = nbytes / self.xfer_rate, own
                fetch_from, mode = reuse_holders[0].ident, "xfer_reuse"
            elif prev is not None and prev in w.resident:
                cost, resume_from, fetch_from, mode = (
                    prefill_n / self.prefill_rate, prev, None, "local")
            else:
                holders = [x for x in self.workers.values()
                           if prev in x.resident] if prev else []
                if holders:
                    nbytes = self.obj_bytes.get(
                        prev, self.estimate_obj_bytes(prev))
                    cost = nbytes / self.xfer_rate + prefill_n / self.prefill_rate
                    resume_from, fetch_from = prev, holders[0].ident
                    mode = "xfer"
                else:
                    cost = len(turn.prefix_ids) / self.prefill_rate
                    resume_from, fetch_from = None, None
                    mode = "recompute"
            costs[w.ident.decode()] = {"mode": mode,
                                       "cost_s": round(cost, 4)}
            cost += {"u": -1e-7, "l": 0.0, "x": 1e-6, "r": 2e-6}[
                "u" if mode == "reuse" else
                "l" if mode == "local" else
                "x" if mode in ("xfer", "xfer_reuse") else "r"]
            if best is None or cost < best[0]:
                best = (cost, w.ident, resume_from, fetch_from, mode,
                        round(costs[w.ident.decode()]["cost_s"], 4))
        if best[4] in ("reuse", "xfer_reuse"):
            self.nrs_reuse[own] = self.nrs_reuse.get(own, 0) + 1
        decision = {"chosen": best[4], "chosen_cost_s": best[5],
                    "prefill_rate": round(self.prefill_rate, 1),
                    "xfer_rate": round(self.xfer_rate, 1),
                    "options": costs}
        return best[1], best[2], best[3], decision

    def estimate_obj_bytes(self, name):
        """Scheduler-side size estimate before the object has been reported."""
        n = KVName.parse(name)
        if n.repr.value == "bf16":
            return n.span_end * self.args.kv_bytes_per_token
        return self.COMPRESSED_OBJ_BYTES

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

        # wait for hellos from all registered workers
        hellos = set()
        while len(hellos) < len(self.workers):
            ident, hdr, _ = msg.recv(sock)
            if hdr.get("type") == "hello":
                hellos.add(hdr["worker_id"])
                w = self.workers.get(ident)
                if w is not None:
                    w.resident = set(hdr.get("resident", []))

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
                break  # no idle worker; retry when a result arrives
            ident, resume_from, fetch_from, decision = chosen
            rid = f"{turn.session}:{turn.turn}"
            w = self.workers[ident]
            if fetch_from is not None:
                self._xfer[rid] = {"stage": "fetch", "target": ident,
                                   "turn": turn, "resume_from": resume_from,
                                   "t_fetch": time.time()}
                msg.send(sock, {"type": "fetch", "name": resume_from},
                         ident=fetch_from)
                w.busy = True   # reserve the target during the transfer
                self.ready.remove(turn)
                continue
            self.send_assign(sock, ident, turn, resume_from,
                             decision=decision)
            self.ready.remove(turn)

    def send_assign(self, sock, ident, turn, resume_from, xfer_bytes=0,
                    xfer_s=0.0, decision=None):
        rid = f"{turn.session}:{turn.turn}"
        w = self.workers[ident]
        w.busy = True
        w.current = rid
        self._xfer.pop(rid, None)
        repr_name = self.repr_of[turn.session]
        is_reuse = resume_from == turn.name(repr_name)
        msg.send(sock, {
            "type": "assign", "request_id": rid,
            "session": turn.session, "turn": turn.turn,
            # resume path prefills the turn's own tokens; recompute path
            # prefills the FULL prefix (doc included — honest recompute);
            # reuse path (doc-turn hit) prefills nothing: the injected
            # object already covers the whole span.
            "prefill_ids": ([] if is_reuse
                            else turn.prefill_ids if resume_from
                            else turn.prefix_ids),
            "cum_tokens": turn.cum_tokens,
            "name": turn.name(repr_name),
            "resume_from": resume_from,
            "repr": repr_name,
            "decode_steps": (0 if turn.turn == -1
                             else self.args.decode_steps),
        }, ident=ident)
        self.records.append({"request_id": rid, "worker": w.ident.decode(),
                             "t_assigned": time.time(),
                             "queue_s": round(time.time() - turn.t_ready, 4),
                             "resume_from": resume_from,
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
            w.busy = hdr.get("busy", False)
            return
        if mtype == "result":
            rid = hdr["request_id"]
            rec = next((r for r in reversed(self.records)
                        if r["request_id"] == rid), None)
            if rec:
                rec.update({k: hdr.get(k) for k in
                            ("ok", "resumed", "prefill_s", "decode_s",
                             "prefill_tokens", "cum_tokens", "error",
                             "obj_bytes", "resident_bytes", "repr",
                             "decoded_ids")})
                rec["latency_s"] = round(time.time() - rec.pop("t_assigned"), 4)
            w.busy = False
            w.current = None
            if hdr.get("ok"):
                session, t = rid.split(":")
                chain_idx = 0 if t == "-1" else int(t) + 1
                name = self.turns_of[session][chain_idx] \
                    .name(self.repr_of[session])
                w.resident.add(name)
                if hdr.get("obj_bytes"):
                    self.obj_bytes[name] = hdr["obj_bytes"]
                # online calibration of the recompute coefficient from
                # non-resumed turns only; doc-turns are pure prefix
                # prefill (cum == prefill length), question-turn
                # recomputes carry the full prefix in prefill_tokens.
                if not rec.get("resumed") and hdr.get("prefill_s"):
                    rate = hdr["prefill_tokens"] / hdr["prefill_s"]
                    self.prefill_rate = 0.7 * self.prefill_rate + 0.3 * rate
            # advance even on failure: the next turn finds no resident
            # predecessor and degrades to full recompute (designed fallback),
            # and finished() must always become reachable.
            self.advance(rid)
            return
        if mtype == "fetched":
            rid, xfer = next(
                ((r, x) for r, x in self._xfer.items()
                 if x["stage"] == "fetch" and x["resume_from"] == hdr["name"]),
                (None, None))
            if not xfer:
                return
            if hdr.get("ok"):
                xfer["stage"] = "deliver"
                xfer["bytes"] = len(payload)
                msg.send(sock, {"type": "deliver", "name": hdr["name"]},
                         payload=payload, ident=xfer["target"])
            else:
                # holder lost the object: fall back to full recompute
                turn = xfer["turn"]
                self._xfer.pop(rid, None)
                self.send_assign(sock, xfer["target"], turn, None)
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
                self.send_assign(sock, xfer["target"], xfer["turn"],
                                 xfer["resume_from"],
                                 xfer_bytes=xfer.get("bytes", 0),
                                 xfer_s=xfer_s)
            return

    def advance(self, request_id):
        session, t = request_id.split(":")
        # chain index: 0 = doc-turn, k = question turn k-1
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
        recompute_tokens = sum(
            r.get("cum_tokens", 0) for r in ok if not r.get("resume_from"))
        out = {
            "config": {k: v for k, v in vars(self.args).items()},
            "policy": self.args.policy,
            "repr_of": self.repr_of,
            "nrs_reuse": self.nrs_reuse,
            "prefill_rate": round(self.prefill_rate, 1),
            "xfer_rate": round(self.xfer_rate, 1),
            "wall_s": round(wall, 2),
            "requests": len(self.records),
            "failed": len(failed),
            "throughput_rps": round(len(ok) / wall, 4),
            "resumed": len(resumed),
            "hit_rate": round(len(resumed) / max(1, len(ok)), 4),
            "recompute_tokens": recompute_tokens,
            "new_tokens_processed": total_new,
            "transfers": self.transfers,
            "transfer_bytes": self.transfer_bytes,
            "avg_latency_s": round(sum(r["latency_s"] for r in ok)
                                   / max(1, len(ok)), 4),
            "records": self.records,
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.args.out,
            f"cluster_{self.args.policy}_{self.args.repr}_"
            f"s{self.args.sessions}t{self.args.turns_per_session}_"
            f"{stamp}.json")
        os.makedirs(self.args.out, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"summary -> {path}")
        for k in ("wall_s", "throughput_rps", "hit_rate", "resumed",
                  "recompute_tokens", "transfers", "transfer_bytes",
                  "avg_latency_s", "failed", "prefill_rate", "xfer_rate"):
            print(f"  {k}: {out[k]}")
        return out


def add_args(ap):
    ap.add_argument("--bind", default="tcp://127.0.0.1:5570")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--policy", default="p1",
                    choices=["p0", "p1", "p2", "rotate"])
    ap.add_argument("--repr", default="m_sp4", choices=["bf16", "m_sp4", "k8v8"])
    ap.add_argument("--mem-budget-gb", type=float, default=8.0,
                    help="per-worker resident KV budget used by the P2 repr "
                         "assignment (system budget = x W workers)")
    ap.add_argument("--kv-bytes-per-token", type=int, default=20480,
                    help="bf16 KV size for this model: 2 heads x 256 dim x "
                         "2 (K,V) x 2 B x 10 full-attn layers")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns-per-session", type=int, default=4)
    ap.add_argument("--doc-chars", type=int, default=4000)
    ap.add_argument("--doc-repeat", type=int, default=1,
                    help="tile the document N times before truncation, to "
                         "synthesize long-context traces (systems replay only; "
                         "generation quality is not measured)")
    ap.add_argument("--doc-repeat-alt", type=int, default=None,
                    help="doc-repeat for odd sessions (mixed-length traces); "
                         "default = same as --doc-repeat")
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "icn_proto"))
    return ap
