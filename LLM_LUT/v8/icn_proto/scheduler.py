#!/usr/bin/env python3
"""Scheduler: trace replay, name-aware placement, metrics aggregation.

Step 2 scope: policies P0 (least-loaded, no locality) and P1 (cache-affinity
with cross-worker object transfer). P2 (ICN cost model + representation
selection) lands in Step 3.

Workload: the first `--sessions` documents of data/multi_turn_prompts_v3.jsonl,
`--turns-per-session` questions each. Turn t of a session continues turn t-1;
a turn is assignable once its predecessor completed (the KV object must
exist). Per-session object chain:
/session/<doc>/turn/<t>/span/0-<cum>/repr/<r>.

A request's resume_from names the session's previous object; the assigned
worker must hold it (resident or after a scheduler-mediated fetch->deliver),
else the worker recomputes from scratch (recompute_tokens = cum_tokens).

Run (spawned by run_cluster.py, which also starts the workers):
    python -m icn_proto.run_cluster --policy p1 ...
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

TRACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "multi_turn_prompts_v3.jsonl")


@dataclass
class Turn:
    session: str
    turn: int
    new_token_ids: list
    cum_tokens: int
    t_ready: float = 0.0      # when the turn became dispatchable

    def name(self, repr_name: str) -> str:
        return (f"/session/{self.session}/turn/{self.turn}"
                f"/span/0-{self.cum_tokens}/repr/{repr_name}")


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
            doc = (sample["document"] * rep)[: self.args.doc_chars]
            prompt = doc
            turns = []
            for t, q in enumerate(sample["questions"][: self.args.turns_per_session]):
                new_text = "\n\n" + q
                ids = tokenizer(new_text, return_tensors="pt").input_ids[0].tolist()
                prompt = prompt + new_text
                cum = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
                turns.append(Turn(session, t, ids, cum))
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

    def choose(self, turn):
        """Returns (ident, resume_from|None, fetch_from|None) or None.

        P0: any idle worker, never resume (full recompute by construction).
        P1: idle worker already holding the previous object -> resume there;
            else an idle worker plus fetch of the object from its holder;
            else any idle worker, no resume (first turn / holder gone).
        rotate (diagnostic): session s turn t goes to worker (s+t) % W, so
            every non-first turn is a cross-worker transfer under full load.
            Used to measure xfer_s, not a real placement policy.
        """
        prev = None
        if turn.turn > 0:
            prev = self.turns_of[turn.session][turn.turn - 1].name(
                self.repr_of[turn.session])
        if self.args.policy == "rotate":
            idx = int("".join(c for c in turn.session if c.isdigit()))
            target = list(self.workers.values())[
                (idx + turn.turn) % len(self.workers)]
            if target.busy:
                return None
            if not prev or prev in target.resident:
                return target.ident, prev, None
            holders = [w for w in self.workers.values()
                       if prev in w.resident]
            if holders:
                return target.ident, prev, holders[0].ident
            return target.ident, None, None
        idle = [w for w in self.workers.values() if not w.busy]
        if not idle:
            return None
        if self.args.policy == "p2":
            return self.choose_p2(turn, prev, idle)
        if self.args.policy == "p0" or not prev:
            return idle[0].ident, None, None
        for w in idle:
            if prev in w.resident:
                return w.ident, prev, None
        holders = [w for w in self.workers.values() if prev in w.resident]
        if holders:
            return idle[0].ident, prev, holders[0].ident
        return idle[0].ident, None, None

    def choose_p2(self, turn, prev, idle):
        """ICN cost model: pick the idle worker minimizing predicted
        execution time of this turn.

          local    : new_tokens / prefill_rate
          transfer : prev_obj_bytes / xfer_rate + new_tokens / prefill_rate
          recompute: cum_tokens / prefill_rate

        The quality penalty (EOS_DELTA[repr]) is identical across workers
        for a chain-homogeneous session, so it shapes repr assignment at
        admission time, not per-turn placement. Ties break toward local
        resume (no transfer bytes on the wire)."""
        new_tokens = max(1, len(turn.new_token_ids))
        best = None
        for w in idle:
            if prev is not None and prev in w.resident:
                cost, resume_from, fetch_from = (
                    new_tokens / self.prefill_rate, prev, None)
            else:
                holders = [x for x in self.workers.values()
                           if prev in x.resident] if prev else []
                if holders:
                    nbytes = self.obj_bytes.get(
                        prev, self.estimate_obj_bytes(prev))
                    cost = nbytes / self.xfer_rate + new_tokens / self.prefill_rate
                    resume_from, fetch_from = prev, holders[0].ident
                else:
                    cost = turn.cum_tokens / self.prefill_rate
                    resume_from, fetch_from = None, None
            # epsilon preference: local < transfer < recompute on ties
            cost += {"l": 0.0, "x": 1e-6, "r": 2e-6}[
                "l" if fetch_from is None and resume_from else
                "x" if fetch_from else "r"]
            if best is None or cost < best[0]:
                best = (cost, w.ident, resume_from, fetch_from)
        return best[1], best[2], best[3]

    def estimate_obj_bytes(self, name):
        """Scheduler-side size estimate before the object has been reported."""
        from icn_proto.kvname import KVName
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
            ident, resume_from, fetch_from = chosen
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
            self.send_assign(sock, ident, turn, resume_from)
            self.ready.remove(turn)

    def send_assign(self, sock, ident, turn, resume_from, xfer_bytes=0,
                    xfer_s=0.0):
        rid = f"{turn.session}:{turn.turn}"
        w = self.workers[ident]
        w.busy = True
        w.current = rid
        self._xfer.pop(rid, None)
        msg.send(sock, {
            "type": "assign", "request_id": rid,
            "session": turn.session, "turn": turn.turn,
            "new_token_ids": turn.new_token_ids,
            "cum_tokens": turn.cum_tokens,
            "resume_from": resume_from,
            "repr": self.repr_of[turn.session],
            "decode_steps": self.args.decode_steps,
        }, ident=ident)
        self.records.append({"request_id": rid, "worker": w.ident.decode(),
                             "t_assigned": time.time(),
                             "queue_s": round(time.time() - turn.t_ready, 4),
                             "resume_from": resume_from,
                             "transfer_bytes": xfer_bytes,
                             "xfer_s": round(xfer_s, 4)})

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
                             "obj_bytes", "resident_bytes", "repr")})
                rec["latency_s"] = round(time.time() - rec.pop("t_assigned"), 4)
            w.busy = False
            w.current = None
            if hdr.get("ok"):
                name = self.turns_of[rid.split(":")[0]][int(rid.split(":")[1])] \
                    .name(self.repr_of[rid.split(":")[0]])
                w.resident.add(name)
                if hdr.get("obj_bytes"):
                    self.obj_bytes[name] = hdr["obj_bytes"]
                # online calibration of the recompute coefficient from fresh
                # (full-prefill) turns only: resumed prefill_s measures the
                # new-question tokens, not the prefix.
                if not rec.get("resumed") and hdr.get("prefill_s"):
                    rate = hdr["cum_tokens"] / hdr["prefill_s"]
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
        nxt = int(t) + 1
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
