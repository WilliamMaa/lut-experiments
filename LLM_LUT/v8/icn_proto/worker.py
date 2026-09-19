#!/usr/bin/env python3
"""Worker process: one model replica, serves scheduler commands.

Lifecycle of one assigned turn:
  1. scheduler sends `assign` {session, turn, prefill_ids, name,
     resume_from, repr, decode_steps}.
  2. if resume_from is set, worker injects its resident object for that name
     (cache continuity); otherwise it starts a fresh cache = full recompute.
  3. worker prefills the new tokens, runs `decode_steps` decode steps, then
     extracts the post-turn object, keeps it resident, and reports metrics.
  4. scheduler may `fetch` an object (bytes) for cross-worker transfer, or
     `deliver` one to this worker.

Resident objects are the per-session latest cache state; workers report
their resident name sets so the scheduler can make placement decisions.

Run (cards are fixed by the launcher via CUDA_VISIBLE_DEVICES):
    python -m icn_proto.worker --scheduler tcp://127.0.0.1:5570 \
        --model-path ... --device balanced_low_0 --repr m_sp4
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zmq

from icn_proto import msg
from icn_proto.kvname import KVName, Repr
from icn_proto.kvcodec import (
    extract_object, inject_object, dumps, loads, place_cache,
)
from icn_proto.presets import cache_factory


class Worker:
    def __init__(self, args):
        self.args = args
        from common.utils import load_model_and_tokenizer
        self.model, self.tokenizer, dev = load_model_and_tokenizer(
            args.model_path, torch_dtype=args.dtype,
            device_map=args.device if not args.device.startswith("cuda:")
            else None,
            device=args.device if args.device.startswith("cuda:") else "cuda:0",
        )
        self.device = str(dev)
        self.config = self.model.config
        self._factories = {}          # repr_name -> make_cache
        self._install_done = False
        self._make_cache(args.repr)   # eager: validates the default path
        self.resident = {}            # name_str -> KVObject (latest per session)
        self.busy = False

    def _make_cache(self, repr_name):
        """Per-representation cache factory, built lazily.

        Only one heavy-hitter patch is ever installed per process (the
        worker's --repr): the stash wrapper is install-idempotent but swaps
        banks on reinstall, so m_sp4 and k8v8 never mix inside one worker.
        bf16 mixes freely (its install is a no-op)."""
        if repr_name not in self._factories:
            make, install, _ = cache_factory(
                repr_name, config=self.config, device=self.device)
            if not self._install_done and repr_name != "bf16":
                install(self.model)
                self._install_done = True
            self._factories[repr_name] = make
        return self._factories[repr_name]()

    # ---- turn execution -------------------------------------------------

    @torch.inference_mode()
    def run_turn(self, session, turn, prefill_ids, resume_from, name,
                 decode_steps, repr_name=None):
        repr_name = repr_name or self.args.repr
        t0 = time.time()
        cache = self._make_cache(repr_name)
        resumed = False
        if resume_from and resume_from in self.resident:
            inject_object(cache, self.resident[resume_from])
            place_cache(cache, self.model)
            resumed = True
        out = None
        prefill_s = 0.0
        if prefill_ids:
            ids = torch.tensor([prefill_ids], dtype=torch.long)
            t1 = time.time()
            out = self.model(input_ids=ids.to(self.device),
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            prefill_s = time.time() - t1
        elif decode_steps:
            # zero-prefill turns are only legal without decode (doc-reuse);
            # question turns must always prefill something to seed logits.
            raise RuntimeError("zero prefill with decode_steps > 0")
        t2 = time.time()
        decoded = []
        for _ in range(decode_steps):
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded.append(int(nxt))
            out = self.model(input_ids=nxt.to(self.device),
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        decode_s = time.time() - t2
        obj = extract_object(cache, KVName.parse(name))
        # eviction: v1 keeps every resident object (allocation-scale
        # footprints fit HBM; eviction policy is a v2 question tied to
        # the memory-pressure cost term, see 03-icn-kv-principle §4.2).
        self.resident[str(obj.name)] = obj
        return {
            "resumed": resumed,
            "repr": repr_name,
            "prefill_tokens": len(prefill_ids),
            "cum_tokens": KVName.parse(name).span_end,
            "prefill_s": round(prefill_s, 4),
            "decode_s": round(decode_s, 4),
            "queue_s": round(time.time() - t0, 4),
            "obj_bytes": obj.nbytes(),
            "resident_bytes": sum(o.nbytes() for o in self.resident.values()),
            "decoded_ids": decoded,
        }

    # ---- command loop ---------------------------------------------------

    def serve(self):
        ctx = zmq.Context()
        sock = msg.dealer(ctx, self.args.scheduler,
                          identity=str(self.args.worker_id))
        msg.send(sock, {"type": "hello", "worker_id": str(self.args.worker_id),
                        "resident": sorted(self.resident)})
        while True:
            _, hdr, payload = msg.recv(sock)
            mtype = hdr.get("type")
            if mtype == "assign":
                self.busy = True
                try:
                    metrics = self.run_turn(
                        hdr["session"], hdr["turn"], hdr["prefill_ids"],
                        hdr.get("resume_from"), hdr["name"],
                        hdr.get("decode_steps", 0), hdr.get("repr"))
                    metrics["type"] = "result"
                    metrics["request_id"] = hdr["request_id"]
                    metrics["ok"] = True
                    msg.send(sock, metrics)
                except Exception as e:  # report, don't die
                    msg.send(sock, {"type": "result",
                                    "request_id": hdr["request_id"],
                                    "ok": False, "error": repr(e)})
                finally:
                    self.busy = False
            elif mtype == "fetch":
                name = hdr["name"]
                if name not in self.resident:
                    msg.send(sock, {"type": "fetched", "name": name,
                                    "ok": False})
                else:
                    msg.send(sock, {"type": "fetched", "name": name,
                                    "ok": True}, dumps(self.resident[name]))
            elif mtype == "deliver":
                obj = loads(payload)
                self.resident[str(obj.name)] = obj
                msg.send(sock, {"type": "delivered", "name": str(obj.name),
                                "ok": True})
            elif mtype == "status":
                msg.send(sock, {"type": "status", "busy": self.busy,
                                "resident": sorted(self.resident)})
            elif mtype == "shutdown":
                msg.send(sock, {"type": "bye"})
                break
        sock.close()
        ctx.term()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="balanced_low_0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--repr", default="m_sp4",
                    choices=["bf16", "m_sp4", "k8v8"])
    ap.add_argument("--worker-id", default=os.environ.get("CUDA_VISIBLE_DEVICES",
                                                          "w"))
    args = ap.parse_args()
    Worker(args).serve()


if __name__ == "__main__":
    main()
