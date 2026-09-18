#!/usr/bin/env python3
"""Worker process: one model replica, serves scheduler commands.

Lifecycle of one assigned turn:
  1. scheduler sends `assign` {session, turn, new_token_ids, resume_from}.
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
        self.make_cache, self.install, _ = cache_factory(
            args.repr, config=self.config, device=self.device)
        self.install(self.model)
        self.resident = {}  # name_str -> KVObject (this worker's HBM cache)
        self.busy = False

    # ---- turn execution -------------------------------------------------

    @torch.inference_mode()
    def run_turn(self, session, turn, new_ids, resume_from, decode_steps,
                 hdr_cum_tokens=None):
        t0 = time.time()
        cache = self.make_cache()
        resumed = False
        if resume_from and resume_from in self.resident:
            inject_object(cache, self.resident[resume_from])
            place_cache(cache, self.model)
            resumed = True
        ids = torch.tensor([new_ids], dtype=torch.long)
        t1 = time.time()
        out = self.model(input_ids=ids.to(self.device),
                         past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        prefill_s = time.time() - t1
        t2 = time.time()
        for _ in range(decode_steps):
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = self.model(input_ids=nxt.to(self.device),
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        decode_s = time.time() - t2
        cum = int(hdr_cum_tokens if hdr_cum_tokens else ids.shape[1])
        name = KVName(session, turn, 0, cum, Repr(self.args.repr))
        obj = extract_object(cache, name)
        self.resident[str(name)] = obj
        return {
            "resumed": resumed,
            "prefill_tokens": int(ids.shape[1]),
            "cum_tokens": cum,
            "prefill_s": round(prefill_s, 4),
            "decode_s": round(decode_s, 4),
            "queue_s": round(t1 - t0, 4),
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
                        hdr["session"], hdr["turn"], hdr["new_token_ids"],
                        hdr.get("resume_from"), hdr.get("decode_steps", 1),
                        hdr.get("cum_tokens"))
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
