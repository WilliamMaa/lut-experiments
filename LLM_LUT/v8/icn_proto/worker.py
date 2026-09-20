#!/usr/bin/env python3
"""Worker process: one model replica, serves scheduler commands (block chain).

Protocol (docs/icn-defined-addressing/05-request-lifecycle.md v3 §2):
  hello   {worker_id, resident: [block name, ...]}
  status  {resident: [...], busy: bool}                # periodic + on change
  fetch   {names: [block name, ...]}  -> holder replies
          {type: fetched, ok: bool, names: [...]} + torch-save payload
          of the KVBlockObj list (order = names order)
  deliver (payload = torch-save of KVBlockObj list)    # blocks to store
  assign  {request_id, session, turn,
           resume_names: [...],    # contiguous block chain [0, E) to inject
           new_block_names: [...], # chain blocks the worker should publish
                                   # (worker extracts the ones it does not
                                   # already hold resident)
           prefill_ids, decode_steps, repr}
          -> result {ok, published: [{name, bytes}], prefill_s,
                     prefill_tokens, decoded_ids, resident_bytes, error}

Resident state is an UNBOUNDED dict of blocks (v1: allocation-scale
footprints fit HBM/RAM; eviction is the slow-path placement controller's
job, 05 §3). GDN linear state travels as a checkpoint on tip blocks.

Run (cards are fixed by the launcher via CUDA_VISIBLE_DEVICES):
    python -m icn_proto.worker --scheduler tcp://127.0.0.1:5570 \
        --model-path ... --device balanced_low_0 --repr bf16
"""

import argparse
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icn_proto import msg
from icn_proto.blkchain import BlockName
from icn_proto.kvcodec_blk import KVBlockObj, extract_blocks, inject_blocks
from icn_proto.presets import cache_factory


class Worker:
    def __init__(self, args):
        self.args = args
        self.device = "cpu"
        self.model = None
        self.config = None
        self.resident = {}          # block name (str) -> KVBlockObj
        self._factories = {}
        self._install_done = False

    # ---- setup ----------------------------------------------------------

    def load_model(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        self.config = AutoConfig.from_pretrained(self.args.model_path,
                                                 trust_remote_code=True)
        dtype = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16}.get(self.args.dtype, torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, dtype=dtype, trust_remote_code=True,
            device_map=self.args.device)
        self.model.eval()
        first = next(self.model.parameters())
        self.device = first.device
        print(f"  first-layer device is {self.device}", flush=True)

    def _make_cache(self, repr_name):
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
    def run_turn(self, session, turn, resume_names, new_block_names,
                 prefill_ids, decode_steps, repr_name=None):
        repr_name = repr_name or self.args.repr
        t0 = time.time()
        wid = self.args.worker_id
        print(f"[{wid}] turn {session}:{turn} start "
              f"resume={len(resume_names)} publish={len(new_block_names)} "
              f"prefill={len(prefill_ids)} decode={decode_steps}", flush=True)
        cache = self._make_cache(repr_name)
        if resume_names:
            blocks = []
            for n in resume_names:
                obj = self.resident.get(n)
                if obj is None:
                    raise RuntimeError(f"resume block not resident: {n}")
                blocks.append(obj)
            inject_blocks(cache, blocks)
            from icn_proto.kvcodec import place_cache
            place_cache(cache, self.model)
            print(f"[{wid}] turn {session}:{turn} injected "
                  f"{len(blocks)} blocks "
                  f"({time.time() - t0:.2f}s)", flush=True)
        out = None
        prefill_s = 0.0
        if prefill_ids:
            ids = torch.tensor([prefill_ids], dtype=torch.long)
            t1 = time.time()
            out = self.model(input_ids=ids.to(self.device),
                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            prefill_s = time.time() - t1
            print(f"[{wid}] turn {session}:{turn} prefilled "
                  f"{len(prefill_ids)} tok in {prefill_s:.2f}s", flush=True)
        elif decode_steps:
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
        # publish: extract the chain blocks this worker does not yet hold,
        # ALWAYS ending with the chain tip (last of new_block_names): the
        # tip carries this turn's GDN checkpoint (extract_blocks attaches
        # it to objs[-1]), so it is re-extracted even when already resident
        # — e.g. delivered earlier as a fetch extension.
        names = list(dict.fromkeys(new_block_names))  # dedupe, order kept
        publish = [BlockName.parse(n) for n in names]
        objs = extract_blocks(cache, publish)
        for obj in objs:
            self.resident[str(obj.name)] = obj
        print(f"[{wid}] turn {session}:{turn} published "
              f"{len(objs)} blocks in {time.time() - t2 - decode_s:.2f}s "
              f"(total {time.time() - t0:.2f}s)", flush=True)
        return {
            "resumed": bool(resume_names),
            "repr": repr_name,
            "prefill_tokens": len(prefill_ids),
            "prefill_s": round(prefill_s, 4),
            "decode_s": round(decode_s, 4),
            "queue_s": round(time.time() - t0, 4),
            "published": [{"name": str(o.name), "bytes": o.nbytes()}
                          for o in objs],
            "resident_bytes": sum(o.nbytes() for o in self.resident.values()),
            "decoded_ids": decoded,
        }

    # ---- command loop ---------------------------------------------------

    def serve(self):
        import zmq
        ctx = zmq.Context()
        sock = msg.dealer(ctx, self.args.scheduler,
                          identity=self.args.worker_id)
        msg.send(sock, {"type": "hello", "worker_id": self.args.worker_id,
                        "resident": list(self.resident),
                        "tips": [n for n, o in self.resident.items()
                                 if o.linear_checkpoint]})
        while True:
            _, hdr, payload = msg.recv(sock)
            mtype = hdr.get("type")
            if mtype == "shutdown":
                break
            if mtype == "fetch":
                names = hdr.get("names", [])
                objs, missing = [], []
                for n in names:
                    obj = self.resident.get(n)
                    (objs if obj is not None else missing).append(
                        obj if obj is not None else n)
                if missing:
                    msg.send(sock, {"type": "fetched", "ok": False,
                                    "missing": missing})
                else:
                    buf = io.BytesIO()
                    torch.save(objs, buf)
                    msg.send(sock, {"type": "fetched", "ok": True,
                                    "names": names}, payload=buf.getvalue())
                continue
            if mtype == "deliver":
                objs = torch.load(io.BytesIO(payload), weights_only=False)
                for obj in objs:
                    self.resident[str(obj.name)] = obj
                print(f"[{self.args.worker_id}] delivered "
                      f"{len(objs)} blocks "
                      f"({sum(o.nbytes() for o in objs) / 1e6:.1f}MB)",
                      flush=True)
                msg.send(sock, {"type": "delivered",
                                "names": [str(o.name) for o in objs]})
                self.report_status(sock)
                continue
            if mtype == "assign":
                hdr_out = {"type": "result",
                           "request_id": hdr["request_id"], "ok": True}
                try:
                    hdr_out.update(self.run_turn(
                        hdr.get("session"), hdr.get("turn"),
                        hdr.get("resume_names", []),
                        hdr.get("new_block_names", []),
                        hdr.get("prefill_ids", []),
                        hdr.get("decode_steps", 0),
                        hdr.get("repr")))
                except Exception as exc:  # noqa: BLE001 - report, don't die
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[{self.args.worker_id}] turn "
                          f"{hdr.get('session')}:{hdr.get('turn')} FAILED\n"
                          f"{tb}", flush=True)
                    hdr_out.update({"ok": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": tb})
                msg.send(sock, hdr_out)
                self.report_status(sock)
                continue

    def report_status(self, sock):
        # no busy field: the scheduler owns the busy flag (set on
        # assign/fetch, cleared on result). A worker-reported busy would
        # be stale by the time it arrives and caused double-booking.
        msg.send(sock, {
            "type": "status",
            "resident": list(self.resident),
            "tips": [n for n, o in self.resident.items()
                     if o.linear_checkpoint]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="balanced_low_0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--repr", default="bf16")
    ap.add_argument("--worker-id", default="w0")
    args = ap.parse_args()
    w = Worker(args)
    w.load_model()
    w.serve()


if __name__ == "__main__":
    main()
