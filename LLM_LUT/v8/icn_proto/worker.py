#!/usr/bin/env python3
"""Worker process: one model replica, serves scheduler commands (block chain).

Protocol (docs/icn-defined-addressing/05-request-lifecycle.md v3 §2):
  hello   {worker_id, resident: [block name, ...]}
  status  {resident: [...], busy: bool}                # periodic + on change
  fetch   {names: [block name, ...]}  -> holder replies
          {type: fetched, ok: bool, names: [...]} + torch-save payload
          of the KVBlockObj list (order = names order)
  deliver (payload = torch-save of KVBlockObj list)    # blocks to store
  evict   {names: [...]}                               # drop from residency
          (placement controller's symmetric criterion; the worker's
          following status is the implicit ack)
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
import json
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
        # E2 (10-e2-backing-tier §3): host-DRAM backing tier. Evicted
        # blocks land here instead of vanishing; recall pulls them back
        # into residency. KVBlockObj tensors already live on CPU
        # (kvcodec_blk.extract_blocks), so spill/recall are dict moves —
        # the bytes are REAL DRAM residency either way.
        cap_mb = getattr(args, "spill_mb", 0.0)
        self.spill_enabled = cap_mb != 0
        self.spill_cap = float("inf") if cap_mb < 0 else cap_mb * 1e6
        self.spill = {}             # block name -> KVBlockObj (LRU order)
        self.spill_bytes = 0
        self.spill_stat = {"evicted_bytes": 0, "dropped_bytes": 0,
                           "recall_count": 0, "recall_bytes": 0}

    def _spill_put(self, name, obj):
        """Eviction landing point: move a block into the tier. Capacity
        pressure drops the oldest entries (LRU). Returns True if the
        block is resident in the tier, False if dropped."""
        if not self.spill_enabled:
            return False
        nb = obj.nbytes()
        while self.spill and self.spill_bytes + nb > self.spill_cap:
            old_name, old = next(iter(self.spill.items()))
            del self.spill[old_name]
            self.spill_bytes -= old.nbytes()
            self.spill_stat["dropped_bytes"] += old.nbytes()
        if nb > self.spill_cap:     # a single object larger than the tier
            self.spill_stat["dropped_bytes"] += nb
            return False
        self.spill[name] = obj
        self.spill_bytes += nb
        self.spill_stat["evicted_bytes"] += nb
        return True

    def _recall(self, names):
        """Pull blocks back from the tier into residency. Same shape as
        a deliver landing — no wire traffic, no GPU involvement."""
        recalled = []
        for n in names:
            obj = self.spill.pop(n, None)
            if obj is None:
                continue
            self.resident[n] = obj
            self.spill_bytes -= obj.nbytes()
            self.spill_stat["recall_count"] += 1
            self.spill_stat["recall_bytes"] += obj.nbytes()
            recalled.append(obj.nbytes())
        if recalled:
            print(f"[{self.args.worker_id}] recalled {len(recalled)} "
                  f"blocks from spill "
                  f"({sum(recalled) / 1e6:.1f}MB)", flush=True)
        return recalled

    # ---- setup ----------------------------------------------------------

    def load_model(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        self.config = AutoConfig.from_pretrained(self.args.model_path,
                                                 trust_remote_code=True)
        dtype = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16}.get(self.args.dtype, torch.bfloat16)
        # explicit_even: the INTENT of balanced_low_0 (balance layers across
        # the visible cards, embeddings on card 0) but deterministic.
        # balanced_low_0 shards by CURRENTLY FREE vram, so neighbour
        # tenants reshuffle our layout on every load — and when the pool
        # cards are too full it silently offloads layers to CPU, which
        # broke place_cache and caused the intermittent cpu-vs-cuda cat
        # crashes. Same balancing principle, no ambient noise.
        # report what accelerate's placement will see: free vram per
        # visible card BEFORE loading. If a card cannot hold its half of
        # the model (~35GB for this 35B bf16), the layout degrades
        # (params on meta/cpu) and every resumed turn crashes.
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"  card {i}: {free/1e9:.1f}GB free of "
                      f"{total/1e9:.1f}GB", flush=True)
        if self.args.device == "explicit_even":
            device_map = self._explicit_even_map()
        else:
            device_map = self.args.device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, dtype=dtype, trust_remote_code=True,
            device_map=device_map)
        self.model.eval()
        first = next(self.model.parameters())
        self.device = first.device
        print(f"  first-layer device is {self.device}", flush=True)
        dm = getattr(self.model, "hf_device_map", None)
        if dm:
            hits = [k for k in dm if k.endswith("layers.0")]
            print(f"  device_map: {len(dm)} entries, layers.0 -> {hits}, "
                  f"sample: {list(dm.items())[:2]}", flush=True)
        else:
            print("  device_map: NONE (single-device fallback)", flush=True)
        n_cpu = sum(1 for p in self.model.parameters()
                    if p.device.type == "cpu")
        n_meta = sum(1 for p in self.model.parameters()
                     if p.device.type == "meta")
        if n_cpu or n_meta:
            print(f"  WARN: {n_cpu} params on cpu, {n_meta} on meta after "
                  f"load — visible cards lack free VRAM, layout DEGRADED. "
                  f"Pick emptier --gpu-pool cards.", flush=True)
        else:
            print("  layout check OK: all params on accelerator",
                  flush=True)

    def _explicit_even_map(self):
        n = self._layer_count()
        half = (n + 1) // 2
        m = {"model.embed_tokens": 0,
             "model.norm": 1,
             "lm_head": 1}
        m.update({f"model.layers.{i}": (0 if i < half else 1)
                  for i in range(n)})
        return m

    def _layer_count(self):
        """Ground truth from the checkpoint's weight index (version-
        independent); config attributes only as fallback — Qwen3_5MoeConfig
        (heterogeneous layers, transformers 5.14) has no
        num_hidden_layers."""
        idx_path = os.path.join(self.args.model_path,
                                "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path, encoding="utf-8") as f:
                wm = json.load(f)["weight_map"]
            ns = {int(k.split("model.layers.")[1].split(".")[0])
                  for k in wm if "model.layers." in k}
            if ns:
                return max(ns) + 1
        cfg = self.config
        for probe in (cfg, getattr(cfg, "text_config", None)):
            if probe is None:
                continue
            n = getattr(probe, "num_hidden_layers", None)
            if isinstance(n, int):
                return n
            lt = getattr(probe, "layer_types", None)
            if lt:
                return len(lt)
        raise RuntimeError(
            f"cannot determine layer count for {type(cfg).__name__}")

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
            self._recall([n for n in resume_names if n in self.spill])
            blocks = []
            missing = []
            for n in resume_names:
                obj = self.resident.get(n)
                if obj is None:
                    missing.append(n)
                else:
                    blocks.append(obj)
            if missing:
                # The scheduler's resident/spilled view was stale when it
                # planned this resume (spill-tier LRU drop not yet
                # reflected). Name EVERY missing block so the scheduler
                # can purge its view and re-plan the turn instead of
                # hard-failing it.
                raise RuntimeError("STALE_RESUME: " + ",".join(missing))
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
            if out.past_key_values is not cache:
                # hybrid models may wrap the passed cache into their own
                # class inside forward — if so, our inject/place targeted
                # a different object than the model actually reads
                print(f"[{wid}] NOTE model replaced cache object in "
                      f"forward: {type(cache).__name__} -> "
                      f"{type(out.past_key_values).__name__}", flush=True)
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
        msg.send(sock, dict({"type": "hello",
                             "worker_id": self.args.worker_id},
                            **self._status_hdr()))
        while True:
            _, hdr, payload = msg.recv(sock)
            mtype = hdr.get("type")
            if mtype == "shutdown":
                break
            if mtype == "fetch":
                names = hdr.get("names", [])
                # a spill holder recalls before sending — residency
                # control may route copies through the tier (E2)
                self._recall([n for n in names if n in self.spill])
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
                # echo repl so the scheduler can attribute this ack to a
                # controller replication rather than a demand fetch
                msg.send(sock, {"type": "delivered",
                                "names": [str(o.name) for o in objs],
                                "repl": hdr.get("repl")})
                self.report_status(sock)
                continue
            if mtype == "evict":
                gone, spilled_b, dropped_b = [], 0, 0
                for n in hdr.get("names", []):
                    obj = self.resident.pop(n, None)
                    if obj is None:
                        continue
                    gone.append(obj)
                    if self._spill_put(n, obj):
                        spilled_b += obj.nbytes()
                    else:
                        dropped_b += obj.nbytes()
                print(f"[{self.args.worker_id}] evicted {len(gone)} blocks "
                      f"({sum(o.nbytes() for o in gone) / 1e6:.1f}MB; "
                      f"spill +{spilled_b / 1e6:.1f}MB, "
                      f"dropped {dropped_b / 1e6:.1f}MB)", flush=True)
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

    def _status_hdr(self):
        # tips = resident tips + spilled tips: a spilled tip still
        # carries its GDN checkpoint and stays resumable (E2), so the
        # scheduler must keep seeing it as a match candidate
        return {
            "resident": list(self.resident),
            "tips": ([n for n, o in self.resident.items()
                      if o.linear_checkpoint]
                     + [n for n, o in self.spill.items()
                        if o.linear_checkpoint]),
            "resident_bytes": sum(o.nbytes()
                                  for o in self.resident.values()),
            "spilled": list(self.spill),
            "spill_bytes": self.spill_bytes,
            "spill": dict(self.spill_stat),
        }

    def report_status(self, sock):
        # no busy field: the scheduler owns the busy flag (set on
        # assign/fetch, cleared on result). A worker-reported busy would
        # be stale by the time it arrives and caused double-booking.
        msg.send(sock, dict({"type": "status"}, **self._status_hdr()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="explicit_even",
                    help="explicit_even (default): deterministic half/half "
                         "split over visible cards — same intent as "
                         "balanced_low_0 without tenant-dependent "
                         "reshuffling. balanced_low_0 kept for reference.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--repr", default="bf16")
    ap.add_argument("--worker-id", default="w0")
    ap.add_argument("--spill-mb", type=float, default=0.0,
                    help="E2 backing tier: host-DRAM spill capacity in MB; "
                         "0 (default) = off (evict drops the block), "
                         "-1 = unlimited; a full tier drops LRU-oldest")
    args = ap.parse_args()
    w = Worker(args)
    w.load_model()
    w.serve()


if __name__ == "__main__":
    main()
