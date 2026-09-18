#!/usr/bin/env python3
"""Step 1 acceptance: extract -> serialize -> inject must be lossless.

For each representation (bf16 / m_sp4 / k8v8):

  inject run: after turn1 (prefill + one decode step to trigger eviction on
              the compact reprs), the cache state is extracted and injected
              DIRECTLY into a fresh cache; turn2 continues from it.
  wire run:   same, but the object goes through dumps() -> bytes -> loads()
              before injection.

Compared logits: turn2 prefill last position and turn2 first decode step.
Both runs share the same transfer semantics (fresh cache, fresh per-turn
scores from the score bank), so any drift is pure codec loss — exactly what
this test is gatekeeping. The never-moved continuous run is reported as
informational context, not asserted: for compact representations its turn-2
eviction reads the stale turn-1 score snapshot that lives on the old cache's
layer objects (a long-lived-cache quirk, see HeavyHitterCache._position_scores),
while a transferred session correctly uses the fresh turn-2 scores — the two
are legitimately allowed to differ.

Usage (on the remote box, e.g. GPU 5 while it is free):
    cd /data/mamingyu/v8
    python icn_proto/tests/test_kvcodec_roundtrip.py \
        --model-path /home/u/downloads/models/<small-model> --device cuda:5

Any instruct-model path works; prompts come from data/multi_turn_prompts_v3.jsonl.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from icn_proto.kvname import KVName, Repr
from icn_proto.kvcodec import extract_object, inject_object, dumps, loads
from icn_proto.presets import cache_factory

TRACE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "multi_turn_prompts_v3.jsonl")

_FACTORIES = {}  # repr -> (make_cache, uninstall), installed once per process


def get_factory(repr_name, config, device, model):
    if repr_name not in _FACTORIES:
        make_cache, install, uninstall = cache_factory(
            repr_name, config=config, device=device)
        install(model)
        _FACTORIES[repr_name] = (make_cache, uninstall)
    return _FACTORIES[repr_name][0]


def build_turns(tokenizer, max_doc_chars=4000):
    """turn1 = doc + q1; turn2 continues with q2. Shared prefix = doc."""
    with open(TRACE, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    doc = sample["document"][:max_doc_chars]
    q1, q2 = sample["questions"][0], sample["questions"][1]
    t1 = tokenizer(doc + "\n\n" + q1, return_tensors="pt").input_ids
    t2 = tokenizer("\n\n" + q2, return_tensors="pt").input_ids
    return t1, t2


@torch.inference_mode()
def forward_turn(model, cache, new_tokens, device):
    out = model(input_ids=new_tokens.to(device),
                past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


@torch.inference_mode()
def run_session(model, repr_name, config, device, t1, t2, mode):
    """mode: "continuous" (never move) | "inject" (direct) | "wire" (bytes)."""
    make_cache = get_factory(repr_name, config, device, model)
    cache = make_cache().to(device)
    name = KVName("accept", turn=1, span_start=0, span_end=t1.shape[1],
                  repr=Repr(repr_name))

    logits = {}
    _, cache = forward_turn(model, cache, t1, device)
    d1 = torch.zeros((1, 1), dtype=torch.long, device=device)
    logits["turn1_decode"], cache = forward_turn(model, cache, d1, device)

    if mode != "continuous":
        obj = extract_object(cache, name)
        if mode == "wire":
            obj = loads(dumps(obj))
        fresh = make_cache()
        inject_object(fresh, obj)
        cache = fresh.to(device)

    logits["turn2_prefill"], cache = forward_turn(model, cache, t2, device)
    d2 = torch.zeros((1, 1), dtype=torch.long, device=device)
    logits["turn2_decode"], _ = forward_turn(model, cache, d2, device)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    from common.utils import load_model_and_tokenizer
    torch_dtype = {"bf16": "bfloat16", "fp16": "float16",
                   "fp32": "float32"}[args.dtype]
    if args.device.startswith("cuda:"):
        model, tok, device = load_model_and_tokenizer(
            args.model_path, torch_dtype=torch_dtype, device=args.device)
    else:
        # e.g. "balanced_low_0": fixed multi-card placement (red-line guard
        # against "auto" lives in the helper). Restrict cards beforehand via
        # CUDA_VISIBLE_DEVICES.
        model, tok, device = load_model_and_tokenizer(
            args.model_path, torch_dtype=torch_dtype, device_map=args.device)
    device = str(device)
    config = model.config

    t1, t2 = build_turns(tok)
    print(f"turn1 tokens={t1.shape[1]}  turn2 tokens={t2.shape[1]}")
    if t1.shape[1] <= 128:
        print("WARNING: turn1 <= 128 tokens; compact-repr eviction will not "
              "trigger and the test exercises serialization only.")

    report = {"model": args.model_path, "device": device,
              "turn1_tokens": t1.shape[1], "representations": {}}
    failed = False
    for repr_name in ("bf16", "m_sp4", "k8v8"):
        cont = run_session(model, repr_name, config, args.device, t1, t2, "continuous")
        inj = run_session(model, repr_name, config, args.device, t1, t2, "inject")
        wire = run_session(model, repr_name, config, args.device, t1, t2, "wire")

        codec_diff = {k: (inj[k].float() - wire[k].float()).abs().max().item()
                      for k in inj}
        cont_diff = {k: (cont[k].float() - wire[k].float()).abs().max().item()
                     for k in inj}
        report["representations"][repr_name] = {
            "codec_max_logit_diff": codec_diff,
            "vs_continuous_max_logit_diff": cont_diff,
        }
        ok = max(codec_diff.values()) < 1e-3
        failed |= not ok
        print(f"[{repr_name:5s}] codec={ {k: f'{v:.2e}' for k, v in codec_diff.items()} } "
              f"vs_continuous={ {k: f'{v:.2e}' for k, v in cont_diff.items()} } "
              f"{'OK' if ok else 'FAIL'}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"kvcodec_roundtrip_{os.path.basename(args.model_path.rstrip('/'))}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"report -> {out}")
    if failed:
        sys.exit(1)
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
