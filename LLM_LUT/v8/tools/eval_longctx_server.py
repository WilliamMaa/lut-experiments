"""Fact-accuracy eval for the v8 compressed-KV vLLM server (docs/28 §8).

Reads docs from a longctx_multi_turn_*.jsonl produced by
tools/gen_longctx_multiturn.py and asks each doc's questions in ONE
multi-turn chat session (document first, then questions sequentially).
Multi-turn is the load-bearing mode: it exercises the compressed cache's
cross-turn recall exactly like the concurrency benchmark does.

Scoring: a question is correct iff every ground-truth string appears in
the model's answer (AND semantics, same as the old harness).

Run against BOTH servers for the 对拍 (docs/28):
  - compressed: V8_COMPRESS_SLOTS=512 python -m vllm_plugin.serve ... --port 18002
  - full-KV baseline: plain `vllm serve` (no plugin) on another port
Red line (docs/28): 64k single-request fact acc must not fall
significantly below 0.734.

Usage (remote, from LLM_LUT/v8):
    python tools/eval_longctx_server.py \
        --base-url http://localhost:18002 \
        --model /home/u/downloads/models/Qwen3.6-35B-A3B \
        --data data/longctx_multi_turn_65536.jsonl \
        --out results/eval_64k_compressed.json

Stdlib only. Exit code 0 iff overall fact acc > 0.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SCRIPT_VERSION = "v1"


def chat(base_url, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        # reasoning models must answer directly, not burn the budget
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="longctx_multi_turn jsonl")
    ap.add_argument("--out", default=None, help="write per-question JSON here")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all docs")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-request timeout seconds")
    args = ap.parse_args()
    global REQ_TIMEOUT
    REQ_TIMEOUT = args.timeout

    print(f"[eval_longctx] script {SCRIPT_VERSION}, data={args.data}")
    docs = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if args.max_docs > 0:
        docs = docs[:args.max_docs]

    per_qtype = {}
    n_correct = n_total = 0
    results = []

    for di, doc in enumerate(docs):
        doc_t0 = time.time()
        messages = [{"role": "user", "content": doc["document"]}]
        doc_correct = 0
        try:
            # anchor turn: model reads the document
            _, dt = chat(args.base_url, args.model, messages, 8)
            messages.append({"role": "assistant", "content": "好的，我已读完。"})
        except Exception as e:
            print(f"[doc {di}] ANCHOR FAIL: {e!r}")
            results.append({"doc_index": di, "error": repr(e)})
            continue

        for qi, (q, gts, qt) in enumerate(zip(doc["questions"],
                                               doc["answers"],
                                               doc["qtype"])):
            messages.append({"role": "user", "content": q})
            try:
                ans, dt = chat(args.base_url, args.model, messages,
                               args.max_tokens)
            except Exception as e:
                ans, dt = f"<ERROR {e!r}>", 0.0
            ok = all(gt in ans for gt in gts)
            n_total += 1
            doc_correct += ok
            n_correct += ok
            per_qtype.setdefault(qt, [0, 0])
            per_qtype[qt][0] += ok
            per_qtype[qt][1] += 1
            results.append({"doc_index": di, "q_index": qi, "qtype": qt,
                            "question": q, "gt": gts, "answer": ans,
                            "correct": ok, "seconds": round(dt, 2)})
            messages.append({"role": "assistant", "content": ans})

        nq = len(doc["questions"])
        print(f"[doc {di}] acc={doc_correct}/{nq} "
              f"({time.time() - doc_t0:.0f}s, "
              f"~{doc['meta']['actual_tokens']:.0f} tokens)")

    acc = n_correct / max(n_total, 1)
    print()
    print(f"OVERALL fact_acc = {acc:.4f}  ({n_correct}/{n_total})")
    for qt, (c, t) in sorted(per_qtype.items()):
        print(f"  {qt:>18s}: {c}/{t} = {c / max(t, 1):.4f}")

    if args.out:
        summary = {"fact_acc": acc, "n_correct": n_correct, "n_total": n_total,
                   "per_qtype": {k: {"correct": v[0], "total": v[1]}
                                 for k, v in per_qtype.items()},
                   "base_url": args.base_url, "data": args.data,
                   "results": results}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")

    sys.exit(0 if n_total > 0 else 1)


REQ_TIMEOUT = 3600

if __name__ == "__main__":
    main()
