"""Concurrency benchmark for the v8 compressed-KV vLLM server (docs/28 §8).

Fixed work, varying concurrency: runs ALL docs of a
longctx_multi_turn_*.jsonl as simultaneous multi-turn chat sessions with a
thread pool of --concurrency workers (each worker drives one doc's full
multi-turn session, anchor + questions sequentially — sessions never share
state client-side). Total work is constant across N, so wall time / session
duration / throughput show how the compressed server scales under
concurrency (KV pressure, scheduling, eviction churn).

Every session prepends a unique nonce to its document copy so identical
docs across workers cannot hit vLLM prefix cache — each request pays the
full prefill (verified pattern: without the nonce, back-to-back identical
docs return in ~1s).

Scoring identical to eval_longctx_server.py: a question is correct iff
every ground-truth string appears in the answer.

Usage (remote, from LLM_LUT/v8, server already running):
    python tools/bench_concurrency.py \
        --base-url http://localhost:18002 \
        --model /home/u/downloads/models/Qwen3.6-35B-A3B \
        --data data/longctx_multi_turn_65536.jsonl \
        --concurrency 8 --out results/bench_c8_slots1024.json

Stdlib only. Exit code 0 iff at least one question was scored.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SCRIPT_VERSION = "v1"

_print_lock = threading.Lock()


def chat(base_url, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
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


def run_session(base_url, model, doc, sid, max_tokens):
    """One full multi-turn session. Returns per-session record."""
    # Unique nonce per session: busts prefix cache between workers that
    # recycle the same doc, and mimics distinct user requests.
    nonce = f"（会话编号 {sid:04d}，请忽略此行。）"
    messages = [{"role": "user",
                 "content": nonce + "\n" + doc["document"]}]
    t0 = time.time()
    turn_seconds = []
    correct = 0
    error = None
    try:
        _, dt = chat(base_url, model, messages, 8)
        turn_seconds.append(dt)
        messages.append({"role": "assistant", "content": "好的，我已读完。"})
        for q, gts in zip(doc["questions"], doc["answers"]):
            messages.append({"role": "user", "content": q})
            try:
                ans, dt = chat(base_url, model, messages, max_tokens)
            except Exception as e:
                ans, dt = f"<ERROR {e!r}>", 0.0
            turn_seconds.append(dt)
            correct += all(gt in ans for gt in gts)
            messages.append({"role": "assistant", "content": ans})
    except Exception as e:
        error = repr(e)
    dur = time.time() - t0
    with _print_lock:
        tag = f" err={error}" if error else ""
        print(f"[sess {sid:03d}] correct={correct}/"
              f"{len(doc['questions'])} dur={dur:.0f}s "
              f"turns={len(turn_seconds)}{tag}", flush=True)
    return {"sid": sid, "doc_index": doc["doc_index"],
            "correct": correct, "n_questions": len(doc["questions"]),
            "seconds": round(dur, 2),
            "turn_seconds": [round(t, 2) for t in turn_seconds],
            "error": error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="longctx_multi_turn jsonl")
    ap.add_argument("--concurrency", type=int, required=True,
                    help="parallel sessions (workers)")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-request timeout seconds")
    args = ap.parse_args()
    global REQ_TIMEOUT
    REQ_TIMEOUT = args.timeout

    print(f"[bench_concurrency] script {SCRIPT_VERSION}, N={args.concurrency}, "
          f"data={args.data}", flush=True)
    docs = []
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["doc_index"] = i
                docs.append(d)
    # sessions = one per doc, recycled round-robin if N > len(docs)
    sessions = [(i, docs[i % len(docs)]) for i in range(len(docs))]

    wall_t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        records = list(ex.map(
            lambda sd: run_session(args.base_url, args.model, sd[1],
                                   sd[0], args.max_tokens),
            sessions))
    wall = time.time() - wall_t0

    n_q = sum(r["n_questions"] for r in records)
    n_c = sum(r["correct"] for r in records)
    durs = [r["seconds"] for r in records if not r["error"]]
    errs = [r for r in records if r["error"]]
    acc = n_c / max(n_q, 1)

    print()
    print(f"N={args.concurrency}  wall={wall:.0f}s  "
          f"sessions={len(records)} (errors={len(errs)})")
    print(f"fact_acc = {acc:.4f}  ({n_c}/{n_q})")
    if durs:
        print(f"session dur: mean={statistics.mean(durs):.0f}s "
              f"median={statistics.median(durs):.0f}s "
              f"max={max(durs):.0f}s "
              f"min={min(durs):.0f}s")
        print(f"throughput: {n_q / wall:.3f} questions/s, "
              f"{len(records) / wall * 3600:.1f} sessions/h")

    if args.out:
        summary = {
            "concurrency": args.concurrency,
            "wall_seconds": round(wall, 2),
            "n_sessions": len(records),
            "n_errors": len(errs),
            "fact_acc": acc,
            "n_correct": n_c, "n_questions": n_q,
            "session_seconds": durs,
            "base_url": args.base_url, "data": args.data,
            "records": records,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")

    sys.exit(0 if n_q > 0 else 1)


REQ_TIMEOUT = 3600

if __name__ == "__main__":
    main()
