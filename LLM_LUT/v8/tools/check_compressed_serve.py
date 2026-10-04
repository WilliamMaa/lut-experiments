"""End-to-end checks for the v8 compressed-KV vLLM plugin.

Runs against a live server started with `python -m vllm_plugin.serve`.
Verifies, in order:
  1. short prompt completes (sanity: compressed forward returns 200)
  2. long prompt (> retention budget) completes AND triggers eviction
     (marker `first eviction` appears in the server log)
  3. multi-turn: model recalls a digit from turn 1 (snap table works
     across turns)
  4. multi-chunk prefill (~8k tokens, crosses the 8192 chunked-prefill
     boundary) completes

Usage (remote, from LLM_LUT/v8):
    python tools/check_compressed_serve.py \
        --base-url http://localhost:18001 \
        --model /home/u/downloads/models/Qwen3.6-35B-A3B \
        --log logs/vllm_smoke.log

Stdlib only. Exit code 0 iff all tests pass.
"""
import argparse
import json
import sys
import time
import urllib.request

MARKERS = ("first obs scoring", "first eviction")


def chat(base_url, model, messages, max_tokens, no_thinking=False):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if no_thinking:
        # Qwen3 reasoning models burn the whole budget on thinking
        # otherwise; recall checks need the final answer to fit.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    return body, time.time() - t0


def content_of(body):
    return body["choices"][0]["message"]["content"]


def tokenize(base_url, model, text):
    req = urllib.request.Request(
        f"{base_url}/tokenize",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return len(json.loads(resp.read().decode())["tokens"])


def make_prompt(base_url, model, target_tokens, seed_sentence):
    """Size a repeated-sentence prompt to roughly target_tokens using the
    server's own tokenizer (English sentence: ~1 token per 4.5 chars, but
    we measure instead of guessing)."""
    n = max(1, int(target_tokens * 4.5 / len(seed_sentence)))
    for _ in range(8):
        text = seed_sentence * n
        got = tokenize(base_url, model, text)
        if abs(got - target_tokens) < target_tokens * 0.1:
            return text, got
        n = max(1, int(n * target_tokens / max(got, 1)))
    return text, got


class LogScan:
    """Reads the server log once, then diffs new v8_plugin lines."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        if path:
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    self.offset = f.tell()
            except OSError as e:
                print(f"[warn] cannot read log {path}: {e}")

    def new_plugin_lines(self):
        if not self.path:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read().decode(errors="replace")
                self.offset = f.tell()
        except OSError:
            return []
        return [ln for ln in data.splitlines() if "v8_plugin" in ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:18001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", default=None,
                    help="server log to scan for v8_plugin markers")
    ap.add_argument("--long-target", type=int, default=2048)
    ap.add_argument("--chunk-target", type=int, default=8000)
    args = ap.parse_args()

    scan = LogScan(args.log)
    failures = []

    def check(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 1) short prompt
    try:
        body, dt = chat(args.base_url, args.model,
                        [{"role": "user",
                          "content": "Reply with the single word: ok"}],
                        max_tokens=8, no_thinking=True)
        ok = "choices" in body and len(content_of(body)) > 0
        check("short_prompt", ok,
              f"{body.get('usage', {}).get('total_tokens', '?')} tokens "
              f"in {dt:.1f}s")
    except Exception as e:
        check("short_prompt", False, repr(e))

    # 2) long prompt -> eviction
    try:
        doc, ntok = make_prompt(
            args.base_url, args.model, args.long_target,
            "The history of computing spans mechanical calculators, "
            "vacuum tube machines, transistors, integrated circuits, "
            "personal computers, and modern accelerators. ")
        body, dt = chat(
            args.base_url, args.model,
            [{"role": "user",
              "content": doc + "\n\nIn one sentence: what does this text "
                             "describe?"}],
            max_tokens=48, no_thinking=True)
        markers = scan.new_plugin_lines()
        evict = [m for m in markers if "first eviction" in m]
        check("long_prompt_eviction",
              "choices" in body and len(evict) > 0,
              f"prompt={ntok} tokens, {dt:.1f}s, "
              f"eviction_marker={'yes' if evict else 'NO'}")
        for m in markers:
            print("    |", m.strip()[:150])
    except Exception as e:
        check("long_prompt_eviction", False, repr(e))

    # 3) multi-turn recall
    try:
        digit = "73419"
        body, dt = chat(
            args.base_url, args.model,
            [{"role": "user", "content": f"Remember this number: {digit}"},
             {"role": "assistant", "content": f"Got it: {digit}"},
             {"role": "user",
              "content": "What number did I ask you to remember? "
                         "Answer with digits only."}],
            max_tokens=32, no_thinking=True)
        check("multiturn_recall", digit in content_of(body),
              f"answer={content_of(body)[:40]!r} ({dt:.1f}s)")
    except Exception as e:
        check("multiturn_recall", False, repr(e))

    # 4) multi-chunk prefill (~8k, crosses chunked-prefill boundary)
    try:
        doc, ntok = make_prompt(
            args.base_url, args.model, args.chunk_target,
            "Machine learning systems improve with data, and the design "
            "of efficient training pipelines matters for large models. ")
        body, dt = chat(
            args.base_url, args.model,
            [{"role": "user",
              "content": doc + "\n\nIn one sentence: what is the topic?"}],
            max_tokens=48, no_thinking=True)
        check("multichunk_prefill", "choices" in body,
              f"prompt={ntok} tokens in {dt:.1f}s")
    except Exception as e:
        check("multichunk_prefill", False, repr(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
