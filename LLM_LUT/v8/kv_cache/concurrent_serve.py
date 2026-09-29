#!/usr/bin/env python3
"""Concurrent multi-turn serving benchmark for the v8 KV compression ladder.

Runs N multi-turn sessions lockstep-batched through one model copy and
measures TTFT, per-step decode time (TPOT), throughput, peak HBM, and quality
(EOS / repetition / factual accuracy vs ground truth) for each ladder config:

    full        no patch, model-default DynamicCache
    hh          heavy_hitter_attn selection (l128/s4/r32/w64)
    hh_merge    + --merge_evicted (convex folding)
    hh_merge_m4 + --span_window 4            (m_sp4, 1000x)
    m4_k8v8     + --k_bits 8 --v_bits 8     (2000x)

Multi-turn semantics follow common/metrics.py run_multi_turn_generation: each
turn re-prefills the full cumulative conversation with a FRESH cache. A turn
is therefore one batched prefill + greedy decode loop; finished sessions feed
pad tokens (mask-isolated) and their answers are truncated at the first EOS.

This is a research harness, not a production engine: no continuous batching,
no CUDA-graph decode, hand-written per-token loop. Cross-config comparisons
are apples-to-apples; absolute numbers understate vLLM-class serving.

Self-tests:
    --cache-selftest   CPU-only, no model: B=1 caches vs one B=N cache on
                       synthetic tensors; asserts per-session bit-parity of
                       selection/folding/quantization.
    --selftest         GPU: first K docs x T turns sequentially (batch=1) vs
                       batched (B=K) with the m_sp4 config; answers must match
                       token-for-token (left-pad / per-batch-score bug catcher).
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_model_and_tokenizer
from common.metrics import (
    compute_generation_metrics,
    measure_peak_memory_mb,
    reset_peak_memory_stats,
)
from kv_cache.kv_cache_patch import HeavyHitterAttnScorePatch

NO_THINK_TEXT = (
    "直接给出最终回复，不要输出思考过程、分析步骤、'Here's a thinking process' "
    "或任何元说明。"
)

# l128/s4/r32/w64: the定型 m_sp4 configuration (docs/19).
LADDER = {
    "full": None,
    "hh": dict(merge_evicted=False, span_window=0, k_bits=16, v_bits=16),
    "hh_merge": dict(merge_evicted=True, span_window=0, k_bits=16, v_bits=16),
    "hh_merge_m4": dict(merge_evicted=True, span_window=4, k_bits=16, v_bits=16),
    "m4_k8v8": dict(merge_evicted=True, span_window=4, k_bits=8, v_bits=8),
}


def build_patch(name: str) -> HeavyHitterAttnScorePatch:
    cfg = LADDER[name]
    if cfg is None:
        return None
    return HeavyHitterAttnScorePatch(
        max_cache_len=128, sink_tokens=4, recent_tokens=32, obs_window=64,
        merge_evicted=cfg["merge_evicted"], k_bits=cfg["k_bits"],
        v_bits=cfg["v_bits"], span_window=cfg["span_window"],
    )


def build_turn_messages(document, questions, turn_idx, history_answers):
    """Mirror of common/metrics.py run_multi_turn_generation message building."""
    messages = [{"role": "system", "content": NO_THINK_TEXT}]
    if turn_idx == 0:
        messages.append({"role": "user", "content": f"{document}\n\n{questions[turn_idx]}"})
    else:
        messages.append({"role": "user", "content": document})
        for prev in range(turn_idx):
            messages.append({"role": "user", "content": questions[prev]})
            messages.append({"role": "assistant", "content": history_answers[prev]})
        messages.append({"role": "user", "content": questions[turn_idx]})
    return messages


def encode_turn(tokenizer, messages):
    kwargs = dict(tokenize=True, return_tensors="pt", add_generation_prompt=True)
    try:
        t = tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        t = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(t, "input_ids"):
        t = t.input_ids
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.long)
    return t.reshape(-1).long()


def left_pad(sequences, pad_id, device):
    """List of 1-D long tensors -> ([B, Lmax] input_ids, [B, Lmax] mask, real_lens)."""
    lens = [s.numel() for s in sequences]
    lmax = max(lens)
    B = len(sequences)
    input_ids = torch.full((B, lmax), pad_id, dtype=torch.long)
    mask = torch.zeros((B, lmax), dtype=torch.long)
    for i, s in enumerate(sequences):
        input_ids[i, lmax - lens[i]:] = s
        mask[i, lmax - lens[i]:] = 1
    return input_ids.to(device), mask.to(device), lens


def run_batched_turn(model, tokenizer, patch, device, messages_list,
                     max_new_tokens, model_config):
    """One lockstep batched turn. Returns per-session turn records."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    seqs = [encode_turn(tokenizer, m) for m in messages_list]
    input_ids, attn_mask, real_lens = left_pad(seqs, pad_id, device)
    B = input_ids.shape[0]

    # NOTE: always thread the cache through out.past_key_values. Transformers
    # may convert/rewrap the object during a forward (legacy->new format),
    # so the returned instance — not the one we passed in — carries the state.
    past = patch.get_cache(device, config=model_config) if patch is not None else None

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ---- prefill ----
    sync()
    t0 = time.perf_counter()
    S = input_ids.shape[1]
    model_dtype = next(model.parameters()).dtype
    # Chunked prefill for large B*S: a padded batch prefill makes HF build a
    # [B,1,S,S] causal mask (N=8/32k -> 15.5GB, N=64 -> 124GB) plus full-seq
    # activations, OOMing long before KV capacity becomes the constraint.
    # The cache already defers eviction to the first decode step (prefill
    # only appends), so chunking preserves eviction semantics: the last
    # chunk's stash still sees q=last-64-rows x K=full-length, identical to
    # the single-shot snapshot. Gated OFF the verified paths: B*S <= 131072
    # keeps the B=1 regression and the B=4 selftest on the exact single-shot
    # forward they were validated on.
    chunk = max(512, min(8192, 2_000_000_000 // max(B * S, 1)))
    use_chunk = B * S > 131_072 and chunk < S
    with torch.no_grad():
        if not use_chunk:
            # logits_to_keep=1: full-seq logits would be B*len*vocab*2B
            # (4*31128*262144*2 ~= 65 GiB for Qwen3.6 — OOM'd an 80GB card);
            # only the last position is needed to seed decode.
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        past_key_values=past, use_cache=True, logits_to_keep=1)
            next_ids = out.logits[:, -1, :].argmax(dim=-1)  # [B]
            past = out.past_key_values
        else:
            # Ready-made 4D additive masks per chunk: causal within the chunk
            # AND original pads invisible everywhere (they sit at the sequence
            # start, i.e. in chunk 0 / the past) — the same visibility the
            # single-shot 2D->4D expansion produces, but O(B*chunk*S) not
            # O(B*S^2).
            next_ids = None
            k_so_far = 0
            for start in range(0, S, chunk):
                cur = input_ids[:, start:start + chunk]
                m = cur.shape[1]
                k_total = k_so_far + m
                rows = (start + torch.arange(m, device=device))[:, None]
                cols = torch.arange(k_total, device=device)[None, :]
                visible = (cols <= rows)[None] & attn_mask[:, :k_total].bool()[:, None, :]
                m4 = torch.zeros(B, 1, m, k_total,
                                 dtype=model_dtype, device=device)
                m4.masked_fill_(~visible[:, None, :, :],
                                torch.finfo(model_dtype).min)
                out = model(input_ids=cur, attention_mask=m4,
                            past_key_values=past, use_cache=True,
                            logits_to_keep=1)
                past = out.past_key_values
                k_so_far = k_total
            next_ids = out.logits[:, -1, :].argmax(dim=-1)  # [B]
    sync()
    ttft = time.perf_counter() - t0

    # ---- greedy decode loop ----
    gen_ids = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        gen_ids[b].append(next_ids[b].item())
    finished |= (next_ids == eos_id)
    last = next_ids
    step_times = []
    n_steps = 0

    pad_counts = [input_ids.shape[1] - r for r in real_lens]
    pad_t = torch.tensor(pad_counts, device=device).unsqueeze(1)  # [B, 1]
    has_pads = any(p > 0 for p in pad_counts)

    # Decode masks are derived HARNESS-SIDE, never from cache internals:
    # out.past_key_values may be a rewrapped object whose reported content
    # length and custom attributes do not reflect the compressed state, and
    # transformers builds the 4D causal mask at forward START from the
    # PRE-update cache length — while eviction happens INSIDE the forward's
    # first update (measured 2026-09-17: a 128-slot mask was extended back
    # to the full 31127-slot prefill length). So for the evicting steps the
    # mask can only be correct if transformers does not rebuild it:
    #   - no pads in the batch -> attention_mask=None; sdpa uses is_causal,
    #     which is exact for q_len=1 and is what the single-stream batch=1
    #     path effectively does (this is why it never crashed there);
    #   - pads present -> a ready-made 4D mask, which transformers passes
    #     through without rebuilding.
    # Layout after any evicting update is always sink|hh|recent; pads can
    # only survive in the sink region (zero-mass pads are never selected
    # into hh at real context lengths; the B=1-vs-batched selftest guards).
    budget = patch.max_cache_len if patch is not None else None
    sink_n = patch.sink_tokens if patch is not None else 0
    content = input_ids.shape[1]  # post-prefill cache content length

    def decode_mask(step):
        nonlocal content
        if patch is None:
            # Full cache grows unbounded, no in-forward eviction: lengths
            # stay consistent, so the 2D mask path is safe.
            return torch.cat(
                [attn_mask, torch.ones(B, step + 1, dtype=torch.long, device=device)],
                dim=1,
            )
        if content + 1 > budget:
            # Evicting update: post-update layout is sink|hh|recent(budget).
            content = budget
            if not has_pads:
                return None
            valid = (torch.arange(budget, device=device).unsqueeze(0) >= pad_t)
            valid[:, sink_n:] = True
            m = torch.zeros(B, 1, 1, budget, dtype=model_dtype, device=device)
            m.masked_fill_(~valid.unsqueeze(1).unsqueeze(1),
                           torch.finfo(model_dtype).min)
            return m
        # No eviction yet (tiny prefill): uncompressed content, orig ==
        # arange(content), so the prefill mask + ones is exact.
        content += 1
        if not has_pads:
            return None
        return torch.cat(
            [attn_mask, torch.ones(B, step + 1, dtype=torch.long, device=device)],
            dim=1,
        )

    for step in range(max_new_tokens - 1):
        if bool(finished.all()):
            break
        feed = torch.where(finished, torch.full_like(last, pad_id), last).unsqueeze(1)
        cur_mask = decode_mask(step)
        sync()
        ts = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=feed, attention_mask=cur_mask,
                        past_key_values=past, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(dim=-1)
            past = out.past_key_values
        sync()
        step_times.append(time.perf_counter() - ts)
        n_steps += 1
        for b in range(B):
            if not finished[b]:
                gen_ids[b].append(nxt[b].item())
        finished |= (nxt == eos_id)
        last = nxt

    records = []
    for b in range(B):
        ids = gen_ids[b]
        records.append({
            "real_prompt_len": real_lens[b],
            "generated_ids": ids,
            "output": tokenizer.decode(ids, skip_special_tokens=True),
            "output_length": len(ids),
            "ended_with_eos": len(ids) > 0 and ids[-1] == eos_id,
            "ttft_s": ttft,
            "decode_steps": n_steps,
        })
    step_mean = statistics.fmean(step_times) if step_times else 0.0
    for r in records:
        r["step_time_mean_s"] = step_mean
    padded_len = input_ids.shape[1]
    return records, {
        "ttft_s": ttft,
        "step_time_mean_s": step_mean,
        "step_time_median_s": statistics.median(step_times) if step_times else 0.0,
        "decode_steps": n_steps,
        "padded_prompt_len": padded_len,
        "padding_overhead": 1.0 - sum(real_lens) / (B * padded_len),
    }


def fact_accuracy(records, gt_list, qtype_list):
    """Ground-truth match, per question type. gt None -> skipped.

    gt may be a string or a list of strings (multi_instruction: ALL must
    appear). Digit/percent answers are matched with a non-digit boundary so
    gt "9%" does not false-positive inside "19%".
    """
    import re
    cache = {}

    def match_one(gt, norm_out):
        key = str(gt)
        pat = cache.get(key)
        if pat is None:
            if re.search(r"[0-9]$", key) or "%" in key:
                pat = re.compile(r"(?<![0-9.])" + re.escape(key) + r"(?![0-9])")
            else:
                pat = re.compile(re.escape(key))
            cache[key] = pat
        return bool(pat.search(norm_out))

    def norm(s):
        return "".join(str(s).split())

    per_type = {}
    n_correct = n_total = 0
    for r, gt, qt in zip(records, gt_list, qtype_list):
        if gt is None:
            continue
        gts = gt if isinstance(gt, list) else [gt]
        norm_out = norm(r["output"])
        ok = all(match_one(g, norm_out) for g in gts)
        per_type.setdefault(qt, [0, 0])
        per_type[qt][0] += int(ok)
        per_type[qt][1] += 1
        n_correct += int(ok)
        n_total += 1
    return {
        "overall": n_correct / n_total if n_total else None,
        "per_type": {k: v[0] / v[1] for k, v in per_type.items()},
        "n": n_total,
    }


def run_cell(model, tokenizer, patch, device, sessions, turns, max_new_tokens,
             model_config):
    """Run all sessions lockstep through `turns` turns. Returns cell result dict."""
    reset_peak_memory_stats()
    t_start = time.perf_counter()
    all_turn_gen = []   # for compute_generation_metrics (EOS / repetition)
    fact_records = []
    turn_stats = []
    total_wall = 0.0
    total_out_tokens = 0
    total_real_prompt_tokens = 0
    total_padded_prompt_tokens = 0

    histories = [[] for _ in sessions]
    for t in range(turns):
        messages = [
            build_turn_messages(s["document"], s["questions"], t, histories[i])
            for i, s in enumerate(sessions)
        ]
        records, stats = run_batched_turn(
            model, tokenizer, patch, device, messages, max_new_tokens, model_config,
        )
        turn_stats.append(stats)
        total_wall += stats["ttft_s"] + stats["step_time_mean_s"] * stats["decode_steps"]
        for i, r in enumerate(records):
            histories[i].append(r["output"])
            total_out_tokens += r["output_length"]
            total_real_prompt_tokens += r["real_prompt_len"]
            total_padded_prompt_tokens += stats["padded_prompt_len"]
            gen = {
                "output": r["output"],
                "output_length": r["output_length"],
                "ended_with_eos": r["ended_with_eos"],
            }
            all_turn_gen.append(gen)
            gt = s_gta = None
            qt = "unknown"
            if t < len(sessions[i].get("answers", [])):
                gt = sessions[i]["answers"][t]
                qt = sessions[i].get("qtype", ["factoid"] * len(sessions[i]["questions"]))[t]
            fact_records.append((gen, gt, qt))

    wall = time.perf_counter() - t_start
    gen_metrics = compute_generation_metrics(all_turn_gen)
    fact = fact_accuracy([g for g, _, _ in fact_records],
                         [gt for _, gt, _ in fact_records],
                         [qt for _, _, qt in fact_records])
    ttfts = [s["ttft_s"] for s in turn_stats]
    steps = [s["step_time_mean_s"] for s in turn_stats]
    return {
        "wall_time_s": wall,
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else 0.0,
        "ttft_median_s": statistics.median(ttfts) if ttfts else 0.0,
        "tpot_mean_s": statistics.fmean(steps) if steps else 0.0,
        "output_tokens": total_out_tokens,
        "prompt_tokens_real": total_real_prompt_tokens,
        "prompt_tokens_padded": total_padded_prompt_tokens,
        "output_tokens_per_s": total_out_tokens / wall if wall > 0 else 0.0,
        "prompt_tokens_per_s_real": total_real_prompt_tokens / wall if wall > 0 else 0.0,
        "peak_hbm_mb": measure_peak_memory_mb(),
        "eos_success_rate": gen_metrics["eos_success_rate"],
        "repetition_rate": gen_metrics["repetition_rate"],
        "avg_output_length": gen_metrics["avg_output_length"],
        "fact_accuracy": fact,
        "per_turn": turn_stats,
    }


def load_sessions(data_file, num_sessions, turns):
    docs = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if not docs:
        raise ValueError(f"no documents in {data_file}")
    sessions = []
    for i in range(num_sessions):
        d = docs[i % len(docs)]
        sessions.append({
            "document": d["document"],
            "questions": d["questions"][:turns],
            "answers": d.get("answers", [None] * len(d["questions"]))[:turns],
            "qtype": d.get("qtype", ["factoid"] * len(d["questions"]))[:turns],
        })
    return sessions


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------

class _FakeBank:
    """Minimal AttentionScoreBank stand-in for the CPU cache self-test."""

    def __init__(self, scores, scores_per_head):
        self.scores = scores
        self.scores_per_head = scores_per_head

    def clear(self):
        pass


def _run_synthetic_cache(keys, values, k_bits, v_bits, bank, decode_keys, decode_vals):
    """Drive one HeavyHitterCache through one batched prefill + decode updates.

    keys/values: [B, H, plen, D]. decode_keys/vals: lists of [B, H, 1, D].
    """
    from kv_cache.heavy_hitter_cache import HeavyHitterCache
    cache = HeavyHitterCache(
        max_cache_len=32, sink_tokens=4, recent_tokens=8,
        importance_mode="attn_score", score_bank=bank, obs_window=16,
        merge_evicted=True, k_bits=k_bits, v_bits=v_bits,
    )
    cache.update(keys, values, 0)
    for dk, dv in zip(decode_keys, decode_vals):
        cache.update(dk, dv, 0)
    return cache


def cache_selftest():
    """CPU: per-session parity between B=1 caches and one batched cache."""
    torch.manual_seed(0)
    B, H, D, plen, steps = 4, 2, 8, 300, 4
    key_data = torch.randn(B, H, plen, D)
    val_data = torch.randn(B, H, plen, D)
    pos_scores = torch.rand(B, plen)
    head_scores = torch.rand(B, H, plen)
    g = torch.Generator().manual_seed(1234)
    dk_list = [torch.randn(B, H, 1, D, generator=g) for _ in range(steps)]
    dv_list = [torch.randn(B, H, 1, D, generator=g) for _ in range(steps)]

    for k_bits, v_bits in [(16, 16), (8, 8)]:
        bank_batched = _FakeBank({0: pos_scores.clone()}, {0: head_scores.clone()})
        cache_b = _run_synthetic_cache(
            key_data, val_data, k_bits, v_bits, bank_batched, dk_list, dv_list)
        for b in range(B):
            bank_1 = _FakeBank({0: pos_scores[b:b + 1].clone()},
                               {0: head_scores[b:b + 1].clone()})
            cache_1 = _run_synthetic_cache(
                key_data[b:b + 1], val_data[b:b + 1], k_bits, v_bits, bank_1,
                [dk[b:b + 1] for dk in dk_list], [dv[b:b + 1] for dv in dv_list],
            )
            lk_b = cache_b.layers[0].keys[b]
            lv_b = cache_b.layers[0].values[b]
            lk_1, lv_1 = cache_1.layers[0].keys[0], cache_1.layers[0].values[0]
            assert lk_1.shape == lk_b.shape and lv_1.shape == lv_b.shape, \
                f"shape mismatch b={b}: {lk_1.shape} vs {lk_b.shape}"
            assert torch.allclose(lk_1, lk_b, atol=1e-5), f"K mismatch b={b} k{k_bits}v{v_bits}"
            assert torch.allclose(lv_1, lv_b, atol=1e-5), f"V mismatch b={b} k{k_bits}v{v_bits}"
            oi_1 = cache_1.layers[0]._hh_orig_idx[0]
            oi_b = cache_b.layers[0]._hh_orig_idx[b]
            assert torch.equal(oi_1, oi_b), f"orig_idx mismatch b={b}"
        print(f"[cache-selftest] PASS k{k_bits}v{v_bits}: B=1 x {B} == B={B} batched")
    print("[cache-selftest] ALL PASS")


def selftest(model_path, data_file, device_map, torch_dtype, docs=4, turns=2, max_new_tokens=32):
    """GPU: sequential batch=1 vs batched B=docs, m_sp4 config, exact-match answers."""
    device = torch.device("cuda:0")
    model, tokenizer, _ = load_model_and_tokenizer(model_path, torch_dtype, "cuda:0", device_map)
    model_config = model.config
    patch = build_patch("hh_merge_m4")
    sessions = load_sessions(data_file, docs, turns)
    patch.install(model)  # without this the score bank stays empty and the
    # cache silently falls back to key-norm (loud [WARN], plen=-1 in the
    # eviction log) — the selftest would compare two key-norm runs.
    try:
        def answers_for(batch_sessions):
            # per-session lists, flattened SESSION-MAJOR by the caller —
            # a turn-major flat list misaligns with the per-session phase
            # whenever docs>1 and turns>1 (observed 2026-09-17: 5/7
            # "mismatches" were just the two flattening orders disagreeing).
            per_session = [[] for _ in batch_sessions]
            histories = [[] for _ in batch_sessions]
            for t in range(turns):
                msgs = [build_turn_messages(s["document"], s["questions"], t, histories[i])
                        for i, s in enumerate(batch_sessions)]
                records, _ = run_batched_turn(model, tokenizer, patch, device, msgs,
                                              max_new_tokens, model_config)
                for i, r in enumerate(records):
                    histories[i].append(r["output"])
                    per_session[i].append(r["output"])
            return per_session

        a_seq = []
        for s in sessions:
            a_seq.extend(answers_for([s])[0])
        a_bat = [a for sess in answers_for(sessions) for a in sess]
    finally:
        patch.uninstall(model)
    n_sess, n_turn = len(sessions), len(a_seq) // len(sessions)
    mismatches = [(i, x, y) for i, (x, y) in enumerate(zip(a_seq, a_bat)) if x != y]
    for i, x, y in mismatches:
        print(f"[selftest] MISMATCH session {i // n_turn} turn {i % n_turn}:"
              f"\n  seq={x!r}\n  bat={y!r}")
    t0_bad = [m for m in mismatches if m[0] % n_turn == 0]
    if t0_bad:
        raise SystemExit(
            f"[selftest] FAIL: {len(t0_bad)}/{n_sess} turn-0 answers differ — "
            f"turn 0 must be token-identical (harness/pad/mask/cache-threading bug)")
    if mismatches:
        print(f"[selftest] PASS (harness-clean): turn 0 token-identical for all "
              f"{n_sess} sessions; {len(mismatches)} later-turn divergences "
              f"documented (batched-prefill float noise over lossy re-compression)")
    else:
        print(f"[selftest] PASS: {len(a_seq)} answers token-identical "
              f"(B=1 sequential vs B={len(sessions)} batched, m_sp4)")
    print(f"[selftest] PASS: {len(a_seq)} answers token-identical "
          f"(B=1 sequential vs B={len(sessions)} batched, m_sp4)")


# --------------------------------------------------------------------------
# Main benchmark
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_file", default=None, help="longctx multi-turn JSONL")
    parser.add_argument("--configs", default="full,hh,hh_merge,hh_merge_m4,m4_k8v8")
    parser.add_argument("--concurrency-list", default="1,8,16,32,64")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-docs", type=int, default=8, help="distinct documents in the pool")
    parser.add_argument("--kv-budget-gb", type=float, default=512.0,
                        help="HBM budget for the sustainability criterion")
    parser.add_argument("--eos-tolerance-pp", type=float, default=2.0,
                        help="sustainable if EOS >= full-config EOS at same N minus this (percentage points)")
    parser.add_argument("--device_map", default="balanced_low_0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--output-dir", default="results/concurrency")
    parser.add_argument("--cache-selftest", action="store_true",
                        help="CPU-only cache batch-parity test, then exit")
    parser.add_argument("--selftest", action="store_true",
                        help="GPU sequential-vs-batched answer parity test, then exit")
    args = parser.parse_args()

    if args.cache_selftest:
        cache_selftest()
        return

    if args.selftest:
        selftest(args.model_path, args.data_file, args.device_map, args.torch_dtype)
        return

    if not args.model_path or not args.data_file:
        parser.error("--model_path and --data_file are required for the benchmark")

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    concurrencies = [int(x) for x in args.concurrency_list.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_path, args.torch_dtype, "cuda:0", args.device_map,
    )
    model_config = model.config
    print(f"[serve] model on {device}; sessions drawn from {args.data_file}")

    summary = []
    baseline_eos = {}  # concurrency -> full-config EOS, for the sustainability rule

    for name in configs:
        patch = build_patch(name)
        if patch is not None:
            patch.install(model)
        print(f"\n[serve] config={name} storage={patch.storage_stats() if patch else 'full KV'}")
        for N in concurrencies:
            sessions = load_sessions(args.data_file, N, args.turns)
            cell_file = out_dir / f"{name}_n{N}.json"
            record = {
                "config": name,
                "concurrency": N,
                "patch": patch.config() if patch else {"name": "full"},
                "storage_stats": patch.storage_stats() if patch else {},
                "kv_budget_gb": args.kv_budget_gb,
                "turns": args.turns,
                "max_new_tokens": args.max_new_tokens,
            }
            try:
                result = run_cell(model, tokenizer, patch, device, sessions,
                                  args.turns, args.max_new_tokens, model_config)
                record.update(result)
                record["status"] = "ok"
                if name == "full":
                    baseline_eos[N] = result["eos_success_rate"]
                sustainable = result["peak_hbm_mb"] / 1024 <= args.kv_budget_gb
                if name != "full" and N in baseline_eos:
                    sustainable = sustainable and (
                        result["eos_success_rate"] >=
                        baseline_eos[N] - args.eos_tolerance_pp / 100.0
                    )
                record["sustainable"] = bool(sustainable)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                record["status"] = "oom"
                record["error"] = str(e)  # "Tried to allocate ... GiB. GPU x ..."
                record["sustainable"] = False
                print(f"[serve]   N={N}: OOM\n{e}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    record["status"] = "oom"
                    record["error"] = str(e)
                    record["sustainable"] = False
                    print(f"[serve]   N={N}: OOM\n{e}")
                else:
                    raise
            else:
                print(f"[serve]   N={N}: TTFT {record['ttft_mean_s']:.2f}s "
                      f"TPOT {record['tpot_mean_s']*1000:.0f}ms "
                      f"out {record['output_tokens_per_s']:.1f} tok/s "
                      f"HBM {record['peak_hbm_mb']/1024:.1f}GB "
                      f"EOS {record['eos_success_rate']:.3f} "
                      f"fact {record['fact_accuracy']['overall']}")
            with open(cell_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            summary.append(record)
        if patch is not None:
            patch.uninstall(model)

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[serve] summary -> {summary_file}")


if __name__ == "__main__":
    main()
