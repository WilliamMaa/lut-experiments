"""Shared env config for the v8 compressed-KV vLLM plugin.

All knobs have the v8 harness Pareto-sweet-spot defaults (docs/26):
512 slots, sink 4, recent 32, obs window 64, span 4 (m4 config).
"""
import os

V8_COMPRESS_SLOTS = int(os.environ.get("V8_COMPRESS_SLOTS", "512"))
V8_SINK_TOKENS = int(os.environ.get("V8_SINK_TOKENS", "4"))
V8_RECENT_TOKENS = int(os.environ.get("V8_RECENT_TOKENS", "32"))
V8_OBS_WINDOW = int(os.environ.get("V8_OBS_WINDOW", "64"))
V8_SPAN_WINDOW = int(os.environ.get("V8_SPAN_WINDOW", "4"))

# Only this arch gets the compressed backend injected. Comma-separated
# substring match against config.architectures.
TARGET_ARCH = os.environ.get(
    "V8_TARGET_ARCH", "Qwen3_5MoeForConditionalGeneration")

# Bumped on every behavioral change; printed at patch() and injection time
# so a remote log alone proves which code version is live.
# 2026-10-06a: docs/31 contract rewrite — identity by request_id (I4),
# kernel-unit addressing with P from the pool (I2), fail-closed frontier
# plan (I3). The n→t heuristic patch chain is gone.
# 2026-10-06b: NewRequestData field is req_id, not request_id (0.19.1).
# 2026-10-06c: FlashAttentionMetadata keeps the row as .block_table
# (built by super().build()); impl reads that, not block_table_tensor.
# 2026-10-06d: _update_and_attend was missing the kv_cache parameter
# (NameError at first forward; caught by repro).
# 2026-10-06e: chunk_start pinned by the metadata builder from the
# scheduler's computed_t. Reading st.snap_len inside the impl doubled the
# chunk for layer >= 1 of a group (layer 0 bumps snap_len mid-step), which
# made the block plan demand 2x the certified frontier (UnitError).
# 2026-10-06f: docs/32 single-plan architecture. The builder produces the
# per-request BlockPlan (plan_write_span, T-unit fail-closed); the impl
# only consumes it (certify_kernel is the one legal P conversion) and
# never re-derives token counts. Local integration harness added:
# vllm_plugin/tests/{fake_vllm,harness}.py + test_integration.py (8 cases
# + random interleaving) + test_forbidden.py (banned-pattern scan).
PLUGIN_VERSION = "2026-10-06f"

# Margin (in blocks) added on top of the retention budget. Eviction keeps
# L <= retention strictly, the margin only covers decode append-then-evict
# transients and debug headroom (~0.3MB per request per layer at bs=16).
BLOCK_MARGIN = 1

# Per-request block cap / prefill allowance (v2026-10-04l). Deferred
# eviction means a request's blocks must cover its WHOLE prompt; prefill
# chunks do not evict below this many tokens. Must be >= the server's
# --max-model-len. Memory cost is paid only as prefill actually advances
# (scheduler allocates lazily); post-prefill decode touches only the
# compact region, but the blocks stay reserved until the request ends.
V8_MAX_SEQ_TOKENS = int(os.environ.get("V8_MAX_SEQ_TOKENS", "131072"))
