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
# 2026-10-06g: rewind condition tightened to `comp < snap_len` (was
# snap_len > comp + C). The old test missed preemption followed by a
# re-scheduled chunk larger than the old snap_len: the stale compact
# layout survived while the scheduler restarted from 0, and the plan
# fail-closed on the inconsistent frontier (found by the harness property
# test, R21 case).
# 2026-10-06h: rewind record completed to max(snap_len, compact_len).
# C==1 chunks never enter the scoring gate, so snap_len stays 0 while
# compact_len grows; after preemption, comp(0) < snap_len(0) looked
# consistent while a stale compact layout survived (harness property
# test, R211 case).
# 2026-10-06i: group layers share one state — the full update (score/
# evict/write) now runs on layer 0 ONLY, marked by st.applied=
# (chunk_start, C); layers 1..N-1 attend over what layer 0 wrote. Every
# layer previously re-appended the chunk to the mutated compact layout:
# the v2026-10-07 16384/8192 double-append crash (10-layer group, caught
# on the 35B repro; the single-layer harness could not see it — the
# multi-layer FakeWorld case closes that gap).
# 2026-10-08j: docs/37 fixed-budget KV allocation. The scheduler now books
# a FIXED B_target manager blocks per request (cdiv(SLOTS+STAGING, B_g) +
# BLOCK_MARGIN), independent of logical length; prefill staging beyond
# SLOTS+STAGING triggers pressure eviction via
# allowance = min(deferred_allowance(...), capacity - C) in
# blockplan.deferred_allowance (single source, builder + impl).
PLUGIN_VERSION = "2026-10-08j"

# Margin (in blocks) added on top of the retention budget. Eviction keeps
# L <= retention strictly, the margin only covers decode append-then-evict
# transients and debug headroom (~0.3MB per request per layer at bs=16).
BLOCK_MARGIN = 1

# Prefill staging budget, in tokens (docs/37). The physical pool per
# request is SLOTS + STAGING tokens: [0, SLOTS) is the steady-state
# compact region, [SLOTS, SLOTS+STAGING) is prefill staging that deferred
# eviction may still grow into before pressure eviction kicks in.
# Default 2x the 8192-token prefill chunk; raise it to trade memory
# savings for quality on prompts that outgrow the staging region.
V8_STAGING_TOKENS = int(os.environ.get("V8_STAGING_TOKENS", "16384"))

# Prefill allowance ceiling BEFORE the fixed-budget capacity clamp
# (v2026-10-04l, kept as the deferred-eviction ceiling). Since
# 2026-10-08j this NO LONGER drives allocation: blocks are booked at the
# fixed B_target (see V8_STAGING_TOKENS), and the per-step allowance is
# min(this ceiling, certified_capacity - C). Memory cost no longer scales
# with prompt length.
V8_MAX_SEQ_TOKENS = int(os.environ.get("V8_MAX_SEQ_TOKENS", "131072"))
