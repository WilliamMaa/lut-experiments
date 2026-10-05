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
PLUGIN_VERSION = "2026-10-04o"

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
