"""CompressedKVImpl: full rewrite of the attention forward (vLLM 0.19.1).

Interface (verified against v0.19.1):
- forward(layer, query, key, value, kv_cache, attn_metadata, output,
          output_scale=None, output_block_scale=None)
  query [T, H, D]; key/value [T, H_kv, D]; output preallocated [T, H, D]
  (None only in exotic paths; Attention.forward always passes it).
- kv_cache (per-layer pool view, 0.19.1 layout): [2, num_blocks, block_size,
  H_kv, D]; K = kv_cache[0], V = kv_cache[1] (see do_kv_cache_update unbind).
- attn_metadata: CompressedKVMetadata with req_states / qsl_cpu /
  seq_lens_cpu attached by the builder.

Semantics (ported from kv_cache/heavy_hitter_cache.py — see eviction.py
for the forbidden-variants list):

- DEFERRED EVICTION (v2026-10-04l, semantics fix after the 64k sweep came
  back 0/64 at every slot count): per-chunk eviction scored keys with only
  the last-64-token tail of each 8192-token chunk, so the actual question
  (at the very end of the prompt) never contributed, and mid-document fact
  records were evicted by filler before the question arrived. The old
  harness instead accumulated attention column sums over the WHOLE prefill
  and compressed once at the first decode step (heavy_hitter_cache.py
  header, lines 11-15). We reproduce that: prefill chunks only score into
  the snapshot table; nothing is evicted while total <= V8_MAX_SEQ_TOKENS;
  the first decode step (C==1) finds total >> budget and evicts once, with
  the question's attention mass present in the snapshot.
- Prefill chunk (q_len > 1): score with the observation window (last
  min(obs, C) query rows) against [compact; chunk], scatter into the
  per-orig-position snapshot, grow compact to the full chunk, write back,
  then attend (compact cols unmasked, chunk cols causal).
- Decode (q_len == 1): allowance = budget; append, then evict when over
  allowance. Decode tokens compete at the prefill mean attention mass
  (never +inf).
- Cost model change, stated plainly: per-request block footprint now
  tracks prompt length up to V8_MAX_SEQ_TOKENS (same as full-KV baseline
  and the old harness). Constant-footprint decode is unchanged. Real
  memory savings need mid-request block freeing — a separate phase.
"""
import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from . import eviction
from . import config


class CompressedKVImpl(FlashAttentionImpl):
    forward_includes_kv_cache_update = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = {
            "retention": config.V8_COMPRESS_SLOTS,
            "sink": config.V8_SINK_TOKENS,
            "recent": config.V8_RECENT_TOKENS,
            "obs": config.V8_OBS_WINDOW,
            "span": config.V8_SPAN_WINDOW,
        }
        # None = not probed yet. SDPA enable_gqa avoids materializing
        # [H, L, D] repeated K/V; probed once per layer with tiny tensors.
        self._gqa_ok = None

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization not supported for CompressedKV")
        if attn_metadata is None:
            return output.fill_(0)

        H_kv = self.num_kv_heads
        bs = kv_cache.shape[2]  # block_size
        qsl = attn_metadata.qsl_cpu.tolist()
        seq_lens = attn_metadata.seq_lens_cpu.tolist()

        for i, st in enumerate(attn_metadata.req_states):
            qs, qe = qsl[i], qsl[i + 1]
            q_i = query[qs:qe]              # [C, H, D]
            k_new = key[qs:qe]              # [C, H_kv, D]
            v_new = value[qs:qe]
            n_computed = seq_lens[i] - (qe - qs)
            self._update_and_attend(st, kv_cache, q_i, k_new, v_new,
                                    n_computed, H_kv, bs, output, qs, qe)

        return output.view(output.shape[0], -1)

    def _update_and_attend(self, st, kv_cache, q, k_new, v_new,
                           n_computed, H_kv, bs, output, qs, qe):
        C = q.shape[0]
        device = q.device
        cfg = self.cfg
        budget = cfg["retention"]
        # Deferred eviction (v2026-10-04l): prefill chunks accumulate into
        # the snapshot but do NOT evict to the budget — the question at the
        # end of the prompt must get to score the keys first (old-harness
        # semantics). Only decode steps (C==1) enforce the tight budget;
        # the first decode step after a long prefill does the one-shot
        # full compress. V8_MAX_SEQ_TOKENS is the prefill safety valve
        # (and the per-request block cap): beyond it, prefill evicts down
        # to the allowance instead of growing without bound.
        allowance = budget if C == 1 else max(budget,
                                              config.V8_MAX_SEQ_TOKENS)
        sink_n = min(cfg["sink"], allowance)
        recent_n = min(cfg["recent"], allowance - sink_n)
        hh_budget = allowance - sink_n - recent_n

        st.ensure_gpu(device)
        # New-request detection was REMOVED here (v2026-10-04p): the
        # n_computed == 0 heuristic false-fired ~3000x under concurrency —
        # async scheduling holds num_computed at 0 until a request's prior
        # step executes, so every prefill chunk looked like a new request
        # and the snapshot table was wiped every step. The builder now
        # decides identity from the block table (append-only prefix; any
        # mismatch on a known blocks[0] = reissued to a new request) and
        # hands impl.py a fresh state when needed. Do NOT reintroduce
        # metadata-based reset here.
        # v2026-10-04n: async scheduling can lag chunk metadata by one chunk
        # (n_computed < positions already scored). For prefill chunks the
        # snapshot length IS the true chunk start; heal and flag it. Decode
        # (C==1) must not take this path (snap_len == prompt length there).
        if C > 1 and n_computed < st.snap_len:
            print(f"[v8_plugin] stale metadata healed: n_computed "
                  f"{n_computed} -> {st.snap_len} C={C}", flush=True)
            n_computed = st.snap_len
        st.grow(max(n_computed + C + 1, allowance + bs), device, H_kv)

        L = st.compact_len
        arange_cap = st._gpu[2]
        # v2026-10-04n: allocate the FULL allowance up front (~1 MiB) instead
        # of rebuilding on demand. Async scheduling can deliver chunk
        # metadata one chunk stale (n_computed lags), which made the
        # on-demand rebuild condition under-allocate and silently truncate
        # arange_cap[:L2] (crash: value [24576] vs target [16385], where
        # 16385 was the previous chunk's need_ar).
        full_ar = config.V8_MAX_SEQ_TOKENS + bs + 16
        if arange_cap is None or arange_cap.numel() < full_ar:
            arange_cap = torch.arange(full_ar, device=device)
            st._gpu = (device, st.blk_tensor, arange_cap)

        # 0.19.1 pool layout: kv_cache = [2, num_blocks, bs, H_kv, D].
        k_cache, v_cache = kv_cache.unbind(0)

        # --- gather compact K/V from the request's private blocks ---
        # slot s -> block slot s//bs -> physical block st.blk[s//bs], offset
        # s%bs.
        k_comp = v_comp = None
        if L > 0:
            jb = arange_cap[:L] // bs
            off = arange_cap[:L] % bs
            bid = st.blk_tensor[jb]
            k_comp = k_cache[bid, off].permute(1, 0, 2)  # [H_kv, L, D]
            v_comp = v_cache[bid, off].permute(1, 0, 2)

        # --- combined (orig-order) view: [compact; new chunk] ---
        orig_new = n_computed + arange_cap[:C]
        if L > 0:
            orig_all = torch.cat([st.orig[:L], orig_new])
            k_all = torch.cat([k_comp, k_new.permute(1, 0, 2)], dim=1)
            v_all = torch.cat([v_comp, v_new.permute(1, 0, 2)], dim=1)
        else:
            orig_all = orig_new
            k_all = k_new.permute(1, 0, 2)
            v_all = v_new.permute(1, 0, 2)
        total_len = L + C

        # --- observation-window scores (prefill chunks only) ---
        if C > 1:
            W = min(cfg["obs"], C)
            q_lastW = q[-W:].permute(1, 0, 2).float()   # [H, W, D]
            causal = eviction.make_causal_add(W, C, device)
            total, per_head = eviction.obs_window_scores(
                q_lastW, k_comp.float() if L else None,
                k_new.permute(1, 0, 2).float(), self.scale, causal)
            # Snapshot table is indexed by ORIGINAL position: scatter, since
            # compact keys sit at non-contiguous orig positions.
            st.snap.index_copy_(0, orig_all, total)
            st.snap_per_head.index_copy_(1, orig_all, per_head)
            st.snap_len = n_computed + C
            if not getattr(self, "_v8_scored_logged", False):
                self._v8_scored_logged = True
                print(f"[v8_plugin] first obs scoring: L={L} C={C} "
                      f"W={W} snap_len={st.snap_len}", flush=True)

        # --- evict (deferred: only over the allowance, see header) ---
        if total_len > allowance:
            mid_end = total_len - recent_n
            mid_orig = orig_all[sink_n:mid_end]
            # Decode-written tokens (orig >= snap_len) compete at the prefill
            # mean — never +inf (they would flush all prefill heavy hitters).
            valid = mid_orig < st.snap_len
            safe = mid_orig.clamp(max=max(st.snap_len - 1, 0))
            scores_mid = torch.where(
                valid, st.snap[safe],
                st.snap[:max(st.snap_len, 1)].mean())
            kept = eviction.select_kept(
                scores_mid, mid_orig, st.snap_len, hh_budget,
                obs_window=cfg["obs"], span_window=cfg["span"])
            hh_k = k_all[:, sink_n:mid_end, :][:, kept, :]
            hh_v = v_all[:, sink_n:mid_end, :][:, kept, :]
            hh_v = eviction.fold_evicted_values(
                hh_v, v_all[:, sink_n:mid_end, :], kept, mid_orig,
                st.snap_per_head)
            new_orig = torch.cat([
                orig_all[:sink_n], mid_orig[kept], orig_all[-recent_n:]])
            k_all = torch.cat(
                [k_all[:, :sink_n, :], hh_k, k_all[:, -recent_n:, :]], dim=1)
            v_all = torch.cat(
                [v_all[:, :sink_n, :], hh_v, v_all[:, -recent_n:, :]], dim=1)
            st.orig[:k_all.shape[1]] = new_orig
            st.compact_len = k_all.shape[1]
            if not getattr(self, "_v8_evict_logged", False):
                self._v8_evict_logged = True
                print(f"[v8_plugin] first eviction: kept="
                      f"{int(kept.shape[0])} L={st.compact_len} "
                      f"orig=[{int(new_orig[0])}..{int(new_orig[-1])}] "
                      f"allowance={allowance} C={C}", flush=True)
        else:
            st.orig[:total_len] = orig_all
            st.compact_len = total_len

        # --- write compact layout back into the private blocks ---
        L2 = st.compact_len
        if k_all.shape[1] != L2:
            # Should be unreachable (both derive from L+C or the eviction
            # cat). v2026-10-04m: print the full现场 and reconcile to the
            # tensor we actually hold.
            print(f"[v8_plugin] LAYOUT MISMATCH st={id(st)} L2={L2} "
                  f"k_all={k_all.shape[1]} L={L} C={C} "
                  f"n_computed={n_computed} allowance={allowance}", flush=True)
            L2 = k_all.shape[1]
            st.compact_len = L2
        jb2 = arange_cap[:L2] // bs
        off2 = arange_cap[:L2] % bs
        bid2 = st.blk_tensor[jb2]
        k_cache[bid2, off2] = k_all.permute(1, 0, 2)
        v_cache[bid2, off2] = v_all.permute(1, 0, 2)

        # --- attention ---
        n_rep = q.shape[1] // H_kv
        k_attn = k_all                            # [H_kv, L2, D]
        v_attn = v_all
        q_h = q.permute(1, 0, 2)                  # [H, C, D]
        if C > 1:
            # K = [compact; chunk]: compact cols unmasked (invariant 4),
            # chunk cols causal.
            k_attn = torch.cat([k_attn, k_new.permute(1, 0, 2)], dim=1)
            v_attn = torch.cat([v_attn, v_new.permute(1, 0, 2)], dim=1)
            # v2026-10-04o: build the additive mask directly in query dtype.
            # The old fp32 [C, L2+C] zeros + .to(bf16) was ~5 GiB of
            # transients at 64k prefill and OOMed with 1.3 GiB free.
            attn_mask = torch.zeros(1, 1, C, L2 + C,
                                    device=device, dtype=q_h.dtype)
            attn_mask[..., L2:] = eviction.make_causal_add(
                C, C, device).to(q_h.dtype)
        else:
            attn_mask = None
        # v2026-10-04o: enable_gqa broadcasts H_kv -> H inside the kernel
        # instead of materializing repeated [H, L, D] K/V (was 2 x 1.7 GiB
        # at 64k). Probe once per layer; fall back to repeat_interleave if
        # the installed torch rejects the flag.
        if self._gqa_ok is None:
            try:
                F.scaled_dot_product_attention(
                    torch.zeros(1, q.shape[1], 2, device=device,
                                dtype=q_h.dtype),
                    torch.zeros(1, H_kv, 4, device=device, dtype=q_h.dtype),
                    torch.zeros(1, H_kv, 4, device=device, dtype=q_h.dtype),
                    enable_gqa=True)
                self._gqa_ok = True
            except RuntimeError:
                self._gqa_ok = False
                print("[v8_plugin] enable_gqa unsupported, falling back to "
                      "repeat_interleave", flush=True)
        if self._gqa_ok:
            o = F.scaled_dot_product_attention(
                q_h.unsqueeze(0), k_attn.unsqueeze(0), v_attn.unsqueeze(0),
                attn_mask=attn_mask, enable_gqa=True)
        else:
            kr = k_attn.repeat_interleave(n_rep, dim=0)
            vr = v_attn.repeat_interleave(n_rep, dim=0)
            o = F.scaled_dot_product_attention(
                q_h.unsqueeze(0), kr.unsqueeze(0), vr.unsqueeze(0),
                attn_mask=attn_mask)
        output[qs:qe] = o[0].permute(1, 0, 2).to(output.dtype)
