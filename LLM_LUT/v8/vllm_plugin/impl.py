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
- Prefill chunk (q_len > 1): score with the observation window (last
  min(obs, C) query rows) against [compact; chunk], evict BEFORE writing
  the chunk (so all compact keys are causally visible to every chunk query
  and the additive mask only covers chunk-internal causality), write the
  new compact layout back into the request's private blocks, then attend.
- Decode (q_len == 1): append; evict only when over budget; decode tokens
  compete at the prefill mean attention mass (never +inf).
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
        sink_n = min(cfg["sink"], budget)
        recent_n = min(cfg["recent"], budget - sink_n)
        hh_budget = budget - sink_n - recent_n

        st.ensure_gpu(device)
        st.grow(max(n_computed + C + 1, budget + bs), device, H_kv)

        L = st.compact_len
        arange_cap = st._gpu[2]
        if arange_cap is None or arange_cap.numel() < budget + bs:
            arange_cap = torch.arange(budget + bs, device=device)
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
                      f"W={W} snap_len={st.snap_len}")

        # --- evict (before the chunk becomes visible: invariant 4) ---
        if total_len > budget:
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
                      f"orig=[{int(new_orig[0])}..{int(new_orig[-1])}]")
        else:
            st.orig[:total_len] = orig_all
            st.compact_len = total_len

        # --- write compact layout back into the private blocks ---
        L2 = st.compact_len
        jb2 = arange_cap[:L2] // bs
        off2 = arange_cap[:L2] % bs
        bid2 = st.blk_tensor[jb2]
        k_cache[bid2, off2] = k_all.permute(1, 0, 2)
        v_cache[bid2, off2] = v_all.permute(1, 0, 2)

        # --- attention ---
        n_rep = q.shape[1] // H_kv
        k_attn = k_all.repeat_interleave(n_rep, dim=0)   # [H, L2(+C), D]
        v_attn = v_all.repeat_interleave(n_rep, dim=0)
        q_h = q.permute(1, 0, 2)                          # [H, C, D]
        if C > 1:
            # K = [compact; chunk]: compact cols unmasked (invariant 4),
            # chunk cols causal.
            k_attn = torch.cat(
                [k_attn, k_new.permute(1, 0, 2).repeat_interleave(n_rep, 0)],
                dim=1)
            v_attn = torch.cat(
                [v_attn, v_new.permute(1, 0, 2).repeat_interleave(n_rep, 0)],
                dim=1)
            causal = eviction.make_causal_add(C, C, device)
            mask = torch.zeros(1, 1, C, L2 + C, device=device,
                               dtype=torch.float32)
            mask[..., L2:] = causal
            # SDPA requires attn_mask dtype == query dtype (bf16 here).
            attn_mask = mask.to(q_h.dtype)
        else:
            attn_mask = None
        o = F.scaled_dot_product_attention(
            q_h.unsqueeze(0), k_attn.unsqueeze(0), v_attn.unsqueeze(0),
            attn_mask=attn_mask)
        output[qs:qe] = o[0].permute(1, 0, 2).to(output.dtype)
