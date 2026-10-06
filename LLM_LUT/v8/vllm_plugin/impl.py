"""CompressedKVImpl: attention forward under the docs/31 contract.

Interface (vLLM 0.19.1):
- forward(layer, query, key, value, kv_cache, attn_metadata, output,
          output_scale=None, output_block_scale=None)
  query [T, H, D]; key/value [T, H_kv, D]; output preallocated [T, H, D]
- kv_cache (per-layer pool view): [2, num_blocks, P, H_kv, D] with
  P = kv_cache.shape[2] the KERNEL page (measured 32 for the target
  model; B_g = 1056 manager units never appears here — the block table
  rows are already kernel units, block_table.py:110-118).
- attn_metadata: CompressedKVMetadata with req_states / req_ids /
  computed_t / scheduled_t attached by the builder (backend.py).

Addressing (I2, the only legal chain): compact slot s -> j = s // P,
r = s % P -> b_j = certified_row[j] -> k_cache[b_j, r]. The certified
row prefix length is the scheduler's frontier for this step
(cdiv(computed_t + scheduled_t, P)); the blockplan check runs BEFORE any
tensor indexing and raises with the full I6 context if the write span
would exceed it (docs/31 I3 — the v2026-10-04t OOB class is impossible by
construction: indices are bounded by the certified prefix).

Semantics (ported from kv_cache/heavy_hitter_cache.py; unchanged by this
rewrite — eviction/attention math only, see eviction.py):
- DEFERRED EVICTION: prefill chunks only score into the snapshot table;
  nothing is evicted while total <= V8_MAX_SEQ_TOKENS; the first decode
  step (C==1) evicts once, with the question's attention mass present.
- Prefill chunk (C > 1): score with the observation window against
  [compact; chunk], scatter into the per-orig-position snapshot, grow
  compact, write back, attend (compact cols unmasked, chunk cols causal).
- Decode (C == 1): allowance = budget; append, then evict when over.
"""
import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from . import eviction
from . import config
from . import units
from .blockplan import build_block_plan


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
        self._pool_page = None      # P, probed from kv_cache at first fwd
        self._logged = False

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization not supported for CompressedKV")
        if attn_metadata is None:
            return output.fill_(0)

        H_kv = self.num_kv_heads
        p_page = int(kv_cache.shape[2])         # P, kernel units
        if self._pool_page is None:
            self._pool_page = p_page
        elif self._pool_page != p_page:
            raise units.UnitError(
                f"[v8_plugin] pool page changed between forwards: "
                f"{self._pool_page} -> {p_page}")
        if not self._logged:
            self._logged = True
            print(f"[v8_plugin] layer cfg: P={p_page} H_kv={H_kv} "
                  f"pool={list(kv_cache.shape)} B_g="
                  f"{getattr(attn_metadata, 'mgr_block_size', '?')}",
                  flush=True)

        qsl = attn_metadata.qsl_cpu.tolist()
        bt = attn_metadata.block_table_tensor

        for i, st in enumerate(attn_metadata.req_states):
            qs, qe = qsl[i], qsl[i + 1]
            q_i = query[qs:qe]              # [C, H, D]
            k_new = key[qs:qe]              # [C, H_kv, D]
            v_new = value[qs:qe]
            self._update_and_attend(
                st, bt[i], attn_metadata.req_ids[i],
                getattr(attn_metadata, "mgr_block_size", 0),
                attn_metadata.computed_t[i],
                attn_metadata.scheduled_t[i],
                q_i, k_new, v_new, H_kv, p_page, output, qs, qe)

        return output.view(output.shape[0], -1)

    def _update_and_attend(self, st, row, req_id, b_g, computed_t,
                           scheduled_t, q, k_new, v_new, H_kv, p_page,
                           output, qs, qe):
        C = q.shape[0]
        device = q.device
        cfg = self.cfg
        budget = cfg["retention"]
        # Deferred eviction: prefill chunks accumulate into the snapshot
        # but do NOT evict to the budget — the question at the end of the
        # prompt must get to score the keys first. Only decode steps
        # (C==1) enforce the tight budget; the first decode step after a
        # long prefill does the one-shot full compress. V8_MAX_SEQ_TOKENS
        # is the prefill safety valve (and the per-request block cap).
        allowance = budget if C == 1 else max(budget,
                                              config.V8_MAX_SEQ_TOKENS)
        sink_n = min(cfg["sink"], allowance)
        recent_n = min(cfg["recent"], allowance - sink_n)
        hh_budget = allowance - sink_n - recent_n

        # Chunk start: v8's own processed record (T). The builder already
        # reset the state if the scheduler frontier ever falls behind it
        # (preemption/recompute), so snap_len <= computed_t + C holds.
        chunk_start = st.snap_len
        if chunk_start > computed_t + C:
            raise units.UnitError(
                f"[v8_plugin] state ahead of scheduler frontier: "
                f"snap_len {chunk_start} > computed {computed_t} + C {C}; "
                f"req={req_id} B_g={b_g} P={p_page}")

        # I3 fail-closed: the write span [0, snap_len + C) may only touch
        # pool blocks inside this step's certified frontier prefix. The row
        # itself is a GPU tensor consumed directly below; only its
        # certified length (in kernel blocks) enters the plan.
        available = units.cdiv(computed_t + scheduled_t, p_page)
        build_block_plan(
            request_id=req_id, group_id=0,
            span_end=units.Qty(chunk_start + C, units.Unit.S),
            computed=units.Qty(computed_t, units.Unit.T),
            scheduled=units.Qty(scheduled_t, units.Unit.T),
            block_row=available,
            pool_page=units.Qty(p_page, units.Unit.P),
            mgr_block_size=b_g)

        st.grow(max(chunk_start + C + 1, allowance + p_page), device, H_kv)
        row_long = row.long()

        # 0.19.1 pool layout: kv_cache = [2, num_blocks, P, H_kv, D].
        k_cache, v_cache = kv_cache.unbind(0)

        # --- gather compact K/V from the certified row prefix ---
        # slot s -> kernel block s//P -> row[s//P], offset s%P. Indices are
        # bounded by plan.required_pool_blocks <= certified prefix.
        L = st.compact_len
        required = units.cdiv(chunk_start + C, p_page)
        arange_cap = st.arange(max(L, required, C) + 1, device)
        k_comp = v_comp = None
        if L > 0:
            jb = arange_cap[:L] // p_page
            bid = row_long[jb]
            off = arange_cap[:L] % p_page
            k_comp = k_cache[bid, off].permute(1, 0, 2)  # [H_kv, L, D]
            v_comp = v_cache[bid, off].permute(1, 0, 2)

        # --- combined (orig-order) view: [compact; new chunk] ---
        orig_new = chunk_start + arange_cap[:C]
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
            # Snapshot table is indexed by ORIGINAL position: scatter.
            st.snap.index_copy_(0, orig_all, total)
            st.snap_per_head.index_copy_(1, orig_all, per_head)
            st.snap_len = chunk_start + C
            if not getattr(self, "_v8_scored_logged", False):
                self._v8_scored_logged = True
                print(f"[v8_plugin] first obs scoring: L={L} C={C} "
                      f"W={W} snap_len={st.snap_len}", flush=True)

        # --- evict (deferred: only over the allowance) ---
        if total_len > allowance:
            mid_end = total_len - recent_n
            mid_orig = orig_all[sink_n:mid_end]
            # Decode-written tokens (orig >= snap_len) compete at the
            # prefill mean — never +inf.
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

        # --- write compact layout back into the certified row prefix ---
        L2 = st.compact_len
        jb2 = arange_cap[:L2] // p_page
        off2 = arange_cap[:L2] % p_page
        bid2 = row_long[jb2]
        k_cache[bid2, off2] = k_all.permute(1, 0, 2)
        v_cache[bid2, off2] = v_all.permute(1, 0, 2)

        # --- attention ---
        n_rep = q.shape[1] // H_kv
        k_attn = k_all                            # [H_kv, L2, D]
        v_attn = v_all
        q_h = q.permute(1, 0, 2)                  # [H, C, D]
        if C > 1:
            k_attn = torch.cat([k_attn, k_new.permute(1, 0, 2)], dim=1)
            v_attn = torch.cat([v_attn, v_new.permute(1, 0, 2)], dim=1)
            attn_mask = torch.zeros(1, 1, C, L2 + C,
                                    device=device, dtype=q_h.dtype)
            attn_mask[..., L2:] = eviction.make_causal_add(
                C, C, device).to(q_h.dtype)
        else:
            attn_mask = None
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
