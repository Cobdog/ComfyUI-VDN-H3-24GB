"""Windowed softmax branch of VDN-H3 for ComfyUI's MiniMax-H3.

Ports the window geometry and softmax semantics of the official VDN release
(github.com/OpenVDN/vdn-minimax-h3, src/models/softmax_attention/) onto ComfyUI's
packed sequence. The released 8-step checkpoint uses radius=1, chunk=5 (chunk-aligned
windows), anchor_frames="both".

The released inference path runs block-sparse FlexAttention over a BlockMask. This
port groups queries that share a window -- under chunk-aligned bounds every frame of
a chunk has the same window -- and runs one dense SDPA call per distinct window, so
it needs no Triton and no torch.compile while keeping the exact same softmax
partition (the official window_softmax_reference is the same arithmetic spelled as
one SDPA per frame instead of per chunk).
"""
import collections
import logging
import os
import warnings
import time

import torch
import torch.nn.functional as F

from vdn_h3_24gb.profiler import (
    section as prof_section, window_deep_profile_enabled,
)

_log = logging.getLogger("comfy.vdn")

ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")


def window_bounds(num_frames, radius, chunk=0):
    """Per-frame inclusive softmax-window bounds [lo, hi], unclamped. Verbatim port.

    chunk == 0: frame mode, centered window |t_q - t_k| <= radius.
    chunk == K: chunk-aligned mode, frame t sees whole chunks [t//K - r, t//K + r].
    """
    if chunk <= 0:
        return [(t - radius, t + radius) for t in range(num_frames)]
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1)
            for t in range(num_frames)]


def full_coverage(bounds, num_frames):
    """True when every window already covers all frames (softmax IS dense and the
    linear branch must go inactive so nothing is counted twice)."""
    return all(lo <= 0 and hi >= num_frames - 1 for lo, hi in bounds)


# --------------------------------------------------- grouped-path plan & scratch --

MAX_CACHED_PLANS = 8
_PLAN_CACHE = collections.OrderedDict()
_KV_SCRATCH = {}


def _window_plan(video_start, video_end, num_frames, tokens_per_frame, bounds,
                 anchor_frames, seq, device):
    """Everything about the window partition that is identical for every block and
    every step of a run: the global-row index, per-group query/window row indices,
    and the anchor-row slices. Cached per layout instead of rebuilding ~50
    arange/cat index tensors per block per step."""
    key = (video_start, video_end, num_frames, tokens_per_frame,
           tuple(map(tuple, bounds)), anchor_frames, seq, str(device))
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        _PLAN_CACHE.move_to_end(key)
        return hit

    def frame_rows(f):
        a = video_start + f * tokens_per_frame
        return torch.arange(a, a + tokens_per_frame, device=device)

    global_idx = torch.cat([torch.arange(video_start, device=device),
                            torch.arange(video_end, seq, device=device)])
    anchors = (0, num_frames - 1)
    anchor_rows = sorted(f for f in anchors if anchor_frames in ("rows", "both"))
    anchor_set = set(anchor_rows)

    grouped = collections.OrderedDict()
    for f in range(num_frames):
        if f in anchor_set:
            continue
        lo = max(bounds[f][0], 0)
        hi = min(bounds[f][1], num_frames - 1)
        grouped.setdefault((lo, hi), []).append(f)

    groups = []
    max_rows = global_idx.numel()
    for (lo, hi), frames in grouped.items():
        extra = [f for f in anchors
                 if anchor_frames in ("columns", "both") and not lo <= f <= hi]
        key_frames = sorted(set(range(lo, hi + 1)) | set(extra))
        win_idx = torch.cat([frame_rows(f) for f in key_frames])
        q_idx = torch.cat([frame_rows(f) for f in frames])
        groups.append((q_idx, win_idx))
        max_rows = max(max_rows, global_idx.numel() + win_idx.numel())
    plan = dict(global_idx=global_idx, groups=groups,
                anchor_slices=[(video_start + f * tokens_per_frame,
                                video_start + (f + 1) * tokens_per_frame)
                               for f in anchor_rows],
                max_kv_rows=max_rows)
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > MAX_CACHED_PLANS:
        _PLAN_CACHE.popitem(last=False)
    return plan


def _kv_scratch(rows, heads, head_dim, device, dtype, retain=True):
    """Retained mode: ONE grow-only k/v buffer pair per (device, dtype), sized
    to the largest window group; every group's gather lands in a slice of it
    instead of a fresh per-group torch.cat allocation. Transient mode (VRAM
    pressure): fresh per call -- the v1.3.1 pattern. Same-stream execution makes
    the reuse safe (each group's SDPA is enqueued before the next group's
    gather overwrites)."""
    if not retain:
        shape = (rows, heads, head_dim)
        return (torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype))
    key = (str(device), dtype)
    pair = _KV_SCRATCH.get(key)
    need = rows * heads * head_dim
    if pair is None or pair[0].numel() < need:
        pair = (torch.empty(need, device=device, dtype=dtype),
                torch.empty(need, device=device, dtype=dtype))
        _KV_SCRATCH[key] = pair
    return (pair[0][:need].view(rows, heads, head_dim),
            pair[1][:need].view(rows, heads, head_dim))


def clear_window_state():
    """Drop cached window plans and the k/v scratch (run interrupt / cleanup)."""
    _PLAN_CACHE.clear()
    _KV_SCRATCH.clear()


def _group_batch_mode():
    """v26 exact group batching control.

    Integer 1..4 preserves the old fixed v8 behavior. ``benchmark``/``autotune``
    lets v26 benchmark B=1..4 on the first real same-shaped interior group set
    and cache the fastest *per-group* exact cuDNN SDPA geometry for that shape.
    """
    raw = os.environ.get("VDN_H3_WINDOW_GROUP_BATCH", "1").strip().lower()
    if raw in ("benchmark", "autotune", "bench"):
        return "benchmark"
    try:
        n = int(raw)
    except Exception:
        n = 1
    return max(1, min(n, 4))


def _group_batch_key(qn, kn, heads, head_dim, device, dtype):
    return (int(qn), int(kn), int(heads), int(head_dim), str(device), str(dtype),
            _forced_backend())


def _benchmark_group_batch(q_batch, k_batch, v_batch, scale, max_b):
    """Choose exact SDPA batch geometry by normalized milliseconds/group.

    Packing is intentionally outside the benchmark: v25 showed group SDPA itself
    is the dominant cost. Every candidate uses identical Q/K/V rows; batch is
    only an independence dimension, so no cross-window attention is introduced.
    """
    key = _group_batch_key(q_batch.shape[1], k_batch.shape[1], q_batch.shape[2],
                           q_batch.shape[3], q_batch.device, q_batch.dtype)
    hit = _GROUP_BATCH_AUTOTUNE.get(key)
    if hit is not None:
        return min(int(hit), int(max_b))
    if q_batch.device.type != "cuda" or not torch.cuda.is_available():
        _GROUP_BATCH_AUTOTUNE[key] = 1
        return 1
    reps = _bench_int_env("VDN_H3_GROUP_BATCH_BENCH_REPS", 3, 1, 8)
    warmup = _bench_int_env("VDN_H3_GROUP_BATCH_BENCH_WARMUP", 1, 1, 3)
    # v27 diagnostic-only custom-kernel feasibility probe on the same real B=1
    # group. It never supplies the production output.
    try:
        from .triton_probe import maybe_probe
        def _v27_ref(qb, kb, vb):
            q4 = qb.permute(0, 2, 1, 3)
            k4 = kb.permute(0, 2, 1, 3)
            v4 = vb.permute(0, 2, 1, 3)
            forced = _resolved_backend(q4, k4, v4, scale)
            return _sdpa_call(q4, k4, v4, scale, backend=forced).permute(0, 2, 1, 3)
        maybe_probe(q_batch, k_batch, v_batch, scale, _v27_ref)
    except Exception as e:
        _log.warning("[vdn-triton-v27] probe setup failed; ignored: %s", e)
    timings = {}
    with torch.no_grad():
        for b in range(1, min(4, int(max_b)) + 1):
            try:
                qb, kb, vb = q_batch[:b], k_batch[:b], v_batch[:b]
                for _ in range(warmup):
                    o = _sdpa_batched(qb, kb, vb, scale)
                torch.cuda.synchronize(q_batch.device)
                st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
                st.record()
                for _ in range(reps):
                    o = _sdpa_batched(qb, kb, vb, scale)
                en.record(); en.synchronize()
                total = float(st.elapsed_time(en)) / reps
                timings[b] = (total, total / b)
                del o
            except Exception:
                continue
    chosen = min(timings, key=lambda b: timings[b][1]) if timings else 1
    _GROUP_BATCH_AUTOTUNE[key] = chosen
    if key not in _GROUP_BATCH_AUTOTUNE_LOGGED:
        _GROUP_BATCH_AUTOTUNE_LOGGED.add(key)
        parts = [f"B{b}={t[0]:.3f}ms/{t[1]:.3f}ms-per-group" for b,t in sorted(timings.items())]
        _log.info("[vdn-group-batch-v26] Q%d K%d H%d D%d -> B%d | %s | warmup=%d reps=%d",
                  q_batch.shape[1], k_batch.shape[1], q_batch.shape[2], q_batch.shape[3],
                  chosen, ", ".join(parts), warmup, reps)
    return min(chosen, int(max_b))


def _sdpa_batched(q_rows, k_rows, v_rows, scale):
    """Exact batched SDPA. Inputs are [B, rows, H, d].

    Each batch element is an independent window group, so batching cannot create
    cross-window attention.  The same exact backend chain used by `_sdpa` is
    retained.
    """
    q4 = q_rows.permute(0, 2, 1, 3)
    k4 = k_rows.permute(0, 2, 1, 3)
    v4 = v_rows.permute(0, 2, 1, 3)
    _log_backend_once(q4, k4, v4, scale)
    forced = _resolved_backend(q4, k4, v4, scale)
    if forced is not None and forced not in _FORCED_BROKEN:
        try:
            return _sdpa_call(q4, k4, v4, scale, backend=forced).permute(0, 2, 1, 3)
        except RuntimeError as e:
            _FORCED_BROKEN.add(forced)
            _log.warning("[vdn] selected window SDPA backend %s unavailable (%s); "
                         "using the default chain from here on", forced, e)
    try:
        from comfy.ops import scaled_dot_product_attention as comfy_sdpa
    except ImportError:
        comfy_sdpa = F.scaled_dot_product_attention
    return comfy_sdpa(q4, k4, v4, scale=scale).permute(0, 2, 1, 3)


def window_softmax_grouped(query, key, value, video_start, video_end,
                           num_frames, tokens_per_frame, bounds, scale,
                           anchor_frames="none", transformer_options=None,
                           retain_buffers=True, profile_block=None):
    """Windowed softmax over the packed sequence [globals | video], one dense SDPA
    call per distinct query group.

    query/key/value: [seq, H, d], already QK-normed and RoPE'd, full sequence.
    Returns [seq, H, d]: every pair involving a global row (text/cond/audio) stays
    dense in both directions; (video, video) pairs are restricted to the window,
    widened by the anchor frames per `anchor_frames` (official semantics: "columns"
    makes frames 0 and F-1 visible to every query, "rows" makes those two frames'
    queries see everything, "both" is exact on both sides).
    """
    heads, head_dim = query.shape[1], query.shape[2]
    seq = query.shape[0]
    out = torch.empty_like(query)
    plan = _window_plan(video_start, video_end, num_frames, tokens_per_frame,
                        bounds, anchor_frames, seq, query.device)
    global_idx = plan["global_idx"]
    deep_prof = window_deep_profile_enabled()
    global_q = query[global_idx]
    g = global_idx.numel()
    if g:
        # globals (text/cond/audio) are dense in both directions: every key
        if deep_prof:
            with prof_section("window_deep_global_sdpa", profile_block):
                global_attended = _sdpa(global_q, key, value, scale, transformer_options)
            with prof_section("window_deep_scatter", profile_block):
                out[global_idx] = global_attended
        else:
            out[global_idx] = _sdpa(global_q, key, value, scale, transformer_options)

    groups = plan["groups"]
    if groups:
        global_k = key[global_idx]
        global_v = value[global_idx]
        batch_mode = _group_batch_mode() if transformer_options is None else 1
        batch_n = 4 if batch_mode == "benchmark" else batch_mode
        if batch_n > 1:
            global _BATCH_LOGGED
            if not _BATCH_LOGGED:
                _BATCH_LOGGED = True
                if batch_mode == "benchmark":
                    _log.info("[vdn] v26 grouped window microbatch: benchmark/autotune B=1..4 (exact SDPA)")
                    if os.environ.get("VDN_H3_TRITON_ATTN_PROBE", "0").strip().lower() in ("1","true","yes","on"):
                        _log.info("[vdn] v27 Triton exact-attention feasibility probe: on (diagnostic-only; production remains cuDNN)")
                else:
                    _log.info("[vdn] grouped window microbatch: %d (exact SDPA, v8)", batch_n)

        if batch_n <= 1:
            k_scratch, v_scratch = _kv_scratch(
                plan["max_kv_rows"], heads, head_dim, key.device, key.dtype,
                retain=retain_buffers)
            if g:
                if deep_prof:
                    with prof_section("window_deep_global_scratch_copy", profile_block):
                        k_scratch[:g].copy_(global_k)
                        v_scratch[:g].copy_(global_v)
                else:
                    k_scratch[:g].copy_(global_k)
                    v_scratch[:g].copy_(global_v)
            for q_idx, win_idx in groups:
                w = win_idx.numel()
                if deep_prof:
                    with prof_section("window_deep_kv_gather", profile_block):
                        torch.index_select(key, 0, win_idx, out=k_scratch[g:g + w])
                        torch.index_select(value, 0, win_idx, out=v_scratch[g:g + w])
                    with prof_section("window_deep_q_gather", profile_block):
                        q_rows = query.index_select(0, q_idx)
                    with prof_section("window_deep_group_sdpa", profile_block):
                        attended = _sdpa(q_rows, k_scratch[:g + w], v_scratch[:g + w],
                                         scale, transformer_options)
                    with prof_section("window_deep_scatter", profile_block):
                        out[q_idx] = attended
                else:
                    torch.index_select(key, 0, win_idx, out=k_scratch[g:g + w])
                    torch.index_select(value, 0, win_idx, out=v_scratch[g:g + w])
                    q_rows = query.index_select(0, q_idx)
                    out[q_idx] = _sdpa(q_rows, k_scratch[:g + w], v_scratch[:g + w],
                                       scale, transformer_options)
        else:
            # Consecutive groups generally share shapes for the interior chunks.
            # Only same-shaped groups are batched so no padding/masking is added.
            i = 0
            while i < len(groups):
                q0, w0 = groups[i]
                qn, kn = q0.numel(), g + w0.numel()
                chunk = [(q0, w0)]
                j = i + 1
                while j < len(groups) and len(chunk) < batch_n:
                    qj, wj = groups[j]
                    if qj.numel() != qn or g + wj.numel() != kn:
                        break
                    chunk.append((qj, wj))
                    j += 1
                if len(chunk) == 1:
                    q_idx, win_idx = chunk[0]
                    k_rows = torch.empty((kn, heads, head_dim), device=key.device, dtype=key.dtype)
                    v_rows = torch.empty_like(k_rows)
                    if deep_prof:
                        with prof_section("window_deep_batch_pack", profile_block):
                            if g:
                                k_rows[:g].copy_(global_k)
                                v_rows[:g].copy_(global_v)
                        with prof_section("window_deep_kv_gather", profile_block):
                            torch.index_select(key, 0, win_idx, out=k_rows[g:])
                            torch.index_select(value, 0, win_idx, out=v_rows[g:])
                        with prof_section("window_deep_q_gather", profile_block):
                            q_rows = query.index_select(0, q_idx)
                        with prof_section("window_deep_group_sdpa", profile_block):
                            attended = _sdpa(q_rows, k_rows, v_rows, scale, transformer_options)
                        with prof_section("window_deep_scatter", profile_block):
                            out[q_idx] = attended
                    else:
                        if g:
                            k_rows[:g].copy_(global_k)
                            v_rows[:g].copy_(global_v)
                        torch.index_select(key, 0, win_idx, out=k_rows[g:])
                        torch.index_select(value, 0, win_idx, out=v_rows[g:])
                        out[q_idx] = _sdpa(query.index_select(0, q_idx), k_rows, v_rows,
                                           scale, transformer_options)
                else:
                    b = len(chunk)
                    q_batch = torch.empty((b, qn, heads, head_dim),
                                          device=query.device, dtype=query.dtype)
                    k_batch = torch.empty((b, kn, heads, head_dim),
                                          device=key.device, dtype=key.dtype)
                    v_batch = torch.empty_like(k_batch)
                    if deep_prof:
                        for bi, (q_idx, win_idx) in enumerate(chunk):
                            with prof_section("window_deep_q_gather", profile_block):
                                torch.index_select(query, 0, q_idx, out=q_batch[bi])
                            if g:
                                with prof_section("window_deep_batch_pack", profile_block):
                                    k_batch[bi, :g].copy_(global_k)
                                    v_batch[bi, :g].copy_(global_v)
                            with prof_section("window_deep_kv_gather", profile_block):
                                torch.index_select(key, 0, win_idx, out=k_batch[bi, g:])
                                torch.index_select(value, 0, win_idx, out=v_batch[bi, g:])
                        if batch_mode == "benchmark":
                            chosen_b = _benchmark_group_batch(q_batch, k_batch, v_batch, scale, b)
                        else:
                            chosen_b = b
                        # If autotune prefers a smaller B than the available same-shape run,
                        # execute exact independent sub-batches at that chosen geometry.
                        attended_parts = []
                        with prof_section("window_deep_group_sdpa", profile_block):
                            for bs in range(0, b, chosen_b):
                                be = min(bs + chosen_b, b)
                                attended_parts.append(_sdpa_batched(q_batch[bs:be], k_batch[bs:be], v_batch[bs:be], scale))
                        attended = torch.cat(attended_parts, dim=0) if len(attended_parts) > 1 else attended_parts[0]
                        for bi, (q_idx, _) in enumerate(chunk):
                            with prof_section("window_deep_scatter", profile_block):
                                out[q_idx] = attended[bi]
                    else:
                        for bi, (q_idx, win_idx) in enumerate(chunk):
                            torch.index_select(query, 0, q_idx, out=q_batch[bi])
                            if g:
                                k_batch[bi, :g].copy_(global_k)
                                v_batch[bi, :g].copy_(global_v)
                            torch.index_select(key, 0, win_idx, out=k_batch[bi, g:])
                            torch.index_select(value, 0, win_idx, out=v_batch[bi, g:])
                        if batch_mode == "benchmark":
                            chosen_b = _benchmark_group_batch(q_batch, k_batch, v_batch, scale, b)
                        else:
                            chosen_b = b
                        attended_parts = []
                        for bs in range(0, b, chosen_b):
                            be = min(bs + chosen_b, b)
                            attended_parts.append(_sdpa_batched(q_batch[bs:be], k_batch[bs:be], v_batch[bs:be], scale))
                        attended = torch.cat(attended_parts, dim=0) if len(attended_parts) > 1 else attended_parts[0]
                        for bi, (q_idx, _) in enumerate(chunk):
                            out[q_idx] = attended[bi]
                i = j if len(chunk) > 1 else i + 1

    for a, b in plan["anchor_slices"]:
        if deep_prof:
            with prof_section("window_deep_anchor_sdpa", profile_block):
                attended = _sdpa(query[a:b], key, value, scale, transformer_options)
            with prof_section("window_deep_scatter", profile_block):
                out[a:b] = attended
        else:
            out[a:b] = _sdpa(query[a:b], key, value, scale, transformer_options)

    return out



def window_softmax_temporal_proxy(query, key, value, video_start, video_end,
                                  num_frames, tokens_per_frame, bounds, scale,
                                  anchor_frames="none", transformer_options=None):
    """v15 cheap fallback for blocks where exact spatial window SDPA is skipped.

    This intentionally preserves an attention-shaped signal instead of replacing
    the released window branch by zero (the v14 experiment showed that zeroing
    even low-norm early windows can destroy denoising).  Global query rows and
    anchor query frames keep exact dense attention.  Other video queries attend
    only to the *same spatial token* across the allowed temporal frames, plus one
    mean global K/V summary token.  Complexity is O(F*S*H*W*d) rather than
    O(F*S*H*(W*S)*d), so it retains temporal/state mixing while removing the
    expensive all-spatial window interaction.

    It is an experimental approximation, not the released VDN math.
    """
    seq, heads, dim = query.shape
    out = torch.empty_like(query)

    # Globals remain exact dense queries.  This is a small fraction of all rows
    # and preserves conditioning exchange better than an identity/zero fallback.
    global_idx = torch.cat([torch.arange(video_start, device=query.device),
                            torch.arange(video_end, seq, device=query.device)])
    if global_idx.numel():
        out[global_idx] = _sdpa(query[global_idx], key, value, scale,
                                transformer_options)
        gk = key[global_idx].mean(dim=0)
        gv = value[global_idx].mean(dim=0)
    else:
        gk = gv = None

    qv = query[video_start:video_end].view(num_frames, tokens_per_frame, heads, dim)
    kv = key[video_start:video_end].view(num_frames, tokens_per_frame, heads, dim)
    vv = value[video_start:video_end].view(num_frames, tokens_per_frame, heads, dim)
    ov = out[video_start:video_end].view(num_frames, tokens_per_frame, heads, dim)

    anchors = {0, num_frames - 1} if anchor_frames in ("rows", "both") else set()

    # Anchor rows retain the released dense-query semantics.
    for f in sorted(anchors):
        a = video_start + f * tokens_per_frame
        b = a + tokens_per_frame
        ov[f].copy_(_sdpa(query[a:b], key, value, scale, transformer_options))

    # Frames sharing the same bounds are processed together.  The SDPA batch
    # dimension is spatial position; heads remain the head dimension.
    grouped = collections.OrderedDict()
    for f in range(num_frames):
        if f in anchors:
            continue
        lo = max(bounds[f][0], 0)
        hi = min(bounds[f][1], num_frames - 1)
        grouped.setdefault((lo, hi), []).append(f)

    allow_anchor_cols = anchor_frames in ("columns", "both")
    for (lo, hi), frames in grouped.items():
        key_frames = list(range(lo, hi + 1))
        if allow_anchor_cols:
            key_frames = sorted(set(key_frames) | {0, num_frames - 1})

        qg = qv[frames]       # [Fq,S,H,D]
        kg = kv[key_frames]   # [Fk,S,H,D]
        vg = vv[key_frames]

        # [S,H,F,D] -- one tiny temporal attention problem per spatial token.
        q4 = qg.permute(1, 2, 0, 3).contiguous()
        k4 = kg.permute(1, 2, 0, 3).contiguous()
        v4 = vg.permute(1, 2, 0, 3).contiguous()

        if gk is not None:
            # Add one compressed global-conditioning token to each spatial
            # temporal sequence rather than all global rows.
            gk4 = gk.view(1, heads, 1, dim).expand(tokens_per_frame, -1, -1, -1)
            gv4 = gv.view(1, heads, 1, dim).expand(tokens_per_frame, -1, -1, -1)
            k4 = torch.cat((k4, gk4), dim=2)
            v4 = torch.cat((v4, gv4), dim=2)

        try:
            from comfy.ops import scaled_dot_product_attention as comfy_sdpa
        except ImportError:
            comfy_sdpa = F.scaled_dot_product_attention
        ag = comfy_sdpa(q4, k4, v4, scale=scale)
        ov[frames] = ag.permute(2, 0, 1, 3)

    return out

# ------------------------------------------------------------- backend visibility --

_BACKEND_LOGGED = False
_BATCH_LOGGED = False
_FORCED_BACKEND = ...  # lazily parsed sentinel
_FORCED_BROKEN = set()
_AUTOTUNE_CACHE = {}
_AUTOTUNE_LOGGED = set()
_GROUP_BATCH_AUTOTUNE = {}
_GROUP_BATCH_AUTOTUNE_LOGGED = set()

# Exact SDPA backends only, in ComfyUI's priority order. The window softmax must
# never route through sage/kitchen int8 overrides (that measurably softens
# output); choosing AMONG the exact kernels is pure perf, no quality surface.
_BACKEND_PRIORITY = ("flash", "cudnn", "mem_efficient", "math")


def _sdpa_backend_enum(name):
    from torch.nn.attention import SDPBackend
    return {"flash": SDPBackend.FLASH_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH}[name]


def _forced_backend():
    """VDN_H3_WINDOW_SDPA controls exact window SDPA dispatch.

    flash|cudnn|mem_efficient|math forces one exact backend.  ``benchmark``
    (alias ``autotune``) runs a tiny shape-local CUDA microbenchmark the first
    time each real window shape is encountered and caches the fastest exact
    backend for the rest of the process.  ``auto`` keeps ComfyUI/PyTorch's
    normal priority chain.
    """
    global _FORCED_BACKEND
    if _FORCED_BACKEND is ...:
        raw = os.environ.get("VDN_H3_WINDOW_SDPA", "auto").strip().lower()
        if raw in ("", "auto"):
            _FORCED_BACKEND = None
        elif raw in ("benchmark", "autotune", "bench"):
            _FORCED_BACKEND = "benchmark"
        elif raw in _BACKEND_PRIORITY:
            _FORCED_BACKEND = raw
        else:
            _log.warning("[vdn] VDN_H3_WINDOW_SDPA=%r not one of %s or benchmark; ignoring",
                         raw, _BACKEND_PRIORITY)
            _FORCED_BACKEND = None
    return _FORCED_BACKEND


def _sdpa_call(q4, k4, v4, scale, backend=None):
    """One SDPA under a single-backend (or default-chain) dispatch context."""
    if backend is None:
        return F.scaled_dot_product_attention(q4, k4, v4, scale=scale)
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel([_sdpa_backend_enum(backend)], set_priority=True):
        return F.scaled_dot_product_attention(q4, k4, v4, scale=scale)


def _bench_int_env(name, default, lo=1, hi=20):
    try:
        value = int(os.environ.get(name, str(default)).strip())
    except Exception:
        value = default
    return max(lo, min(value, hi))


def _autotune_key(q4, k4, v4):
    # Include every property that can materially change SDPA kernel selection.
    return (str(q4.device), str(q4.dtype), tuple(q4.shape), tuple(k4.shape),
            tuple(v4.shape))


def _benchmark_backend(q4, k4, v4, scale):
    """Return the fastest available exact CUDA SDPA backend for this shape.

    The benchmark is deliberately small and happens only once per distinct real
    window shape.  It never benchmarks approximation kernels and never changes
    Q/K/V or attention geometry.  Math is used only as a last-resort fallback,
    because timing it on large video windows can be both slow and memory hungry.
    """
    key = _autotune_key(q4, k4, v4)
    hit = _AUTOTUNE_CACHE.get(key)
    if hit is not None:
        return hit

    # CPU/unit-test fallback: no CUDA events, no reason to autotune.
    if q4.device.type != "cuda" or not torch.cuda.is_available():
        _AUTOTUNE_CACHE[key] = None
        return None

    warmup = _bench_int_env("VDN_H3_SDPA_BENCH_WARMUP", 1, 1, 5)
    reps = _bench_int_env("VDN_H3_SDPA_BENCH_REPS", 3, 1, 10)
    candidates = ("flash", "cudnn", "mem_efficient")
    timings = {}

    # A rejected backend may emit warnings for every probe. Keep the Comfy log
    # readable; availability and timings are reported below.
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        for name in candidates:
            try:
                for _ in range(warmup):
                    out = _sdpa_call(q4, k4, v4, scale, backend=name)
                torch.cuda.synchronize(q4.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(reps):
                    out = _sdpa_call(q4, k4, v4, scale, backend=name)
                end.record()
                end.synchronize()
                timings[name] = float(start.elapsed_time(end)) / reps
                del out
            except Exception:
                continue

    if timings:
        chosen = min(timings, key=timings.get)
    else:
        # Preserve correctness if no accelerated backend accepts this shape.
        try:
            _sdpa_call(q4, k4, v4, scale, backend="math")
            torch.cuda.synchronize(q4.device)
            chosen = "math"
            timings["math"] = float("nan")
        except Exception:
            chosen = None  # Let PyTorch's default chain raise/report normally.

    _AUTOTUNE_CACHE[key] = chosen
    if key not in _AUTOTUNE_LOGGED:
        _AUTOTUNE_LOGGED.add(key)
        shape = (f"B{q4.shape[0]} H{q4.shape[1]} Q{q4.shape[2]} "
                 f"K{k4.shape[2]} D{q4.shape[3]}")
        parts = []
        for name in candidates:
            if name in timings:
                parts.append(f"{name}={timings[name]:.3f}ms")
            else:
                parts.append(f"{name}=NA")
        if chosen == "math":
            parts.append("math=fallback")
        _log.info("[vdn-sdpa-bench] %s -> %s | %s | warmup=%d reps=%d",
                  shape, chosen or "default", ", ".join(parts), warmup, reps)
    return chosen


def _resolved_backend(q4, k4, v4, scale):
    mode = _forced_backend()
    if mode == "benchmark":
        return _benchmark_backend(q4, k4, v4, scale)
    return mode


def _log_backend_once(q4, k4, v4, scale):
    """Name the exact kernel the priority chain actually picks for these window
    shapes: probe flash -> cuDNN -> mem-efficient -> math once, in order."""
    global _BACKEND_LOGGED
    if _BACKEND_LOGGED:
        return
    _BACKEND_LOGGED = True
    forced = _forced_backend()
    if forced == "benchmark":
        _log.info("[vdn] window SDPA backend: benchmark/autotune (v18; exact per-shape CUDA microbenchmark)")
        return
    if forced is not None:
        _log.info("[vdn] window SDPA backend: %s (forced via "
                  "VDN_H3_WINDOW_SDPA)", forced)
        return
    chosen = "math"
    # torch emits a UserWarning per rejected backend while probing ("not used
    # because ...", "runtime disabled"); users don't need the spam -- the info
    # line below reports the winner.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in _BACKEND_PRIORITY:
            try:
                _sdpa_call(q4, k4, v4, scale, backend=name)
                chosen = name
                break
            except Exception:
                continue
    _log.info("[vdn] window SDPA backend: %s (priority flash -> cuDNN -> "
              "mem-efficient; override with VDN_H3_WINDOW_SDPA=benchmark|flash|cudnn|"
              "mem_efficient)", chosen)


def _sdpa(q_rows, k_rows, v_rows, scale, transformer_options=None):
    """[rows, H, d] x [keys, H, d] -> [rows, H, d] via one dense attention.

    With transformer options the call goes through ComfyUI's dispatched
    attention, so any optimized_attention_override on the model (e.g. a sage
    patch) applies to the window groups exactly as it does to the base model's
    dense attention. The dispatched functions scale by head_dim ** -0.5
    internally, which is the `scale` the caller passes. Without them (unit
    tests, CPU) raw SDPA keeps the path dependency-free."""
    if transformer_options is not None:
        from comfy.ldm.modules import attention as comfy_attention
        rows, heads, dim = q_rows.shape
        out = comfy_attention.optimized_attention(
            q_rows.reshape(1, rows, heads * dim),
            k_rows.reshape(1, -1, heads * dim),
            v_rows.reshape(1, -1, heads * dim),
            heads, transformer_options=transformer_options)
        return out.reshape(rows, heads, dim)
    # No override: still dispatch through comfy's backend-priority chain
    # (flash -> cuDNN -> mem-efficient), since Windows torch builds ship without
    # the flash kernel and raw F.sdpa lands on the slow mem-efficient backend.
    q4 = q_rows.permute(1, 0, 2).unsqueeze(0)
    k4 = k_rows.permute(1, 0, 2).unsqueeze(0)
    v4 = v_rows.permute(1, 0, 2).unsqueeze(0)
    _log_backend_once(q4, k4, v4, scale)
    forced = _resolved_backend(q4, k4, v4, scale)
    if forced is not None and forced not in _FORCED_BROKEN:
        try:
            return _sdpa_call(q4, k4, v4, scale, backend=forced) \
                .squeeze(0).permute(1, 0, 2)
        except RuntimeError as e:
            _FORCED_BROKEN.add(forced)
            _log.warning("[vdn] selected window SDPA backend %s unavailable (%s); "
                         "using the default chain from here on", forced, e)
    try:
        from comfy.ops import scaled_dot_product_attention as comfy_sdpa
    except ImportError:                      # unit tests run without comfy on path
        comfy_sdpa = F.scaled_dot_product_attention
    attended = comfy_sdpa(q4, k4, v4, scale=scale)
    return attended.squeeze(0).permute(1, 0, 2)


# ---------------------------------------------------------------- flex path --

_FLEX = None
_BM_CACHE = {}


def _build_window_tables(seq, video_start, video_end, num_frames,
                         tokens_per_frame, bounds, device):
    """Per-token [lo, hi] allowed video-frame ranges; the table both the flex
    mask_mod and the dense test oracle index into."""
    lo = torch.zeros(seq, dtype=torch.long, device=device)
    hi = torch.full((seq,), num_frames - 1, dtype=torch.long, device=device)
    for f in range(num_frames):
        a = video_start + f * tokens_per_frame
        lo[a:a + tokens_per_frame] = max(bounds[f][0], 0)
        hi[a:a + tokens_per_frame] = min(bounds[f][1], num_frames - 1)
    return lo, hi


def _window_mask_mod(video_start, video_end, num_frames, tokens_per_frame,
                     lo, hi, anchor_frames):
    """mask_mod over token indices: globals dense both ways, video restricted to
    its chunk window, plus anchor columns and/or anchor rows."""
    allow_k = anchor_frames in ("columns", "both")
    allow_q = anchor_frames in ("rows", "both")

    def mask_mod(b, h, q, kv):
        gq = (q < video_start) | (q >= video_end)
        gk = (kv < video_start) | (kv >= video_end)
        qf = (q - video_start) // tokens_per_frame
        kf = (kv - video_start) // tokens_per_frame
        allowed = gq | gk | ((kf >= lo[q]) & (kf <= hi[q]))
        if allow_k:
            allowed = allowed | (kf == 0) | (kf == num_frames - 1)
        if allow_q:
            allowed = allowed | (qf == 0) | (qf == num_frames - 1)
        return allowed

    return mask_mod


def window_softmax_flex(query, key, value, video_start, video_end, num_frames,
                        tokens_per_frame, bounds, scale, anchor_frames="none"):
    """The same window partition as window_softmax_grouped, executed as one fused
    FlexAttention kernel over the full sequence with a BlockMask -- the official
    release's softmax architecture, minus its FA4 backend. Needs torch.compile +
    triton; the first call per sequence shape compiles (and the BlockMask is
    cached per shape)."""
    global _FLEX
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention)
    seq = query.shape[0]
    ck = (seq, video_start, video_end, num_frames, tokens_per_frame,
          anchor_frames, tuple(tuple(b) for b in bounds), query.device.type)
    bm = _BM_CACHE.get(ck)
    if bm is None:
        lo, hi = _build_window_tables(seq, video_start, video_end, num_frames,
                                      tokens_per_frame, bounds, query.device)
        bm = create_block_mask(
            _window_mask_mod(video_start, video_end, num_frames,
                             tokens_per_frame, lo, hi, anchor_frames),
            None, None, seq, seq, query.device, _compile=True)
        _BM_CACHE[ck] = bm
    out = _FLEX(query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                block_mask=bm, scale=scale)
    return out.squeeze(0).transpose(0, 1)
