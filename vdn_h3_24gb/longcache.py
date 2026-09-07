"""v49 AutoLongCache tail-block cache engine.

This is intentionally conservative and deterministic.  It is inspired by the
public TE-Speed MiniMax-H3 block-loop cache idea, but the implementation here is
purpose-built for VDN-H3 and for memory-constrained 24 GiB cards:

* cache only on selected NFE numbers (default 3,5,7),
* recompute a duration-selected leading portion of the 50-block DiT,
* store only the skipped-tail residual on CPU,
* stream that residual back in small row chunks on cache steps so no ~GiB-sized
  temporary residual has to live on the GPU.

The MiniMax-H3 core needs a small block-loop hook.  tools/install_minimax_block_loop_hook.py
installs it and writes a backup next to ComfyUI's model.py.
"""

import logging
import os

import torch

_log = logging.getLogger("comfy.vdn")


def _env_on(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _parse_steps(raw):
    out = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except Exception:
            continue
        if n > 0:
            out.add(n)
    return out


class ControlledTailCache:
    """Deterministic block-loop replacement for the v49 AutoLongCache.

    A FULL+CAPTURE step runs blocks [0:k), copies target-stream rows of the
    midpoint to CPU, then runs [k:N) and stores (final-midpoint) on CPU.  A CACHE
    step runs only [0:k) and adds the previously stored tail residual back to
    audio/video rows in small H2D chunks.
    """

    def __init__(self, depth=0.40, cache_steps=(3, 5, 7), copy_rows=512):
        self.depth = max(0.0, min(0.95, float(depth)))
        self.cache_steps = set(int(x) for x in cache_steps)
        self.copy_rows = max(64, int(copy_rows))
        self.last_sigma = None
        self.step = 0
        self.last_mode = "full"
        self.residuals = None
        self.ranges = None
        self.block_count = None
        self._logged_setup = False

    def reset(self):
        self.last_sigma = None
        self.step = 0
        self.last_mode = "full"
        self.residuals = None
        self.ranges = None
        self.block_count = None

    @staticmethod
    def _sigma(args):
        opts = args.get("transformer_options") or {}
        sigma = opts.get("sigmas")
        if sigma is None:
            return None
        try:
            return float(torch.as_tensor(sigma).flatten()[0].float())
        except Exception:
            return None

    def _advance_step(self, sigma):
        if sigma is None:
            # Fallback for custom samplers that do not expose sigma.  The normal
            # H3 sampler does, but this still keeps the laboratory usable.
            self.step += 1
            return True
        if self.last_sigma is None or sigma > self.last_sigma + 1e-6:
            # New denoise run (sigma jumps back upward).
            self.residuals = None
            self.ranges = None
            self.block_count = None
            self.step = 1
            self.last_sigma = sigma
            return True
        if abs(sigma - self.last_sigma) <= 1e-6:
            return False
        self.step += 1
        self.last_sigma = sigma
        return True

    def _warm_blocks(self, n):
        return max(1, min(n - 1, round(n * (1.0 - self.depth))))

    @staticmethod
    def _normalize_ranges(ranges, rows):
        cleaned = []
        for item in ranges or ():
            try:
                a, b = int(item[0]), int(item[1])
            except Exception:
                continue
            a = max(0, min(rows, a))
            b = max(a, min(rows, b))
            if b > a:
                cleaned.append((a, b))
        return cleaned

    @staticmethod
    def _cpu_copy(x):
        # Blocking CPU copies are deliberate: they keep lifetime/stream rules
        # simple on Windows and avoid retaining extra GPU tensors across blocks.
        return x.detach().to(device="cpu", copy=True)

    def _capture_residual(self, h_mid, h_final, ranges):
        residuals = []
        total_bytes = 0
        for a, b in ranges:
            mid = self._cpu_copy(h_mid[a:b])
            fin = self._cpu_copy(h_final[a:b])
            fin.sub_(mid)
            total_bytes += fin.numel() * fin.element_size()
            residuals.append(fin)
            del mid
        self.residuals = residuals
        self.ranges = list(ranges)
        return total_bytes

    def _add_residual(self, h):
        if not self.residuals or not self.ranges:
            return h
        for (a, b), residual in zip(self.ranges, self.residuals):
            n = b - a
            for off in range(0, n, self.copy_rows):
                end = min(n, off + self.copy_rows)
                chunk = residual[off:end].to(device=h.device, dtype=h.dtype)
                h[a + off:a + end].add_(chunk)
                del chunk
        return h

    def __call__(self, args, kwargs):
        original = kwargs.get("original_block")
        if original is None:
            raise RuntimeError("v42 long-cache hook did not receive original_block")

        block_count = int(args.get("block_count", 0))
        h0 = args.get("img")
        if block_count < 2 or h0 is None:
            return original(args)

        ranges = self._normalize_ranges(args.get("cache_ranges"), h0.shape[0])
        if not ranges:
            if not self._logged_setup:
                self._logged_setup = True
                _log.warning("[vdn-longcache] no audio/video cache ranges from MiniMax block-loop hook; running FULL")
            return original(args)

        sigma = self._sigma(args)
        advanced = self._advance_step(sigma)
        # Same-sigma duplicate forwards should reproduce the same choice without
        # advancing the NFE pattern.
        nfe = max(1, self.step)
        k = self._warm_blocks(block_count)
        can_cache = (nfe in self.cache_steps and self.residuals is not None and
                     self.ranges == ranges and self.block_count == block_count)

        if not self._logged_setup:
            self._logged_setup = True
            _log.info(
                "[vdn-longcache] v49 AutoLongCache: depth=%.2f warm=%d/%d cache_steps=%s copy_rows=%d device=cpu",
                self.depth, k, block_count,
                ",".join(str(x) for x in sorted(self.cache_steps)), self.copy_rows)

        if can_cache:
            h = original({**args, "start": 0, "end": k})["img"]
            h = self._add_residual(h)
            self.last_mode = "cache"
            _log.info("[vdn-longcache] NFE %d CACHE warm=%d skipped=%d", nfe, k, block_count - k)
            return {"img": h}

        # Only the FULL steps immediately preceding a configured CACHE step need
        # to pay the midpoint-copy cost.  For the default 3/5/7 pattern those are
        # 2/4/6.  NFE1 and NFE8 remain ordinary full passes.
        need_capture = (nfe + 1) in self.cache_steps
        if need_capture:
            h_mid = original({**args, "start": 0, "end": k})["img"]
            # Preserve only target-stream rows before the tail mutates h_mid in
            # place.  CPU memory is plentiful; GPU peak remains essentially flat.
            mids = [self._cpu_copy(h_mid[a:b]) for a, b in ranges]
            h_final = original({**args, "img": h_mid, "start": k, "end": block_count})["img"]
            residuals = []
            total_bytes = 0
            for (a, b), mid in zip(ranges, mids):
                fin = self._cpu_copy(h_final[a:b])
                fin.sub_(mid)
                total_bytes += fin.numel() * fin.element_size()
                residuals.append(fin)
            self.residuals = residuals
            self.ranges = list(ranges)
            self.block_count = block_count
            self.last_mode = "full"
            _log.info(
                "[vdn-longcache] NFE %d FULL+CAPTURE warm=%d tail=%d residual=%.3f GiB",
                nfe, k, block_count - k, total_bytes / (1024.0 ** 3))
            return {"img": h_final}

        h = original(args)["img"]
        self.last_mode = "full"
        if nfe >= 8:
            # The released DMD stage uses an 8-NFE cycle; free the CPU residual
            # immediately so a subsequent prompt starts clean.
            self.residuals = None
            self.ranges = None
            self.block_count = None
        _log.info("[vdn-longcache] NFE %d FULL", nfe)
        return {"img": h}


def install_long_cache(model, depth=None):
    """Attach v49 AutoLongCache to an already VDN-patched model.

    ``depth`` is the duration-selected production value.  An explicit
    VDN_H3_LONG_CACHE_DEPTH environment value overrides it for calibration.
    """
    if not (_env_on("VDN_H3_LONG_CACHE_ENABLE", "0") or _env_on("VDN_H3_LONG_CACHE_LAB", "0")):
        return False

    dm = model.get_model_object("diffusion_model")
    if not hasattr(dm, "_run_blocks"):
        _log.warning(
            "[vdn-longcache] MiniMax block-loop hook is not installed; long cache disabled. "
            "Run tools/install_minimax_block_loop_hook.py and restart ComfyUI.")
        return False

    raw_depth = os.environ.get("VDN_H3_LONG_CACHE_DEPTH", "").strip()
    if raw_depth:
        try:
            depth = float(raw_depth)
        except Exception:
            _log.warning("[vdn-longcache] invalid VDN_H3_LONG_CACHE_DEPTH=%r; using AutoLongCache depth", raw_depth)
    if depth is None:
        depth = 0.60
    steps = _parse_steps(os.environ.get("VDN_H3_LONG_CACHE_STEPS", "3,5,7")) or {3, 5, 7}
    try:
        copy_rows = int(os.environ.get("VDN_H3_LONG_CACHE_COPY_ROWS", "512"))
    except Exception:
        copy_rows = 512

    cache = ControlledTailCache(depth=depth, cache_steps=steps, copy_rows=copy_rows)
    model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    return True
