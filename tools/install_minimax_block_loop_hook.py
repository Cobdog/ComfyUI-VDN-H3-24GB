#!/usr/bin/env python
"""Install the MiniMax-H3 block-loop slicing hook required by ComfyUI-VDN-H3-24GB.

Supports the current ComfyUI 0.33.x MiniMax-H3 loop (2026-08/09 layout) and
keeps the patch limited to comfy/ldm/minimax/model.py. A backup is created
before the first change. Re-running is idempotent. Use --revert to restore it.
"""

import argparse
import ast
import re
import shutil
from pathlib import Path

TARGET = Path("comfy/ldm/minimax/model.py")
BACKUP_SUFFIX = ".vdn_longcache.bak"

# This helper mirrors the current upstream loop exactly, but allows a start/end
# slice.  It intentionally preserves patches_replace and ComfyUI prefetch.
RUN_BLOCKS = '''    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        device = h.device
        start = max(0, int(start))
        end = len(self.blocks) if end is None else min(len(self.blocks), int(end))
        if end <= start:
            return h
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks[start:end]), device, transformer_options)
        for i in range(start, end):
            block = self.blocks[i]
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        return h

'''

HOOK_LOOP = '''        # blocks (VDN/TE-compatible block-loop hook)
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        cache_ranges = [(a, b) for a, b, kind in layout.segments if kind in ("audio", "video")]
        if ("block_loop", 0) in blocks_replace:
            def block_loop_wrap(args):
                return {"img": self._run_blocks(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                                args["transformer_options"], args.get("start", 0), args.get("end"))}
            h = blocks_replace[("block_loop", 0)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options, "cache_ranges": cache_ranges,
                 "block_count": len(self.blocks)},
                {"original_block": block_loop_wrap})["img"]
        else:
            h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)
'''

# Current ComfyUI 0.33.x block loop. Keep the regex deliberately strict enough
# to avoid touching an unknown future implementation, but tolerant of whitespace.
LOOP_RE_CURRENT = re.compile(
    r'        # blocks\n'
    r'        patches_replace = transformer_options\.get\("patches_replace", \{\}\)\n'
    r'        blocks_replace = patches_replace\.get\("dit", \{\}\)\n'
    r'        prefetch_queue = comfy\.model_prefetch\.make_prefetch_queue\(list\(self\.blocks\), device, transformer_options\)\n'
    r'        for i, block in enumerate\(self\.blocks\):\n'
    r'.*?'
    r'        if prefetch_queue is not None:\n'
    r'            comfy\.model_prefetch\.prefetch_queue_pop\(prefetch_queue, device, None\)\n',
    re.DOTALL,
)

# Earlier tested build used malloc_scope="block". Preserve backwards support.
LOOP_RE_OLD = re.compile(
    r'        # blocks\n'
    r'        patches_replace = transformer_options\.get\("patches_replace", \{\}\)\n'
    r'        blocks_replace = patches_replace\.get\("dit", \{\}\)\n'
    r'        prefetch_queue = comfy\.model_prefetch\.make_prefetch_queue\(list\(self\.blocks\), device, transformer_options\)\n'
    r'.*?'
    r'        comfy\.model_prefetch\.prefetch_queue_pop\(prefetch_queue, device, None, malloc_scope="block"\)\n',
    re.DOTALL,
)

FORWARD_ANCHOR = "    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):"


def find_target(root):
    root = Path(root).resolve()
    candidates = [root / TARGET, root / "ComfyUI" / TARGET]
    for p in candidates:
        if p.is_file():
            return p
    raise SystemExit(f"VDN-H3-24GB: could not find {TARGET} under {root}")


def status(text):
    return "def _run_blocks(self, h, t_emb" in text and '("block_loop", 0) in blocks_replace' in text


def patch_text(text):
    if status(text):
        return text
    matcher = LOOP_RE_CURRENT.search(text)
    loop_re = LOOP_RE_CURRENT
    if matcher is None:
        matcher = LOOP_RE_OLD.search(text)
        loop_re = LOOP_RE_OLD
    if matcher is None:
        raise SystemExit(
            "VDN-H3-24GB: could not safely locate the MiniMax-H3 block loop in this ComfyUI build; "
            "no file was changed. Please provide comfy/ldm/minimax/model.py.")
    text = loop_re.sub(HOOK_LOOP, text, count=1)
    if "def _run_blocks(self, h, t_emb" not in text:
        if FORWARD_ANCHOR not in text:
            raise SystemExit("VDN-H3-24GB: MiniMax-H3 _forward anchor not found; no file was changed.")
        text = text.replace(FORWARD_ANCHOR, RUN_BLOCKS + FORWARD_ANCHOR, 1)
    ast.parse(text)
    if not status(text):
        raise SystemExit("VDN-H3-24GB: internal hook validation failed; no file was changed.")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-ui", default=".", help="ComfyUI root (default: current directory)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ns = ap.parse_args()
    target = find_target(ns.comfy_ui)
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    text = target.read_text(encoding="utf-8")

    if ns.revert:
        if not backup.is_file():
            raise SystemExit(f"VDN-H3-24GB: no backup found at {backup}")
        shutil.copy2(backup, target)
        print(f"VDN-H3-24GB: restored {target} from {backup.name}")
        return

    if ns.check:
        print(f"VDN-H3-24GB: block-loop hook {'present' if status(text) else 'missing'} in {target}")
        return

    if status(text):
        print(f"VDN-H3-24GB: block-loop hook already present in {target}")
        return

    patched = patch_text(text)
    if not backup.is_file():
        shutil.copy2(target, backup)
        print(f"VDN-H3-24GB: backup written to {backup}")
    target.write_text(patched, encoding="utf-8")
    print(f"VDN-H3-24GB: block-loop hook installed in {target}")


if __name__ == "__main__":
    main()
