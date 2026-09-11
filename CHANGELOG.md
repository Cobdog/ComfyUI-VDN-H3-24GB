# Changelog

## Unreleased

- Added `Start_VDN_H3_24GB.sh`, a Linux launcher mirroring the BAT that
  finds the ComfyUI Python environment by itself, in this order:
  `--python`/`VDN_PYTHON`, `VDN_CONDA_ENV`, the active environment,
  `venv/` or `.venv/` in the ComfyUI root (uv-created included), a uniquely
  `*comfy*`-named conda environment, then `python3` — each validated with an
  `import sqlalchemy` check. Probes SageAttention before enabling it, and
  supports `--dry-run` (fully non-mutating), `--revert-hook`, `--uv-sync`
  and `--` passthrough. Requires bash 4 or newer. Personal launch arguments
  can be saved in `Start_VDN_H3_24GB.args` (one per line); they override the
  VDN defaults, and `--` arguments override the file.
- Added `Check_Installation_24GB.sh` and a `.gitattributes` keeping shell
  scripts LF-only and `.bat` files CRLF.
- Corrected stale README instructions: the `VDN_CONDA_ENV` line the Windows
  section told users to edit does not exist in the BAT (conda users should
  use the included template), and the sage-attention fallback described
  there applies to the new Linux script, not to the BAT, which always passes
  `--use-sage-attention`.

## 1.1.0 — 2026-09-10

Public release version: **1.1.0**.

- Restored token-refiner attention adapter mapping for ComfyUI's fused QKV
  layout. Released adapter application increases from `default=100, turbo=204`
  to `default=104, turbo=208`.
- Added bounded complete-frame batching for frame-statistics preparation,
  limiting temporary workspace to approximately 1 GiB on long clips.
- Corrected cross-stream lifetime tracking for prefetched quantized and plain
  storages when using CUDA allocators including `cudaMallocAsync`.
- Validated continuous 5/10/15/20-second generation at 0.4 MP, 10 seconds at
  0.8 MP, SHORT/LONG transitions and character/style LoRA composition on an
  RTX 3090 24 GB system.
- Added synchronized visual comparison media and detailed validation results.

## 1.0.0

- Initial public 24 GB-optimized VDN-H3 release.
