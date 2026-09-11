# Changelog

## Unreleased

- Added `Start_VDN_H3_24GB.sh`, a Linux launcher mirroring the BAT that
  auto-detects the ComfyUI Python environment (active environment, `venv/`,
  `.venv/` including uv-managed ones, conda via `VDN_CONDA_ENV` or a unique
  `*comfy*` name, then `python3`), validates each candidate with an
  `import sqlalchemy` check, probes SageAttention before enabling it, and
  supports `--dry-run`, `--revert-hook`, `--uv-sync` and `--` passthrough.
- Added `Check_Installation_24GB.sh` and a `.gitattributes` entry keeping
  shell scripts LF-only.

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
