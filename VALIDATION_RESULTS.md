# Version 1.1.0 (v50) validation — 2026-09-10

Test system: RTX 3090 24 GB, Windows, eight-step DMD, MiniMax-H3 FL2VA pruned
INT8 ConvRot base, INT8 ConvRot VDN stage, `branch_weights=stream`, transient
buffers, cuDNN window SDPA and the established v49 AutoMemory/AutoLongCache
production policy.

## Completed v50 runs

| Duration | Resolution class | Sampling | Mean / step | Result |
|---:|---:|---:|---:|---|
| 5 s | 0.4 MP | 1:09 cold / 1:05 warm | 8.70 / 8.18 s | Pass |
| 6.583333 s (`length=158`) | 0.4 MP | 1:27 | 10.89 s | Pass |
| 10 s | 0.4 MP | 2:05 | 15.74 s | Pass |
| 15 s | 0.4 MP | 3:21 | 25.22 s | Pass |
| 20 s | 0.4 MP | 4:20 | 32.57 s | Pass |
| 10 s | 0.8 MP | 5:18 | 39.84 s | Pass |
| 10 s + character/style LoRA | 0.4 MP | 2:11 | 16.38 s | Pass |

The first 0.8 MP attempt contained a one-off host paging/offload stall at NFE 2
and took 11:29. An immediate repeat completed normally in 5:18; full NFEs were
approximately 48–50 seconds and cached NFEs approximately 25 seconds.

The 15-to-5-second transition completed successfully. AutoMemory unloaded stale
LONG residency, restored the SHORT profile and completed the following render
without an OOM, CUDA error, NaN or Inf.

## Previous release versus v50

Same prompt, seed, workflow, 0.4 MP geometry and eight-step schedule:

| Test | Previous release | v50 | Difference |
|---|---:|---:|---:|
| 10 s sampling | 2:00 (15.11 s/step) | 2:05 (15.74 s/step) | v50 about 4.2% slower |
| 10 s full execution | 186.93 s | 189.01 s | v50 about 1.1% slower |
| 6.583333 s sampling | 1:23 (10.46 s/step) | 1:27 (10.89 s/step) | v50 about 4.1% slower |

The released adapter application count changed from `default=100, turbo=204`
to `default=104, turbo=208`. The restored token-refiner mappings produced a
material prompt-adherence improvement in two paired reviews. In the more
demanding railway-platform test, v50 better preserved the empty carriage,
subject position outside the train, turn toward the train and final composition,
while substantially reducing an explicitly forbidden human reflection.

The unusual-duration connection-reset report from another installation did not
reproduce locally on either version and is not claimed as fixed by v50.
