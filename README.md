# ComfyUI-VDN-H3-24GB

24 GB-oriented MiniMax-H3 VDN runtime for ComfyUI, based on the v49 memory-policy build tested on an RTX 3090 Ti 24 GB.

## What this release contains

- Unique ComfyUI node IDs: `ApplyVDNH3_24GB` and `ApplyVDNH3Advanced_24GB`.
- Duration-aware AutoMemory and SHORT/LONG/VERY_LONG residency transitions.
- AutoLongCache for long H3 sequences.
- Selective VDN branch loading and the tested Ampere 24 GB launch profile.
- A reversible MiniMax-H3 block-loop hook installer required by LongCache.

## Installation

Extract/clone the repository so the final path is exactly:

```text
<ComfyUI>/custom_nodes/ComfyUI-VDN-H3-24GB/
```

Inside that folder you should directly see:

```text
__init__.py
vdn_h3_24gb/
tools/
Start_VDN_H3_24GB.bat
```

Do **not** leave an extra nested directory such as:

```text
custom_nodes/ComfyUI-VDN-H3-24GB/
```

`Check_Installation_24GB.bat` can verify the basic layout.


## VDN checkpoint

The VDN stage checkpoint is distributed separately on Hugging Face:

[**speach1sdef178/VDN-H3-INT8-ConvRot-ComfyUI**](https://huggingface.co/speach1sdef178/VDN-H3-INT8-ConvRot-ComfyUI)

Download the complete folder:

```text
stage-dmd-step-250-int8_convrot_comfyui
```

and place it at:

```text
ComfyUI/models/vdn/stage-dmd-step-250-int8_convrot_comfyui/
```

Do not download only the linear-branch `.safetensors`; the complete stage also contains `model_spec.json`, the default adapter, and the turbo adapter.

## Starting ComfyUI

The included `Start_VDN_H3_24GB.bat` can be run directly from the node folder. It resolves the ComfyUI root from its own location and does not depend on the Command Prompt working directory.

If your ComfyUI uses Conda, either activate the environment before launching or edit this line near the top of the BAT:

```bat
set "VDN_CONDA_ENV="
```

to your environment name, for example:

```bat
set "VDN_CONDA_ENV=ComfyUI_Krea2"
```

The BAT verifies Python, resolves the hook path, patches/checks the MiniMax block loop, and only then starts ComfyUI. If SageAttention is installed, the BAT enables it automatically; otherwise it launches without the SageAttention flag and warns that performance may differ.


## Tested node preset

For the 24 GB profile, use the **Apply VDN-H3 24GB Optimized** node with:

```text
vdn_checkpoint      = stage-dmd-step-250-int8_convrot_comfyui
apply_turbo_adapter = true
strength            = 1.0
lora_mode           = merge
branch_weights      = stream
retain_buffers      = auto
attention_backend   = grouped
```

Connect the **same H3 LATENT** that goes to the sampler to the node's `auto_memory_latent` input. This connection is required for the tested duration-aware 24 GB AutoMemory/AutoLongCache policy. The included example workflow is configured this way.

`branch_weights=stream` is the tested 24 GB default. `auto` remains available for experimentation, but it is not the published 3090 Ti preset.

## Core hook

LongCache needs a small hook in:

```text
comfy/ldm/minimax/model.py
```

`tools/install_minimax_block_loop_hook.py` is deliberately conservative: it only patches a recognized MiniMax-H3 block-loop layout, validates the result with Python AST parsing, creates `model.py.vdn_longcache.bak` before the first modification, and is safe to run repeatedly.

To restore the backup manually:

```bat
python custom_nodes\ComfyUI-VDN-H3-24GB\tools\install_minimax_block_loop_hook.py --comfy-ui . --revert
```

## Tested profile

RTX 3090 Ti 24 GB, Windows, ComfyUI 0.33.x-era MiniMax-H3 implementation, 0.4 MP, 8-step DMD, H3 FL2VA INT8 ConvRot. Tests completed at 5 s, 10 s, 12 s and 15 s. This is a tested configuration, not a guarantee for every 24 GB GPU or every future ComfyUI build.

## Compatibility note

The public node IDs and Python package namespace are distinct from other VDN-H3 ports so both packages can be installed without sharing the same ComfyUI node IDs. The MiniMax core hook is a ComfyUI-core patch and therefore remains a shared runtime modification; the installer is idempotent and creates a backup.

## License / attribution

This project is derived from the released VideoDeltaNet/OpenVDN work and an existing ComfyUI VDN-H3 port. Preserve the included `LICENSE` notices. MiniMax-H3 model weights and VDN checkpoints may have separate licenses; review them before redistribution or commercial use.


## Tested 24 GB preset

Use the Optimized node with:

```text
vdn_checkpoint      = stage-dmd-step-250-int8_convrot_comfyui
apply_turbo_adapter = true
strength            = 1.0
lora_mode           = merge
branch_weights      = stream
retain_buffers      = auto
attention_backend   = grouped
auto_memory_latent  = the SAME H3 latent used by the sampler
```

The 24 GB launcher sets the memory policy used for our validated Ampere tests.

### Validated release-build result

- GPU: RTX 3090 Ti 24 GB
- Base: MiniMax H3 FL2VA INT8 ConvRot
- Resolution: 0.4 MP
- Duration: 10 s
- Steps: 8
- Sampling: **2:08 total, 16.06 s/it**

Longer 5 s / 10 s / 12 s / 15 s workloads were also completed during development on the same 24 GB system.

## Windows launcher

`Start_VDN_H3_24GB.bat` expects `python` to resolve to the same working environment you normally use for ComfyUI.  
If you use Conda, activate that environment first, then run the BAT.

A template is also included:

```text
Start_VDN_H3_24GB_CONDA_TEMPLATE.bat
```

Edit `YOUR_COMFY_ENV` to your own environment name.

The launcher checks the MiniMax block-loop hook before starting ComfyUI.

