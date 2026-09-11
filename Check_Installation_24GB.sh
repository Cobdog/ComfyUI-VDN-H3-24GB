#!/usr/bin/env bash
# Linux mirror of Check_Installation_24GB.bat.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
echo "[VDN-H3-24GB] Node folder: \"$HERE\""

MISSING=0
check() {
    if [[ -f $HERE/$2 ]]; then echo "[OK] $1"; else echo "[MISSING] $1"; MISSING=1; fi
}
check __init__.py __init__.py
check "vdn_h3_24gb/nodes.py" vdn_h3_24gb/nodes.py
check "hook installer" tools/install_minimax_block_loop_hook.py

if [[ -f $HERE/../../main.py ]]; then
    echo "[OK] ComfyUI root: \"$(cd -- "$HERE/../.." && pwd)\""
else
    echo "[WARNING] main.py not found two levels above this folder."
fi
exit "$MISSING"
