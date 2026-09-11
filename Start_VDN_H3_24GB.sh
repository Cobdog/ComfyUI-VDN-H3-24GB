#!/usr/bin/env bash
# Linux launcher for ComfyUI-VDN-H3-24GB. Mirrors Start_VDN_H3_24GB.bat and
# finds the ComfyUI Python environment (venv, .venv/uv, conda, system) by itself.
set -euo pipefail
shopt -s nullglob
if ((BASH_VERSINFO[0] < 4)); then
    echo "[VDN-H3-24GB] ERROR: bash >= 4 is required (run: bash $0)" >&2
    exit 1
fi

TAG="[VDN-H3-24GB]"
HOOK_NAME=install_minimax_block_loop_hook.py

usage() {
    cat <<EOF
Usage: ${0##*/} [--dry-run] [--revert-hook] [--uv-sync] [--python PATH] [-- MAIN_ARGS...]

  --dry-run      resolve and validate everything, print the launch command, don't launch
  --revert-hook  undo the MiniMax block-loop hook and exit
  --uv-sync      if uv.lock exists but no venv does, run "uv sync" before detection
  --python PATH  use this interpreter (--python=PATH also works; or set VDN_PYTHON)
  -- ...         remaining arguments are passed to ComfyUI's main.py

Conda environments can be chosen with VDN_CONDA_ENV (name or path).
ComfyUI launch arguments can be saved, one per line, in Start_VDN_H3_24GB.args
next to this script; they override the VDN defaults, and -- args override them.
Exit codes: 0 success, 1 runtime error, 2 usage error. Requires bash >= 4.
EOF
}

die() { echo; echo "$TAG ERROR: $*" >&2; echo >&2; exit 1; }

DRY_RUN=0 REVERT_HOOK=0 UV_SYNC=0 PYTHON_SEEN=0
PYTHON_OVERRIDE=${VDN_PYTHON:-}
EXTRA_ARGS=()
while (($#)); do
    case $1 in
        --dry-run) DRY_RUN=1 ;;
        --revert-hook) REVERT_HOOK=1 ;;
        --uv-sync) UV_SYNC=1 ;;
        --python) if (($# < 2)) || [[ $2 == -* ]]; then echo "$TAG ERROR: --python needs a path" >&2; exit 2; fi
                  PYTHON_OVERRIDE=$2; PYTHON_SEEN=1; shift ;;
        --python=*) PYTHON_OVERRIDE=${1#*=}; PYTHON_SEEN=1 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$TAG ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
if ((PYTHON_SEEN && ${#PYTHON_OVERRIDE} == 0)); then
    echo "$TAG ERROR: --python needs a path" >&2; exit 2
fi
if ((REVERT_HOOK && UV_SYNC)); then
    echo "$TAG ERROR: --revert-hook and --uv-sync are mutually exclusive." >&2; exit 2
fi
if [[ -n $PYTHON_OVERRIDE ]]; then  # absolutize before any cd
    if [[ $PYTHON_OVERRIDE == /* ]]; then :
    elif [[ $PYTHON_OVERRIDE == */* ]]; then PYTHON_OVERRIDE=$PWD/$PYTHON_OVERRIDE
    else
        PYTHON_OVERRIDE=$(command -v -- "$PYTHON_OVERRIDE" 2>/dev/null || printf '%s' "$PYTHON_OVERRIDE")
        [[ $PYTHON_OVERRIDE == /* ]] || PYTHON_OVERRIDE=$PWD/$PYTHON_OVERRIDE
    fi
fi

# ---- Locate ComfyUI root and node folder (same search as the BAT) ----
# Try the as-invoked path first (like %~dp0), then the symlink-resolved one.
SELF_DIR=$(dirname -- "$0")
CANON_DIR=$(dirname -- "$(readlink -f -- "$0")")

find_root() {
    local base d root
    for base in "$SELF_DIR" "$CANON_DIR"; do
        for d in "$base" "$base/ComfyUI" "$base/../.." "$base/../../.."; do
            if [[ -f $d/main.py ]]; then
                root=$(cd -P -- "$d" 2>/dev/null && pwd || true)
                if [[ -n $root ]]; then printf '%s\n' "$root"; return 0; fi
            fi
        done
    done
    return 1
}

find_node() {
    local c=$1/custom_nodes
    local exact=$c/ComfyUI-VDN-H3-24GB d e
    if [[ -f $exact/__init__.py ]]; then printf '%s\n' "$exact"; return 0; fi
    for d in "$c"/ComfyUI-VDN-H3-24GB*/; do
        if [[ -f $d/__init__.py && -f $d/tools/$HOOK_NAME ]]; then printf '%s\n' "${d%/}"; return 0; fi
    done
    for d in "$c"/ComfyUI-VDN-H3-24GB*/; do
        for e in "$d"ComfyUI-VDN-H3-24GB*/; do
            if [[ -f $e/__init__.py && -f $e/tools/$HOOK_NAME ]]; then printf '%s\n' "${e%/}"; return 0; fi
        done
    done
    return 1
}

COMFY_ROOT=$(find_root) || die "could not locate ComfyUI root containing main.py.
Script location: \"$SELF_DIR\"
Expected the script inside <ComfyUI>/custom_nodes/ComfyUI-VDN-H3-24GB[/nested]."
echo "$TAG ComfyUI root: \"$COMFY_ROOT\""

NODE_DIR=$(find_node "$COMFY_ROOT") || die "release node folder was not found.
Expected a folder matching ComfyUI-VDN-H3-24GB* under:
\"$COMFY_ROOT/custom_nodes\""
HOOK=$NODE_DIR/tools/$HOOK_NAME
echo "$TAG Node folder: \"$NODE_DIR\""

# ---- VDN-H3 v49 24GB production settings (identical to the BAT) ----
export VDN_H3_AUTO_MEMORY=1
export VDN_H3_VERY_LONG_HEADROOM_GIB=2.50
export VDN_H3_OUTPROJ_CACHE_GIB=0
export VDN_H3_LONG_CACHE_ENABLE=1
export VDN_H3_LONG_CACHE_STEPS=3,5,7
export VDN_H3_LONG_CACHE_COPY_ROWS=512
export VDN_H3_PROFILE=0
export VDN_H3_MEM_PROFILE=0
export VDN_H3_IMPORTANCE_PROFILE=0
export VDN_H3_WINDOW_IMPORTANCE_PROFILE=0
export VDN_H3_DELTA_DEEP_PROFILE=0
export VDN_H3_DELTA_SOLVER_LAB=0
export VDN_H3_DELTA_REALPATH_LAB=0
export VDN_H3_OUTPROJ_GEOMETRY_LAB=0
export VDN_H3_REUSE_PROFILE=0
export VDN_H3_QKV_EXEC_LAB=0
export VDN_H3_QKV_DEEP_PROFILE=0
export VDN_H3_WINDOW_DEEP_PROFILE=0
export VDN_H3_TRITON_ATTN_PROBE=0
export VDN_H3_TRITON_ATTN_AUTOTUNE=0
export VDN_H3_OUTPROJ_MAT_PROFILE=0
export VDN_H3_OUTPROJ_BLOCK_PROFILE=0
export VDN_H3_OUTPROJ_CACHE_FORMAT_PROFILE=0
export VDN_H3_STREAM_PREFETCH=0
export VDN_H3_BRANCH_OVERLAP=0
export VDN_H3_DELTA_SOLVE=direct
export VDN_H3_ABLATION=hybrid
export VDN_H3_WINDOW_GROUP_BATCH=1
export VDN_H3_HYBRID_STRIDE=1
export VDN_H3_HYBRID_BLOCKS=0,2,4,5,9,12,13,15,17,21,24,27,30,36,42,48
export VDN_H3_WINDOW_INNER_TRIM=4
export VDN_H3_SELECTIVE_WEIGHT_LOAD=1
export VDN_H3_WINDOW_SDPA=cudnn
export VDN_H3_OUTPROJ_CACHE_CLEAR_AFTER=8
export VDN_H3_OUTPROJ_EXECUTION=materialized
export VDN_H3_OUTPROJ_GEMM=mm_bias
export VDN_H3_QKV_EXECUTION=native
# The BAT clears these (cmd "set VAR=" unsets); an empty value is not equivalent.
unset VDN_H3_LONG_CACHE_DEPTH VDN_H3_WINDOW_BLOCKS VDN_H3_WINDOW_FALLBACK VDN_H3_OUTPROJ_CACHE_BLOCKS

# ---- Find a Python that is the working ComfyUI environment ----
TRIED=()

try_python() {
    if [[ ! -x $2 ]]; then TRIED+=("$1: not found ($2)"); return 1; fi
    if ! "$2" -c "import sqlalchemy" >/dev/null 2>&1; then
        TRIED+=("$1: failed sqlalchemy check, not the ComfyUI environment ($2)")
        return 1
    fi
    echo "$TAG Python [$1]: \"$2\""
    PYTHON=$2
}

PYTHON=
p=

if [[ -n $PYTHON_OVERRIDE ]]; then
    try_python "--python / VDN_PYTHON override" "$PYTHON_OVERRIDE" \
        || die "the interpreter you specified is not a working ComfyUI environment:
$PYTHON_OVERRIDE
It must be executable and able to import sqlalchemy."
fi

# VDN_CONDA_ENV is an explicit choice: it outranks the heuristics below, and a bad value aborts.
if [[ -z $PYTHON && -n ${VDN_CONDA_ENV:-} ]]; then
    if [[ $VDN_CONDA_ENV == /* ]]; then
        try_python "conda env at $VDN_CONDA_ENV" "$VDN_CONDA_ENV/bin/python" \
            || die "VDN_CONDA_ENV is not a usable ComfyUI environment: $VDN_CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then
        mapfile -t CONDA_ENVS < <(conda env list --json 2>/dev/null | sed -n '/"envs":/,/]/p' | grep -o '"/[^"]*"' | tr -d '"')
        CONDA_HIT=
        for e in ${CONDA_ENVS[@]+"${CONDA_ENVS[@]}"}; do
            if [[ $(basename -- "$e") == "$VDN_CONDA_ENV" ]]; then CONDA_HIT=$e; break; fi
        done
        if [[ -n $CONDA_HIT ]]; then
            try_python "conda env '$VDN_CONDA_ENV'" "$CONDA_HIT/bin/python" \
                || die "conda env '$VDN_CONDA_ENV' failed the sqlalchemy check: $CONDA_HIT"
        else
            die "no conda environment named '$VDN_CONDA_ENV' was found."
        fi
    else
        die "VDN_CONDA_ENV=$VDN_CONDA_ENV but conda was not found on PATH."
    fi
fi

if [[ -z $PYTHON && -n ${VIRTUAL_ENV:-} ]]; then
    try_python "active virtualenv" "$VIRTUAL_ENV/bin/python" || true
fi
if [[ -z $PYTHON && -n ${CONDA_PREFIX:-} ]]; then
    try_python "active conda environment" "$CONDA_PREFIX/bin/python" || true
fi

if [[ -z $PYTHON ]]; then
    for v in venv .venv; do
        if try_python "$v/ in ComfyUI root" "$COMFY_ROOT/$v/bin/python"; then break; fi
    done
fi

if [[ -z $PYTHON ]] && command -v conda >/dev/null 2>&1; then
    mapfile -t CONDA_ENVS < <(conda env list --json 2>/dev/null | sed -n '/"envs":/,/]/p' | grep -o '"/[^"]*"' | tr -d '"')
    MATCHES=()
    for e in ${CONDA_ENVS[@]+"${CONDA_ENVS[@]}"}; do
        if [[ $(basename -- "${e,,}") == *comfy* ]]; then MATCHES+=("$e"); fi
    done
    if ((${#MATCHES[@]} == 1)); then
        try_python "conda env '$(basename -- "${MATCHES[0]}")'" "${MATCHES[0]}/bin/python" || true
    elif ((${#MATCHES[@]} > 1)); then
        die "multiple conda environments match *comfy*:
$(printf '  %s\n' "${MATCHES[@]}")
Set VDN_CONDA_ENV to choose one."
    else
        TRIED+=("conda: no environment name contains 'comfy'")
    fi
fi

# uv project without a venv yet: ask for (or run) uv sync. Never blocks --revert-hook.
UV_GATE=0
if (( ! REVERT_HOOK )) && [[ -z $PYTHON && ! -d $COMFY_ROOT/venv && ! -d $COMFY_ROOT/.venv && -f $COMFY_ROOT/uv.lock ]] \
   && command -v uv >/dev/null 2>&1; then
    UV_GATE=1
    if ((UV_SYNC)); then
        if ((DRY_RUN)); then
            echo "$TAG dry run: would run 'uv sync' in \"$COMFY_ROOT\", then use its .venv."
            exit 0
        fi
        (cd -- "$COMFY_ROOT" && uv sync) || die "uv sync failed."
        try_python ".venv after uv sync" "$COMFY_ROOT/.venv/bin/python" || true
    else
        echo "$TAG uv.lock found but the ComfyUI root has no venv."
        echo "$TAG Run 'uv sync' in \"$COMFY_ROOT\", or rerun with --uv-sync."
        if ((DRY_RUN)); then
            echo "$TAG dry run: a real run stops here until the venv exists."
            exit 0
        fi
        exit 1
    fi
fi
if ((UV_SYNC && ! UV_GATE)); then
    echo "$TAG --uv-sync ignored: it only applies when uv.lock exists and no venv does."
fi

if [[ -z $PYTHON ]]; then
    p=$(command -v python3 2>/dev/null || true)
    try_python "python3 from PATH" "${p:-python3}" || true
fi

# Recovery path: the hook installer is stdlib-only, so plain python3 suffices.
if [[ -z $PYTHON && -n $p ]] && ((REVERT_HOOK)); then
    echo "$TAG no ComfyUI environment found; using plain python3 for --revert-hook."
    PYTHON=$p
fi

if [[ -z $PYTHON ]]; then
    echo >&2
    echo "$TAG ERROR: no usable ComfyUI Python environment was found. Tried:" >&2
    printf '  - %s\n' "${TRIED[@]}" >&2
    echo "$TAG Activate the environment you normally use for ComfyUI, or set one of:" >&2
    echo "$TAG   VDN_PYTHON=/path/to/python $0" >&2
    echo "$TAG   VDN_CONDA_ENV=<conda env name> $0" >&2
    echo >&2
    exit 1
fi

# ---- Revert path ----
if ((REVERT_HOOK)); then
    if ((DRY_RUN)); then
        echo "$TAG dry run: would run: \"$PYTHON\" \"$HOOK\" --comfy-ui \"$COMFY_ROOT\" --revert"
    else
        exec "$PYTHON" "$HOOK" --comfy-ui "$COMFY_ROOT" --revert
    fi
    exit 0
fi

cd -- "$COMFY_ROOT"

# ---- Hook ----
if ((DRY_RUN)); then
    HOOK_OUT=$("$PYTHON" "$HOOK" --comfy-ui "$COMFY_ROOT" --check) \
        || die "block-loop hook check failed."
    echo "$HOOK_OUT"
    if [[ $HOOK_OUT == "VDN-H3-24GB: block-loop hook missing "* ]]; then
        echo "$TAG note: a real run will install the hook (backup kept alongside)."
    fi
else
    "$PYTHON" "$HOOK" --comfy-ui "$COMFY_ROOT" || die "block-loop hook installation/check failed."
fi

# ---- Launch ----
# User-saved arguments (one per line, #-line comments) override the VDN
# defaults; -- passthrough overrides the file.
USER_ARGS=()
ARGS_FILE=$NODE_DIR/Start_VDN_H3_24GB.args
if [[ -f $ARGS_FILE ]]; then
    while IFS= read -r line || [[ -n $line ]]; do
        line=${line%$'\r'}
        line=${line#"${line%%[![:space:]]*}"}
        line=${line%"${line##*[![:space:]]}"}
        if [[ -z $line || $line == \#* ]]; then continue; fi
        USER_ARGS+=("$line")
    done < "$ARGS_FILE"
    if ((${#USER_ARGS[@]} > 0)); then
        echo "$TAG adding ${#USER_ARGS[@]} launch argument(s) from ${ARGS_FILE##*/}"
    fi
fi

ARGS=(main.py --listen 127.0.0.1 --port 8191 --disable-dynamic-vram --disable-async-offload)
if "$PYTHON" -c "from sageattention import sageattn" >/dev/null 2>&1; then
    ARGS+=(--use-sage-attention)
    echo "$TAG SageAttention detected; enabling --use-sage-attention."
else
    echo "$TAG SageAttention not found; launching without it. Performance may differ."
fi
ARGS+=(${USER_ARGS[@]+"${USER_ARGS[@]}"})
ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

if ((DRY_RUN)); then
    echo "$TAG dry run OK. Would run in \"$COMFY_ROOT\":"
    printf '  %q ' "$PYTHON" "${ARGS[@]}"
    echo
    exit 0
fi
echo
echo "$TAG Starting ComfyUI..."
exec "$PYTHON" "${ARGS[@]}"
