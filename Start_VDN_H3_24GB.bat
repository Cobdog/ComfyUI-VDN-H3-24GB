@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem ComfyUI-VDN-H3-24GB public launcher
rem Tested preset for 24 GB GPUs
rem ============================================================

set "BAT_DIR=%~dp0"
set "COMFY_ROOT="
set "NODE_DIR="
set "HOOK="

rem ---- Locate ComfyUI root ----
if exist "%BAT_DIR%main.py" (
    set "COMFY_ROOT=%BAT_DIR%"
    goto :root_found
)

if exist "%BAT_DIR%ComfyUI\main.py" (
    set "COMFY_ROOT=%BAT_DIR%ComfyUI"
    goto :root_found
)

for %%I in ("%BAT_DIR%..\..") do (
    if exist "%%~fI\main.py" (
        set "COMFY_ROOT=%%~fI"
        goto :root_found
    )
)

for %%I in ("%BAT_DIR%..\..\..") do (
    if exist "%%~fI\main.py" (
        set "COMFY_ROOT=%%~fI"
        goto :root_found
    )
)

echo.
echo [VDN-H3-24GB] ERROR: could not locate ComfyUI root containing main.py.
echo BAT location: "%BAT_DIR%"
echo.
pause
exit /b 1

:root_found
for %%I in ("%COMFY_ROOT%") do set "COMFY_ROOT=%%~fI"
echo [VDN-H3-24GB] ComfyUI root: "%COMFY_ROOT%"

rem ---- Locate node folder ----
set "CUSTOM_NODES=%COMFY_ROOT%\custom_nodes"

if exist "%CUSTOM_NODES%\ComfyUI-VDN-H3-24GB\__init__.py" (
    set "NODE_DIR=%CUSTOM_NODES%\ComfyUI-VDN-H3-24GB"
    goto :node_found
)

for /d %%D in ("%CUSTOM_NODES%\ComfyUI-VDN-H3-24GB*") do (
    if exist "%%~fD\__init__.py" (
        if exist "%%~fD\tools\install_minimax_block_loop_hook.py" (
            set "NODE_DIR=%%~fD"
            goto :node_found
        )
    )
)

for /d %%D in ("%CUSTOM_NODES%\ComfyUI-VDN-H3-24GB*") do (
    for /d %%E in ("%%~fD\ComfyUI-VDN-H3-24GB*") do (
        if exist "%%~fE\__init__.py" (
            if exist "%%~fE\tools\install_minimax_block_loop_hook.py" (
                set "NODE_DIR=%%~fE"
                goto :node_found
            )
        )
    )
)

echo.
echo [VDN-H3-24GB] ERROR: release node folder was not found.
echo Expected a folder matching ComfyUI-VDN-H3-24GB* under:
echo "%CUSTOM_NODES%"
echo.
pause
exit /b 1

:node_found
set "HOOK=%NODE_DIR%\tools\install_minimax_block_loop_hook.py"
echo [VDN-H3-24GB] Node folder: "%NODE_DIR%"
echo [VDN-H3-24GB] Hook installer: "%HOOK%"

rem ---- VDN-H3 v49 24GB production settings ----
set VDN_H3_AUTO_MEMORY=1
set VDN_H3_VERY_LONG_HEADROOM_GIB=2.50
set VDN_H3_OUTPROJ_CACHE_GIB=0
set VDN_H3_LONG_CACHE_ENABLE=1
set VDN_H3_LONG_CACHE_DEPTH=
set VDN_H3_LONG_CACHE_STEPS=3,5,7
set VDN_H3_LONG_CACHE_COPY_ROWS=512

set VDN_H3_PROFILE=0
set VDN_H3_MEM_PROFILE=0
set VDN_H3_IMPORTANCE_PROFILE=0
set VDN_H3_WINDOW_IMPORTANCE_PROFILE=0
set VDN_H3_DELTA_DEEP_PROFILE=0
set VDN_H3_DELTA_SOLVER_LAB=0
set VDN_H3_DELTA_REALPATH_LAB=0
set VDN_H3_OUTPROJ_GEOMETRY_LAB=0
set VDN_H3_REUSE_PROFILE=0
set VDN_H3_QKV_EXEC_LAB=0
set VDN_H3_QKV_DEEP_PROFILE=0
set VDN_H3_WINDOW_DEEP_PROFILE=0
set VDN_H3_TRITON_ATTN_PROBE=0
set VDN_H3_TRITON_ATTN_AUTOTUNE=0
set VDN_H3_OUTPROJ_MAT_PROFILE=0
set VDN_H3_OUTPROJ_BLOCK_PROFILE=0
set VDN_H3_OUTPROJ_CACHE_FORMAT_PROFILE=0
set VDN_H3_STREAM_PREFETCH=0
set VDN_H3_BRANCH_OVERLAP=0

set VDN_H3_DELTA_SOLVE=direct
set VDN_H3_ABLATION=hybrid
set VDN_H3_WINDOW_GROUP_BATCH=1
set VDN_H3_HYBRID_STRIDE=1
set VDN_H3_HYBRID_BLOCKS=0,2,4,5,9,12,13,15,17,21,24,27,30,36,42,48
set VDN_H3_WINDOW_BLOCKS=
set VDN_H3_WINDOW_FALLBACK=
set VDN_H3_WINDOW_INNER_TRIM=4
set VDN_H3_SELECTIVE_WEIGHT_LOAD=1
set VDN_H3_WINDOW_SDPA=cudnn
set VDN_H3_OUTPROJ_CACHE_CLEAR_AFTER=8
set VDN_H3_OUTPROJ_CACHE_BLOCKS=
set VDN_H3_OUTPROJ_EXECUTION=materialized
set VDN_H3_OUTPROJ_GEMM=mm_bias
set VDN_H3_QKV_EXECUTION=native

cd /d "%COMFY_ROOT%"

rem ---- Use current Python from PATH / active environment ----
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [VDN-H3-24GB] ERROR: python was not found in PATH.
    echo Activate the same Python/Conda environment you normally use for ComfyUI,
    echo then run this BAT again.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_EXE=%%P"
echo [VDN-H3-24GB] Python: "%PYTHON_EXE%"

python -c "import sqlalchemy" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [VDN-H3-24GB] ERROR: the active Python environment does not look like your working ComfyUI environment.
    echo Missing import: sqlalchemy
    echo.
    echo Activate the SAME Python/Conda environment you normally use to launch ComfyUI,
    echo then run this BAT again.
    echo.
    pause
    exit /b 1
)

echo [VDN-H3-24GB] Checking MiniMax block-loop hook...
python "%HOOK%" --comfy-ui "%COMFY_ROOT%"
if errorlevel 1 (
    echo.
    echo [VDN-H3-24GB] ERROR: block-loop hook installation/check failed.
    echo.
    pause
    exit /b 1
)

echo.
echo [VDN-H3-24GB] Starting ComfyUI...
python main.py --listen 127.0.0.1 --port 8191 --disable-dynamic-vram --disable-async-offload --use-sage-attention

echo.
echo [VDN-H3-24GB] ComfyUI exited.
pause
