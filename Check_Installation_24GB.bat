@echo off
setlocal EnableExtensions
set "HERE=%~dp0"
echo [VDN-H3-24GB] Node folder: "%HERE%"
if exist "%HERE%__init__.py" (echo [OK] __init__.py) else (echo [MISSING] __init__.py)
if exist "%HERE%vdn_h3_24gb\nodes.py" (echo [OK] vdn_h3_24gb\nodes.py) else (echo [MISSING] vdn_h3_24gb\nodes.py)
if exist "%HERE%tools\install_minimax_block_loop_hook.py" (echo [OK] hook installer) else (echo [MISSING] hook installer)
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
if exist "%ROOT%\main.py" (echo [OK] ComfyUI root: "%ROOT%") else (echo [WARNING] main.py not found two levels above this folder.)
echo.
pause
