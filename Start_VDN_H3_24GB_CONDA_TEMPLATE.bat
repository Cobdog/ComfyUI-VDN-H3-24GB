@echo off
setlocal

rem Edit this to match your working ComfyUI Conda environment.
set "COMFY_ENV=YOUR_COMFY_ENV"

call conda activate %COMFY_ENV%
if errorlevel 1 (
  echo Failed to activate Conda environment: %COMFY_ENV%
  pause
  exit /b 1
)

call "%~dp0Start_VDN_H3_24GB.bat"
