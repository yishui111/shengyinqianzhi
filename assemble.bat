@echo off
rem ==================================================
rem  SHENGYINQIANZHI - one-key preflight for big assets
rem  Guarantee flow: (A) copy original project folder
rem  with big assets (fastest), or (B) clone this repo
rem  then run this script; details: see DEPLOY.md top.
rem ==================================================
setlocal
cd /d "%~dp0"
set "MISSING=0"
echo Checking required big assets...
if exist "runtime\py312" (echo   OK   runtime\py312) else (echo   MISS runtime\py312 ^& set MISSING=1)
if exist "ffmpeg\bin" (echo   OK   ffmpeg\bin) else (echo   MISS ffmpeg\bin ^& set MISSING=1)
if exist "models\pymss" (echo   OK   models\pymss) else (echo   MISS models\pymss ^& set MISSING=1)
if exist "models\asr" (echo   OK   models\asr) else (echo   MISS models\asr ^& set MISSING=1)
echo.
if %MISSING%==0 (
  echo ALL big assets present. Run start.bat now.
) else (
  echo Some big assets missing. See DEPLOY.md (top section
  "Deployment guarantee") for download instructions.
)
pause
