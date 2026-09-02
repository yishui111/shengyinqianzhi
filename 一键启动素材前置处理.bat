@echo off
REM ============================================================
REM  Voice Material Preprocess - one-click start (port 8070)
REM  Audio/Video -> clean vocals for training
REM  Stop : double-click Close-Preprocess.bat
REM ============================================================
setlocal
set "ROOT=%~dp0"
if not defined PRE_PORT set "PRE_PORT=8070"
set "PATH=%ROOT%ffmpeg\bin;%PATH%"
echo ============================================
echo   Voice Material Preprocess
echo   Web UI  : http://127.0.0.1:%PRE_PORT%/
echo   Output  : %ROOT%ziliao\output\
echo ============================================
"%ROOT%runtime\py312\python.exe" "%ROOT%pre_service\preprocess_api.py"
echo.
pause
