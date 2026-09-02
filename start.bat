@echo off
rem ============================================================
rem  Voice Material Preprocess - start server (port 8070)
rem  Audio/Video material -> clean vocals for AI training
rem  Stop: stop.bat (or the original close script)
rem ============================================================
setlocal
set "ROOT=%~dp0"
if not defined PRE_PORT set "PRE_PORT=8070"

rem --- port already in use? just open the browser ---
netstat -ano | findstr /r /c:":%PRE_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo Service is already running on port %PRE_PORT%.
    start "" "http://127.0.0.1:%PRE_PORT%/"
    exit /b 0
)

if not exist "%ROOT%runtime\py312\python.exe" (
    echo [ERROR] runtime\py312\python.exe not found.
    echo Please prepare the big folders first - runtime / ffmpeg / models - see DEPLOY.md.
    pause
    exit /b 1
)

echo ============================================
echo   Voice Material Preprocess
echo   Web UI  : http://127.0.0.1:%PRE_PORT%/
echo   Output  : %ROOT%ziliao\output\
echo ============================================
set "PATH=%ROOT%ffmpeg\bin;%PATH%"
start "Voice Material Preprocess (port %PRE_PORT%)" "%ROOT%runtime\py312\python.exe" "%ROOT%pre_service\preprocess_api.py"

echo Waiting for the server to boot ...
timeout /t 8 /nobreak >nul
start "" "http://127.0.0.1:%PRE_PORT%/"
echo Started. Server log is shown in the new console window.
echo If the browser did not open, visit http://127.0.0.1:%PRE_PORT%/ manually.
exit /b 0
