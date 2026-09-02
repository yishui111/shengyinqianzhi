@echo off
echo Stopping Voice Preprocess (port 8070) ...
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8070 -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output 'Preprocess stopped' } else { Write-Output 'Preprocess is not running' }"
echo.
pause
