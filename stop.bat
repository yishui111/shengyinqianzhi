@echo off
rem ============================================================
rem  Voice Material Preprocess - stop server (port 8070)
rem ============================================================
setlocal
if not defined PRE_PORT set "PRE_PORT=8070"
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort $env:PRE_PORT -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output ('Service on port ' + $env:PRE_PORT + ' stopped') } else { Write-Output ('Service on port ' + $env:PRE_PORT + ' is not running') }"
exit /b 0
