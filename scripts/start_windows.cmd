@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1" %*
exit /b %errorlevel%
