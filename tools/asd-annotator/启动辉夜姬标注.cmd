@echo off
setlocal
chcp 65001 >nul
set "ANNOTATOR_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%ANNOTATOR_PY%" (
    "%ANNOTATOR_PY%" "%~dp0start_windows.py" %*
) else (
    py -3 "%~dp0start_windows.py" %*
)
if errorlevel 1 pause
endlocal
