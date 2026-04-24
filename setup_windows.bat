@echo off
setlocal
cd /d "%~dp0"

powershell -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"

echo.
if %ERRORLEVEL% EQU 0 (
    echo Wildlife Detection setup completed successfully.
) else (
    echo Wildlife Detection setup failed. Review the messages above.
)
pause
