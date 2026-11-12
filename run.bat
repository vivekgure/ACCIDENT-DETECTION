@echo off
title AURA-VISION // AI Vehicle Accident Detection System
cd /d "%~dp0"

set "PYTHON_BIN=python"
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
)

echo =========================================================================
echo    AURA-VISION // AI VEHICLE ACCIDENT DETECTION PLATFORM
echo =========================================================================
echo Using Python: %PYTHON_BIN%
echo Checking environment...
"%PYTHON_BIN%" check_env.py
if %ERRORLEVEL% neq 0 (
    echo [!] Environment check reported warnings or missing components.
)
echo.
echo Launching Flask Web Server at http://127.0.0.1:5000 ...
start "" "http://127.0.0.1:5000"
"%PYTHON_BIN%" app.py
pause
