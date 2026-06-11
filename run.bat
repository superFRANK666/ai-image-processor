@echo off
REM AI Image Processor - Launcher
chcp 65001 >nul

setlocal
cd /d "%~dp0"

echo ========================================
echo   AI Image Processor v1.0.0
echo ========================================
echo.

:: 1. Check venv
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Setting up...
    echo.
    
    if exist "scripts\setup.bat" (
        call scripts\setup.bat
    ) else (
        echo [ERROR] Missing scripts\setup.bat
        pause
        exit /b 1
    )
    
    if errorlevel 1 (
        echo.
        echo [ERROR] Setup failed.
        pause
        exit /b 1
    )
    echo.
)

:: 2. Use venv Python directly. This is faster than activating the shell first.
set "PYTHON=venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Missing %PYTHON%
    pause
    exit /b 1
)

:: 3. Run
echo [OK] Starting application...
echo.

"%PYTHON%" main.py

:: 4. Crash Handling
if errorlevel 1 (
    echo.
    echo [CRITICAL] Application terminated unexpectedly.
    echo [HINT] Consider downloading models:
    echo        python scripts/download_all_models.py
    echo.
    pause
)

endlocal

