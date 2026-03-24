@echo off
REM AI Image Processor - Launcher
chcp 65001 >nul

setlocal
cd /d "%~dp0"

echo ========================================
echo   AI Image Processor v1.1.0
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

:: 2. Activate
echo [OK] Activating environment...
call venv\Scripts\activate.bat

:: 3. Run
echo [OK] Starting application...
echo.

python main.py

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
