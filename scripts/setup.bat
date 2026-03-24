@echo off
REM AI Image Processor - Setup Script
chcp 65001 >nul

echo ========================================
echo   AI Image Processor - Environment Setup
echo ========================================
echo.

cd /d "%~dp0.."

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check Virtual Environment
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/3] Virtual environment already exists.
)

REM Activate Environment
echo [2/3] Activating environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Cannot find activate.bat.
    pause
    exit /b 1
)

REM Install Dependencies
echo [3/3] Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip -q
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies. Please check your internet.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup Completed Successfully!
echo ========================================
echo.
echo Usage:
echo   1. Run run.bat to start.
echo   2. Or manually: venv\Scripts\activate
echo      Then: python main.py
echo.
pause
