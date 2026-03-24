#!/bin/bash
# AI Image Processor - Setup Script (Linux/macOS)

echo "========================================"
echo "  AI Image Processor - Environment Setup"
echo "========================================"
echo

# Determine project root
cd "$(dirname "$0")/.."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.9+"
    exit 1
fi

# Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create venv."
        exit 1
    fi
    echo "      Done."
else
    echo "[1/3] Virtual environment already exists."
fi

# Activate Environment
echo "[2/3] Activating environment..."
source venv/bin/activate

# Install Dependencies
echo "[3/3] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi

echo
echo "========================================"
echo "  Setup Completed Successfully!"
echo "========================================"
echo
echo "Usage:"
echo "  1. Run ./run.sh to start."
echo "  2. Or manually: source venv/bin/activate"
echo "     Then: python3 main.py"
echo
