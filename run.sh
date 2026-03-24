#!/bin/bash
# AI Image Processor - Launcher (Linux/macOS)

# Ensure script runs from project root
cd "$(dirname "$0")"

echo "========================================"
echo "  AI Image Processor v1.1.0"
echo "========================================"
echo

# 1. Check venv
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Virtual environment not found."
    echo "[INFO] Starting automatic environment setup..."
    echo
    
    if [ -f "scripts/setup.sh" ]; then
        bash scripts/setup.sh
    else
        echo "[ERROR] Missing 'scripts/setup.sh'."
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Setup failed."
        exit 1
    fi
    echo
fi

# 2. Activate
echo "[OK] Activating environment..."
source venv/bin/activate

# 3. Run
echo "[OK] Starting application..."
echo

python3 main.py

# 4. Error Handling
if [ $? -ne 0 ]; then
    echo
    echo "[CRITICAL] Application terminated unexpectedly."
    echo "[HINT] Consider downloading models:"
    echo "       python3 scripts/download_all_models.py"
    echo
fi
