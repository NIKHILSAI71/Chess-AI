@echo off
REM Chess AI Setup Script for Windows
REM This script sets up your environment and gets you ready to train and play

echo.
echo ======================================
echo     CHESS AI SETUP SCRIPT
echo ======================================
echo.

REM Check if Python is installed
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
) else (
    echo ✓ Python is installed
)

REM Check if pip is installed
echo [2/4] Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
) else (
    echo ✓ pip is available
)

REM Install dependencies
echo [3/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some packages may not have installed correctly
    echo You can try running: pip install -r requirements.txt
) else (
    echo ✓ Dependencies installed successfully
)

REM Check if everything is working
echo [4/4] Testing installation...
python -c "import torch; import flask; print('✓ Core dependencies working')"
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may not be working correctly
) else (
    echo ✓ Installation test passed
)

echo.
echo ======================================
echo     SETUP COMPLETE!
echo ======================================
echo.
echo Quick start options:
echo.
echo 1. One-click launcher:     python launch.py
echo 2. Train AI only:         python train_ai.py
echo 3. Play game only:        cd play ^&^& python chess_game_server.py
echo.
echo For more information, see CONSOLIDATION_GUIDE.md
echo.
pause
