@echo off
REM Chess Game Quick Launcher
REM This script starts the chess game with proper directory navigation

echo.
echo ======================================
echo    Chess AI Game Launcher
echo ======================================
echo.

REM Check if we're in the right directory
if not exist "chess_game_server.py" (
    echo Error: chess_game_server.py not found!
    echo Please run this script from the play/ directory
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Starting Chess Game Server...
echo.
echo The game will open in your browser at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python chess_game_server.py

pause
