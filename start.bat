@echo off
title Annabeth Startup
cd /d "%~dp0"

echo ========================================
echo   Starting Annabeth Desktop Companion
echo ========================================
echo.

:: Start TTS Server
echo [1/3] Starting TTS Server...
start "TTS Server" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate && python third_party\GPT-SoVITS\api_v2.py"

:: Wait for TTS
echo Waiting 15 seconds for TTS...
timeout /t 15 /nobreak >nul

:: Start Chat Server  
echo [2/3] Starting Chat Server...
start "Chat Server" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate && python -m server.main_chat"

:: Wait for Chat
echo Waiting 5 seconds for Chat...
timeout /t 5 /nobreak >nul

:: Start Desktop Companion
echo [3/3] Starting Desktop Companion...
start "Desktop Companion" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate && python client\desktop_companion_webview.py"

echo.
echo ========================================
echo   Annabeth is starting up!
echo ========================================
echo.
echo Controls (when companion window focused):
echo   1     - Active mode (listening)
echo   2     - Idle mode
echo   3     - Dance Beat mode  
echo   4     - Dance Full mode
echo   S     - Toggle silence
echo   D     - Cycle dance modes
echo   ESC   - Close companion
echo.
echo Close the 3 command windows to shut down.
echo.
pause
