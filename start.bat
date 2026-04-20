@echo off
title Annabeth Desktop Companion
cd /d "%~dp0"

echo ========================================
echo   Starting Annabeth Desktop Companion
echo ========================================
echo.

if not exist "%~dp0start_annabeth.ps1" (
	echo start_annabeth.ps1 not found.
	exit /b 1
)

echo Launching Annabeth (all services start automatically)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_annabeth.ps1" -ProjectRoot "%cd%"
exit /b %errorlevel%
