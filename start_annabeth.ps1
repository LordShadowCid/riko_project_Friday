# Annabeth Desktop Companion - Full Startup Script
# Run this from PowerShell to start everything

$ProjectRoot = "c:\Users\blakd\OneDrive\Desktop\Anabeth"
$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Annabeth Desktop Companion  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Start GPT-SoVITS TTS Server (Docker)
Write-Host "`n[1/3] Starting GPT-SoVITS TTS Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; docker-compose -f docker-compose.gpt-sovits.yml up"

# Wait for TTS server to be ready
Write-Host "Waiting for TTS server (15 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 15

# 2. Start Main Chat Server
Write-Host "`n[2/3] Starting Main Chat Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; & '$VenvPython' -m server.main_chat"

# Wait for chat server to initialize
Write-Host "Waiting for chat server (5 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# 3. Start Desktop Companion (Avatar + Audio)
Write-Host "`n[3/3] Starting Desktop Companion..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\client'; & '$VenvPython' desktop_companion_webview.py"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Annabeth is starting up!             " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nControls (when companion window focused):"
Write-Host "  S     - Toggle chat silence"
Write-Host "  D     - Cycle dance modes"
Write-Host "  1-4   - Quick mode select"
Write-Host "  Space - Cycle all modes"
Write-Host "  F5    - Reload avatar"
Write-Host "  ESC   - Close companion"
Write-Host "`nClose all 3 PowerShell windows to shut down."
