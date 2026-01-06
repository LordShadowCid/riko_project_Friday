# Annabeth Desktop Companion - Full Startup Script
# Run this from PowerShell to start everything

$ProjectRoot = "c:\Users\blakd\OneDrive\Desktop\Anabeth"
$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"
$GPT_SOVITS_PYTHON = "$ProjectRoot\third_party\GPT-SoVITS\.venv\Scripts\python.exe"

# Add cuDNN to PATH for GPU-accelerated Whisper
$env:PATH = "$ProjectRoot\.venv\Lib\site-packages\nvidia\cudnn\bin;$ProjectRoot\.venv\Lib\site-packages\nvidia\cublas\bin;$env:PATH"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Annabeth Desktop Companion  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Start GPT-SoVITS TTS Server (Native - faster than Docker!)
Write-Host "`n[1/3] Starting GPT-SoVITS TTS Server (Native)..." -ForegroundColor Yellow
$ttsStartup = @"
cd '$ProjectRoot\third_party\GPT-SoVITS'
`$env:CUDA_VISIBLE_DEVICES = '0'
`$env:is_half = 'true'
`$env:PATH = '$ProjectRoot\.venv\Lib\site-packages\nvidia\cudnn\bin;' + `$env:PATH
& '$GPT_SOVITS_PYTHON' -u api_v2.py -a 127.0.0.1 -p 9880
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ttsStartup

# Wait for TTS server to be ready (native starts faster than Docker)
Write-Host "Waiting for TTS server (20 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 20

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
