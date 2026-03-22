# Annabeth Desktop Companion - Full Startup Script
# Run this from PowerShell to start everything
# Usage: .\start_annabeth.ps1           (Unity frontend - default)
#        .\start_annabeth.ps1 -Legacy   (PyQt6 WebView frontend)
param([switch]$Legacy)

$ProjectRoot = "c:\Users\blakd\OneDrive\Desktop\Anabeth"
$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"
$GPT_SOVITS_PYTHON = "$ProjectRoot\third_party\GPT-SoVITS\.venv\Scripts\python.exe"
$UnityBuild = "C:\Users\blakd\unit\Builds\AnnabethTest\Annabeth.exe"

# Add cuDNN to PATH for GPU-accelerated Whisper
$env:PATH = "$ProjectRoot\.venv\Lib\site-packages\nvidia\cudnn\bin;$ProjectRoot\.venv\Lib\site-packages\nvidia\cublas\bin;$env:PATH"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Annabeth Desktop Companion  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Start GPT-SoVITS TTS Server (Native - faster than Docker!)
Write-Host "`n[1/3] Starting GPT-SoVITS TTS Server (Native)..." -ForegroundColor Yellow
$ttsStartup = @"
cd '$ProjectRoot\third_party\GPT-SoVITS'
`$env:CUDA_VISIBLE_DEVICES = '1'
`$env:is_half = 'true'
`$env:PATH = '$ProjectRoot\.venv\Lib\site-packages\nvidia\cudnn\bin;' + `$env:PATH
& '$GPT_SOVITS_PYTHON' -u api_v2.py -a 127.0.0.1 -p 9880
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ttsStartup

# Wait for TTS server to be ready (model loading takes time, especially first run with OneDrive)
Write-Host "Waiting for TTS server to load models..." -ForegroundColor Gray
$ttsReady = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:9880/docs" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ttsReady = $true; break }
    } catch {}
    if ($i % 6 -eq 5) { Write-Host "  Still loading TTS models... ($([int](($i+1)*5))s elapsed)" -ForegroundColor Gray }
}
if ($ttsReady) {
    Write-Host "TTS server is ready!" -ForegroundColor Green
} else {
    Write-Host "TTS server not responding after 5 minutes - continuing anyway" -ForegroundColor Yellow
}

# 2. Start Main Chat Server
Write-Host "`n[2/3] Starting Main Chat Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; & '$VenvPython' -m server.main_chat"

# Wait for chat server to initialize
Write-Host "Waiting for chat server (5 seconds)..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# 3. Start Frontend (Unity or Legacy PyQt6)
if ($Legacy) {
    Write-Host "`n[3/3] Starting Desktop Companion (PyQt6 Legacy)..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\client'; & '$VenvPython' desktop_companion_webview.py"
} else {
    Write-Host "`n[3/3] Starting Desktop Companion (Unity)..." -ForegroundColor Yellow
    if (Test-Path $UnityBuild) {
        Start-Process $UnityBuild
    } else {
        Write-Host "Unity build not found at: $UnityBuild" -ForegroundColor Red
        Write-Host "Build from Unity: Annabeth > Build Standalone, or use -Legacy flag" -ForegroundColor Yellow
        Write-Host "Falling back to PyQt6..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\client'; & '$VenvPython' desktop_companion_webview.py"
    }
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Annabeth is starting up!             " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nControls (when companion window focused):"
Write-Host "  1     - Active mode"
Write-Host "  2     - Idle mode"
Write-Host "  3     - Procedural dance"
Write-Host "  4     - Shikanoko dance"
Write-Host "  S     - Toggle chat silence"
Write-Host "  D     - Cycle dance modes"
Write-Host "  Q     - Pause read-aloud"
Write-Host "  R     - Resume read-aloud"
Write-Host "  Space - Interrupt"
Write-Host "  F1    - Toggle debug overlay"
Write-Host "  Ctrl+Shift+X - Interrupt speech (global)"
Write-Host "`nClose all windows to shut down."
