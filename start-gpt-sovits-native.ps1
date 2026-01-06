# GPT-SoVITS Native Windows Launcher
# Runs GPT-SoVITS directly on Windows without Docker for ~20-30% speed improvement
# Requires: Python 3.11, CUDA 12.x, and pre-installed venv at third_party/GPT-SoVITS/.venv

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Paths
$GPT_SOVITS_DIR = Join-Path $ProjectRoot "third_party\GPT-SoVITS"
$VENV_PYTHON = Join-Path $GPT_SOVITS_DIR ".venv\Scripts\python.exe"

# Verify venv exists
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: GPT-SoVITS venv not found at $VENV_PYTHON" -ForegroundColor Red
    Write-Host "Please run the installation first (see documentation)" -ForegroundColor Yellow
    exit 1
}

# Set environment variables for native mode
$env:CUDA_VISIBLE_DEVICES = "0"  # Use RTX A4000 (GPU 0)
$env:is_half = "true"

# Add cuDNN to PATH if available (for faster CUDA operations)
$cuDNNPath = Join-Path $ProjectRoot ".venv\Lib\site-packages\nvidia\cudnn\bin"
if (Test-Path $cuDNNPath) {
    $env:PATH = "$cuDNNPath;$env:PATH"
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  GPT-SoVITS Native Windows Mode" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting GPT-SoVITS API on http://127.0.0.1:9880 ..." -ForegroundColor Green
Write-Host "GPU: CUDA device 0 (RTX A4000)" -ForegroundColor Yellow
Write-Host ""

# Change to GPT-SoVITS directory and run the API
Set-Location $GPT_SOVITS_DIR

# Run the API server
try {
    & $VENV_PYTHON -u api_v2.py -a 127.0.0.1 -p 9880
}
catch {
    Write-Host "ERROR: GPT-SoVITS failed to start: $_" -ForegroundColor Red
    exit 1
}
