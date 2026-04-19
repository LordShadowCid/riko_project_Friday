# GPT-SoVITS Native Windows Launcher
# Runs GPT-SoVITS directly on Windows without Docker for ~20-30% speed improvement
# Requires: Python 3.11, CUDA 12.x, and pre-installed venv at third_party/GPT-SoVITS/.venv

param(
    [string]$Gpu = "1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Paths
$GPT_SOVITS_DIR = Join-Path $ProjectRoot "third_party\GPT-SoVITS"
$SHARED_MODELS_DIR = Join-Path $ProjectRoot "gpt_sovits_models"
$VENV_PYTHON = Join-Path $GPT_SOVITS_DIR ".venv\Scripts\python.exe"
$RequiredAssets = @(
    "GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth",
    "GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large",
    "GPT_SoVITS\pretrained_models\chinese-hubert-base"
)
$SharedAssetLinks = @(
    @{
        Target = "GPT_SoVITS\pretrained_models"
        Source = "pretrained_models"
    },
    @{
        Target = "GPT_SoVITS\text\G2PWModel"
        Source = "G2PWModel"
    }
)

# Verify venv exists
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: GPT-SoVITS venv not found at $VENV_PYTHON" -ForegroundColor Red
    Write-Host "Please run the installation first (see documentation)" -ForegroundColor Yellow
    exit 1
}

foreach ($mapping in $SharedAssetLinks) {
    $targetPath = Join-Path $GPT_SOVITS_DIR $mapping.Target
    $sourcePath = Join-Path $SHARED_MODELS_DIR $mapping.Source

    if ((-not (Test-Path $targetPath)) -and (Test-Path $sourcePath)) {
        $parentPath = Split-Path -Parent $targetPath
        if (-not (Test-Path $parentPath)) {
            New-Item -ItemType Directory -Path $parentPath -Force | Out-Null
        }

        New-Item -ItemType Junction -Path $targetPath -Target $sourcePath | Out-Null
    }
}

$missingAssets = @()
foreach ($relativePath in $RequiredAssets) {
    $fullPath = Join-Path $GPT_SOVITS_DIR $relativePath
    if (-not (Test-Path $fullPath)) {
        $missingAssets += $relativePath
    }
}

if ($missingAssets.Count -gt 0) {
    Write-Host "ERROR: GPT-SoVITS pretrained assets are missing:" -ForegroundColor Red
    foreach ($asset in $missingAssets) {
        Write-Host ("  - " + $asset) -ForegroundColor Yellow
    }
    Write-Host "Download the GPT-SoVITS pretrained models into third_party\GPT-SoVITS before using native mode." -ForegroundColor Yellow
    exit 1
}

# Set environment variables for native mode
$env:CUDA_VISIBLE_DEVICES = $Gpu
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
Write-Host "GPU: CUDA device $Gpu" -ForegroundColor Yellow
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
