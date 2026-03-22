# ============================================================================
# Annabeth - Complete Setup Script for Fresh Windows PC
# ============================================================================
# Run as Administrator:  Right-click PowerShell -> Run as Administrator
#   cd "c:\Users\blakd\OneDrive\Desktop\Anabeth"
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\setup_annabeth.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$ProjectRoot = "c:\Users\blakd\OneDrive\Desktop\Anabeth"
$sep = '=' * 70
$stepNum = 0

function Step {
    param([string]$msg)
    $script:stepNum++
    Write-Host ''
    Write-Host $sep -ForegroundColor Cyan
    Write-Host "  STEP $script:stepNum : $msg" -ForegroundColor Cyan
    Write-Host $sep -ForegroundColor Cyan
}

function OK   { param([string]$msg); Write-Host "  [OK] $msg" -ForegroundColor Green }
function WARN { param([string]$msg); Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function FAIL { param([string]$msg); Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function INFO { param([string]$msg); Write-Host "  $msg" -ForegroundColor Gray }

# --------------------------------------------------------------------------
Step "Install Visual C++ Build Tools (winget)"
# --------------------------------------------------------------------------
# Needed for compiling Python packages like pyopenjtalk
$progX86 = [Environment]::GetFolderPath('ProgramFilesX86')
$vsWhere = Join-Path $progX86 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vsWhere) {
    OK 'Visual Studio / Build Tools already installed'
} else {
    INFO 'Installing Visual Studio Build Tools via winget...'
    INFO 'This installs the C++ desktop workload needed for Python native extensions.'
    winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    if ($LASTEXITCODE -eq 0) { OK 'Build Tools installed' } else { WARN 'Build Tools install may need manual finish - check Windows Updates' }
}

# --------------------------------------------------------------------------
Step "Install ffmpeg (winget)"
# --------------------------------------------------------------------------
$ffmpegCheck = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpegCheck) {
    OK ('ffmpeg already on PATH: ' + $ffmpegCheck.Source)
} else {
    INFO 'Installing ffmpeg via winget...'
    winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -eq 0) {
        # Refresh PATH for this session
        $machinePath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
        $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
        $env:PATH = "$machinePath;$userPath"
        OK 'ffmpeg installed'
    } else {
        WARN 'ffmpeg install may have failed - you can install manually from gyan.dev/ffmpeg/builds'
    }
}

# --------------------------------------------------------------------------
Step "Install / Verify cuDNN"
# --------------------------------------------------------------------------
# cuDNN will be installed via pip nvidia-cudnn-cu12 into the venv later.
# But we check if it is already system-installed.
$cudnnFound = $false
$cudaPathEnv = $env:CUDA_PATH
if ($cudaPathEnv) {
    $bins = Get-ChildItem (Join-Path $cudaPathEnv 'bin\cudnn*.dll') -ErrorAction SilentlyContinue
    if ($bins) { $cudnnFound = $true }
}
if (Test-Path 'C:\Program Files\NVIDIA\CUDNN') {
    $bins2 = Get-ChildItem 'C:\Program Files\NVIDIA\CUDNN\*\bin\cudnn*.dll' -ErrorAction SilentlyContinue
    if ($bins2) { $cudnnFound = $true }
}

if ($cudnnFound) {
    OK 'cuDNN found on system'
} else {
    INFO 'cuDNN not found system-wide - will install via pip (nvidia-cudnn-cu12) into venv'
    INFO 'This is the recommended approach for Python projects'
}

# --------------------------------------------------------------------------
Step "Create Python virtual environment (.venv)"
# --------------------------------------------------------------------------
$venvDir = Join-Path $ProjectRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (Test-Path $venvPython) {
    OK "Venv already exists at $venvDir"
    $pyVer = & $venvPython --version 2>&1
    INFO "Venv Python: $pyVer"
} else {
    INFO "Creating venv at $venvDir ..."
    python -m venv $venvDir
    if (Test-Path $venvPython) { OK 'Venv created' } else { FAIL 'Failed to create venv'; exit 1 }
}

# Upgrade pip and install uv in the venv
INFO 'Upgrading pip and installing uv...'
& $venvPython -m pip install --upgrade pip --quiet 2>$null
& $venvPython -m pip install --upgrade uv --quiet 2>$null
OK 'pip and uv upgraded'

# --------------------------------------------------------------------------
Step "Install PyTorch + torchaudio (CUDA)"
# --------------------------------------------------------------------------
# System has CUDA 13.2. PyTorch cu130 wheels are forward-compatible.
INFO 'Installing PyTorch + torchaudio with CUDA 13.0 support...'
& $venvPython -m uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -eq 0) { OK 'PyTorch + torchaudio installed (CUDA)' } else { WARN 'PyTorch install had issues' }

# --------------------------------------------------------------------------
Step "Install NVIDIA cuDNN + cuBLAS (pip, for Whisper GPU)"
# --------------------------------------------------------------------------
INFO 'Installing nvidia-cudnn-cu12 and nvidia-cublas-cu12 into venv...'
& $venvPython -m uv pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
if ($LASTEXITCODE -eq 0) { OK 'cuDNN + cuBLAS installed via pip' } else { WARN 'cuDNN pip install had issues' }

# --------------------------------------------------------------------------
Step "Install Annabeth client requirements"
# --------------------------------------------------------------------------
$extraReqFile = Join-Path $ProjectRoot 'extra-req.txt'
$clientReqFile = Join-Path $ProjectRoot 'requirements-client.txt'

INFO 'Installing extra-req.txt ...'
& $venvPython -m uv pip install -r $extraReqFile
INFO 'Installing requirements-client.txt ...'
& $venvPython -m uv pip install -r $clientReqFile
OK 'Client requirements installed'

# --------------------------------------------------------------------------
Step "Install PyQt6 + WebEngine (Desktop Companion)"
# --------------------------------------------------------------------------
INFO 'Installing PyQt6, PyQt6-WebEngine, keyboard, pyperclip, pyautogui...'
& $venvPython -m uv pip install PyQt6 PyQt6-WebEngine PyQt6-sip keyboard pyperclip pyautogui aiohttp
if ($LASTEXITCODE -eq 0) { OK 'Desktop companion dependencies installed' } else { WARN 'Some desktop deps may have failed' }

# --------------------------------------------------------------------------
Step "Install GPT-SoVITS requirements (TTS server)"
# --------------------------------------------------------------------------
$sovitsDir = Join-Path $ProjectRoot 'third_party\GPT-SoVITS'
$sovitsVenv = Join-Path $sovitsDir '.venv'
$sovitsPython = Join-Path $sovitsVenv 'Scripts\python.exe'
$sovitsReqs = Join-Path $ProjectRoot 'requirements.txt'

if (Test-Path $sovitsPython) {
    OK 'GPT-SoVITS venv already exists'
} else {
    INFO "Creating GPT-SoVITS venv at $sovitsVenv ..."
    python -m venv $sovitsVenv
}

INFO 'Upgrading pip and installing uv in GPT-SoVITS venv...'
& $sovitsPython -m pip install --upgrade pip --quiet 2>$null
& $sovitsPython -m pip install --upgrade uv --quiet 2>$null

INFO 'Installing PyTorch in GPT-SoVITS venv...'
& $sovitsPython -m uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130

INFO 'Installing GPT-SoVITS requirements (requirements.txt) - pass 1 (no deps)...'
& $sovitsPython -m uv pip install -r $sovitsReqs --no-deps
INFO 'Installing GPT-SoVITS requirements (requirements.txt) - pass 2 (with deps)...'
& $sovitsPython -m uv pip install -r $sovitsReqs
if ($LASTEXITCODE -eq 0) { OK 'GPT-SoVITS requirements installed' } else { WARN 'Some GPT-SoVITS deps may need manual attention' }

# Install NLTK data needed by GPT-SoVITS
INFO 'Downloading NLTK data (averaged_perceptron_tagger, cmudict)...'
& $sovitsPython -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')"

# --------------------------------------------------------------------------
Step "Pull Ollama model"
# --------------------------------------------------------------------------
$ollamaCheck = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCheck) {
    INFO 'Pulling mannix/llama3.1-8b-abliterated (may take a while on first download)...'
    ollama pull mannix/llama3.1-8b-abliterated
    if ($LASTEXITCODE -eq 0) { OK 'Ollama model pulled' } else { WARN 'Ollama pull had issues - is the Ollama service running?' }
} else {
    WARN 'Ollama not found on PATH. Install from https://ollama.com then run:  ollama pull mannix/llama3.1-8b-abliterated'
}

# --------------------------------------------------------------------------
Step "Set environment variables"
# --------------------------------------------------------------------------
# CUDA_HOME
$cudaHome = [Environment]::GetEnvironmentVariable('CUDA_HOME', 'Machine')
if (-not $cudaHome) {
    $cudaPathVal = $env:CUDA_PATH
    if ($cudaPathVal) {
        [Environment]::SetEnvironmentVariable('CUDA_HOME', $cudaPathVal, 'Machine')
        $env:CUDA_HOME = $cudaPathVal
        OK "CUDA_HOME set to $cudaPathVal"
    } else {
        WARN 'CUDA_PATH not set, cannot set CUDA_HOME'
    }
} else {
    OK "CUDA_HOME already set: $cudaHome"
}

# HF_HUB_DISABLE_SYMLINKS_WARNING (Windows Whisper workaround)
[Environment]::SetEnvironmentVariable('HF_HUB_DISABLE_SYMLINKS_WARNING', '1', 'User')
OK 'HF_HUB_DISABLE_SYMLINKS_WARNING set'

# --------------------------------------------------------------------------
Step "Verify installations"
# --------------------------------------------------------------------------
Write-Host ''
INFO '--- Main Annabeth venv ---'
& $venvPython -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
& $venvPython -c "import faster_whisper; print('  faster_whisper: OK')"
& $venvPython -c "import ctranslate2; print('  ctranslate2: OK | CUDA devices:', ctranslate2.get_cuda_device_count())"
& $venvPython -c "import sounddevice; print('  sounddevice: OK')"
& $venvPython -c "import PyQt6.QtWidgets; print('  PyQt6: OK')"
& $venvPython -c "import ollama; print('  ollama (python): OK')"

Write-Host ''
INFO '--- GPT-SoVITS venv ---'
& $sovitsPython -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
& $sovitsPython -c "import fastapi; print('  fastapi: OK')"

Write-Host ''
INFO '--- System tools ---'
$ffCheck = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffCheck) { OK ('ffmpeg: ' + $ffCheck.Source) } else { WARN 'ffmpeg: NOT FOUND - may need to restart shell or install manually' }
$gitCheck = Get-Command git -ErrorAction SilentlyContinue
if ($gitCheck) { OK 'git: OK' } else { WARN 'git: NOT FOUND' }
$dockerCheck = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCheck) { OK 'docker: OK' } else { WARN 'docker: NOT FOUND' }
$olCheck = Get-Command ollama -ErrorAction SilentlyContinue
if ($olCheck) { OK 'ollama: OK' } else { WARN 'ollama: NOT FOUND' }

# --------------------------------------------------------------------------
Write-Host ''
Write-Host $sep -ForegroundColor Green
Write-Host '  SETUP COMPLETE!' -ForegroundColor Green
Write-Host $sep -ForegroundColor Green
Write-Host ''
Write-Host '  Next steps:' -ForegroundColor Yellow
Write-Host '    1. Close and reopen PowerShell (to pick up PATH changes)'
Write-Host '    2. If you use OpenAI, set your API key:'
Write-Host '       [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")'
Write-Host '    3. Run Annabeth:'
Write-Host '       cd "c:\Users\blakd\OneDrive\Desktop\Anabeth"'
Write-Host '       .\start_annabeth.ps1'
Write-Host ''
Write-Host '  Hardware reference saved to: PC_HARDWARE_REFERENCE.md' -ForegroundColor Gray
Write-Host ''
