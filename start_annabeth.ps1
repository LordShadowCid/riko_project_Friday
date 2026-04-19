# Annabeth Desktop Companion - Full Startup Script
# Run this from PowerShell to start everything
# Usage: .\start_annabeth.ps1                            (Unity frontend - default)
#        .\start_annabeth.ps1 -Legacy                    (PyQt6 WebView frontend)
#        .\start_annabeth.ps1 -ProjectRoot D:\Annabeth  (override repo location)
#        .\start_annabeth.ps1 -ReuseRunningOllama       (skip Ollama restart/pinning)
param(
    [switch]$Legacy,
    [string]$ProjectRoot,
    [string]$UnityBuild,
    [string]$OllamaGpu = "0",
    [string]$TtsGpu = "1",
    [switch]$ReuseRunningOllama
)

# Win32 API for hiding the launcher console window (so it doesn't sit behind the character)
Add-Type -Name ConsoleUtils -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern IntPtr GetConsoleWindow();
[DllImport("user32.dll")]
public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
'@

if (-not $ProjectRoot) {
    if ($PSScriptRoot) {
        $ProjectRoot = $PSScriptRoot
    } else {
        $ProjectRoot = (Get-Location).Path
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

# ── Startup logging ──────────────────────────────────────────────────────
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogFile = Join-Path $LogDir ("startup_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log")
Start-Transcript -Path $LogFile -Force | Out-Null
# Keep only the 10 most recent startup logs
Get-ChildItem $LogDir -Filter "startup_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 10 | Remove-Item -Force -ErrorAction SilentlyContinue

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GptSovitsRoot = Join-Path $ProjectRoot "third_party\GPT-SoVITS"
$SharedGptSovitsModelRoot = Join-Path $ProjectRoot "gpt_sovits_models"
$GPT_SOVITS_PYTHON = Join-Path $ProjectRoot "third_party\GPT-SoVITS\.venv\Scripts\python.exe"
$CharacterConfigPath = Join-Path $ProjectRoot "character_config.yaml"
if (-not $UnityBuild) {
    $UnityBuild = Join-Path $env:USERPROFILE "unit\Builds\AnnabethTest\Annabeth.exe"
}

$RequiredGptSovitsAssets = @(
    "GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS\pretrained_models\gsv-v2final-pretrained\s2G2333k.pth",
    "GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large",
    "GPT_SoVITS\pretrained_models\chinese-hubert-base"
)
$OptionalSharedGptSovitsLinks = @(
    @{
        Target = "GPT_SoVITS\pretrained_models"
        Source = "pretrained_models"
    },
    @{
        Target = "GPT_SoVITS\text\G2PWModel"
        Source = "G2PWModel"
    }
)

function Test-OneDrivePath {
    param([string]$Path)

    return $Path -match '(?i)[\\/]OneDrive([\\/]|$)'
}

function Wait-ForHttpReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$Attempts = 60,
        [int]$SleepSeconds = 5,
        [string]$ProgressLabel = "Service"
    )

    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        Start-Sleep -Seconds $SleepSeconds
        try {
            $response = Invoke-WebRequest $Url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {}

        if ($attempt % 6 -eq 5) {
            $elapsed = [int](($attempt + 1) * $SleepSeconds)
            Write-Host "  Still waiting for $ProgressLabel... (${elapsed}s elapsed)" -ForegroundColor Gray
        }
    }

    return $false
}

function Wait-ForProcessHttpReady {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$Attempts = 30,
        [int]$SleepSeconds = 1,
        [string]$ProgressLabel = "Service"
    )

    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        if ($Process.HasExited) {
            return $false
        }

        try {
            $response = Invoke-WebRequest $Url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {}

        Start-Sleep -Seconds $SleepSeconds

        if ($attempt % 5 -eq 4) {
            $elapsed = [int](($attempt + 1) * $SleepSeconds)
            Write-Host "  Still waiting for $ProgressLabel... (${elapsed}s elapsed)" -ForegroundColor Gray
        }
    }

    return $false
}

function Start-HiddenPowerShell {
    param(
        [Parameter(Mandatory = $true)][string]$Command
    )

    Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $Command
}

function Stop-ProcessTree {
    param(
        [int[]]$ProcessIds
    )

    foreach ($processId in $ProcessIds | Where-Object { $_ }) {
        try { & taskkill /F /T /PID $processId 2>$null | Out-Null } catch {}
    }
}

function Ensure-GptSovitsSharedAssets {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)][string]$SharedRoot
    )

    foreach ($mapping in $OptionalSharedGptSovitsLinks) {
        $targetPath = Join-Path $RuntimeRoot $mapping.Target
        $sourcePath = Join-Path $SharedRoot $mapping.Source

        if (Test-Path $targetPath) {
            continue
        }

        if (-not (Test-Path $sourcePath)) {
            continue
        }

        $parentPath = Split-Path -Parent $targetPath
        if (-not (Test-Path $parentPath)) {
            New-Item -ItemType Directory -Path $parentPath -Force | Out-Null
        }

        New-Item -ItemType Junction -Path $targetPath -Target $sourcePath | Out-Null
    }
}

function Test-GptSovitsNativePrereqs {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath
    )

    $missing = @()
    foreach ($relativePath in $RequiredGptSovitsAssets) {
        $fullPath = Join-Path $RootPath $relativePath
        if (-not (Test-Path $fullPath)) {
            $missing += $relativePath
        }
    }

    return $missing
}

# Add cuDNN to PATH for GPU-accelerated Whisper
$env:PATH = "$ProjectRoot\.venv\Lib\site-packages\nvidia\cudnn\bin;$ProjectRoot\.venv\Lib\site-packages\nvidia\cublas\bin;$env:PATH"

if (-not (Test-Path $VenvPython)) {
    throw "Project venv Python not found at: $VenvPython"
}

if (-not (Test-Path $GPT_SOVITS_PYTHON)) {
    throw "GPT-SoVITS venv Python not found at: $GPT_SOVITS_PYTHON"
}

Ensure-GptSovitsSharedAssets -RuntimeRoot $GptSovitsRoot -SharedRoot $SharedGptSovitsModelRoot

$missingGptSovitsAssets = Test-GptSovitsNativePrereqs -RootPath $GptSovitsRoot
if ($missingGptSovitsAssets.Count -gt 0) {
    $missingList = $missingGptSovitsAssets -join ", "
    throw "GPT-SoVITS pretrained assets are missing: $missingList. Install/download the GPT-SoVITS pretrained models before running the native launcher."
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Annabeth Desktop Companion  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot" -ForegroundColor Gray
if (Test-OneDrivePath $ProjectRoot) {
    Write-Host "WARNING: Project is running from OneDrive. Moving it to D: will improve model load and file I/O latency." -ForegroundColor Yellow
}
Write-Host "Whisper/TTS GPU: $TtsGpu" -ForegroundColor Gray
Write-Host "Preferred Ollama GPU: $OllamaGpu" -ForegroundColor Gray

$ollamaModel = "mannix/llama3.1-8b-abliterated"
if (Test-Path $CharacterConfigPath) {
    try {
        $configuredModel = & $VenvPython -c "from pathlib import Path; import yaml; config = yaml.safe_load(Path(r'$CharacterConfigPath').read_text(encoding='utf-8')) or {}; print((config.get('ollama') or {}).get('model') or '')"
        if ($LASTEXITCODE -eq 0 -and $configuredModel) {
            $ollamaModel = $configuredModel.Trim()
        }
    } catch {
        Write-Host "Unable to read configured Ollama model from character_config.yaml; using launcher default." -ForegroundColor Yellow
    }
}
Write-Host "Ollama warm model: $ollamaModel" -ForegroundColor Gray

$ollamaModelsDir = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "Process")
if (-not $ollamaModelsDir) {
    $ollamaModelsDir = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
}
if (-not $ollamaModelsDir) {
    $ollamaModelsDir = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "Machine")
}
if ($ollamaModelsDir) {
    Write-Host "Ollama models: $ollamaModelsDir" -ForegroundColor Gray
}

$ollamaStartedByScript = $false
$ollamaProc = $null
$ollamaServerReachable = $false
$ttsStartedByScript = $false
$ttsProc = $null
$ttsServerReachable = $false
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
try {
    $ollamaPing = Invoke-WebRequest "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    if ($ollamaPing.StatusCode -eq 200) {
        $ollamaServerReachable = $true
        Write-Host "Ollama server detected on port 11434" -ForegroundColor Green
    }
} catch {
}

if ($ollamaServerReachable -and -not $ReuseRunningOllama) {
    Write-Host "Restarting existing Ollama server to enforce GPU $OllamaGpu..." -ForegroundColor Yellow
    $ollamaProcessIds = @()
    $ollamaProcessIds += @(Get-Process ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
    $ollamaProcessIds += @(Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess)
    $ollamaProcessIds = @($ollamaProcessIds | Sort-Object -Unique)
    Stop-ProcessTree -ProcessIds $ollamaProcessIds
    Start-Sleep -Seconds 2
    $ollamaServerReachable = $false
}

if (-not $ollamaServerReachable) {
    if ($ollamaCmd) {
        Write-Host "Starting Ollama server pinned to GPU $OllamaGpu..." -ForegroundColor Yellow
        $ollamaStartup = @"
`$env:CUDA_VISIBLE_DEVICES = '$OllamaGpu'
& '$($ollamaCmd.Source)' serve
"@
        $ollamaProc = Start-HiddenPowerShell -Command $ollamaStartup
        if (Wait-ForHttpReady -Url "http://127.0.0.1:11434/api/tags" -Attempts 18 -SleepSeconds 2 -ProgressLabel "Ollama") {
            $ollamaStartedByScript = $true
            Write-Host "Ollama server is ready!" -ForegroundColor Green
        } else {
            Write-Host "Ollama did not become ready after 36 seconds - warm-up may fail" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Ollama executable not found on PATH - startup will rely on an already running service" -ForegroundColor Yellow
    }
} else {
    Write-Host "Reusing existing Ollama server; GPU placement may differ from the preferred launcher setting." -ForegroundColor Yellow
}

try {
    $ttsPing = Invoke-WebRequest "http://127.0.0.1:9880/docs" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    if ($ttsPing.StatusCode -eq 200) {
        $ttsServerReachable = $true
        Write-Host "GPT-SoVITS server detected on port 9880" -ForegroundColor Green
    }
} catch {
}

# 1. Start GPT-SoVITS TTS Server (Native - faster than Docker!)
Write-Host "`n[1/3] Starting GPT-SoVITS TTS Server (Native)..." -ForegroundColor Yellow
if (-not $ttsServerReachable) {
    $ttsStartup = @"
cd '$ProjectRoot\third_party\GPT-SoVITS'
`$env:CUDA_VISIBLE_DEVICES = '$TtsGpu'
`$env:is_half = 'true'
`$env:PATH = '$ProjectRoot\.venv\Lib\site-packages\nvidia\cudnn\bin;' + `$env:PATH
& '$GPT_SOVITS_PYTHON' -u api_v2.py -a 127.0.0.1 -p 9880
"@
    $ttsProc = Start-HiddenPowerShell -Command $ttsStartup

    # Wait for TTS server to be ready (model loading takes time, especially first run with OneDrive)
    Write-Host "Waiting for TTS server to load models..." -ForegroundColor Gray
    $ttsReady = Wait-ForHttpReady -Url "http://127.0.0.1:9880/docs" -ProgressLabel "TTS models"
    if ($ttsReady) {
        $ttsStartedByScript = $true
        Write-Host "TTS server is ready!" -ForegroundColor Green
    } else {
        Write-Host "TTS server not responding after 5 minutes - aborting startup" -ForegroundColor Red
        if ($ttsProc -and !$ttsProc.HasExited) {
            try { & taskkill /F /T /PID $ttsProc.Id 2>$null | Out-Null } catch {}
        }
        if ($ollamaStartedByScript -and $ollamaProc -and !$ollamaProc.HasExited) {
            try { & taskkill /F /T /PID $ollamaProc.Id 2>$null | Out-Null } catch {}
        }
        throw "GPT-SoVITS did not become ready on port 9880. Fix TTS before starting Annabeth."
    }
} else {
    Write-Host "Reusing existing GPT-SoVITS server; GPU placement may differ from the preferred launcher setting." -ForegroundColor Yellow
}

# 2. Start Main Chat Server
Write-Host "`n[2/3] Starting Main Chat Server..." -ForegroundColor Yellow
$chatStartup = @"
cd '$ProjectRoot'
`$env:ANNABETH_PROJECT_ROOT = '$ProjectRoot'
& '$VenvPython' -m server.main_chat
"@
$chatProc = Start-HiddenPowerShell -Command $chatStartup

# Wait for chat server to initialize and expose the avatar HTTP surface
Write-Host "Waiting for chat server to open the avatar endpoint..." -ForegroundColor Gray
$chatReady = Wait-ForProcessHttpReady -Process $chatProc -Url "http://127.0.0.1:8765/" -Attempts 30 -SleepSeconds 1 -ProgressLabel "chat server"

if (-not $chatReady) {
    if ($ttsStartedByScript -and $ttsProc -and !$ttsProc.HasExited) {
        try { & taskkill /F /T /PID $ttsProc.Id 2>$null | Out-Null } catch {}
    }
    if ($ollamaStartedByScript -and $ollamaProc -and !$ollamaProc.HasExited) {
        try { & taskkill /F /T /PID $ollamaProc.Id 2>$null | Out-Null } catch {}
    }

    if ($chatProc -and !$chatProc.HasExited) {
        try { & taskkill /F /T /PID $chatProc.Id 2>$null | Out-Null } catch {}
    }

    throw "Chat server did not become ready on port 8765. Check the backend console logs before launching the frontend."
}

# Pre-warm Ollama context allocation (forces model to allocate 3072-token KV cache
# so the first real user request doesn't pay the ~1-2s reallocation penalty)
Write-Host "Pre-warming Ollama LLM context..." -ForegroundColor Gray
try {
    $warmBody = @{ 
        model = $ollamaModel
        messages = @(@{ role = "user"; content = "hi" })
        stream = $false
        keep_alive = -1
        options = @{ num_ctx = 8192; num_predict = 1 }
    } | ConvertTo-Json -Depth 4 -Compress
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/chat" -Method POST -Body $warmBody -ContentType "application/json" -TimeoutSec 30 -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "Ollama context pre-warmed (8192 tokens allocated)" -ForegroundColor Green
} catch {
    Write-Host "Ollama warm-up skipped (model may need first-request allocation)" -ForegroundColor Yellow
}

# 3. Start Frontend (Unity or Legacy PyQt6)
if ($Legacy) {
    Write-Host "`n[3/3] Starting Desktop Companion (PyQt6 Legacy)..." -ForegroundColor Yellow
    $frontendProc = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList "-Command", "cd '$ProjectRoot\client'; & '$VenvPython' desktop_companion_webview.py"
} else {
    Write-Host "`n[3/3] Starting Desktop Companion (Unity)..." -ForegroundColor Yellow
    if (Test-Path $UnityBuild) {
        $frontendProc = Start-Process $UnityBuild -PassThru
    } else {
        Write-Host "Unity build not found at: $UnityBuild" -ForegroundColor Red
        Write-Host "Build from Unity: Annabeth > Build Standalone, or use -Legacy flag" -ForegroundColor Yellow
        Write-Host "Falling back to PyQt6..." -ForegroundColor Yellow
        $frontendProc = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList "-Command", "cd '$ProjectRoot\client'; & '$VenvPython' desktop_companion_webview.py"
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

# Hide this launcher console so it doesn't appear as a background window behind the character
$consoleHwnd = [Win32.ConsoleUtils]::GetConsoleWindow()
[Win32.ConsoleUtils]::ShowWindow($consoleHwnd, 0) | Out-Null  # SW_HIDE = 0

# Wait for the frontend (Unity) to exit, then clean up background processes
if ($frontendProc -and !$frontendProc.HasExited) {
    $frontendProc.WaitForExit()
}

# Kill the server process TREES using taskkill /F /T (kills children too - catches Python inside PowerShell wrapper)
foreach ($proc in @($(if ($ttsStartedByScript) { $ttsProc } else { $null }), $chatProc, $(if ($ollamaStartedByScript) { $ollamaProc } else { $null }))) {
    if ($proc -and !$proc.HasExited) {
        try { & taskkill /F /T /PID $proc.Id 2>$null } catch {}
    }
}

# Fallback: also kill by port in case process handles were lost
foreach ($port in 9880, 8765) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { & taskkill /F /T /PID $_.OwningProcess 2>$null }
}

Stop-Transcript -ErrorAction SilentlyContinue | Out-Null