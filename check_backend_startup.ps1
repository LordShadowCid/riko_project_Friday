param(
    [string]$TtsGpu = "1"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python virtual environment not found at $python"
}

$ttsTerminalId = $null

try {
    $ttsReachable = $false
    try {
        $response = Invoke-WebRequest "http://127.0.0.1:9880/docs" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        $ttsReachable = $response.StatusCode -eq 200
    } catch {
        $ttsReachable = $false
    }

    if (-not $ttsReachable) {
        Write-Host "Starting GPT-SoVITS for backend startup check..." -ForegroundColor Cyan
        $process = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $repoRoot "start-gpt-sovits-native.ps1"),
            "-Gpu", $TtsGpu
        )

        $deadline = (Get-Date).AddSeconds(90)
        do {
            Start-Sleep -Seconds 2
            try {
                $response = Invoke-WebRequest "http://127.0.0.1:9880/docs" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
                if ($response.StatusCode -eq 200) {
                    $ttsReachable = $true
                    break
                }
            } catch {}
        } while ((Get-Date) -lt $deadline -and -not $process.HasExited)

        if (-not $ttsReachable) {
            throw "GPT-SoVITS did not become reachable on http://127.0.0.1:9880/docs"
        }
    }

    Write-Host "Running backend self-check-only mode..." -ForegroundColor Cyan
    $env:ANNABETH_SELF_CHECK_ONLY = "1"
    & $python -m server.main_chat --self-check-only
    $exitCode = $LASTEXITCODE
    Remove-Item Env:ANNABETH_SELF_CHECK_ONLY -ErrorAction SilentlyContinue

    if ($exitCode -ne 0) {
        throw "Backend self-check-only mode failed with exit code $exitCode"
    }

    Write-Host "Backend startup self-check passed." -ForegroundColor Green
}
finally {
    Remove-Item Env:ANNABETH_SELF_CHECK_ONLY -ErrorAction SilentlyContinue
    powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot "stop_annabeth.ps1") | Out-Null
}