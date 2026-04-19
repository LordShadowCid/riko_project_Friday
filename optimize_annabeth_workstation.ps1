param(
    [string]$WorkspaceRoot = "D:\Annabeth",
    [string]$OllamaModelsPath = "D:\AI\Models\Ollama",
    [string]$UnityBuildRoot = "C:\Users\blakd\unit\Builds\AnnabethTest"
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-Directory {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Annabeth Workstation Optimization     " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Ensure-Directory -Path $WorkspaceRoot
Ensure-Directory -Path $OllamaModelsPath

[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $OllamaModelsPath, "User")
Write-Host "Set user OLLAMA_MODELS to $OllamaModelsPath" -ForegroundColor Green

$exclusionPaths = @($WorkspaceRoot, $OllamaModelsPath, $UnityBuildRoot) | Where-Object { Test-Path $_ }

if (-not (Test-IsAdministrator)) {
    Write-Host "Run this script as Administrator to apply Windows Defender exclusions." -ForegroundColor Yellow
    Write-Host "Planned exclusions:" -ForegroundColor Gray
    $exclusionPaths | ForEach-Object { Write-Host ("  " + $_) -ForegroundColor Gray }
    exit 0
}

$preferences = Get-MpPreference
$existing = @($preferences.ExclusionPath)

foreach ($path in $exclusionPaths) {
    if ($existing -contains $path) {
        Write-Host "Defender exclusion already present: $path" -ForegroundColor Gray
        continue
    }

    Add-MpPreference -ExclusionPath $path
    Write-Host "Added Defender exclusion: $path" -ForegroundColor Green
}

Write-Host "Optimization pass complete." -ForegroundColor Green