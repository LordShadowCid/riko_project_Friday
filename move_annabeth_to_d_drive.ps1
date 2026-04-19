param(
    [string]$SourceRoot,
    [string]$DestinationRoot = "D:\Annabeth"
)

$ErrorActionPreference = "Stop"

if (-not $SourceRoot) {
    if ($PSScriptRoot) {
        $SourceRoot = $PSScriptRoot
    } else {
        $SourceRoot = (Get-Location).Path
    }
}

$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)
$DestinationRoot = [System.IO.Path]::GetFullPath($DestinationRoot)

if (-not (Test-Path $SourceRoot)) {
    throw "Source root not found: $SourceRoot"
}

if ($DestinationRoot.StartsWith($SourceRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination cannot be inside the source folder. Choose a separate path such as D:\Annabeth"
}

$destDrive = [System.IO.Path]::GetPathRoot($DestinationRoot)
if (-not (Test-Path $destDrive)) {
    throw "Destination drive not found: $destDrive"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Annabeth Workspace Migration Helper   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Source      : $SourceRoot" -ForegroundColor Gray
Write-Host "Destination : $DestinationRoot" -ForegroundColor Gray
Write-Host ""
Write-Host "Stop Annabeth, Unity builds, Ollama model pulls, and editors writing into this folder before mirroring." -ForegroundColor Yellow

if (-not (Test-Path $DestinationRoot)) {
    New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
}

$roboArgs = @(
    $SourceRoot,
    $DestinationRoot,
    "/MIR",
    "/COPY:DAT",
    "/DCOPY:DAT",
    "/R:2",
    "/W:2",
    "/XJ",
    "/FFT",
    "/NFL",
    "/NDL",
    "/NP",
    "/XD", ".vs"
)

Write-Host "Mirroring workspace with robocopy..." -ForegroundColor Green
& robocopy @roboArgs
$robocopyExit = $LASTEXITCODE

if ($robocopyExit -ge 8) {
    throw "Robocopy reported a failure (exit code $robocopyExit)"
}

Write-Host ""
Write-Host "Mirror complete." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open the destination folder in VS Code" -ForegroundColor Gray
Write-Host "  2. Run .\setup_annabeth.ps1 if you want a dependency re-check" -ForegroundColor Gray
Write-Host "  3. Start with .\start_annabeth.ps1 -ProjectRoot '$DestinationRoot'" -ForegroundColor Gray
Write-Host ""
Write-Host "The source folder was not deleted." -ForegroundColor Yellow