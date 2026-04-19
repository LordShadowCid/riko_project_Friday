param(
    [switch]$IncludeOllama,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

$targets = @(
    @{ Name = "Annabeth backend"; Port = 8765 },
    @{ Name = "Legacy companion HTTP"; Port = 8766 },
    @{ Name = "GPT-SoVITS"; Port = 9880 }
)

if ($IncludeOllama) {
    $targets += @{ Name = "Ollama"; Port = 11434 }
}

function Get-ListeningProcessInfo {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -Property OwningProcess -Unique

    foreach ($connection in $connections) {
        try {
            $process = Get-Process -Id $connection.OwningProcess -ErrorAction Stop
            [PSCustomObject]@{
                Port = $Port
                ProcessId = $process.Id
                ProcessName = $process.ProcessName
                Path = $process.Path
            }
        }
        catch {
            [PSCustomObject]@{
                Port = $Port
                ProcessId = $connection.OwningProcess
                ProcessName = "<unknown>"
                Path = $null
            }
        }
    }
}

function Stop-ProcessTreeSafe {
    param([int]$ProcessId, [string]$Label)

    if ($WhatIf) {
        Write-Host "[WhatIf] Would stop $Label (PID $ProcessId)" -ForegroundColor Yellow
        return
    }

    try {
        & taskkill /F /T /PID $ProcessId 2>$null | Out-Null
        Write-Host "Stopped $Label (PID $ProcessId)" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to stop $Label (PID $ProcessId): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stopping Annabeth Processes          " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$foundAny = $false

foreach ($target in $targets) {
    $infos = @(Get-ListeningProcessInfo -Port $target.Port)
    if ($infos.Count -eq 0) {
        Write-Host "$($target.Name): nothing listening on port $($target.Port)" -ForegroundColor Gray
        continue
    }

    $foundAny = $true
    foreach ($info in $infos) {
        $label = "$($target.Name) on port $($info.Port) [$($info.ProcessName)]"
        Stop-ProcessTreeSafe -ProcessId $info.ProcessId -Label $label
    }
}

$frontendCandidates = @(
    @(Get-Process -Name "Annabeth" -ErrorAction SilentlyContinue)
    @(Get-Process -Name "AnnabethTest" -ErrorAction SilentlyContinue)
) | Where-Object { $_ } | Sort-Object Id -Unique

foreach ($process in $frontendCandidates) {
    $foundAny = $true
    Stop-ProcessTreeSafe -ProcessId $process.Id -Label "Unity frontend [$($process.ProcessName)]"
}

if (-not $foundAny) {
    Write-Host "No Annabeth-owned listeners or frontend processes were found." -ForegroundColor Gray
}