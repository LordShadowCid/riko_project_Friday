param(
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python virtual environment not found at $python"
}

$tests = @(
    "check_backend_startup.ps1",
    "test_read_aloud.py",
    "test_avatar_state_sync.py",
    "test_avatar_message_broadcast.py",
    "test_system_integrity.py"
)

$passed = 0
$failed = 0

foreach ($test in $tests) {
    Write-Host "`n============================================================"
    Write-Host "Running $test"
    Write-Host "============================================================"

    if ($test.EndsWith('.ps1')) {
        powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot $test)
    }
    else {
        & $python (Join-Path $repoRoot $test)
    }
    if ($LASTEXITCODE -eq 0) {
        $passed += 1
        Write-Host "[PASS] $test" -ForegroundColor Green
    }
    else {
        $failed += 1
        Write-Host "[FAIL] $test (exit $LASTEXITCODE)" -ForegroundColor Red
        if ($StopOnFailure) {
            break
        }
    }
}

Write-Host "`n============================================================"
Write-Host "Runtime check summary: $passed passed, $failed failed"
Write-Host "============================================================"

if ($failed -gt 0) {
    exit 1
}

exit 0