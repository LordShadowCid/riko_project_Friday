# Download Free VRMA Animation Packs
# Run this script to download free VRM animation files for the avatar

$ErrorActionPreference = "Stop"
$animationsDir = Join-Path $PSScriptRoot "animations"

Write-Host "=== VRMA Animation Downloader ===" -ForegroundColor Cyan
Write-Host ""

# Create animations directory
if (-not (Test-Path $animationsDir)) {
    New-Item -ItemType Directory -Path $animationsDir | Out-Null
    Write-Host "Created animations folder: $animationsDir" -ForegroundColor Green
}

# Temporary download folder
$tempDir = Join-Path $env:TEMP "vrma_downloads"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir | Out-Null
}

Write-Host ""
Write-Host "NOTE: BOOTH requires a web browser to download files." -ForegroundColor Yellow
Write-Host "I'll open the download pages for you. Save the files to:" -ForegroundColor Yellow
Write-Host "  $animationsDir" -ForegroundColor White
Write-Host ""

# Free VRMA packs to download
$packs = @(
    @{
        Name = "VRoid Project Official 7-Pack (FREE)"
        URL = "https://booth.pm/en/items/5512385"
        Files = @("VRMA_01.vrma", "VRMA_02.vrma", "VRMA_03.vrma", "VRMA_04.vrma", "VRMA_05.vrma", "VRMA_06.vrma", "VRMA_07.vrma")
        Description = "Show body, Greeting, Peace sign, Shoot, Spin, Model pose, Squat"
    },
    @{
        Name = "fumi2kick Motion Pack (FREE - CC0)"
        URL = "https://booth.pm/en/items/5527394"
        Files = @("001_motion_pose.vrma", "002_dogeza.vrma", "003_humidai.vrma", "004_hello_1.vrma", "005_smartphone.vrma", "006_drinkwater.vrma", "007_gekirei.vrma", "008_gatan.vrma")
        Description = "Pose, Bow, Step, Hello/Wave, Smartphone, Drink, Encourage, React"
    },
    @{
        Name = "Wakatuya 5 Poses (FREE)"
        URL = "https://booth.pm/en/items/5876268"
        Files = @()
        Description = "5 static poses for photos"
    }
)

Write-Host "Available FREE VRMA packs:" -ForegroundColor Cyan
Write-Host ""

foreach ($pack in $packs) {
    Write-Host "  $($pack.Name)" -ForegroundColor White
    Write-Host "    $($pack.Description)" -ForegroundColor Gray
    Write-Host "    URL: $($pack.URL)" -ForegroundColor Blue
    Write-Host ""
}

Write-Host ""
$choice = Read-Host "Open BOOTH download pages in browser? (y/n)"

if ($choice -eq 'y' -or $choice -eq 'Y') {
    foreach ($pack in $packs) {
        Write-Host "Opening: $($pack.Name)..." -ForegroundColor Green
        Start-Process $pack.URL
        Start-Sleep -Seconds 1
    }
    
    Write-Host ""
    Write-Host "=== INSTRUCTIONS ===" -ForegroundColor Cyan
    Write-Host "1. Click 'Free Download' on each BOOTH page" -ForegroundColor White
    Write-Host "2. You may need to create a free BOOTH/pixiv account" -ForegroundColor White
    Write-Host "3. Extract the ZIP files to:" -ForegroundColor White
    Write-Host "   $animationsDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Expected folder structure:" -ForegroundColor Cyan
    Write-Host "  animations/" -ForegroundColor White
    Write-Host "    VRMA_01.vrma  (Show body)" -ForegroundColor Gray
    Write-Host "    VRMA_02.vrma  (Greeting) <-- Great for voice assistant!" -ForegroundColor Green
    Write-Host "    VRMA_03.vrma  (Peace sign)" -ForegroundColor Gray
    Write-Host "    004_hello_1.vrma  (Hello/Wave) <-- Also great!" -ForegroundColor Green
    Write-Host "    ..." -ForegroundColor Gray
    Write-Host ""
}

# Check if any VRMA files already exist
$existingFiles = Get-ChildItem -Path $animationsDir -Filter "*.vrma" -ErrorAction SilentlyContinue
if ($existingFiles) {
    Write-Host "=== Found VRMA files ===" -ForegroundColor Green
    foreach ($file in $existingFiles) {
        Write-Host "  $($file.Name)" -ForegroundColor White
    }
    Write-Host ""
    Write-Host "These animations will appear in the avatar viewer!" -ForegroundColor Cyan
} else {
    Write-Host "No VRMA files found yet in animations folder." -ForegroundColor Yellow
    Write-Host "Download and extract the packs to see animation buttons in the viewer." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "After downloading, restart the avatar server to load animations." -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
