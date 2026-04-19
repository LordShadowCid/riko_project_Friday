# Annabeth PC Hardware Inventory
# PowerShell 5.1 compatible

$sep = '=' * 70

Write-Host ''
Write-Host $sep
Write-Host '  ANNABETH - PC HARDWARE & SOFTWARE INVENTORY'
Write-Host ('  ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Host $sep

Write-Host ''
Write-Host '>>> OPERATING SYSTEM'
Get-CimInstance Win32_OperatingSystem | Format-List Caption, Version, BuildNumber, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory

Write-Host '>>> CPU'
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, CurrentClockSpeed

Write-Host '>>> MEMORY (RAM)'
$ram = Get-CimInstance Win32_PhysicalMemory
$ram | Format-Table DeviceLocator, Capacity, Speed, MemoryType, Manufacturer -AutoSize
$totalGB = [math]::Round(($ram | Measure-Object Capacity -Sum).Sum / 1GB, 1)
Write-Host ('  Total Installed RAM: ' + $totalGB + ' GB')

Write-Host ''
Write-Host '>>> GPU(s)'
Get-CimInstance Win32_VideoController | Format-List Name, DriverVersion, AdapterRAM, VideoProcessor, CurrentHorizontalResolution, CurrentVerticalResolution, Status

Write-Host '>>> NVIDIA DRIVER / CUDA (nvidia-smi)'
try {
    $nvOut = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) { Write-Host ($nvOut | Out-String) }
    else { Write-Host '  nvidia-smi returned error (no NVIDIA GPU or driver not installed)' }
} catch {
    Write-Host '  nvidia-smi not found - NVIDIA driver likely not installed yet'
}

Write-Host ''
Write-Host '>>> STORAGE DRIVES'
Get-CimInstance Win32_DiskDrive | Format-Table Model, MediaType, @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}}, InterfaceType -AutoSize

Write-Host '>>> DRIVE PARTITIONS / FREE SPACE'
Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
    Format-Table DeviceID, VolumeName,
        @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}},
        @{N='FreeGB';E={[math]::Round($_.FreeSpace/1GB,1)}},
        @{N='FreePct';E={[math]::Round($_.FreeSpace/$_.Size*100,1)}} -AutoSize

Write-Host '>>> AUDIO DEVICES'
Get-CimInstance Win32_SoundDevice | Format-List Name, Manufacturer, Status, DeviceID

Write-Host '>>> NETWORK ADAPTERS (active)'
Get-NetAdapter | Where-Object {$_.Status -eq 'Up'} | Format-Table Name, InterfaceDescription, LinkSpeed, MacAddress -AutoSize

Write-Host '>>> USB DEVICES (summary)'
Get-CimInstance Win32_USBHub | Select-Object -First 15 | Format-Table Name, DeviceID -AutoSize

Write-Host ''
Write-Host $sep
Write-Host '  SOFTWARE / RUNTIME CHECKS'
Write-Host $sep

Write-Host ''
Write-Host '>>> PYTHON'
try { $pyVer = & python --version 2>&1; Write-Host ('  ' + $pyVer) }
catch { Write-Host '  Python NOT found on PATH' }

Write-Host ''
Write-Host '>>> PIP'
try { $pipOut = & python -m pip --version 2>&1; Write-Host ('  ' + $pipOut) }
catch { Write-Host '  pip not available' }

Write-Host ''
Write-Host '>>> GIT'
try { $gitVer = & git --version 2>&1; Write-Host ('  ' + $gitVer) }
catch { Write-Host '  Git NOT found on PATH' }

Write-Host ''
Write-Host '>>> NODE.JS'
try { $nodeVer = & node --version 2>&1; Write-Host ('  Node: ' + $nodeVer) }
catch { Write-Host '  Node.js NOT found' }

Write-Host ''
Write-Host '>>> DOCKER'
try { $dkVer = & docker --version 2>&1; Write-Host ('  ' + $dkVer) }
catch { Write-Host '  Docker NOT found' }

Write-Host ''
Write-Host '>>> OLLAMA'
try { $olVer = & ollama --version 2>&1; Write-Host ('  ' + $olVer) }
catch { Write-Host '  Ollama NOT found on PATH' }

Write-Host ''
Write-Host '>>> FFMPEG'
try { $ffVer = & ffmpeg -version 2>&1 | Select-Object -First 1; Write-Host ('  ' + $ffVer) }
catch { Write-Host '  ffmpeg NOT found on PATH' }

Write-Host ''
Write-Host '>>> CUDA TOOLKIT (nvcc)'
try { $nvcc = & nvcc --version 2>&1 | Select-String 'release'; Write-Host ('  ' + $nvcc) }
catch { Write-Host '  nvcc not found - CUDA toolkit may not be installed' }

Write-Host ''
Write-Host '>>> cuDNN CHECK'
$cudnnFound = $false
$cudaPath = $env:CUDA_PATH
if ($cudaPath) {
    $binFiles = Get-ChildItem (Join-Path $cudaPath 'bin\cudnn*.dll') -ErrorAction SilentlyContinue
    $hdrFiles = Get-ChildItem (Join-Path $cudaPath 'include\cudnn*.h') -ErrorAction SilentlyContinue
    if ($binFiles) { $binFiles | ForEach-Object { Write-Host ('  Found: ' + $_.FullName) }; $cudnnFound = $true }
    if ($hdrFiles) { $hdrFiles | ForEach-Object { Write-Host ('  Found: ' + $_.FullName) }; $cudnnFound = $true }
}
$nvidiaPath = 'C:\Program Files\NVIDIA\CUDNN'
if (Test-Path $nvidiaPath) {
    Get-ChildItem "$nvidiaPath\*\bin\cudnn*.dll" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('  Found: ' + $_.FullName); $cudnnFound = $true }
}
if (-not $cudnnFound) { Write-Host '  cuDNN libraries NOT found in common locations' }

Write-Host ''
Write-Host '>>> VISUAL C++ BUILD TOOLS'
$progX86 = [Environment]::GetFolderPath('ProgramFilesX86')
$vsWhere = Join-Path $progX86 'Microsoft Visual Studio\Installer\vswhere.exe'
$knownBuildToolsPath = Join-Path $progX86 'Microsoft Visual Studio\2022\BuildTools'
if (Test-Path $vsWhere) {
    $installs = & $vsWhere -all -format json 2>$null | ConvertFrom-Json
    foreach ($inst in $installs) {
        Write-Host ('  ' + $inst.displayName + ' - ' + $inst.installationVersion)
    }
}
elseif (Test-Path $knownBuildToolsPath) {
    Write-Host ('  Microsoft Visual Studio 2022 Build Tools - installed at ' + $knownBuildToolsPath)
}
else {
    Write-Host '  Visual Studio / Build Tools NOT detected'
}

Write-Host ''
Write-Host '>>> KEY ENVIRONMENT VARIABLES'
$envCheck = @('CUDA_PATH', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'OPENAI_API_KEY')
foreach ($varName in $envCheck) {
    $val = [Environment]::GetEnvironmentVariable($varName, 'Machine')
    if ($varName -eq 'OPENAI_API_KEY') {
        if ($val) { Write-Host '  OPENAI_API_KEY: (set)' } else { Write-Host '  OPENAI_API_KEY: (not set)' }
    } else {
        Write-Host ('  ' + $varName + ': ' + $val)
    }
}

Write-Host ''
Write-Host '  PATH entries (CUDA/Python/NVIDIA related):'
$sysPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
if ($sysPath) {
    $pattern = 'cuda|python|ollama|ffmpeg|nvidia'
    $sysPath -split ';' | Where-Object { $_ -match $pattern } | ForEach-Object { Write-Host ('    ' + $_) }
}

Write-Host ''
Write-Host $sep
Write-Host '  INVENTORY COMPLETE - Share this output for setup planning'
Write-Host $sep
