# Riko Project - GPT-SoVITS Docker Setup Script (PowerShell)
Write-Host "Starting Riko Project with GPT-SoVITS Docker..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
$dockerComposeAvailable = $false
try {
    docker compose version | Out-Null
    $dockerComposeAvailable = $true
} catch {
    try {
        docker-compose --version | Out-Null
    } catch {
        Write-Host "Docker Compose is not available. Please install Docker Compose." -ForegroundColor Red
        exit 1
    }
}

# Create models directory if it doesn't exist
if (!(Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}

Write-Host "Starting GPT-SoVITS container..." -ForegroundColor Yellow

# Use docker compose if available, otherwise fall back to docker-compose
if ($dockerComposeAvailable) {
    docker compose up -d gpt-sovits
} else {
    docker-compose up -d gpt-sovits
}

Write-Host "Waiting for GPT-SoVITS to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if the service is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9880/health" -TimeoutSec 5
    Write-Host "GPT-SoVITS is running on http://localhost:9880" -ForegroundColor Green
} catch {
    Write-Host "GPT-SoVITS is starting up... This may take a few minutes on first run." -ForegroundColor Yellow
    Write-Host "Check logs with: docker logs riko-gpt-sovits" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Wait for GPT-SoVITS to fully start (check logs: docker logs riko-gpt-sovits)"
Write-Host "2. Test the API: Invoke-WebRequest http://localhost:9880/health"
Write-Host "3. Run your Riko chat: C:/Users/blakd/OneDrive/Desktop/Anabeth/.venv/Scripts/python.exe server/main_chat.py"
Write-Host ""
Write-Host "To stop: docker compose down" -ForegroundColor Red