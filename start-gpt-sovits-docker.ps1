$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$gptDir = Join-Path $repoRoot 'third_party\GPT-SoVITS'

if (-not (Test-Path $gptDir)) {
  Write-Host "Cloning GPT-SoVITS into $gptDir ..."
  git clone https://github.com/RVC-Boss/GPT-SoVITS $gptDir
} else {
  Write-Host "GPT-SoVITS repo already present at $gptDir"
}

Write-Host "Starting GPT-SoVITS container on http://127.0.0.1:9880 ..."
docker compose -f (Join-Path $repoRoot 'docker-compose.gpt-sovits.yml') up -d

Write-Host "\nCheck:"
Write-Host "  curl http://127.0.0.1:9880/tts?text=hi&text_lang=en&ref_audio_path=/data/ref/main_sample.wav&prompt_lang=en&prompt_text=hi" 
