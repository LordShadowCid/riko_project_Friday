# Helper installer for Windows PowerShell
# Assumes you already activated your venv (recommended).
# Example:
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"

Write-Host "Upgrading pip + installing uv..."
python -m pip install --upgrade pip
python -m pip install --upgrade uv

Write-Host "Installing dependencies (extra-req.txt, requirements-client.txt)..."
uv pip install -r extra-req.txt
uv pip install -r requirements-client.txt

Write-Host "Checking CTranslate2 CUDA visibility..."
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"

Write-Host "Done. If CUDA devices is > 0, set whisper.device=cuda in character_config.yaml." 
