# Annabeth Workstation — Hardware & Software Reference
> Scanned: 2026-03-15

## Hardware

| Component | Details |
|-----------|---------|
| **CPU** | Intel i9-12900 — 16 cores / 24 threads @ 2.4 GHz |
| **RAM** | 64 GB DDR5-5400 (2 x 32 GB Samsung) |
| **GPU 0** | NVIDIA RTX A4000 — 16 GB VRAM (Bus 01:00.0) |
| **GPU 1** | NVIDIA RTX A4000 — 16 GB VRAM (Bus 02:00.0) |
| **iGPU** | Intel UHD Graphics 770 |
| **Boot Drive (C:)** | WD PC SN740 NVMe 477 GB — ~332 GB free |
| **AI_DATA (D:)** | Storage Space 10 TB — ~10 TB free |
| **DATA_1TB (E:)** | Storage Space 930 GB — ~930 GB free |
| **Audio** | Realtek USB Audio (ASUS), Logitech USB Audio (BCC950), NVIDIA HDMI Audio x2 |
| **Network** | Marvell AQtion 10G + Intel I226-V (both active at 1 Gbps) |
| **USB** | Logitech BCC950 webcam/speaker, ASUS peripherals |

## NVIDIA / CUDA

| Item | Value |
|------|-------|
| Driver | 595.79 |
| CUDA (runtime, nvidia-smi) | 13.2 |
| CUDA Toolkit (nvcc) | 13.2 (V13.2.51) |
| CUDA_PATH | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` |
| cuDNN | 9.20.0.48 (via pip nvidia-cudnn-cu12) |
| CUDA_HOME | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` |

## Installed Software (as of 2026-03-21)

| Software | Version | Notes |
|----------|---------|-------|
| Windows | 11 Pro Build 26200 | 64-bit |
| Python | 3.13.12 | `C:\Users\blakd\AppData\Local\Programs\Python\Python313` |
| pip | 25.3 | |
| Git | 2.53.0 | |
| Node.js | 24.14.0 | |
| Docker | 29.2.1 | |
| Ollama | 0.18.0 | Running on both GPUs |
| NVIDIA Driver | 595.79 | |
| ffmpeg | 8.1 (Gyan.FFmpeg via winget) | On PATH |
| cuDNN | 9.20.0.48 (nvidia-cudnn-cu12 pip) | DLLs in venv, added to PATH by start script |
| VS Build Tools | 2022 Build Tools | Installed at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |

## GPU Assignment Plan

| GPU | Assigned To |
|-----|-------------|
| GPU 0 (01:00.0) | Ollama / LLM inference |
| GPU 1 (02:00.0) | Faster-Whisper ASR + GPT-SoVITS TTS |

## Port Allocations

| Port | Service |
|------|---------|
| 9880 | GPT-SoVITS TTS API |
| 8765 | Avatar WebSocket |
| 8766 | Desktop Companion HTTP |
| 11434 | Ollama LLM |
