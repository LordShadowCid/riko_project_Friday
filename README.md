# Project Annabeth

Project Annabeth is a anime focused LLM project (forked from Project Riko by Just Rayen). She listens, and remembers your conversations. It combines Ollama/OpenAI LLM, GPT-SoVITS voice synthesis, Faster-Whisper ASR, and a 3D VRM avatar into a fully configurable conversational pipeline.

**tested with python 3.10 Windows >10 and Linux Ubuntu**

## 🧾 Reference snapshots

This repo includes a `reference/` folder containing snapshots from before the upstream re-sync (kept for comparison and for reusing old code if needed):

- `reference/backup_pre_upstream_reset_2025-12-31/src/`: snapshot of the pre-reset branch
- `reference/stash_pre_upstream_reset_2025-12-31/`: snapshot of the pre-reset working tree (includes uncommitted changes)

## ✨ Features

- 💬 **LLM-based dialogue** using OpenAI API (configurable system prompts)
- 🧠 **Conversation memory** to keep context during interactions
- 🔊 **Voice generation** via GPT-SoVITS API
- 🎧 **Speech recognition** using Faster-Whisper
- 📁 Clean YAML-based config for personality configuration


## ⚙️ Configuration

All prompts and parameters are stored in `character_config.yaml`.

## Workstation Layout

For the best Windows performance, keep the repo, model folders, and generated assets on a local non-OneDrive drive such as `D:`. Running large model files from OneDrive-backed paths increases startup latency and makes Python environments more fragile.

Recommended split on this workstation class:

- Ollama / LLM on GPU 0
- Faster-Whisper + GPT-SoVITS on GPU 1
- Models, caches, Unity builds, and third-party weights on `D:`

The startup launcher now auto-detects the repo root from its own location, so the project can be moved without editing hardcoded paths.

If you want to stage the repo onto a faster drive, use `move_annabeth_to_d_drive.ps1`. It mirrors the current workspace to `D:\Annabeth` by default and leaves the source folder untouched.

For a Windows optimization pass after moving, run `optimize_annabeth_workstation.ps1` as Administrator. It pins the user `OLLAMA_MODELS` path to `D:\AI\Models\Ollama` and adds Windows Defender exclusions for the Annabeth workspace, Ollama model storage, and Unity build output.

```yaml
OPENAI_API_KEY: sk-YOURAPIKEY
history_file: chat_history.json
model: "gpt-4.1-mini"
presets:
  default:
    system_prompt: |
      You are a helpful assistant named Riko.
      You speak like a snarky anime girl.
      Always refer to the user as "senpai".

sovits_ping_config:
  text_lang: en
  prompt_lang : en
  ref_audio_path : D:\PyProjects\waifu_project\riko_project\character_files\main_sample.wav
  prompt_text : This is a sample voice for you to just get started with because it sounds kind of cute but just make sure this doesn't have long silences.
  
````

You can define personalities by modiying the config file.

### Windows 11 notes (audio devices + GPU)

You can optionally pick which microphone/speaker devices to use (helpful if you have multiple audio devices):

```yaml
audio:
  input_device:  # e.g. 1  OR  "Realtek"  (substring match)
  output_device: # e.g. 5  OR  "Speakers" (substring match)
```

To see your available device names/indexes:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Whisper can run on CPU or GPU. For example, on CUDA:

```yaml
whisper:
  model: small.en
  device: cuda
  compute_type: float16
  # cuda_visible_devices: "0"  # optionally pick GPU 0 vs 1
```


## 🛠️ Setup

### Install Dependencies

```bash
pip install uv 
uv pip install -r extra-req.txt
uv pip install -r requirements-client.txt
```

If you are also running the full GPT-SoVITS stack locally, you may need the larger dependency set in `requirements.txt` (it is intentionally heavy).

### Faster-Whisper on GPU (Windows 11)

This project uses Faster-Whisper (CTranslate2). Per the Faster-Whisper documentation, GPU execution requires NVIDIA CUDA 12 + cuDNN 9 (and cuBLAS for CUDA 12).

After installing your NVIDIA driver + CUDA 12 + cuDNN 9, you can validate that CTranslate2 sees your GPUs:

```bash
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

If this prints `CUDA devices: 0`, Faster-Whisper will run on CPU.

#### Troubleshooting (when `CUDA devices: 0`)

- Confirm your NVIDIA driver is installed and `nvidia-smi` shows your GPU(s).
- Confirm you installed CUDA 12 + cuDNN 9 (older CUDA/cuDNN combos typically won’t work with the newest CTranslate2 GPU wheels).
- If you have multiple GPUs, you can force which one Faster-Whisper sees via `whisper.cuda_visible_devices` in `character_config.yaml`.

#### CPU fallback

If you just want it to work (slower, but simplest), set:

```yaml
whisper:
  device: cpu
  compute_type: int8
```

### Quick verification (Windows)

Use these quick checks to confirm your local install is healthy:

```bash
# List audio devices (helps pick audio.input_device/audio.output_device)
python -c "import sounddevice as sd; print(sd.query_devices())"

# Confirm Faster-Whisper is installed
python -c "import faster_whisper; print('faster_whisper import: OK')"

# Confirm your OpenAI API key is present if you plan to use OpenAI instead of local Ollama
python -c "import os; print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))"

# Confirm GPT-SoVITS server docs are reachable (optional if you want the launcher to start TTS for you)
python -c "import requests; print('TTS server HTTP:', requests.get('http://127.0.0.1:9880/docs', timeout=2).status_code)"

# Start the full stack from the repo folder (Unity frontend by default)
.\start_annabeth.ps1

# Or start the Legacy PyQt6 frontend explicitly
.\start_annabeth.ps1 -Legacy

# Or start it after moving the repo to another drive
.\start_annabeth.ps1 -ProjectRoot D:\Annabeth -OllamaGpu 0 -TtsGpu 1
```

Then set your config to use CUDA:

```yaml
whisper:
  device: cuda
  compute_type: float16
```

**If you want to use GPU support for Faster whisper** Make sure you also have:

* CUDA & cuDNN installed correctly (for Faster-Whisper GPU support)
* `ffmpeg` installed (for audio processing)

For the focused runtime regression slice after startup, Unity, avatar, or read-aloud changes, run:

```powershell
.\run_runtime_checks.ps1
```

To validate backend startup without entering the microphone loop, run:

```powershell
.\check_backend_startup.ps1
```


## 🧪 Usage

### 1. Preferred launch path

The canonical Windows startup path is:

```powershell
.\start_annabeth.ps1
```

What it does:

1. Reuses or starts Ollama on `127.0.0.1:11434`
2. Starts GPT-SoVITS on `127.0.0.1:9880`
3. Starts the Python backend with `python -m server.main_chat`
4. Launches the Unity companion if the configured build exists
5. Falls back to the Legacy PyQt6 frontend if the Unity build is missing, or if you pass `-Legacy`

Use these variants when needed:

```powershell
# Force the Legacy frontend
.\start_annabeth.ps1 -Legacy

# Reuse an already-running Ollama server without restarting it
.\start_annabeth.ps1 -ReuseRunningOllama

# Override the Unity build path
.\start_annabeth.ps1 -UnityBuild "C:\Path\To\Annabeth.exe"
```

If you end up with a stale backend, TTS server, or frontend after a crash, use:

```powershell
.\stop_annabeth.ps1
```

Add `-IncludeOllama` only if you also want to stop the local Ollama server.

### 2. Optional manual service launch

This repo calls a GPT-SoVITS WebAPI at `http://127.0.0.1:9880/tts`.

**Docker (recommended on Windows):**

1. Ensure Docker Desktop is installed and GPU support is enabled (WSL2 + NVIDIA Container Toolkit).
2. From the repo root, run:

```powershell
./start-gpt-sovits-docker.ps1
```

This will clone the official GPT-SoVITS repo into `third_party/GPT-SoVITS/` (if missing) and start a container exposing port `9880`.

Quick reachability check:

```bash
python -c "import requests; print('TTS server HTTP:', requests.get('http://127.0.0.1:9880/tts', timeout=2).status_code)"
```

Notes:
- The container must be able to read the reference audio. This repo mounts `./character_files` into the container at `/data/ref`, and `character_config.yaml` uses `/data/ref/main_sample.wav`.
- GPT-SoVITS may require downloading its pretrained models inside `third_party/GPT-SoVITS/` (see the GPT-SoVITS docs if `/tts` returns a model/config error).
- If this workspace already has models staged under `gpt_sovits_models/`, the native launchers will automatically junction that shared store into `third_party/GPT-SoVITS/` at startup.

### 3. Manual backend entrypoint


```bash
python -m server.main_chat
```

Use the direct Python entrypoint only when you are intentionally launching services by hand for debugging. For normal desktop use, prefer `start_annabeth.ps1` so the backend and frontend stay on the same startup contract.

The flow:

1. Annabeth listens to your voice via microphone (push to talk)
2. Transcribes it with Faster-Whisper
3. Passes it to Ollama or OpenAI (with history)
4. Generates a response
5. Synthesizes Annabeth's voice using GPT-SoVITS
6. Plays the output back to you

### Read-Aloud Notes

For "read this" / selected-text read-aloud:

- Keep the app with the selected text focused when you give the voice command, so Annabeth copies from the selected app instead of the companion window.
- In the Legacy PyQt6 frontend, you can also use `Ctrl+Shift+R` to trigger read-aloud for the current selection.
- If Annabeth says she does not see any selected text, reselect the text and try again without bringing the companion window to the foreground first.

For a repeatable end-to-end manual verification flow, use [docs/RUNTIME_SMOKE_CHECKLIST.md](d:\Annabeth\docs\RUNTIME_SMOKE_CHECKLIST.md).


## 📌 TODO / Future Improvements

* [ ] GUI or web interface
* [ ] Live microphone input support
* [ ] Emotion or tone control in speech synthesis
* [ ] VRM model frontend


## 🧑‍🎤 Credits

* Voice synthesis powered by [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* ASR via [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* Language model via [OpenAI GPT](https://platform.openai.com)


## 📜 License

MIT — feel free to clone, modify, and build your own waifu voice companion.


