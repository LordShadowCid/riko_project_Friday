# Project Riko

Project Riko is a anime focused LLM project by Just Rayen. She listens, and remembers your conversations. It combines OpenAI’s GPT, GPT-SoVITS voice synthesis, and Faster-Whisper ASR into a fully configurable conversational pipeline.

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

# Confirm your OpenAI API key is present (required for LLM calls)
python -c "import os; print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))"

# Confirm GPT-SoVITS server is reachable (required for TTS)
python -c "import requests; print('TTS server HTTP:', requests.get('http://127.0.0.1:9880').status_code)"
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


## 🧪 Usage

### 1. Launch the GPT-SoVITS API 

### 2. Run the main script:


```bash
python server/main_chat.py
```

The flow:

1. Riko listens to your voice via microphone (push to talk)
2. Transcribes it with Faster-Whisper
3. Passes it to GPT (with history)
4. Generates a response
5. Synthesizes Riko's voice using GPT-SoVITS
6. Plays the output back to you


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


