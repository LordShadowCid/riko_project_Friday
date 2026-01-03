from server.process.asr_func.asr_push_to_talk import record_and_transcribe, transcribe_file
from server.process.llm_funcs.llm_scr import llm_response
from server.process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import os
import time
import asyncio
import threading
### transcribe audio 
import uuid
import soundfile as sf

from server.riko_config import load_config, repo_root, resolve_repo_path

# Avatar server integration
avatar_api = None
avatar_loop = None

def _start_avatar_server():
    """Start the avatar WebSocket server in a background thread"""
    global avatar_api, avatar_loop
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "client"))
        from avatar_server import start_avatar_server, get_avatar_api
        
        avatar_loop = asyncio.new_event_loop()
        
        def run_server():
            asyncio.set_event_loop(avatar_loop)
            avatar_loop.run_until_complete(start_avatar_server())
            avatar_loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        # Give server time to start
        time.sleep(0.5)
        
        avatar_api = get_avatar_api()
        print("[Avatar] Server started at http://localhost:8765")
        
    except ImportError as e:
        print(f"[Avatar] Could not start avatar server (missing aiohttp?): {e}")
        print("[Avatar] Install with: pip install aiohttp")
    except Exception as e:
        print(f"[Avatar] Could not start avatar server: {e}")


def avatar_speak_start(text: str = None):
    """Notify avatar that speaking is starting"""
    if avatar_api and avatar_loop:
        asyncio.run_coroutine_threadsafe(
            avatar_api['speak_start'](text), 
            avatar_loop
        )


def avatar_speak_end():
    """Notify avatar that speaking has ended"""
    if avatar_api and avatar_loop:
        asyncio.run_coroutine_threadsafe(
            avatar_api['speak_end'](), 
            avatar_loop
        )


def _prepare_whisper_model_source(model_name: str) -> str:
    """Ensure Faster-Whisper model is loadable on Windows without symlink privileges.

    Hugging Face cache normally uses symlinks. On Windows without Developer Mode/admin,
    symlink creation can fail (WinError 1314). To avoid that, download into a local
    folder using file copies and load the model from that folder.
    """
    try:
        # If a local path was provided, just use it.
        p = Path(model_name)
        if p.exists():
            return str(p)
    except Exception:
        pass

    # If user provided a repo_id (e.g. "Systran/faster-whisper-base.en"), let
    # faster-whisper handle it.
    if "/" in str(model_name) or "\\" in str(model_name):
        return model_name

    # Windows-only workaround.
    if os.name != "nt":
        return model_name

    repo_id = f"Systran/faster-whisper-{model_name}"
    local_dir = repo_root() / "models" / "faster_whisper" / str(model_name)
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return str(local_dir)
    except Exception as e:
        print(f"NOTE: Whisper model pre-download failed ({repo_id}): {e}")
        return model_name


def _startup_self_check(char_config: dict, input_device, output_device, whisper_cfg: dict):
    print("\n--- Startup self-check ---")

    # API key sanity
    api_key = str(os.environ.get('OPENAI_API_KEY') or char_config.get('OPENAI_API_KEY', '') or '')
    if not api_key or api_key.strip() in {"sk-YOURAPIKEY", "YOUR_API_KEY"}:
        print("WARNING: OPENAI_API_KEY is not set (set env var OPENAI_API_KEY or set it in character_config.yaml)")

    # Ref audio sanity
    try:
        ref_audio = (char_config.get('sovits_ping_config') or {}).get('ref_audio_path')
        if ref_audio:
            # If user set a container/Linux path (e.g. /data/ref/main_sample.wav),
            # skip local filesystem existence checks.
            if isinstance(ref_audio, str) and ref_audio.strip().startswith("/"):
                print(f"TTS: ref_audio_path is a container path: {ref_audio}")
            else:
                ref_audio_abs = resolve_repo_path(ref_audio)
                if not Path(ref_audio_abs).exists():
                    print(f"WARNING: ref_audio_path not found: {ref_audio_abs}")
    except Exception:
        pass

    # Audio device visibility
    try:
        import sounddevice as sd

        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
        print(f"Audio: input_device={input_device!r} (default={default_in}), output_device={output_device!r} (default={default_out})")
    except Exception:
        print(f"Audio: input_device={input_device!r}, output_device={output_device!r}")

    # Whisper settings recap
    print(
        "Whisper config: "
        + f"model={whisper_cfg.get('model', 'base.en')} "
        + f"device={whisper_cfg.get('device', 'cpu')} "
        + f"compute_type={whisper_cfg.get('compute_type', 'float32')}"
    )

    # TTS server reachability (best-effort)
    try:
        import requests

        requests.get("http://127.0.0.1:9880/", timeout=1)
    except Exception:
        print("NOTE: GPT-SoVITS server not detected at http://127.0.0.1:9880 (start it before chatting)")

    # Repo root recap
    try:
        print(f"Repo root: {repo_root()}")
    except Exception:
        pass

    print("--- End self-check ---\n")


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')

# Start avatar server
_start_avatar_server()

char_config = load_config()

whisper_cfg = char_config.get('whisper', {}) or {}
cuda_visible = whisper_cfg.get('cuda_visible_devices')
if cuda_visible:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)

from faster_whisper import WhisperModel

whisper_model_name = whisper_cfg.get('model', 'base.en')
whisper_device = whisper_cfg.get('device', 'cpu')
whisper_compute_type = whisper_cfg.get('compute_type', 'float32')

whisper_model_source = _prepare_whisper_model_source(str(whisper_model_name))
print(f"Whisper: model={whisper_model_name} device={whisper_device} compute_type={whisper_compute_type}")
if whisper_model_source != whisper_model_name:
    print(f"Whisper: using local model folder: {whisper_model_source}")
try:
    whisper_model = WhisperModel(whisper_model_source, device=whisper_device, compute_type=whisper_compute_type)
except Exception as e:
    msg = str(e)
    cuda_requested = str(whisper_device).lower() == "cuda"
    maybe_cudnn = ("cudnn" in msg.lower()) or ("cublas" in msg.lower()) or ("Could not locate" in msg)
    allow_fallback = bool(whisper_cfg.get("fallback_to_cpu", True))

    if cuda_requested and maybe_cudnn and allow_fallback:
        print("\nWARNING: Whisper CUDA init failed; falling back to CPU.")
        print("This usually means CUDA 12 is installed but cuDNN 9 DLLs are missing from PATH.")
        print("Fix: install cuDNN 9 for CUDA 12 and ensure the cuDNN 'bin' folder is on PATH (contains cudnn_ops64_9.dll).")
        print(f"Original error: {e}\n")
        whisper_device = "cpu"
        whisper_compute_type = "int8"
        whisper_model = WhisperModel(whisper_model_source, device=whisper_device, compute_type=whisper_compute_type)
    else:
        raise

audio_cfg = char_config.get('audio', {}) or {}
input_device = audio_cfg.get('input_device')
output_device = audio_cfg.get('output_device')

_startup_self_check(char_config, input_device, output_device, whisper_cfg)

while True:
    conversation_recording = Path("audio") / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    output_wav_path = None

    try:
        try:
            user_spoken_text = record_and_transcribe(
                whisper_model,
                conversation_recording,
                input_device=input_device,
            )
        except Exception as e:
            msg = str(e)
            cudnn_like = ("cudnn" in msg.lower()) or ("cudnn_ops64_9.dll" in msg) or ("cudnnCreateTensorDescriptor" in msg)
            cuda_requested = str(whisper_cfg.get('device', '')).lower() == 'cuda'
            allow_fallback = bool(whisper_cfg.get('fallback_to_cpu', True))

            if cudnn_like and cuda_requested and allow_fallback:
                print("\nWARNING: Whisper GPU transcription failed (cuDNN missing). Falling back to CPU for ASR.")
                print("Fix: install cuDNN 9 for CUDA 12 and add its 'bin' folder to PATH (contains cudnn_ops64_9.dll).")
                print(f"Original error: {e}\n")

                whisper_model = WhisperModel(whisper_model_source, device='cpu', compute_type='int8')
                if Path(conversation_recording).exists():
                    user_spoken_text = transcribe_file(whisper_model, str(conversation_recording))
                    print(f"Transcription: {user_spoken_text}")
                else:
                    raise
            else:
                raise
        if not user_spoken_text:
            print("No transcription captured; try again.")
            continue

        llm_output = llm_response(user_spoken_text)
        if not llm_output:
            print("LLM returned empty output; try again.")
            continue

        print(f"Riko: {llm_output}")

        # Generate a unique filename
        uid = uuid.uuid4().hex
        filename = f"output_{uid}.wav"
        output_wav_path = Path("audio") / filename
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        gen_aud_path = sovits_gen(llm_output, output_wav_path)
        if gen_aud_path is None:
            print("TTS generation failed (is GPT-SoVITS running on 127.0.0.1:9880?)")
            continue

        # Notify avatar to start lip-sync
        avatar_speak_start(llm_output)
        
        play_audio(output_wav_path, output_device=output_device)
        
        # Notify avatar to stop lip-sync
        avatar_speak_end()

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print("Error during chat loop:", e)
    finally:
        # clean up audio files
        try:
            for fp in Path("audio").glob("*.wav"):
                if fp.is_file():
                    fp.unlink()
        except Exception:
            pass
    # # Example
    # duration = get_wav_duration(output_wav_path)

    # print("waiting for audio to finish...")
    # time.sleep(duration)