"""
Main chat loop for Annabeth Desktop Companion.

Handles:
- Speech recognition (Whisper)
- LLM conversation
- TTS synthesis (GPT-SoVITS)
- Avatar integration
"""
# Logging must be configured before any other server imports so that
# modules which call logging.getLogger(__name__) at import time pick up
# the correct handlers.
from server.logging_config import configure_logging
configure_logging()

# Temporary test-session debug logger — remove after testing is complete.
from server.debug_logger import setup_test_logging
setup_test_logging()

import logging
_log = logging.getLogger(__name__)

from server.process.asr_func.asr_push_to_talk import record_and_transcribe, transcribe_file
from server.process.asr_func.asr_vad import (
    record_vad_and_transcribe, 
    get_interrupt_flag,
    get_speaking_flag,
)
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    keyboard = None
    HAS_KEYBOARD = False
from server.process.llm_funcs.llm_scr import llm_response, llm_response_streaming
from server.process.tts_func.sovits_ping import sovits_gen, play_audio
from server.process.read_aloud.text_capture import estimate_word_timings, capture_selected_text, get_last_capture_debug
from pathlib import Path
import os
import sys
import time
import asyncio
import threading
import queue
import uuid
import re
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from server.annabeth_config import load_config, repo_root, resolve_repo_path
from server.process.memory.feedback import log_feedback
from server.utils import configure_windows_cuda_runtime, describe_port_listener

# Add parent directory for shared imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared import (
    CompanionMode,
    get_config,
    get_companion_state,
    get_read_aloud_manager,
)

# Get shared instances
_shared_config = get_config()
_state = get_companion_state()

# Avatar server integration
avatar_api: Optional[dict] = None
avatar_loop: Optional[asyncio.AbstractEventLoop] = None
SELF_CHECK_ONLY = (
    "--self-check-only" in sys.argv
    or str(os.environ.get("ANNABETH_SELF_CHECK_ONLY", "")).strip().lower() in {"1", "true", "yes", "on"}
)

# Module-level state — initialized here so _start_avatar_server() can reference them
# before the full initialisation block lower in the file.
_improvement_scheduler = None

def _start_avatar_server() -> None:
    """Start the avatar WebSocket server in a background thread"""
    global avatar_api, avatar_loop

    # Import from client directory
    client_dir = _shared_config.paths.client_dir
    if str(client_dir) not in sys.path:
        sys.path.insert(0, str(client_dir))

    from avatar_server import start_avatar_server, get_avatar_api

    avatar_loop = asyncio.new_event_loop()
    startup_ready = threading.Event()
    startup_error = []

    def run_server():
        try:
            asyncio.set_event_loop(avatar_loop)
            avatar_loop.run_until_complete(start_avatar_server())
        except Exception as exc:
            startup_error.append(exc)
        finally:
            startup_ready.set()

        if startup_error:
            return

        avatar_loop.run_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    if not startup_ready.wait(timeout=5.0):
        raise RuntimeError("Avatar server startup timed out before binding port 8765")

    if startup_error:
        error = startup_error[0]
        if isinstance(error, OSError) and getattr(error, "errno", None) == 10048:
            port = _shared_config.server.avatar_port
            detail = describe_port_listener(port, expected_text="Annabeth Avatar")
            raise RuntimeError(
                f"Avatar server failed to start because {detail} Stop the existing Annabeth backend or use the running instance instead."
            ) from error
        raise RuntimeError(f"Avatar server failed to start: {error}") from error

    avatar_api = get_avatar_api()
    print(f"[Avatar] Server started at {_shared_config.server.avatar_http_url}")

    # Register face broadcast for facial expression timeline (Phase 2)
    try:
        from server.process.llm_funcs.facial_expressions import set_face_broadcast
        set_face_broadcast(avatar_loop, avatar_api['broadcast'])
        print("[FacialExpr] Broadcast registered")
    except Exception as _fe_err:
        print(f"[FacialExpr] Broadcast registration failed (non-fatal): {_fe_err}")

    # Start self-improvement scheduler inside the avatar event loop (Phase 10)
    if _improvement_scheduler is not None:
        try:
            asyncio.run_coroutine_threadsafe(
                _start_improvement_scheduler(), avatar_loop
            )
        except Exception as _si_err:
            print(f"[SelfImprovement] Scheduler start failed (non-fatal): {_si_err}")


def avatar_speak_start(text: Optional[str] = None) -> None:
    """Notify avatar that speaking is starting"""
    if avatar_api and avatar_loop:
        _state.speaking = True
        asyncio.run_coroutine_threadsafe(
            avatar_api['speak_start'](text), 
            avatar_loop
        )


def avatar_speak_end() -> None:
    """Notify avatar that speaking has ended"""
    if avatar_api and avatar_loop:
        _state.speaking = False
        asyncio.run_coroutine_threadsafe(
            avatar_api['speak_end'](), 
            avatar_loop
        )


def avatar_debug_status(status: str, user_text: str = "", response_text: str = "") -> None:
    """Send debug overlay info to Unity"""
    if avatar_api and avatar_loop:
        asyncio.run_coroutine_threadsafe(
            avatar_api['send_debug_status'](status, user_text, response_text),
            avatar_loop
        )


def is_listening_paused() -> bool:
    """Check if listening should be paused (dance/idle modes or silenced)"""
    # Use shared state - this is updated by avatar_server when S key is pressed
    return _state.is_listening_paused()


async def _start_improvement_scheduler():
    """Coroutine that starts the self-improvement scheduler inside the avatar event loop."""
    if _improvement_scheduler is not None:
        _improvement_scheduler.start()
        print("[SelfImprovement] Scheduler running (interval: weekly).")


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
    local_dir = _shared_config.paths.models_dir / "faster_whisper" / str(model_name)
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


def _startup_self_check(char_config: dict, input_device, output_device, whisper_cfg: dict) -> None:
    print("\n--- Startup self-check ---")
    startup_errors = []

    # Backend sanity
    api_key = str(os.environ.get('OPENAI_API_KEY') or char_config.get('OPENAI_API_KEY', '') or '')
    if not api_key or api_key.strip() in {"sk-YOURAPIKEY", "YOUR_API_KEY"}:
        print("LLM: OPENAI_API_KEY not set; local Ollama backend will be used")
    else:
        print("LLM: OPENAI_API_KEY detected; OpenAI backend may override local Ollama")

    # Ref audio sanity (warning only — pyttsx3 fallback handles missing TTS server)
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
                    print(f"WARNING: TTS ref_audio_path not found: {ref_audio_abs} (will use pyttsx3 fallback)")
        else:
            print("WARNING: TTS ref_audio_path is not configured (will use pyttsx3 fallback)")
    except Exception:
        print("WARNING: TTS ref_audio_path could not be validated (will use pyttsx3 fallback)")

    # Audio device visibility
    try:
        import sounddevice as sd

        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
        print(f"Audio: input_device={input_device!r} (default={default_in}), output_device={output_device!r} (default={default_out})")
    except Exception:
        print(f"Audio: input_device={input_device!r}, output_device={output_device!r}")

    # Whisper settings recap
    w_device = whisper_cfg.get('device', 'cpu')
    w_device_idx = whisper_cfg.get('device_index', 0)
    device_str = f"{w_device}" + (f" (GPU {w_device_idx})" if w_device == 'cuda' else "")
    print(
        "Whisper config: "
        + f"model={whisper_cfg.get('model', 'base.en')} "
        + f"device={device_str} "
        + f"compute_type={whisper_cfg.get('compute_type', 'float32')}"
    )

    # TTS server reachability (best-effort — warning only, pyttsx3 fallback exists)
    tts_url = _shared_config.server.tts_url
    try:
        import requests
        r = requests.get(tts_url + "/docs", timeout=3)
        if r.status_code == 200:
            print(f"TTS: GPT-SoVITS server reachable at {tts_url} [OK]")
        else:
            print(f"WARNING: GPT-SoVITS returned status {r.status_code} — pyttsx3 fallback active")
    except Exception:
        print(f"WARNING: GPT-SoVITS not detected at {tts_url} — pyttsx3 fallback active")

    # Ollama reachability (retry up to 3 times for slow starts)
    ollama_host = char_config.get('ollama', {}).get('host', 'http://127.0.0.1:11434')
    ollama_model = char_config.get('ollama', {}).get('model') or char_config.get('model', 'mannix/llama3.1-8b-abliterated')
    ollama_ok = False
    ollama_model_found = False
    for _attempt in range(3):
        try:
            import requests
            r = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                if any(ollama_model in m for m in models):
                    print(f"Ollama: {ollama_model} loaded [OK]")
                    ollama_model_found = True
                else:
                    startup_errors.append(
                        f"Ollama running but model '{ollama_model}' not found. Available: {', '.join(models[:5])}"
                    )
                ollama_ok = True
                break
            else:
                startup_errors.append(f"Ollama returned status {r.status_code}")
        except Exception:
            if _attempt < 2:
                print(f"Ollama not ready, retrying ({_attempt + 1}/3)...")
                time.sleep(2)
            else:
                startup_errors.append(f"Ollama not detected at {ollama_host} (is it running?)")

    # Pre-warm Ollama model — force-load into VRAM so first real query is fast
    if ollama_ok and ollama_model_found:
        try:
            import requests
            keep_alive = char_config.get('ollama', {}).get('keep_alive', -1)
            print(f"Ollama: Pre-warming {ollama_model} (loading into VRAM)...")
            r = requests.post(f"{ollama_host}/api/generate", json={
                "model": ollama_model,
                "prompt": "hi",
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_predict": 1},
            }, timeout=120)
            if r.status_code == 200:
                print(f"Ollama: Model pre-warmed [OK]")
            else:
                print(f"Ollama: Pre-warm got status {r.status_code}")
        except Exception as e:
            print(f"Ollama: Pre-warm failed (non-fatal): {e}")

    # Audio input/output device validation
    try:
        import sounddevice as sd
        if input_device:
            from server.process.asr_func.asr_vad import _resolve_device as resolve_input
            resolved_in = resolve_input(input_device, kind='input')
            if resolved_in is not None:
                print(f"Audio input: '{input_device}' -> device {resolved_in} [OK]")
            else:
                print(f"WARNING: Input device '{input_device}' not found — will use default")
        if output_device:
            from server.process.tts_func.sovits_ping import _resolve_device as resolve_output
            resolved_out = resolve_output(output_device, kind='output')
            if resolved_out is not None:
                print(f"Audio output: '{output_device}' -> device {resolved_out} [OK]")
            else:
                print(f"WARNING: Output device '{output_device}' not found — will use default")
    except Exception as e:
        print(f"Audio device check error: {e}")

    # Repo root recap
    print(f"Repo root: {_shared_config.paths.project_root}")

    if startup_errors:
        print("--- End self-check ---\n")
        raise RuntimeError("Startup self-check failed:\n - " + "\n - ".join(startup_errors))

    print("--- End self-check ---\n")


def get_wav_duration(path) -> float:
    """Get duration of a WAV file in seconds."""
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


def clean_text_for_tts(text: str) -> str:
    """Clean up LLM output for natural TTS playback.
    
    Removes:
    - Asterisk actions like *laughs* or *sighs*
    - ALL CAPS words (converts to lowercase)
    - Multiple exclamation/question marks
    - Emotion state tags: {happy 8.5} or malformed amused 7.1})
    - Percentage emotion annotations: 25% happy, 40% annoyed
    """
    if not text:
        return text
    
    # Remove asterisk-wrapped actions like *laughs* or *OH BOY*
    # This handles both actions and emphasized words
    text = re.sub(r'\*[^*]+\*', '', text)
    
    # Strip all curly-brace blocks — catches {happy 8.5} emotion tags that
    # slipped past the on_sentence wrapper (belt-and-suspenders)
    text = re.sub(r'\{[^}]*\}', '', text)

    # Strip percentage-format emotion annotations: "25% happy", "40% annoyed"
    _EMOTION_WORDS = (
        r'happy|sad|angry|annoyed|anxious|amused|curious|relaxed|'
        r'fear|disgust|surprised|love|neutral|devotion|arousal'
    )
    text = re.sub(
        r'\b\d+(?:\.\d+)?%\s+(?:' + _EMOTION_WORDS + r')\b',
        '', text, flags=re.IGNORECASE
    )

    # Strip orphaned closing-brace fragments — the model often produces malformed
    # emotion annotations like:  "amused 7.1})"  "disappointed resignation 8.0}))"
    # Pattern: one or two words followed by a decimal number and closing brace/paren
    text = re.sub(
        r'\b\w+(?:\s+\w+)?\s+-?\d+(?:\.\d+)?\s*[})]+',
        '', text, flags=re.IGNORECASE
    )

    # Strip "Feeling: word NUMBER}" and "Feeling: word" annotations
    text = re.sub(r'\bFeeling:\s*[\w\s,.:]+\d[)},]*', '', text, flags=re.IGNORECASE)

    # Strip trailing orphan punctuation left by removals  e.g. "):  " -> ""
    text = re.sub(r'^[)\]},:\s]+', '', text)
    text = re.sub(r'[)\]},:\s]+$', '', text)

    # Convert ALL CAPS words (3+ letters) to title case
    def fix_caps(match):
        word = match.group(0)
        return word.capitalize()
    
    text = re.sub(r'\b[A-Z]{3,}\b', fix_caps, text)
    
    # Reduce multiple punctuation (!!!! -> !)
    text = re.sub(r'([!?]){2,}', r'\1', text)
    
    # Clean up extra whitespace from removals
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# =============================================================================
# READ ALOUD FUNCTIONALITY
# =============================================================================

def process_read_aloud_queue(output_device) -> bool:
    """
    Process pending read-aloud sentences with pre-buffering.
    
    Uses a background thread to pre-generate TTS for the NEXT sentence
    while the current one is playing, eliminating gaps.
    Interruption is via Ctrl+Shift+X hotkey (interrupt flag polled by play_audio).
    
    Returns True if read-aloud was processed, False otherwise.
    """
    try:
        read_aloud = get_read_aloud_manager()
    except Exception:
        return False
    
    if not read_aloud.state.is_reading:
        return False
    
    print("\n[ReadAloud] Reading selected text...")
    audio_dir = _shared_config.paths.audio_dir
    sentence_idx = 0
    
    # Pre-buffer queue: holds (sentence, audio_path) for next sentence
    next_audio = [None]  # [0] = (cleaned_text, audio_path) or None
    prefetch_thread = [None]
    prefetch_lock = threading.Lock()
    
    def prefetch_next_sentence(idx: int):
        """Pre-generate TTS for the next sentence in background."""
        import time as _time
        _prefetch_start = _time.time()
        
        # Get sentence at idx (without advancing state)
        sentences = read_aloud.state.sentences
        if idx >= len(sentences):
            with prefetch_lock:
                next_audio[0] = None
            print(f"  [Prefetch] Sentence {idx}: no more sentences")
            return
        
        sentence = sentences[idx]
        cleaned = clean_text_for_tts(sentence)
        if not cleaned:
            with prefetch_lock:
                next_audio[0] = ('', None)  # Empty sentence
            print(f"  [Prefetch] Sentence {idx}: empty after cleaning")
            return
        
        # Generate TTS
        uid = uuid.uuid4().hex
        output_path = audio_dir / f"read_prefetch_{uid}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        gen_path = sovits_gen(cleaned, output_path)
        _prefetch_elapsed = _time.time() - _prefetch_start
        
        with prefetch_lock:
            if gen_path:
                next_audio[0] = (cleaned, output_path)
                print(f"  [Prefetch] Sentence {idx}: TTS ready in {_prefetch_elapsed:.2f}s")
            else:
                next_audio[0] = (cleaned, None)  # TTS failed
                print(f"  [Prefetch] Sentence {idx}: TTS FAILED after {_prefetch_elapsed:.2f}s")
    
    # Start prefetching first sentence
    first_idx = read_aloud.state.current_index
    prefetch_thread[0] = threading.Thread(target=prefetch_next_sentence, args=(first_idx,), daemon=True)
    prefetch_thread[0].start()
    
    while read_aloud.state.is_reading:
        # Check for pause request
        if read_aloud.state.pause_requested:
            read_aloud.state.complete_pause()
            print("\n[ReadAloud] [PAUSED] Paused - ask questions or press R to resume")
            return True
        
        # Wait for prefetched audio
        import time as _time
        _wait_start = _time.time()
        if prefetch_thread[0]:
            prefetch_thread[0].join(timeout=30)  # Max wait 30s
        _wait_elapsed = _time.time() - _wait_start
        
        with prefetch_lock:
            prefetched = next_audio[0]
            next_audio[0] = None
        
        if _wait_elapsed > 0.1:
            print(f"  [ReadAloud] ⏳ Waited {_wait_elapsed:.2f}s for TTS prefetch")
        
        if prefetched is None:
            # No more sentences
            break
        
        cleaned, output_path = prefetched
        
        if not cleaned:
            # Empty sentence - advance and continue
            read_aloud.state.advance_index()
            # Start prefetch for next
            next_idx = read_aloud.state.current_index
            prefetch_thread[0] = threading.Thread(target=prefetch_next_sentence, args=(next_idx,), daemon=True)
            prefetch_thread[0].start()
            continue
        
        print(f"  [>] {cleaned}")
        
        # Send highlight data to browser clients
        try:
            duration_estimate = max(len(cleaned.split()) * 0.35, 1.0)
            word_timings = [
                {"word": w, "start": s, "end": e}
                for (w, s, e) in estimate_word_timings(cleaned, duration_estimate)
            ]
            if avatar_api and avatar_loop:
                asyncio.run_coroutine_threadsafe(
                    avatar_api.get('send_read_highlight')(cleaned, word_timings, sentence_idx),
                    avatar_loop,
                )
        except Exception as e:
            print(f"[ReadAloud] Highlight send failed: {e}")
        
        if output_path is None:
            print("  (TTS failed for this sentence)")
            read_aloud.state.advance_index()
            # Start prefetch for next
            next_idx = read_aloud.state.current_index
            prefetch_thread[0] = threading.Thread(target=prefetch_next_sentence, args=(next_idx,), daemon=True)
            prefetch_thread[0].start()
            continue
        
        # Notify avatar
        avatar_speak_start(cleaned)
        
        # Set speaking flag (Ctrl+Shift+X hotkey sets interrupt flag)
        get_interrupt_flag().clear()
        get_speaking_flag().set()
        
        # START PREFETCH FOR NEXT SENTENCE (runs in parallel with playback!)
        next_idx = read_aloud.state.current_index + 1
        prefetch_thread[0] = threading.Thread(target=prefetch_next_sentence, args=(next_idx,), daemon=True)
        prefetch_thread[0].start()
        
        # Play audio (while next is being generated)
        was_interrupted = not play_audio(
            output_path,
            output_device=output_device,
            interrupt_flag=get_interrupt_flag(),
        )
        
        get_speaking_flag().clear()
        
        # Cleanup this audio file
        try:
            output_path.unlink()
        except Exception:
            pass
        
        if was_interrupted:
            # User interrupted - request pause
            read_aloud.request_pause()
            print("\n[ReadAloud] [PAUSED] Interrupted - finishing sentence...")
            read_aloud.state.complete_pause()
            print("[ReadAloud] [PAUSED] Paused - ask questions or press R to resume")
            avatar_speak_end()
            # Clean up prefetched audio if any
            with prefetch_lock:
                if next_audio[0] and next_audio[0][1]:
                    try:
                        Path(next_audio[0][1]).unlink()
                    except Exception:
                        pass
                next_audio[0] = None
            try:
                if avatar_api and avatar_loop:
                    asyncio.run_coroutine_threadsafe(avatar_api.get('send_read_clear')(), avatar_loop)
            except Exception:
                pass
            return True
        
        # Advance to next sentence
        read_aloud.state.advance_index()
        sentence_idx += 1
    
    avatar_speak_end()
    try:
        if avatar_api and avatar_loop:
            asyncio.run_coroutine_threadsafe(avatar_api.get('send_read_clear')(), avatar_loop)
    except Exception:
        pass
    
    if not read_aloud.state.is_paused:
        print("[ReadAloud] Finished reading [OK]")
    
    return True


print(' \n ========= Starting Chat... ================ \n')

# Start avatar server
_start_avatar_server()

char_config = load_config()

configure_windows_cuda_runtime()

whisper_cfg = char_config.get('whisper', {}) or {}
cuda_visible = whisper_cfg.get('cuda_visible_devices')
if cuda_visible:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)

from faster_whisper import WhisperModel

whisper_model_name = whisper_cfg.get('model', 'base.en')
whisper_device = whisper_cfg.get('device', 'cpu')
whisper_device_index = whisper_cfg.get('device_index', 0)  # Which GPU to use
whisper_compute_type = whisper_cfg.get('compute_type', 'float32')

whisper_model_source = _prepare_whisper_model_source(str(whisper_model_name))
device_info = f"device={whisper_device}"
if whisper_device == 'cuda':
    device_info += f" (GPU {whisper_device_index})"
print(f"Whisper: model={whisper_model_name} {device_info} compute_type={whisper_compute_type}")
if whisper_model_source != whisper_model_name:
    print(f"Whisper: using local model folder: {whisper_model_source}")
try:
    whisper_model = WhisperModel(
        whisper_model_source,
        device=whisper_device,
        device_index=whisper_device_index,
        compute_type=whisper_compute_type
    )
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

# VAD configuration - can be added to character_config.yaml later
vad_cfg = char_config.get('vad', {}) or {}
use_vad = vad_cfg.get('enabled', True)  # Default to hands-free mode
vad_aggressiveness = vad_cfg.get('aggressiveness', 3)  # 0-3, higher = more aggressive filtering
silence_threshold = vad_cfg.get('silence_threshold_sec', 1.0)
vad_min_energy = vad_cfg.get('min_energy', 300)  # Minimum RMS energy to consider as speech (filters noise)
# Interrupt detection settings - higher values = less sensitive (fewer false interrupts)
# Speaker identification settings
speaker_id_cfg = char_config.get('speaker_id', {}) or {}
use_speaker_id = speaker_id_cfg.get('enabled', True)
speaker_id_threshold = speaker_id_cfg.get('threshold', 0.75)
current_speaker = None  # Track who is currently speaking

# Pre-warm models for faster first request
if use_speaker_id:
    print("[Warmup] Loading speaker encoder and profiles...")
    from server.process.asr_func.speaker_id import _get_encoder, load_speaker_profiles
    _get_encoder()  # Force-load the voice encoder now
    load_speaker_profiles()  # Load all speaker profiles into memory

_startup_self_check(char_config, input_device, output_device, whisper_cfg)

if SELF_CHECK_ONLY:
    print("[Startup] Self-check-only mode complete; exiting before microphone loop.")
    raise SystemExit(0)

# Start autonomous reflection loop
try:
    from server.process.memory.reflection_loop import start_reflection_loop, get_proactive_queue
    def _is_annabeth_idle():
        return not _state.speaking and not get_speaking_flag().is_set()
    _reflection_loop = start_reflection_loop(is_idle_fn=_is_annabeth_idle)
    _proactive_queue = get_proactive_queue()
except Exception as _e:
    print(f"[Reflection] Failed to start (non-fatal): {_e}")
    _reflection_loop = None
    _proactive_queue = None

# Start emotion decay loop so emotions gradually return to baseline
try:
    from server.process.memory.emotion_state import start_decay_loop
    start_decay_loop(interval_seconds=60)
except Exception as _e:
    print(f"[EmotionState] Failed to start decay loop (non-fatal): {_e}")

# Start self-improvement scheduler (Phase 10)
try:
    from server.process.self_improvement.scheduler import ImprovementScheduler, SchedulerConfig
    from pathlib import Path as _Path
    _improvement_scheduler = ImprovementScheduler(
        config=SchedulerConfig(src_path=_Path("server")),
        on_improvement_ready=lambda imp: print(
            f"[SelfImprovement] Opportunity: {imp.opportunity.description} "
            f"in {imp.opportunity.file_path}:{imp.opportunity.line_number}"
        ),
    )
    # Scheduler.start() requires an asyncio event loop — deferred to the async startup.
except Exception as _e:
    print(f"[SelfImprovement] Failed to initialise (non-fatal): {_e}")
    _improvement_scheduler = None

if use_vad:
    print("\n[MIC] HANDS-FREE MODE enabled - just start speaking!")
    print("   (Annabeth will listen and respond automatically)")
    print("   (Press Ctrl+Shift+X to interrupt her while she's speaking)\n")
else:
    print("\n[PTT] PUSH-TO-TALK MODE - press ENTER to record\n")

# Hotkey-based interruption (replaces mic-based BackgroundListener)
# Ctrl+Shift+X sets the interrupt flag; play_audio() polls it every 50ms.
def _hotkey_interrupt():
    flag = get_interrupt_flag()
    if get_speaking_flag().is_set():
        flag.set()
        print("[STOP] Hotkey interrupt!")

if HAS_KEYBOARD:
    keyboard.add_hotkey('ctrl+shift+x', _hotkey_interrupt, suppress=False)
else:
    print("[STOP] 'keyboard' module not installed; global interrupt hotkey disabled")

_prev_llm_thread = None  # Track previous LLM thread for join-on-interrupt

while True:
    # =========================================================================
    # CHECK FOR SHUTDOWN REQUEST (from Unity OnApplicationQuit)
    # =========================================================================
    if _state.shutdown_requested:
        print("\n[Shutdown] Frontend requested shutdown — exiting...")
        break

    # =========================================================================
    # CHECK PROACTIVE THOUGHT QUEUE (reflection loop)
    # =========================================================================
    if _proactive_queue is not None and not _proactive_queue.empty() and not is_listening_paused():
        try:
            thought = _proactive_queue.get_nowait()
            if thought:
                print(f"\n[Proactive] {thought}")
                # Broadcast as idle_thought speech bubble to Unity (Phase 6)
                if avatar_api and avatar_loop:
                    asyncio.run_coroutine_threadsafe(
                        avatar_api['broadcast']({"type": "idle_thought", "text": thought}),
                        avatar_loop
                    )
                avatar_speak_start(thought)
                uid = uuid.uuid4().hex
                _proactive_audio_dir = _shared_config.paths.audio_dir
                _proactive_audio_dir.mkdir(parents=True, exist_ok=True)
                _pq_wav = sovits_gen(thought, _proactive_audio_dir / f"proactive_{uid}.wav")
                if _pq_wav:
                    play_audio(_pq_wav, output_device=output_device)
                    try:
                        Path(_pq_wav).unlink(missing_ok=True)
                    except Exception:
                        pass
                avatar_speak_end()
        except Exception as _pq_e:
            print(f"[Proactive] Error: {_pq_e}")

    # =========================================================================
    # CHECK FOR READ-ALOUD QUEUE
    # =========================================================================
    # Process any pending read-aloud sentences before checking for new input
    try:
        if process_read_aloud_queue(output_device):
            # Read-aloud was processed
            read_aloud = get_read_aloud_manager()
            if read_aloud.state.is_paused:
                # Stay in loop - user can ask questions via voice
                # Continue below to normal voice input handling
                pass
            else:
                # Reading complete, go back to normal loop
                continue
    except Exception as e:
        print(f"[ReadAloud] Error: {e}")
    
    # =========================================================================
    # CHECK PAUSE STATE
    # =========================================================================
    # Check if we should pause listening (dance/idle modes or silenced)
    paused = is_listening_paused()
    if paused:
        # Use spaces to clear the line when overwriting with \r
        print("[Main] Listening paused - waiting...          ", end='\r', flush=True)
        time.sleep(0.1)  # Check again in 100ms (faster S key response)
        continue
    
    # Clear the "waiting" message when resuming
    print("                                                  ", end='\r', flush=True)
    
    audio_dir = _shared_config.paths.audio_dir
    conversation_recording = audio_dir / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    output_wav_path = None
    llm_input_override = None  # set by read-aloud stop handler
    
    # Timing diagnostics
    t_start = time.time()

    try:
        # Clear any previous interrupt flags
        get_interrupt_flag().clear()
        
        speaker_name = None
        t_record_start = time.time()
        avatar_debug_status("[MIC] Listening...")
        try:
            if use_vad:
                # Hands-free VAD-based recording with speaker identification
                user_spoken_text, speaker_name = record_vad_and_transcribe(
                    whisper_model,
                    str(conversation_recording),
                    input_device=input_device,
                    sample_rate=16000,
                    vad_aggressiveness=vad_aggressiveness,
                    silence_threshold_sec=silence_threshold,
                    identify_speaker=use_speaker_id,
                    speaker_threshold=speaker_id_threshold,
                    min_audio_energy=vad_min_energy,
                )
                current_speaker = speaker_name
            else:
                # Traditional push-to-talk
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
            avatar_debug_status("[X] No speech captured", user_text="(empty)")
            continue
        
        # Send user input to Unity overlay
        speaker_tag = f"[{speaker_name}]" if speaker_name else ""
        avatar_debug_status("[...] Processing...", user_text=f"{speaker_tag} {user_spoken_text}")
        
        # Timing: record + transcribe complete
        t_transcribe_done = time.time()
        print(f"\n[Timing] Record+Transcribe: {t_transcribe_done - t_record_start:.2f}s")

        # =====================================================================
        # CHECK FOR READ-ALOUD INTENT
        # =====================================================================
        # Detect if user wants to read selected text
        read_intent_phrases = [
            "read that", "read this", "read the selected", "read selected",
            "read it", "read aloud", "read to me", "read what i selected",
            "read the text", "can you read", "please read",
            "read this for me", "read that for me", "read it for me",
            "read what's highlighted", "read the highlighted",
            "read my selection", "read what i highlighted",
        ]
        user_lower = user_spoken_text.lower().strip()
        is_read_intent = any(phrase in user_lower for phrase in read_intent_phrases)

        # ----- RESUME / STOP intents while paused -----
        resume_phrases = [
            "keep reading", "continue reading", "go on", "resume reading",
            "read on", "carry on", "keep going", "continue where you left off",
        ]
        stop_phrases = [
            "stop reading", "that's enough", "never mind", "cancel reading",
            "forget it", "done reading", "quit reading",
        ]
        is_resume_intent = any(p in user_lower for p in resume_phrases)
        is_stop_intent   = any(p in user_lower for p in stop_phrases)

        read_aloud = get_read_aloud_manager()

        # Handle stop while paused or reading
        if is_stop_intent and read_aloud.is_active:
            read_aloud.stop()
            print("[ReadAloud] Stopped by voice command")
            # Let LLM respond naturally (omit the raw stop phrase)
            llm_input_override = "The user asked me to stop reading."

        # Handle resume while paused
        elif is_resume_intent and read_aloud.state.is_paused:
            read_aloud.resume()
            print("[ReadAloud] Resumed by voice command")
            continue  # process_read_aloud_queue picks it up next loop

        elif is_read_intent:
            print("[ReadAloud] Detected read intent - capturing text...")
            captured_text = capture_selected_text(restore_clipboard=True)
            
            if captured_text and len(captured_text) > 5:
                print(f"[ReadAloud] Captured {len(captured_text)} characters")

                # TTS acknowledgment before reading
                ack_text = "Sure, let me read that for you."
                print(f"Annabeth: {ack_text}")
                ack_path = audio_dir / f"read_ack_{uuid.uuid4().hex}.wav"
                ack_path.parent.mkdir(parents=True, exist_ok=True)
                ack_wav = sovits_gen(ack_text, ack_path)
                if ack_wav:
                    avatar_speak_start(ack_text)
                    play_audio(ack_wav, output_device=output_device)
                    avatar_speak_end()
                    try:
                        ack_path.unlink()
                    except Exception:
                        pass

                read_aloud.state.start_reading(captured_text)
                # Go back to top of loop to process the queue
                continue
            else:
                print("[ReadAloud] No text captured - make sure text is selected!")
                capture_debug = get_last_capture_debug()
                print(
                    "[ReadAloud] Capture debug: "
                    f"hwnd={capture_debug.get('foreground_hwnd')} "
                    f"title={capture_debug.get('foreground_title')!r} "
                    f"target_was_companion={capture_debug.get('target_was_companion')} "
                    f"clipboard_had_text_before={capture_debug.get('clipboard_had_text_before')}"
                )
                # Tell the user through TTS
                fail_text = "I don't see any text selected. Try highlighting the text you want me to read first."
                print(f"Annabeth: {fail_text}")
                fail_path = audio_dir / f"read_fail_{uuid.uuid4().hex}.wav"
                fail_path.parent.mkdir(parents=True, exist_ok=True)
                fail_wav = sovits_gen(fail_text, fail_path)
                if fail_wav:
                    avatar_speak_start(fail_text)
                    play_audio(fail_wav, output_device=output_device)
                    avatar_speak_end()
                    try:
                        fail_path.unlink()
                    except Exception:
                        pass
                continue

        # =====================================================================
        # CHECK FOR READ-ALOUD Q&A CONTEXT
        # =====================================================================
        # If read-aloud is paused, add context about what was being read
        qa_context = ""
        try:
            if read_aloud.state.is_paused:
                qa_context = read_aloud.get_qa_context()
                if qa_context:
                    print("[ReadAloud] Answering question about the text...")
        except Exception:
            pass
        
        # Combine Q&A context with user input if available
        llm_input = llm_input_override if llm_input_override else user_spoken_text
        llm_input_override = None  # reset
        if qa_context:
            llm_input = f"{qa_context}\n\nThe user's question: {user_spoken_text}"

        # Use streaming for faster response - speak each sentence as it arrives
        # Pipeline: LLM generates -> TTS synthesizes -> Audio plays (all overlapped)
        print("Annabeth: ", end="", flush=True)
        
        # Timing for LLM
        t_llm_start = time.time()
        t_first_sentence = [None]  # Use list to capture from closure
        t_first_audio = [None]
        
        sentence_queue = queue.Queue()
        audio_queue = queue.Queue()  # Queue of (sentence, audio_path) tuples
        full_response = []
        llm_done = threading.Event()
        tts_done = threading.Event()
        tts_cancel = threading.Event()  # Signal TTS pipeline to stop
        
        def on_sentence(sentence: str):
            """Called for each sentence from the LLM."""
            if t_first_sentence[0] is None:
                t_first_sentence[0] = time.time()
            # Clean LLM output before display (strip *actions*, ALL CAPS, etc.)
            cleaned = clean_text_for_tts(sentence)
            if cleaned:
                full_response.append(cleaned)
                sentence_queue.put(cleaned)
        
        def run_llm():
            """Run LLM in background thread."""
            try:
                llm_response_streaming(llm_input, on_sentence=on_sentence, speaker_name=speaker_name)
            finally:
                llm_done.set()
                sentence_queue.put(None)  # Signal end
        
        def run_tts_pipeline():
            """Run TTS in background - prefetch audio with parallel TTS submissions."""
            max_ahead = 3  # Submit up to 3 TTS jobs ahead of playback
            pending = []   # List of (sentence, future, output_path) in order
            end_signal = False
            
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts") as tts_pool:
                while not tts_cancel.is_set():
                    # Fill pending queue up to max_ahead
                    while len(pending) < max_ahead and not end_signal and not tts_cancel.is_set():
                        # Use a very short timeout for first sentence (latency critical)
                        wait = 0.01 if not pending else 0.03
                        try:
                            sentence = sentence_queue.get(timeout=wait)
                        except queue.Empty:
                            if llm_done.is_set() and sentence_queue.empty():
                                end_signal = True
                            break
                        
                        if sentence is None:
                            end_signal = True
                            break
                        
                        cleaned_sentence = clean_text_for_tts(sentence)
                        if not cleaned_sentence:
                            continue
                        
                        uid = uuid.uuid4().hex
                        filename = f"output_{uid}.wav"
                        output_path = audio_dir / filename
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        future = tts_pool.submit(sovits_gen, cleaned_sentence, output_path)
                        pending.append((sentence, future, output_path))
                    
                    # Drain completed futures in order
                    while pending:
                        sentence, future, output_path = pending[0]
                        if future.done():
                            pending.pop(0)
                            result = future.result()
                            if result:
                                audio_queue.put((sentence, output_path))
                            else:
                                audio_queue.put((sentence, None))
                        else:
                            # Wait briefly for the head-of-line future
                            try:
                                future.result(timeout=0.03)
                            except Exception:
                                pass
                            break
                    
                    # Exit when all done
                    if end_signal and not pending:
                        break
                    
                    time.sleep(0.01)
            
            tts_done.set()
            audio_queue.put(None)  # Signal end
        
        # Ensure previous turn's LLM thread finished saving history
        # (prevents race condition where interrupt leaves daemon thread running)
        if _prev_llm_thread is not None and _prev_llm_thread.is_alive():
            _prev_llm_thread.join(timeout=8)

        # Start LLM generation in background
        llm_thread = threading.Thread(target=run_llm, daemon=True)
        llm_thread.start()
        _prev_llm_thread = llm_thread

        # Gate Grillo beats during active conversation (Phase 6)
        try:
            from server.process.memory.reflection_loop import set_conversation_active as _set_conv_active
            _set_conv_active(True)
        except Exception:
            pass

        # Block self-improvement writes during active conversation (Phase 10)
        if _improvement_scheduler is not None:
            _improvement_scheduler.conversation_active = True
        
        # Start TTS pipeline in background
        tts_thread = threading.Thread(target=run_tts_pipeline, daemon=True)
        tts_thread.start()
        
        # Process audio as it arrives - plays while next is being generated
        first_sentence = True
        was_interrupted = False
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
            except queue.Empty:
                if tts_done.is_set():
                    break
                continue
            
            if item is None:
                break
            
            sentence, output_wav_path = item
            
            # Timing: first audio ready to play
            if t_first_audio[0] is None:
                t_first_audio[0] = time.time()
                # Print timing summary
                if t_first_sentence[0]:
                    print(f"\n[Timing] LLM first sentence: {t_first_sentence[0] - t_llm_start:.2f}s, TTS: {t_first_audio[0] - t_first_sentence[0]:.2f}s, Total to audio: {t_first_audio[0] - t_start:.2f}s")
            
            # Print the sentence
            if first_sentence:
                print(sentence, end="", flush=True)
                first_sentence = False
            else:
                print(f" {sentence}", end="", flush=True)
            
            if output_wav_path is None:
                print("\n(TTS generation failed)")
                continue
            
            # Notify avatar to start lip-sync (first sentence only)
            if first_sentence or not get_speaking_flag().is_set():
                avatar_speak_start(sentence)
            
            # Set speaking flag and clear interrupt before each sentence
            get_interrupt_flag().clear()
            get_speaking_flag().set()
            
            # Play audio with interruption support (Ctrl+Shift+X to interrupt)
            was_interrupted = not play_audio(
                output_wav_path, 
                output_device=output_device,
                interrupt_flag=get_interrupt_flag(),
            )
            
            # Clean up this audio file
            try:
                output_wav_path.unlink()
            except Exception:
                pass
            
            if was_interrupted:
                break
        
        print()  # Newline after response
        
        # Stop speaking flag and avatar
        get_speaking_flag().clear()
        avatar_speak_end()
        
        # Send final response to debug overlay
        full_resp_text = " ".join(full_response)
        avatar_debug_status("[MIC] Listening...", user_text=f"{speaker_tag} {user_spoken_text}" if user_spoken_text else "", response_text=full_resp_text)
        
        # --- Implicit feedback logging ---
        t_end = time.time()
        _fb_speaker = speaker_name or "Unknown"
        try:
            if was_interrupted:
                log_feedback("interruption", speaker=_fb_speaker)
            else:
                log_feedback("turn_complete", value=round(t_end - t_start, 2), speaker=_fb_speaker)
        except Exception:
            pass

        # Notify reflection loop that activity just happened
        try:
            if _reflection_loop is not None:
                _reflection_loop.notify_activity()
        except Exception:
            pass

        # Clear conversation-active gate (Phase 6)
        try:
            from server.process.memory.reflection_loop import set_conversation_active as _set_conv_active
            _set_conv_active(False)
        except Exception:
            pass

        # Re-enable self-improvement writes (Phase 10)
        if _improvement_scheduler is not None:
            _improvement_scheduler.conversation_active = False

        if was_interrupted:
            print("(Annabeth was interrupted - listening for your input...)")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print("Error during chat loop:", e)
        import traceback
        traceback.print_exc()
    finally:
        # Signal TTS pipeline to stop, wait briefly for it to finish
        try:
            tts_cancel.set()
            tts_thread.join(timeout=2.0)
        except (NameError, AttributeError):
            pass
        # clean up audio files (safe now - TTS thread has exited or timed out)
        try:
            for fp in audio_dir.glob("*.wav"):
                if fp.is_file():
                    fp.unlink()
        except Exception:
            pass

# =========================================================================
# POST-LOOP CLEANUP — runs after the main loop exits (shutdown or Ctrl+C)
# =========================================================================
print("[Shutdown] Cleaning up...")

# Stop the avatar WebSocket server
if avatar_loop is not None:
    avatar_loop.call_soon_threadsafe(avatar_loop.stop)

# Stop the emotion decay loop
try:
    from server.process.memory.emotion_state import stop_decay_loop
    stop_decay_loop()
except Exception:
    pass

# Stop the self-improvement scheduler
if _improvement_scheduler is not None:
    try:
        _improvement_scheduler.stop()
    except Exception:
        pass

# Stop the reflection (grillo) loop
if _reflection_loop is not None:
    try:
        _reflection_loop.stop()
    except Exception:
        pass

print("[Shutdown] Annabeth stopped.")