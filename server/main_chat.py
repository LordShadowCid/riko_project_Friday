"""
Main chat loop for Annabeth Desktop Companion.

Handles:
- Speech recognition (Whisper)
- LLM conversation
- TTS synthesis (GPT-SoVITS)
- Avatar integration
"""
from server.process.asr_func.asr_push_to_talk import record_and_transcribe, transcribe_file
from server.process.asr_func.asr_vad import (
    record_vad_and_transcribe, 
    BackgroundListener,
    get_interrupt_flag,
    get_speaking_flag,
)
from server.process.llm_funcs.llm_scr import llm_response, llm_response_streaming
from server.process.tts_func.sovits_ping import sovits_gen, play_audio
from server.process.read_aloud.text_capture import estimate_word_timings, capture_selected_text
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
from typing import Optional

from server.annabeth_config import load_config, repo_root, resolve_repo_path

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

def _start_avatar_server() -> None:
    """Start the avatar WebSocket server in a background thread"""
    global avatar_api, avatar_loop
    
    try:
        # Import from client directory
        client_dir = _shared_config.paths.client_dir
        if str(client_dir) not in sys.path:
            sys.path.insert(0, str(client_dir))
        
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
        port = _shared_config.server.avatar_port
        print(f"[Avatar] Server started at http://localhost:{port}")
        
    except ImportError as e:
        print(f"[Avatar] Could not start avatar server (missing aiohttp?): {e}")
        print("[Avatar] Install with: pip install aiohttp")
    except Exception as e:
        print(f"[Avatar] Could not start avatar server: {e}")


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


def is_listening_paused() -> bool:
    """Check if listening should be paused (dance/idle modes or silenced)"""
    # Use shared state - this is updated by avatar_server when S key is pressed
    return _state.is_listening_paused()


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
    w_device = whisper_cfg.get('device', 'cpu')
    w_device_idx = whisper_cfg.get('device_index', 0)
    device_str = f"{w_device}" + (f" (GPU {w_device_idx})" if w_device == 'cuda' else "")
    print(
        "Whisper config: "
        + f"model={whisper_cfg.get('model', 'base.en')} "
        + f"device={device_str} "
        + f"compute_type={whisper_cfg.get('compute_type', 'float32')}"
    )

    # TTS server reachability (best-effort)
    tts_url = _shared_config.server.tts_url
    try:
        import requests
        requests.get(tts_url + "/", timeout=1)
    except Exception:
        print(f"NOTE: GPT-SoVITS server not detected at {tts_url} (start it before chatting)")

    # Repo root recap
    print(f"Repo root: {_shared_config.paths.project_root}")

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
    """
    if not text:
        return text
    
    # Remove asterisk-wrapped actions like *laughs* or *OH BOY*
    # This handles both actions and emphasized words
    text = re.sub(r'\*[^*]+\*', '', text)
    
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

def process_read_aloud_queue(output_device, bg_listener) -> bool:
    """
    Process pending read-aloud sentences.
    
    Returns True if read-aloud was processed, False otherwise.
    """
    try:
        read_aloud = get_read_aloud_manager()
    except Exception:
        return False
    
    if not read_aloud.state.is_reading:
        return False
    
    print("\n[ReadAloud] 📖 Reading selected text...")
    audio_dir = _shared_config.paths.audio_dir
    sentence_idx = 0
    
    while read_aloud.state.is_reading:
        # Get next sentence
        sentence = read_aloud.get_next_sentence()
        
        if sentence is None:
            # Either paused or done
            if read_aloud.state.is_paused:
                print("\n[ReadAloud] ⏸️ Paused - ask questions or press R to resume")
                return True  # Let main loop handle Q&A
            break
        
        # Clean and speak the sentence
        cleaned = clean_text_for_tts(sentence)
        if not cleaned:
            # Empty after cleanup - skip to next
            read_aloud.state.advance_index()
            continue
        
        print(f"  📖 {cleaned}")
        
        # Send highlight data to browser clients (approx timings)
        try:
            duration_estimate = max(len(cleaned.split()) * 0.35, 1.0)  # rough estimate
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
        
        # Generate TTS
        uid = uuid.uuid4().hex
        output_path = audio_dir / f"read_{uid}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        gen_path = sovits_gen(cleaned, output_path)
        if not gen_path:
            print("  (TTS failed for this sentence)")
            # Skip to next sentence
            read_aloud.state.advance_index()
            continue
        
        # Notify avatar
        avatar_speak_start(cleaned)
        
        # Start background listener for interruption
        get_interrupt_flag().clear()
        get_speaking_flag().set()
        bg_listener.start()
        
        # Play audio
        was_interrupted = not play_audio(
            output_path,
            output_device=output_device,
            interrupt_flag=get_interrupt_flag(),
        )
        
        # Stop listener
        bg_listener.stop()
        get_speaking_flag().clear()
        
        # Cleanup
        try:
            output_path.unlink()
        except Exception:
            pass
        
        if was_interrupted:
            # User interrupted - request pause
            read_aloud.request_pause()
            print("\n[ReadAloud] ⏸️ Interrupted - finishing sentence...")
            # Complete the pause
            read_aloud.state.complete_pause()
            print("[ReadAloud] ⏸️ Paused - ask questions or press R to resume")
            avatar_speak_end()
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
        print("[ReadAloud] ✅ Finished reading")
    
    return True


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
# Interrupt detection settings - higher values = less sensitive (fewer false interrupts)
interrupt_aggressiveness = vad_cfg.get('interrupt_aggressiveness', 3)  # Max filtering for interrupts
interrupt_speech_frames = vad_cfg.get('interrupt_speech_frames', 15)  # ~450ms of sustained speech needed
interrupt_min_energy = vad_cfg.get('interrupt_min_energy', 500)  # Minimum volume to trigger interrupt

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

if use_vad:
    print("\n🎤 HANDS-FREE MODE enabled - just start speaking!")
    print("   (Annabeth will listen and respond automatically)")
    print("   (You can interrupt her while she's speaking)\n")
else:
    print("\n🔘 PUSH-TO-TALK MODE - press ENTER to record\n")

# Background listener for interruption detection (uses stricter settings to avoid false triggers)
bg_listener = BackgroundListener(
    input_device=input_device,
    sample_rate=16000,
    vad_aggressiveness=interrupt_aggressiveness,
    speech_frames_threshold=interrupt_speech_frames,
    min_audio_energy=interrupt_min_energy,
)

while True:
    # =========================================================================
    # CHECK FOR READ-ALOUD QUEUE
    # =========================================================================
    # Process any pending read-aloud sentences before checking for new input
    try:
        if process_read_aloud_queue(output_device, bg_listener):
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
        time.sleep(0.5)  # Check again in 500ms
        continue
    
    # Clear the "waiting" message when resuming
    print("                                                  ", end='\r', flush=True)
    
    audio_dir = _shared_config.paths.audio_dir
    conversation_recording = audio_dir / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    output_wav_path = None

    try:
        # Clear any previous interrupt flags
        get_interrupt_flag().clear()
        
        speaker_name = None
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
            continue

        # =====================================================================
        # CHECK FOR READ-ALOUD INTENT
        # =====================================================================
        # Detect if user wants to read selected text
        read_intent_phrases = [
            "read that", "read this", "read the selected", "read selected",
            "read it", "read aloud", "read to me", "read what i selected",
            "read the text", "can you read", "please read"
        ]
        user_lower = user_spoken_text.lower().strip()
        is_read_intent = any(phrase in user_lower for phrase in read_intent_phrases)
        
        if is_read_intent:
            print("[ReadAloud] Detected read intent - capturing text...")
            captured_text = capture_selected_text(restore_clipboard=True)
            
            if captured_text and len(captured_text) > 5:
                print(f"[ReadAloud] Captured {len(captured_text)} characters")
                read_aloud = get_read_aloud_manager()
                read_aloud.state.start_reading(captured_text)
                # Go back to top of loop to process the queue
                continue
            else:
                print("[ReadAloud] No text captured - make sure text is selected!")
                # Let LLM respond about the failure

        # =====================================================================
        # CHECK FOR READ-ALOUD Q&A CONTEXT
        # =====================================================================
        # If read-aloud is paused, add context about what was being read
        qa_context = ""
        try:
            read_aloud = get_read_aloud_manager()
            if read_aloud.state.is_paused:
                qa_context = read_aloud.get_qa_context()
                if qa_context:
                    print("[ReadAloud] 💬 Answering question about the text...")
        except Exception:
            pass
        
        # Combine Q&A context with user input if available
        llm_input = user_spoken_text
        if qa_context:
            llm_input = f"{qa_context}\n\nThe user's question: {user_spoken_text}"

        # Use streaming for faster response - speak each sentence as it arrives
        # Pipeline: LLM generates -> TTS synthesizes -> Audio plays (all overlapped)
        print("Annabeth: ", end="", flush=True)
        
        sentence_queue = queue.Queue()
        audio_queue = queue.Queue()  # Queue of (sentence, audio_path) tuples
        full_response = []
        llm_done = threading.Event()
        tts_done = threading.Event()
        
        def on_sentence(sentence: str):
            """Called for each sentence from the LLM."""
            full_response.append(sentence)
            sentence_queue.put(sentence)
        
        def run_llm():
            """Run LLM in background thread."""
            try:
                llm_response_streaming(llm_input, on_sentence=on_sentence, speaker_name=speaker_name)
            finally:
                llm_done.set()
                sentence_queue.put(None)  # Signal end
        
        def run_tts_pipeline():
            """Run TTS in background - pre-generate audio while previous plays."""
            while True:
                try:
                    sentence = sentence_queue.get(timeout=0.1)
                except queue.Empty:
                    if llm_done.is_set():
                        break
                    continue
                
                if sentence is None:
                    break
                
                # Clean up the text for natural TTS (remove *actions*, ALL CAPS, etc.)
                cleaned_sentence = clean_text_for_tts(sentence)
                if not cleaned_sentence:
                    continue  # Skip empty sentences after cleanup
                
                # Generate TTS for this sentence
                uid = uuid.uuid4().hex
                filename = f"output_{uid}.wav"
                output_path = audio_dir / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                gen_aud_path = sovits_gen(cleaned_sentence, output_path)
                if gen_aud_path:
                    audio_queue.put((sentence, output_path))  # Keep original for display
                else:
                    audio_queue.put((sentence, None))  # TTS failed
            
            tts_done.set()
            audio_queue.put(None)  # Signal end
        
        # Start LLM generation in background
        llm_thread = threading.Thread(target=run_llm, daemon=True)
        llm_thread.start()
        
        # Start TTS pipeline in background
        tts_thread = threading.Thread(target=run_tts_pipeline, daemon=True)
        tts_thread.start()
        
        # Process audio as it arrives - plays while next is being generated
        first_sentence = True
        was_interrupted = False
        
        while True:
            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if tts_done.is_set():
                    break
                continue
            
            if item is None:
                break
            
            sentence, output_wav_path = item
            
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
            
            # Start background listener for interruption
            get_interrupt_flag().clear()
            get_speaking_flag().set()
            bg_listener.start()
            
            # Play audio with interruption support
            was_interrupted = not play_audio(
                output_wav_path, 
                output_device=output_device,
                interrupt_flag=get_interrupt_flag(),
            )
            
            # Stop background listener
            bg_listener.stop()
            
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
        # clean up audio files
        try:
            for fp in audio_dir.glob("*.wav"):
                if fp.is_file():
                    fp.unlink()
        except Exception:
            pass