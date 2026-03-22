import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd
from pathlib import Path

from server.annabeth_config import load_config, resolve_repo_path
from server.utils import resolve_device as _resolve_device

char_config = load_config()

# HTTP session for connection pooling (reuses TCP connections for faster requests)
_tts_session: requests.Session = None

def _get_tts_session() -> requests.Session:
    """Get or create a persistent HTTP session for TTS requests."""
    global _tts_session
    if _tts_session is None:
        _tts_session = requests.Session()
        # Set default timeout
        _tts_session.headers.update({'Content-Type': 'application/json'})
    return _tts_session


def play_audio(path, output_device=None, interrupt_flag=None):
    """
    Play audio file with optional interruption support.
    
    Args:
        path: Path to WAV file
        output_device: Output device (name, index, or None)
        interrupt_flag: Optional threading.Event - if set, playback stops
        
    Returns:
        True if played fully, False if interrupted
    """
    import numpy as np
    
    data, samplerate = sf.read(path)
    device = _resolve_device(output_device, kind='output')
    
    # Query device capabilities to handle channel mismatch
    try:
        if device is not None:
            dev_info = sd.query_devices(device, 'output')
            max_out_ch = dev_info.get('max_output_channels', 2)
        else:
            dev_info = sd.query_devices(kind='output')
            max_out_ch = dev_info.get('max_output_channels', 2)
        
        # Convert stereo to mono if device only supports mono
        if data.ndim == 2 and data.shape[1] == 2 and max_out_ch == 1:
            data = data.mean(axis=1)  # stereo -> mono
        # Convert mono to stereo if device requires stereo
        elif data.ndim == 1 and max_out_ch >= 2:
            data = np.column_stack([data, data])  # mono -> stereo
    except Exception as e:
        print(f"Warning: could not query device capabilities: {e}")
    
    if interrupt_flag is None:
        # Simple playback without interruption support
        sd.play(data, samplerate, device=device)
        sd.wait()
        return True
    else:
        # Interruptible playback - check flag periodically
        sd.play(data, samplerate, device=device)
        
        # Calculate total duration
        duration = len(data) / samplerate
        check_interval = 0.05  # Check every 50ms
        elapsed = 0
        
        while elapsed < duration:
            if interrupt_flag.is_set():
                sd.stop()
                print("[STOP] Playback interrupted!")
                return False
            time.sleep(check_interval)
            elapsed += check_interval
        
        sd.wait()
        return True

def sovits_gen(in_text, output_wav_pth = "output.wav"):
    url = "http://127.0.0.1:9880/tts"

    ref_audio_path = char_config['sovits_ping_config']['ref_audio_path']
    # If the user provided a Linux/container path (e.g. /data/ref/main_sample.wav),
    # do NOT rewrite it to a Windows absolute path.
    if isinstance(ref_audio_path, str) and ref_audio_path.strip().startswith("/"):
        ref_audio_path = ref_audio_path.strip()
    else:
        ref_audio_path = resolve_repo_path(ref_audio_path)

    payload = {
        "text": in_text,
        "text_lang": char_config['sovits_ping_config']['text_lang'],
        "ref_audio_path": ref_audio_path,
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": char_config['sovits_ping_config']['prompt_lang']
    }

    try:
        # Use session pooling for faster connection reuse
        session = _get_tts_session()
        response = session.post(url, json=payload, timeout=30)
        response.raise_for_status()  # throws if not 200

        # Save the response audio if it's binary
        with open(output_wav_pth, "wb") as f:
            f.write(response.content)

        return output_wav_pth

    except Exception as e:
        print(f"Error in sovits_gen: {e}")
        print("[TTS] Trying pyttsx3 fallback...")
        return _fallback_tts(in_text, output_wav_pth)


def _fallback_tts(text, output_wav_pth):
    """Fallback TTS using pyttsx3 (Windows SAPI5) when GPT-SoVITS is unavailable."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.save_to_file(text, str(output_wav_pth))
        engine.runAndWait()
        if Path(output_wav_pth).exists() and Path(output_wav_pth).stat().st_size > 0:
            return output_wav_pth
    except Exception as e2:
        print(f"[TTS] Fallback also failed: {e2}")
    return None



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)


