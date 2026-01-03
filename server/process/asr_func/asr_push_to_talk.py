import os
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


def _resolve_device(device, kind='input'):
    """Resolve a sounddevice input/output device selector.

    - None: use default
    - int: treated as device index
    - str: case-insensitive substring match against device names
    - kind: 'input' or 'output' - filters to only match devices with channels for that direction
    """
    if device is None or device == "":
        return None
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        devices = sd.query_devices()
        needle = device.lower().strip()
        for idx, d in enumerate(devices):
            name = str(d.get("name", "")).lower()
            if needle and needle in name:
                # Filter by device capability
                if kind == 'output' and d.get('max_output_channels', 0) > 0:
                    return idx
                elif kind == 'input' and d.get('max_input_channels', 0) > 0:
                    return idx
                # If kind doesn't match, keep searching
    return None

def record_and_transcribe(model, output_file="recording.wav", samplerate=44100, input_device=None):
    """
    Simple push-to-talk recorder: record -> save -> transcribe -> return text
    """
    
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("Press ENTER to start recording...")
    input()
    
    print("🔴 Recording... Press ENTER to stop")
    
    device = _resolve_device(input_device, kind='input')

    # Record audio directly
    recording = sd.rec(
        int(60 * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=device,
    )
    input()  # Wait for stop
    sd.stop()
    sd.wait()
    
    print("⏹️  Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)

    if (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0:
        raise RuntimeError(f"Recorded file missing or empty: {output_file}")
    
    print("🎯 Transcribing...")

    transcription = transcribe_file(model, output_file)
    
    print(f"Transcription: {transcription}")
    return transcription.strip()


def transcribe_file(model, audio_path: str) -> str:
    if (not os.path.exists(audio_path)) or os.path.getsize(audio_path) == 0:
        raise RuntimeError(f"Audio file not found or empty: {audio_path}")
    segments, _ = model.transcribe(audio_path)
    return " ".join([segment.text for segment in segments]).strip()


# Example usage
if __name__ == "__main__":
    model = WhisperModel("base.en", device="cpu", compute_type="float32")
    result = record_and_transcribe(model)
    print(f"Got: '{result}'")
    