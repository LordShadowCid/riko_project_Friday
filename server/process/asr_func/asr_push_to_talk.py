import os
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


def _resolve_device(device):
    """Resolve a sounddevice input/output device selector.

    - None: use default
    - int: treated as device index
    - str: case-insensitive substring match against device names
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
                return idx
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
    
    device = _resolve_device(input_device)

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
    
    print("⏹️  Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)
    
    print("🎯 Transcribing...")
    
    # Transcribe
    segments, _ = model.transcribe(output_file)
    transcription = " ".join([segment.text for segment in segments])
    
    print(f"Transcription: {transcription}")
    return transcription.strip()


# Example usage
if __name__ == "__main__":
    model = WhisperModel("base.en", device="cpu", compute_type="float32")
    result = record_and_transcribe(model)
    print(f"Got: '{result}'")
    