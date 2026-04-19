import os
import sounddevice as sd
import soundfile as sf
from server.utils import configure_windows_cuda_runtime, resolve_device

configure_windows_cuda_runtime()

from faster_whisper import WhisperModel

def record_and_transcribe(model, output_file="recording.wav", samplerate=44100, input_device=None):
    """
    Simple push-to-talk recorder: record -> save -> transcribe -> return text
    """
    
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("Press ENTER to start recording...")
    input()
    
    print("[REC] Recording... Press ENTER to stop")
    
    device = resolve_device(input_device, kind='input')

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
    
    print("[STOP] Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)

    if (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0:
        raise RuntimeError(f"Recorded file missing or empty: {output_file}")
    
    print("[ASR] Transcribing...")

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
    