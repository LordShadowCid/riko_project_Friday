import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd

from riko_config import load_config, resolve_repo_path

char_config = load_config()


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


def play_audio(path, output_device=None):
    data, samplerate = sf.read(path)
    device = _resolve_device(output_device)
    sd.play(data, samplerate, device=device)
    sd.wait()  # Wait until playback is finished

def sovits_gen(in_text, output_wav_pth = "output.wav"):
    url = "http://127.0.0.1:9880/tts"

    ref_audio_path = char_config['sovits_ping_config']['ref_audio_path']
    ref_audio_path = resolve_repo_path(ref_audio_path)

    payload = {
        "text": in_text,
        "text_lang": char_config['sovits_ping_config']['text_lang'],
        "ref_audio_path": ref_audio_path,
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": char_config['sovits_ping_config']['prompt_lang']
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # throws if not 200

        print(response)

        # Save the response audio if it's binary
        with open(output_wav_pth, "wb") as f:
            f.write(response.content)
        # print("Audio saved as output.wav")

        return output_wav_pth

    except Exception as e:
        print("Error in sovits_gen:", e)
        return None



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)


