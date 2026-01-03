import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd

from server.annabeth_config import load_config, resolve_repo_path

char_config = load_config()


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


def play_audio(path, output_device=None):
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
            import numpy as np
            data = np.column_stack([data, data])  # mono -> stereo
    except Exception as e:
        print(f"Warning: could not query device capabilities: {e}")
    
    sd.play(data, samplerate, device=device)
    sd.wait()  # Wait until playback is finished

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


