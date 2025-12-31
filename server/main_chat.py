from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import os
import time
### transcribe audio 
import uuid
import soundfile as sf

from riko_config import load_config


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')

char_config = load_config()

whisper_cfg = char_config.get('whisper', {}) or {}
cuda_visible = whisper_cfg.get('cuda_visible_devices')
if cuda_visible:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)

from faster_whisper import WhisperModel

whisper_model_name = whisper_cfg.get('model', 'base.en')
whisper_device = whisper_cfg.get('device', 'cpu')
whisper_compute_type = whisper_cfg.get('compute_type', 'float32')

print(f"Whisper: model={whisper_model_name} device={whisper_device} compute_type={whisper_compute_type}")
whisper_model = WhisperModel(whisper_model_name, device=whisper_device, compute_type=whisper_compute_type)

audio_cfg = char_config.get('audio', {}) or {}
input_device = audio_cfg.get('input_device')
output_device = audio_cfg.get('output_device')

while True:
    conversation_recording = Path("audio") / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    output_wav_path = None

    try:
        user_spoken_text = record_and_transcribe(
            whisper_model,
            conversation_recording,
            input_device=input_device,
        )
        if not user_spoken_text:
            print("No transcription captured; try again.")
            continue

        llm_output = llm_response(user_spoken_text)
        if not llm_output:
            print("LLM returned empty output; try again.")
            continue

        # Generate a unique filename
        uid = uuid.uuid4().hex
        filename = f"output_{uid}.wav"
        output_wav_path = Path("audio") / filename
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        gen_aud_path = sovits_gen(llm_output, output_wav_path)
        if gen_aud_path is None:
            print("TTS generation failed (is GPT-SoVITS running on 127.0.0.1:9880?)")
            continue

        play_audio(output_wav_path, output_device=output_device)

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