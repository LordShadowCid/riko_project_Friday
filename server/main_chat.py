from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import os
import time
### transcribe audio 
import uuid
import soundfile as sf


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')
whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")

while True:
    conversation_recording = Path("audio") / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    output_wav_path = None

    try:
        user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)
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

        play_audio(output_wav_path)

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