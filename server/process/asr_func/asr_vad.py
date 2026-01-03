"""
Voice Activity Detection (VAD) based ASR module.
Enables hands-free voice activation - no button press needed.
Uses webrtcvad to detect when user starts/stops speaking.
"""
import os
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from typing import Tuple, Optional


# Shared state for interruption
_interrupt_flag = threading.Event()
_is_speaking = threading.Event()  # True when TTS is playing


def get_interrupt_flag():
    """Get the interrupt flag for checking if user started speaking."""
    return _interrupt_flag


def get_speaking_flag():
    """Get the flag that indicates if TTS is currently playing."""
    return _is_speaking


def _resolve_device(device, kind='input'):
    """Resolve a sounddevice input/output device selector."""
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
                if kind == 'output' and d.get('max_output_channels', 0) > 0:
                    return idx
                elif kind == 'input' and d.get('max_input_channels', 0) > 0:
                    return idx
    return None


class VADRecorder:
    """
    Hands-free voice recorder using Voice Activity Detection.
    Automatically starts recording when speech is detected,
    and stops after a period of silence.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_threshold_sec: float = 1.0,
        pre_speech_padding_sec: float = 0.3,
        min_speech_duration_sec: float = 0.5,
        input_device=None,
    ):
        """
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000 for webrtcvad)
            frame_duration_ms: Frame size in ms (must be 10, 20, or 30 for webrtcvad)
            vad_aggressiveness: 0-3, higher = more aggressive filtering of non-speech
            silence_threshold_sec: How long to wait after speech stops before stopping recording
            pre_speech_padding_sec: How much audio to keep before speech started
            min_speech_duration_sec: Minimum speech duration to consider valid
            input_device: Sounddevice input device (name substring, index, or None for default)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_threshold_sec = silence_threshold_sec
        self.pre_speech_padding_sec = pre_speech_padding_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.input_device = input_device
        
        # Frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Pre-speech buffer size in frames
        self.pre_speech_frames = int(pre_speech_padding_sec * 1000 / frame_duration_ms)
        
        # Silence threshold in frames
        self.silence_frames_threshold = int(silence_threshold_sec * 1000 / frame_duration_ms)
        
        # Min speech frames
        self.min_speech_frames = int(min_speech_duration_sec * 1000 / frame_duration_ms)
        
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
    def record_until_silence(self, cancel_event: threading.Event = None) -> np.ndarray:
        """
        Listen for speech, record it, and return when silence is detected.
        
        Args:
            cancel_event: Optional event to signal cancellation
            
        Returns:
            Numpy array of audio data (float32, mono)
        """
        device = _resolve_device(self.input_device, kind='input')
        
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[VAD] Audio status: {status}")
            audio_queue.put(indata.copy())
        
        # Pre-speech ring buffer
        pre_speech_buffer = []
        
        # Collected speech frames
        speech_frames = []
        
        # State
        is_recording_speech = False
        silence_frame_count = 0
        speech_frame_count = 0
        
        print("🎤 Listening... (speak when ready)")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_size,
            device=device,
            callback=audio_callback,
        ):
            while True:
                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    print("[VAD] Cancelled")
                    return None
                
                try:
                    frame = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Convert to bytes for webrtcvad
                frame_bytes = frame.tobytes()
                
                # Check if frame contains speech
                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                except Exception:
                    is_speech = False
                
                if not is_recording_speech:
                    # Not yet recording - maintain pre-speech buffer
                    pre_speech_buffer.append(frame)
                    if len(pre_speech_buffer) > self.pre_speech_frames:
                        pre_speech_buffer.pop(0)
                    
                    if is_speech:
                        # Speech detected! Start recording
                        print("🔴 Speech detected - recording...")
                        is_recording_speech = True
                        speech_frames = list(pre_speech_buffer)  # Include pre-speech audio
                        speech_frames.append(frame)
                        speech_frame_count = 1
                        silence_frame_count = 0
                        
                        # If TTS is playing, signal interruption
                        if _is_speaking.is_set():
                            _interrupt_flag.set()
                            print("⚡ Interrupting playback...")
                else:
                    # Currently recording
                    speech_frames.append(frame)
                    
                    if is_speech:
                        speech_frame_count += 1
                        silence_frame_count = 0
                    else:
                        silence_frame_count += 1
                        
                        if silence_frame_count >= self.silence_frames_threshold:
                            # Enough silence - stop recording
                            print("⏹️ Silence detected - processing...")
                            
                            if speech_frame_count >= self.min_speech_frames:
                                # Valid speech detected
                                break
                            else:
                                # Too short - probably noise, reset
                                print("(Too short, ignoring...)")
                                is_recording_speech = False
                                speech_frames = []
                                pre_speech_buffer = []
        
        if not speech_frames:
            return None
        
        # Combine all frames
        audio_data = np.concatenate(speech_frames, axis=0)
        
        # Convert int16 to float32
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float.flatten()
    
    def record_and_save(self, output_path: str, cancel_event: threading.Event = None) -> str:
        """Record speech and save to a WAV file."""
        audio = self.record_until_silence(cancel_event)
        
        if audio is None or len(audio) == 0:
            return None
        
        sf.write(output_path, audio, self.sample_rate)
        return output_path


def record_vad_and_transcribe(
    model,
    output_file: str = "recording.wav",
    input_device=None,
    sample_rate: int = 16000,
    vad_aggressiveness: int = 2,
    silence_threshold_sec: float = 1.0,
    cancel_event: threading.Event = None,
    identify_speaker: bool = True,
    speaker_threshold: float = 0.75,
) -> Tuple[str, Optional[str]]:
    """
    Record audio using VAD (hands-free), identify speaker, and transcribe it.
    
    Args:
        model: Whisper model instance
        output_file: Path to save the recorded audio
        input_device: Audio input device
        sample_rate: Sample rate for recording
        vad_aggressiveness: VAD sensitivity (0-3)
        silence_threshold_sec: Seconds of silence before stopping
        cancel_event: Optional event for cancellation
        identify_speaker: Whether to identify who is speaking
        speaker_threshold: Similarity threshold for speaker identification
        
    Returns:
        Tuple of (transcription, speaker_name or None)
    """
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    recorder = VADRecorder(
        sample_rate=sample_rate,
        vad_aggressiveness=vad_aggressiveness,
        silence_threshold_sec=silence_threshold_sec,
        input_device=input_device,
    )
    
    # Record audio (returns raw audio data)
    audio_data = recorder.record_until_silence(cancel_event)
    
    if audio_data is None or len(audio_data) == 0:
        return "", None
    
    # Save audio file for transcription
    sf.write(output_file, audio_data, sample_rate)
    
    # Identify speaker (do this while preparing transcription)
    speaker_name = None
    if identify_speaker:
        try:
            from server.process.asr_func.speaker_id import identify_speaker as speaker_id_func
            speaker_name, confidence = speaker_id_func(audio_data, sample_rate, speaker_threshold)
            if speaker_name:
                print(f"👤 Speaker: {speaker_name} ({confidence:.0%} confidence)")
            else:
                print(f"👤 Speaker: Unknown (best match: {confidence:.0%})")
        except Exception as e:
            print(f"[Speaker ID] Error: {e}")
    
    print("🎯 Transcribing...")
    
    # Use Silero VAD filter to skip silence and improve transcription speed
    segments, _ = model.transcribe(
        output_file,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    transcription = " ".join([segment.text for segment in segments]).strip()
    
    print(f"Transcription: {transcription}")
    return transcription, speaker_name


# Background listener for interruption during TTS
class BackgroundListener:
    """
    Listens in background during TTS playback to detect user interruption.
    Uses high aggressiveness, volume threshold, and requires sustained speech to avoid false triggers.
    """
    
    def __init__(self, input_device=None, sample_rate=16000, vad_aggressiveness=3, 
                 speech_frames_threshold=15, min_audio_energy=500):
        """
        Args:
            input_device: Audio input device
            sample_rate: Sample rate (16000 recommended)
            vad_aggressiveness: 0-3, higher = less sensitive (3 recommended for interruption)
            speech_frames_threshold: Number of consecutive speech frames required to interrupt
                                     At 30ms per frame, 15 frames = ~450ms of sustained speech
            min_audio_energy: Minimum RMS audio energy to consider as potential speech
                             This filters out quiet background noise that VAD might mistake for speech
                             Typical values: 300-1000 (higher = requires louder speech)
        """
        self.input_device = input_device
        self.sample_rate = sample_rate
        # Use maximum aggressiveness (3) by default to filter out background noise
        self.vad = webrtcvad.Vad(min(vad_aggressiveness, 3))
        self.frame_size = int(sample_rate * 30 / 1000)  # 30ms frames
        self.speech_frames_threshold = speech_frames_threshold
        self.min_audio_energy = min_audio_energy
        self._stop_event = threading.Event()
        self._thread = None
        
    def start(self):
        """Start listening in background."""
        _interrupt_flag.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop background listening."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def _listen_loop(self):
        """Background loop to detect voice."""
        device = _resolve_device(self.input_device, kind='input')
        
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())
        
        consecutive_speech_frames = 0
        # Require sustained, loud speech to trigger interrupt
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_size,
                device=device,
                callback=audio_callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        frame = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    # Calculate audio energy (RMS) to filter out quiet background noise
                    frame_int = frame.astype(np.int32)
                    rms_energy = np.sqrt(np.mean(frame_int ** 2))
                    
                    # Only check VAD if audio is loud enough
                    if rms_energy < self.min_audio_energy:
                        consecutive_speech_frames = 0
                        continue
                    
                    frame_bytes = frame.tobytes()
                    
                    try:
                        is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    except Exception:
                        is_speech = False
                    
                    if is_speech:
                        consecutive_speech_frames += 1
                        if consecutive_speech_frames >= self.speech_frames_threshold:
                            _interrupt_flag.set()
                            return  # Exit after triggering interrupt
                    else:
                        consecutive_speech_frames = 0
                        
        except Exception as e:
            print(f"[BackgroundListener] Error: {e}")
