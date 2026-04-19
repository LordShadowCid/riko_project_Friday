"""
System Audio Analyzer for Desktop Companion.
Captures system audio via WASAPI loopback and performs beat/frequency analysis.
Sends analysis data to the avatar via WebSocket.
Uses PyAudioWPatch for Windows WASAPI loopback support.
"""
import numpy as np
import threading
import time
import json
from collections import deque

try:
    import webrtcvad as _webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("[Audio] PyAudioWPatch not installed. Run: pip install PyAudioWPatch")


# ============================================================================
# VADProcessor — WebRTC-based speech/non-speech classifier (Phase 4)
# ============================================================================


class VADProcessor:
    """
    Lightweight wrapper around webrtcvad for chunk-level speech detection.
    Converts float32 audio to int16 PCM, runs VAD on 30ms frames, and
    returns True when the fraction of frames with speech exceeds a threshold.

    Falls back gracefully to energy-based detection when webrtcvad is absent.
    """

    # webrtcvad requires exactly 8000, 16000, 32000, or 48000 Hz.
    _VALID_RATES = {8000, 16000, 32000, 48000}
    # Valid frame durations in ms.
    _FRAME_MS = 30

    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2, speech_ratio: float = 0.2):
        """
        Args:
            sample_rate:   Audio sample rate.  Must be one of 8k/16k/32k/48k.
            aggressiveness: 0-3.  Higher = more aggressive non-speech filtering.
            speech_ratio:  Fraction of frames that must be speech to return True.
        """
        # Read from settings registry if available
        try:
            from server.settings_registry import registry
            aggressiveness = int(registry.get("VAD_AGGRESSIVENESS"))
            speech_ratio = float(registry.get("VAD_SPEECH_RATIO"))
        except Exception:
            pass

        self.sample_rate = sample_rate if sample_rate in self._VALID_RATES else 16000
        self.aggressiveness = max(0, min(3, aggressiveness))
        self.speech_ratio = max(0.0, min(1.0, speech_ratio))
        self._vad = None
        if HAS_WEBRTCVAD:
            self._vad = _webrtcvad.Vad(self.aggressiveness)

    def detect_speech(self, audio_float32: "np.ndarray") -> bool:
        """
        Return True if *audio_float32* (mono, float32) contains speech.

        Falls back to RMS energy check if webrtcvad is unavailable or the
        chunk cannot be resampled to a supported rate.
        """
        if self._vad is None or not HAS_WEBRTCVAD:
            # Fallback: energy threshold
            return float(np.sqrt(np.mean(audio_float32 ** 2))) > 0.01

        # Convert float32 → int16 PCM
        pcm = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        frame_samples = int(self.sample_rate * self._FRAME_MS / 1000)
        frame_bytes = frame_samples * 2  # 2 bytes per int16 sample

        if len(pcm) < frame_bytes:
            return False  # Too short to classify

        speech_frames = 0
        total_frames = 0
        for start in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[start: start + frame_bytes]
            total_frames += 1
            try:
                if self._vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
            except Exception:
                pass

        if total_frames == 0:
            return False
        return (speech_frames / total_frames) >= self.speech_ratio


# ============================================================================
class SystemAudioAnalyzer:
    """Captures and analyzes system audio in real-time using WASAPI loopback."""
    
    def __init__(self, chunk_size=1024, preferred_device_name: str = ""):
        self.chunk_size = chunk_size
        self.preferred_device_name = (preferred_device_name or "").strip()
        self.running = False
        self.thread = None
        
        # Analysis results (updated in real-time)
        self.bass_energy = 0.0
        self.mid_energy = 0.0
        self.high_energy = 0.0
        self.overall_energy = 0.0
        self.is_beat = False
        self.beat_intensity = 0.0
        
        # Beat detection state
        self.energy_history = deque(maxlen=43)  # ~1 second of history
        self.last_beat_time = 0
        self.beat_cooldown = 0.12  # Minimum time between beats (faster detection)
        self.beat_threshold = 1.3  # Energy must be this much above average (more sensitive)
        
        # Smoothing
        self.smooth_bass = 0.0
        self.smooth_mid = 0.0
        self.smooth_high = 0.0
        self.smooth_factor = 0.3
        
        # Feature #24: Configurable noise floor and app filter
        self.noise_floor = 0.02
        self.filter_apps = []  # List of app names to filter (empty = all)
        
        # PyAudio
        self.p = None
        self.stream = None
        self.sample_rate = 48000
        self.channels = 2
        self.loopback_device = None
        
        # Callbacks for sending data
        self.on_analysis_update = None

        # WebRTC VAD pre-filter (Phase 4)
        # Enabled by default when webrtcvad-wheels is installed.
        self._vad_processor: VADProcessor | None = None
        self.vad_enabled: bool = HAS_WEBRTCVAD
        if HAS_WEBRTCVAD:
            self._vad_processor = VADProcessor(sample_rate=16000)
    
    def _find_loopback_device(self):
        """Find the WASAPI loopback device matching the default output."""
        if not HAS_PYAUDIO:
            return None
        
        try:
            self.p = pyaudio.PyAudio()
            
            # Collect all loopback devices first
            loopback_devices = []
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                if dev.get('isLoopbackDevice', False):
                    loopback_devices.append(dev)
                    print(f"[Audio] Found loopback: {dev['name']}")
            
            if not loopback_devices:
                print("[Audio] No loopback devices found")
                return None

            # 1. Preferred device from config (partial, case-insensitive match)
            if self.preferred_device_name:
                hint = self.preferred_device_name.lower()
                for dev in loopback_devices:
                    if hint in dev['name'].lower():
                        print(f"[Audio] Selected loopback (config preferred): {dev['name']}")
                        return dev
                print(f"[Audio] Preferred device '{self.preferred_device_name}' not found, trying default")

            # 2. Match the Windows default output device
            default_name = ''
            try:
                default_output = self.p.get_default_output_device_info()
                default_name = default_output.get('name', '').lower()
                print(f"[Audio] Default output device: {default_output.get('name')}")
            except Exception as e:
                print(f"[Audio] Could not get default output: {e}")

            if default_name:
                for dev in loopback_devices:
                    dev_name = dev['name'].lower()
                    # Loopback names often have " [Loopback]" suffix; strip it for comparison
                    dev_base = dev_name.replace(' [loopback]', '')
                    if default_name in dev_name or dev_base in default_name:
                        print(f"[Audio] Selected loopback (matches default output): {dev['name']}")
                        return dev

            # 3. Fallback: prefer non-conference/headset devices (likely music speakers)
            #    BUT if the preferred device was a conference device, don't exclude them
            _non_conf_keywords = ('webcam', 'microphone')
            if not self.preferred_device_name:
                # Only exclude conference/headset when no preferred device was specified
                _non_conf_keywords = ('conference', 'speakerphone', 'webcam', 'headset', 'headphone', 'microphone')
            non_conf = [d for d in loopback_devices
                        if not any(kw in d['name'].lower() for kw in _non_conf_keywords)]
            if non_conf:
                print(f"[Audio] Using first non-conference loopback: {non_conf[0]['name']}")
                return non_conf[0]

            # 4. Last resort: first loopback device
            print(f"[Audio] Using first loopback (fallback): {loopback_devices[0]['name']}")
            return loopback_devices[0]
            
        except Exception as e:
            print(f"[Audio] Error finding loopback: {e}")
            return None
    
    def start(self):
        """Start capturing and analyzing audio."""
        if self.running:
            return True
        
        if not HAS_PYAUDIO:
            print("[Audio] PyAudioWPatch not available")
            return False
        
        self.loopback_device = self._find_loopback_device()
        if not self.loopback_device:
            return False
        
        self.sample_rate = int(self.loopback_device['defaultSampleRate'])
        self.channels = int(self.loopback_device['maxInputChannels'])
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.loopback_device['index'],
                frames_per_buffer=self.chunk_size
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print(f"[Audio] Started capturing from {self.loopback_device['name']}")
            return True
            
        except Exception as e:
            print(f"[Audio] Failed to open stream: {e}")
            return False
    
    def stop(self):
        """Stop capturing audio."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
        
        print("[Audio] Stopped")
    
    def _capture_loop(self):
        """Main capture and analysis loop."""
        while self.running:
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                
                # Convert to mono if stereo
                if self.channels > 1:
                    audio = audio.reshape(-1, self.channels).mean(axis=1)

                # VAD pre-filter: skip analysis on non-speech frames
                if self.vad_enabled and self._vad_processor is not None:
                    # Downsample to 16 kHz for VAD if captured at higher rate
                    import scipy.signal as _sig
                    target_sr = 16000
                    if self.sample_rate != target_sr:
                        num_samples = int(len(audio) * target_sr / self.sample_rate)
                        vad_audio = _sig.resample(audio, num_samples).astype(np.float32)
                    else:
                        vad_audio = audio
                    if not self._vad_processor.detect_speech(vad_audio):
                        continue  # Skip silent / non-speech chunk

                # Analyze
                self._analyze_chunk(audio)
                
                # Callback
                if self.on_analysis_update:
                    self.on_analysis_update(self.get_analysis())
                    
            except OSError as e:
                # Stream closed (-9988) or device error — stop the loop
                if self.running:
                    print(f"[Audio] Stream closed, stopping capture: {e}")
                self.running = False
                break
            except Exception as e:
                if self.running:
                    print(f"[Audio] Capture error: {e}")
                time.sleep(0.01)
    
    def _analyze_chunk(self, audio_data):
        """Analyze a chunk of audio data."""
        # Apply window
        windowed = audio_data * np.hanning(len(audio_data))
        
        # FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(audio_data), 1.0 / self.sample_rate)
        
        # Normalize
        fft = fft / (len(audio_data) / 2 + 1)
        
        # Frequency bands
        bass_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 2000)
        high_mask = freqs >= 2000
        
        # Calculate raw energy in each band
        # Reduced multipliers from 80/60/40 to prevent ambient noise from
        # registering as music (was causing avatar to dance with no audio).
        raw_bass = np.sqrt(np.mean(fft[bass_mask]**2)) * 20 if np.any(bass_mask) else 0
        raw_mid = np.sqrt(np.mean(fft[mid_mask]**2)) * 15 if np.any(mid_mask) else 0
        raw_high = np.sqrt(np.mean(fft[high_mask]**2)) * 10 if np.any(high_mask) else 0
        
        # Noise gate: zero out energy below ambient noise floor
        nf = self.noise_floor
        if raw_bass < nf: raw_bass = 0.0
        if raw_mid < nf: raw_mid = 0.0
        if raw_high < nf: raw_high = 0.0
        
        # Smooth
        self.smooth_bass = self.smooth_bass * (1 - self.smooth_factor) + raw_bass * self.smooth_factor
        self.smooth_mid = self.smooth_mid * (1 - self.smooth_factor) + raw_mid * self.smooth_factor
        self.smooth_high = self.smooth_high * (1 - self.smooth_factor) + raw_high * self.smooth_factor
        
        # Clamp to 0-1
        self.bass_energy = min(1.0, max(0.0, self.smooth_bass))
        self.mid_energy = min(1.0, max(0.0, self.smooth_mid))
        self.high_energy = min(1.0, max(0.0, self.smooth_high))
        
        # Overall energy
        self.overall_energy = self.bass_energy * 0.5 + self.mid_energy * 0.3 + self.high_energy * 0.2
        
        # Beat detection
        current_energy = self.bass_energy * 2 + self.mid_energy
        self.energy_history.append(current_energy)
        
        if len(self.energy_history) > 5:
            avg_energy = sum(self.energy_history) / len(self.energy_history)
            current_time = time.time()
            
            if (current_energy > avg_energy * self.beat_threshold and 
                current_time - self.last_beat_time > self.beat_cooldown and
                avg_energy > 0.02):  # Lower threshold for better quiet music detection
                self.is_beat = True
                self.beat_intensity = min(1.0, (current_energy / max(avg_energy, 0.01) - 1))
                self.last_beat_time = current_time
            else:
                self.is_beat = False
                self.beat_intensity = max(0, self.beat_intensity - 0.15)
    
    def get_analysis(self):
        """Get the current analysis results as a dictionary."""
        return {
            'type': 'audio_analysis',
            'bass': round(self.bass_energy, 3),
            'mid': round(self.mid_energy, 3),
            'high': round(self.high_energy, 3),
            'energy': round(self.overall_energy, 3),
            'beat': self.is_beat,
            'beatIntensity': round(self.beat_intensity, 3)
        }
    
    def get_analysis_json(self):
        """Get the current analysis results as a JSON string."""
        return json.dumps(self.get_analysis())

    def update_config(self, sound_threshold=None, filter_apps=None):
        """Feature #24: Update audio config from Unity settings."""
        if sound_threshold is not None:
            self.noise_floor = max(0.0, min(0.5, float(sound_threshold)))
            print(f"[Audio] Noise floor set to {self.noise_floor:.3f}")
        if filter_apps is not None:
            if isinstance(filter_apps, str):
                self.filter_apps = [a.strip().lower() for a in filter_apps.split(",") if a.strip()]
            else:
                self.filter_apps = [str(a).lower() for a in filter_apps]
            print(f"[Audio] App filter: {self.filter_apps if self.filter_apps else '(all)'}")


# Standalone test with visual feedback
if __name__ == "__main__":
    print("=" * 60)
    print("System Audio Analyzer Test")
    print("Play some music on YouTube/Pandora/etc and watch the bars!")
    print("=" * 60)
    
    analyzer = SystemAudioAnalyzer()
    
    def print_analysis(data):
        def bar(v, width=20):
            filled = int(v * width)
            return '█' * filled + '░' * (width - filled)
        
        beat_marker = " * BEAT! *" if data['beat'] else ""
        print(f"\rBass: {bar(data['bass'])} | Mid: {bar(data['mid'])} | High: {bar(data['high'])}{beat_marker}    ", end='', flush=True)
    
    analyzer.on_analysis_update = print_analysis
    
    if analyzer.start():
        print("\nListening... Press Ctrl+C to stop\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
            analyzer.stop()
    else:
        print("Failed to start audio capture")
