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
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("[Audio] PyAudioWPatch not installed. Run: pip install PyAudioWPatch")


class SystemAudioAnalyzer:
    """Captures and analyzes system audio in real-time using WASAPI loopback."""
    
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size
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
        
        # PyAudio
        self.p = None
        self.stream = None
        self.sample_rate = 48000
        self.channels = 2
        self.loopback_device = None
        
        # Callbacks for sending data
        self.on_analysis_update = None
    
    def _find_loopback_device(self):
        """Find the WASAPI loopback device matching the default output."""
        if not HAS_PYAUDIO:
            return None
        
        try:
            self.p = pyaudio.PyAudio()
            
            # Get the default output device name
            try:
                default_output = self.p.get_default_output_device_info()
                default_name = default_output.get('name', '').lower()
                print(f"[Audio] Default output device: {default_output.get('name')}")
            except Exception as e:
                print(f"[Audio] Could not get default output: {e}")
                default_name = ''
            
            # Collect all loopback devices
            loopback_devices = []
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                if dev.get('isLoopbackDevice', False):
                    loopback_devices.append(dev)
                    print(f"[Audio] Found loopback: {dev['name']}")
            
            if not loopback_devices:
                print("[Audio] No loopback devices found")
                return None
            
            # Try to find loopback that matches default output
            for dev in loopback_devices:
                dev_name = dev['name'].lower()
                # Check if the loopback device name contains the default output name
                # (loopback names often have " [Loopback]" suffix)
                if default_name and (default_name in dev_name or 
                    dev_name.replace(' [loopback]', '') in default_name or
                    default_name.replace('(b', '(bcc950') in dev_name):  # Handle truncated names
                    print(f"[Audio] Selected loopback (matches default): {dev['name']}")
                    return dev
            
            # Fallback: just use the first loopback device
            print(f"[Audio] Using first loopback (no match found): {loopback_devices[0]['name']}")
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
                
                # Analyze
                self._analyze_chunk(audio)
                
                # Callback
                if self.on_analysis_update:
                    self.on_analysis_update(self.get_analysis())
                    
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
        
        # Calculate raw energy in each band - boosted for better sensitivity
        # User uses BCC950 ConferenceCam as only audio device
        raw_bass = np.sqrt(np.mean(fft[bass_mask]**2)) * 80 if np.any(bass_mask) else 0
        raw_mid = np.sqrt(np.mean(fft[mid_mask]**2)) * 60 if np.any(mid_mask) else 0
        raw_high = np.sqrt(np.mean(fft[high_mask]**2)) * 40 if np.any(high_mask) else 0
        
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
        
        beat_marker = " ♪ BEAT! ♪" if data['beat'] else ""
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
