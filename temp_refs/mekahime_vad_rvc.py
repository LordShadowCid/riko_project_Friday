"""
SOURCE: https://github.com/zeekk0/MekaHime-Pipeline-V1/blob/main/MKHM_Pipeline_V1.py
REPO: MekaHime-Pipeline-V1 (zeekk0)
PURPOSE: Pattern reference for WebRTC VAD speech detection and RVC voice conversion.
         Annabeth integration points:
           - VAD: augment/replace client/audio_analyzer.py silence detection
           - RVC: add as post-processing stage after GPT-SoVITS TTS output
           
ORIGINAL DEPS:
    pip install webrtcvad torch torchaudio librosa soundfile
    pip install git+https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

NOTE on RVC import:
    RVC doesn't have a clean pip package. Either:
    1. Clone RVC repo into third_party/rvc/ and add to sys.path
    2. Use the infer_cli.py approach (subprocess call)
    3. Use rvc-python wrapper: pip install rvc-python

ANNABETH INTEGRATION STRATEGY:
    1. WebRTC VAD drops in as a secondary filter on top of existing WASAPI loopback.
       Current: audio_analyzer.py uses energy threshold silence detection.
       Add: webrtcvad.Vad(aggressiveness=2) on each 480-sample frame before buffering.
       
    2. RVC post-processes XTTS/GPT-SoVITS output.
       Current: server saves TTS WAV → sends path to Unity → Unity plays audio.
       Add: after TTS write, call convert_with_rvc(tts_output_path) → overwrite or save 
       with _rvc suffix → send updated path to Unity.
       Hook: server/process/llm_funcs/llm_tts.py or wherever TTS is written.
"""

import os
import sys
import logging
import asyncio
import struct
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple


# ============================================================
# VAD (WebRTC Voice Activity Detection)
# ============================================================

class VADProcessor:
    """
    WebRTC VAD wrapper for speech detection.
    
    ORIGINAL: CLAUDE_AI_GF.detect_speech_with_vad()
    
    Frame parameters (must be exact for webrtcvad):
        sample_rate = 16000  # 16 kHz ONLY (or 8k/32k/48k)
        frame_duration_ms = 30  # 10, 20, or 30 ms ONLY  
        frame_size = 16000 * 30 // 1000  # = 480 samples
        
    Aggressiveness 0-3:
        0 = least aggressive (more false positives)
        2 = balanced (recommended for Annabeth)
        3 = most aggressive (fewer false positives, may miss soft speech)
    
    Usage in Annabeth:
        vad = VADProcessor(aggressiveness=2)
        is_speech = vad.detect_speech(audio_float32_16khz)
        # Returns True if >= 20% of 30ms frames contain speech
    """
    
    SAMPLE_RATE = 16000         # webrtcvad only supports 8k, 16k, 32k, 48k
    FRAME_DURATION_MS = 30      # must be 10, 20, or 30ms
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples
    SPEECH_RATIO_THRESHOLD = 0.2  # 20% of frames must be speech
    
    def __init__(self, aggressiveness: int = 2):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.logger = logging.getLogger(__name__)
            self._available = True
        except ImportError:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("webrtcvad not installed — VAD disabled. pip install webrtcvad")
            self._available = False
    
    def detect_speech(self, audio_float32: np.ndarray) -> bool:
        """
        Detect if audio buffer contains speech.
        
        Args:
            audio_float32: 16kHz float32 numpy array
        Returns:
            True if speech detected (>= SPEECH_RATIO_THRESHOLD of frames)
        """
        if not self._available:
            return True  # Fall through to existing energy detection
        
        # Convert float32 → int16 PCM (webrtcvad needs PCM bytes)
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        raw_bytes = audio_int16.tobytes()
        
        # Split into 30ms frames
        bytes_per_frame = self.FRAME_SIZE * 2  # 2 bytes per int16 sample
        frames = [raw_bytes[i:i+bytes_per_frame] 
                  for i in range(0, len(raw_bytes) - bytes_per_frame + 1, bytes_per_frame)]
        
        if not frames:
            return False
        
        # Count speech frames
        speech_count = 0
        for frame in frames:
            if len(frame) == bytes_per_frame:
                try:
                    if self.vad.is_speech(frame, self.SAMPLE_RATE):
                        speech_count += 1
                except Exception:
                    pass
        
        speech_ratio = speech_count / len(frames)
        return speech_ratio >= self.SPEECH_RATIO_THRESHOLD
    
    def detect_speech_bytes(self, pcm_bytes: bytes) -> bool:
        """Detect speech from raw int16 PCM bytes (16kHz)."""
        if not self._available:
            return True
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767
        return self.detect_speech(audio_np)


# ============================================================
# RVC (Retrieval-based Voice Conversion)
# ============================================================

class RVCProcessor:
    """
    RVC voice conversion wrapper.
    
    ORIGINAL: CLAUDE_AI_GF.convert_with_rvc()
    
    RVC parameters:
        sid = 0                    # Speaker ID (usually 0)
        f0_up_key = 0              # Pitch shift in semitones (0 = no change)
        f0_method = "rmvpe"        # Best quality; "harvest" or "crepe" as fallback
        file_index = "logs/model.index"  # Feature index for voice timbre
        index_rate = 0.8           # How much to use feature index (0-1)
        filter_radius = 3          # Median filter radius (3 = slight smoothing)
        resample_sr = 0            # 0 = no resample, or set target sample rate
        rms_mix_rate = 0.8         # Mix RMS envelope (1.0 = full original envelope)
        protect = 0.33             # Protect voiceless consonants (0.33 = balanced)
    
    Annabeth model setup:
        1. Download pre-trained RVC model (.pth) → models/rvc/annabeth.pth
        2. Generate feature index: logs/annabeth.index
        3. Set RVC_MODEL_PATH and RVC_INDEX_PATH in character_config.yaml
    
    Two integration methods:
        A) Direct (import VC from rvc library) — faster, requires RVC to be on sys.path
        B) Subprocess (call infer_cli.py) — safer, easier to manage
    
    This class implements Method B (subprocess) as it's cleaner for Annabeth's architecture.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.8,
        protect: float = 0.33,
        rvc_root: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.index_path = index_path or ""
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.protect = protect
        self.rvc_root = rvc_root or str(Path(__file__).parent.parent / "third_party" / "rvc")
        
        self._vc = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize RVC (Method A: direct import).
        Call this once at startup.
        
        Requires RVC-WebUI cloned to third_party/rvc/:
            git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git third_party/rvc
        """
        try:
            if self.rvc_root not in sys.path:
                sys.path.insert(0, self.rvc_root)
            
            from configs.config import Config           # type: ignore
            from vc_infer_pipeline import VC            # type: ignore
            
            config = Config()
            self._vc = VC(config)
            self._vc.get_vc(self.model_path)
            self._initialized = True
            self.logger.info(f"RVC initialized with model: {self.model_path}")
            return True
        except Exception as e:
            self.logger.warning(f"RVC direct init failed: {e}. RVC post-processing will be skipped.")
            return False
    
    def convert(self, input_path: str, output_path: str) -> bool:
        """
        Convert audio file using RVC (Method A: direct).
        
        Args:
            input_path: Path to TTS-generated WAV file
            output_path: Where to save RVC-processed WAV
        Returns:
            True if successful
        """
        if not self._initialized or self._vc is None:
            return False
        
        try:
            # Check if rmvpe model available, fall back to harvest
            f0_method = self.f0_method
            rmvpe_path = Path(self.rvc_root) / "rmvpe.pt"
            if not rmvpe_path.exists():
                f0_method = "harvest"
                self.logger.warning("rmvpe.pt not found, using harvest method")
            
            result = self._vc.vc_single(
                sid=0,
                input_audio_path=input_path,
                f0_up_key=self.f0_up_key,
                f0_file=None,
                f0_method=f0_method,
                file_index=self.index_path,
                file_index2="",
                index_rate=self.index_rate,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.8,
                protect=self.protect
            )
            
            # result = (info_str, (target_sr, audio_output_np))
            if result and len(result) == 2:
                info, (target_sr, audio_output) = result
                sf.write(output_path, audio_output, target_sr)
                self.logger.debug(f"RVC conversion complete: {output_path}")
                return True
        except Exception as e:
            self.logger.error(f"RVC conversion failed: {e}")
        return False
    
    async def convert_subprocess(self, input_path: str, output_path: str) -> bool:
        """
        Convert audio using RVC via subprocess (Method B: safer).
        Requires: third_party/rvc/infer_cli.py
        
        Example usage in Annabeth's TTS pipeline:
            rvc = RVCProcessor(model_path="annabeth.pth")
            success = await rvc.convert_subprocess(tts_output, rvc_output)
        """
        rvc_cli = Path(self.rvc_root) / "infer_cli.py"
        if not rvc_cli.exists():
            self.logger.warning(f"RVC CLI not found at {rvc_cli}")
            return False
        
        cmd = [
            sys.executable, str(rvc_cli),
            "--f0up_key", str(self.f0_up_key),
            "--input_path", input_path,
            "--output_path", output_path,
            "--pth_path", self.model_path or "",
            "--index_path", self.index_path,
            "--f0method", self.f0_method,
            "--index_rate", str(self.index_rate),
            "--protect", str(self.protect),
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                return True
            else:
                self.logger.error(f"RVC subprocess error: {stderr.decode()}")
        except asyncio.TimeoutError:
            self.logger.error("RVC subprocess timed out")
        except Exception as e:
            self.logger.error(f"RVC subprocess failed: {e}")
        return False


# ============================================================
# INTEGRATION EXAMPLE — how to patch into Annabeth's TTS pipeline
# ============================================================
"""
In server/process/tts/ (wherever TTS writes its WAV):

    async def generate_tts_and_convert(text: str, out_path: str) -> str:
        # 1. Generate TTS (GPT-SoVITS or XTTS)
        tts_path = out_path.replace(".wav", "_tts.wav")
        await gpt_sovits.synthesize(text, tts_path)
        
        # 2. RVC post-process (if enabled in config)
        if rvc_enabled and rvc_processor.initialized:
            rvc_path = out_path.replace(".wav", "_rvc.wav")
            success = rvc_processor.convert(tts_path, rvc_path)
            if success:
                return rvc_path  # Send RVC audio to Unity
        
        return tts_path  # Fall back to TTS audio

In client/audio_analyzer.py — add VAD filter before saving buffer:
    
    vad = VADProcessor(aggressiveness=2)
    
    async def process_audio_chunk(chunk_bytes: bytes):
        # ... existing WASAPI capture ...
        if vad.detect_speech_bytes(chunk_bytes):
            speech_buffer.append(chunk_bytes)
        else:
            # Silence — flush buffer if long enough
            if len(speech_buffer) >= min_speech_frames:
                await send_to_stt(speech_buffer)
            speech_buffer.clear()
"""
