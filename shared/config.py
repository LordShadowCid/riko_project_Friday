"""
Shared configuration for Annabeth Desktop Companion.

This module provides centralized configuration that's shared between
the Python backend and can be exported to JavaScript frontend.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


# =============================================================================
# ENUMS - Explicit types instead of magic strings
# =============================================================================

class CompanionMode(str, Enum):
    """Avatar companion modes."""
    ACTIVE = "active"      # Normal chat mode - listening and responding
    IDLE = "idle"          # Idle mode - not listening
    DANCE_BEAT = "dance_beat"  # Procedural beat-reactive dance
    DANCE_FULL = "dance_full"  # Full choreographed VRMA dance


class MessageType(str, Enum):
    """WebSocket message types for client-server communication."""
    # Client -> Server
    MODE_CHANGE = "mode_change"
    TOGGLE_SILENCE = "toggle_silence"
    SET_SILENCE = "set_silence"
    
    # Server -> Client
    SPEAK_START = "speak_start"
    SPEAK_END = "speak_end"
    EMOTION = "emotion"
    AUDIO_ANALYSIS = "audio_analysis"


class Emotion(str, Enum):
    """Avatar emotion states."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ServerConfig:
    """Server configuration settings."""
    # Avatar WebSocket server
    avatar_host: str = "0.0.0.0"
    avatar_port: int = 8765
    
    # Desktop companion HTTP server
    companion_http_port: int = 8766
    
    # TTS server (GPT-SoVITS)
    tts_host: str = "localhost"
    tts_port: int = 9880
    
    @property
    def avatar_ws_url(self) -> str:
        """WebSocket URL for avatar server."""
        return f"ws://127.0.0.1:{self.avatar_port}/ws"
    
    @property
    def avatar_http_url(self) -> str:
        """HTTP URL for avatar server."""
        return f"http://localhost:{self.avatar_port}"
    
    @property
    def tts_url(self) -> str:
        """URL for TTS server."""
        return f"http://{self.tts_host}:{self.tts_port}"


@dataclass
class AudioConfig:
    """Audio capture and analysis configuration."""
    # Sample rate for audio capture
    sample_rate: int = 16000
    
    # Audio analysis thresholds (0.0 - 1.0)
    audio_threshold: float = 0.05  # Minimum energy to trigger dance
    beat_threshold: float = 1.3    # Threshold for beat detection
    beat_cooldown_sec: float = 0.12  # Minimum time between beats
    
    # Frequency band multipliers for sensitivity
    bass_multiplier: float = 80.0
    mid_multiplier: float = 60.0
    high_multiplier: float = 40.0
    
    # Minimum average energy to consider as "audio present"
    min_avg_energy: float = 0.02
    
    # Audio broadcast frame rate
    audio_broadcast_fps: int = 30  # Send audio data at ~30 FPS
    
    # VAD (Voice Activity Detection) settings
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    silence_threshold_sec: float = 1.0  # Silence duration to end speech


@dataclass  
class AnimationConfig:
    """Animation timing configuration."""
    # Fade durations (seconds)
    fade_in_duration: float = 0.5
    fade_out_duration: float = 0.5
    
    # Blink timing
    blink_duration: float = 0.15
    blink_interval_min: float = 2.0
    blink_interval_max: float = 6.0
    
    # Idle animation
    idle_breathing_speed: float = 0.8
    idle_breathing_amplitude: float = 0.01
    
    # Dance intensity multipliers
    dance_beat_intensity: float = 1.0
    dance_full_intensity: float = 1.8


@dataclass
class PathConfig:
    """File path configuration."""
    # Project root (computed from this file's location)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def vrm_model_path(self) -> Path:
        return self.models_dir / "vrm" / "claire_avatar.vrm"
    
    @property
    def animations_dir(self) -> Path:
        return self.project_root / "animations"
    
    @property
    def shikanoko_dance_path(self) -> Path:
        return self.animations_dir / "shikanoko_dance.vrma"
    
    @property
    def audio_dir(self) -> Path:
        return self.project_root / "audio"
    
    @property
    def client_dir(self) -> Path:
        return self.project_root / "client"


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

@dataclass
class AnnabeConfig:
    """Main configuration container."""
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Singleton config instance
_config: Optional[AnnabeConfig] = None


def get_config() -> AnnabeConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AnnabeConfig()
    return _config


def reset_config() -> None:
    """Reset config to defaults (useful for testing)."""
    global _config
    _config = None


# =============================================================================
# EXPORT TO JAVASCRIPT
# =============================================================================

def config_to_js() -> str:
    """Export configuration as JavaScript constants."""
    cfg = get_config()
    return f"""
// Auto-generated from shared/config.py - DO NOT EDIT MANUALLY
const CONFIG = {{
    server: {{
        avatarWsUrl: '{cfg.server.avatar_ws_url}',
        avatarHttpUrl: '{cfg.server.avatar_http_url}',
        companionHttpPort: {cfg.server.companion_http_port},
    }},
    audio: {{
        threshold: {cfg.audio.audio_threshold},
        beatThreshold: {cfg.audio.beat_threshold},
        beatCooldownSec: {cfg.audio.beat_cooldown_sec},
        bassMultiplier: {cfg.audio.bass_multiplier},
        midMultiplier: {cfg.audio.mid_multiplier},
        highMultiplier: {cfg.audio.high_multiplier},
        minAvgEnergy: {cfg.audio.min_avg_energy},
    }},
    animation: {{
        fadeInDuration: {cfg.animation.fade_in_duration},
        fadeOutDuration: {cfg.animation.fade_out_duration},
        blinkDuration: {cfg.animation.blink_duration},
        blinkIntervalMin: {cfg.animation.blink_interval_min},
        blinkIntervalMax: {cfg.animation.blink_interval_max},
        danceBeatIntensity: {cfg.animation.dance_beat_intensity},
        danceFullIntensity: {cfg.animation.dance_full_intensity},
    }},
    paths: {{
        vrmModel: '/models/vrm/claire_avatar.vrm',
        shikanokoDance: '/animations/shikanoko_dance.vrma',
    }},
    modes: {{
        ACTIVE: '{CompanionMode.ACTIVE.value}',
        IDLE: '{CompanionMode.IDLE.value}',
        DANCE_BEAT: '{CompanionMode.DANCE_BEAT.value}',
        DANCE_FULL: '{CompanionMode.DANCE_FULL.value}',
    }},
    messageTypes: {{
        MODE_CHANGE: '{MessageType.MODE_CHANGE.value}',
        TOGGLE_SILENCE: '{MessageType.TOGGLE_SILENCE.value}',
        SET_SILENCE: '{MessageType.SET_SILENCE.value}',
        SPEAK_START: '{MessageType.SPEAK_START.value}',
        SPEAK_END: '{MessageType.SPEAK_END.value}',
        EMOTION: '{MessageType.EMOTION.value}',
        AUDIO_ANALYSIS: '{MessageType.AUDIO_ANALYSIS.value}',
    }},
}};

// Freeze to prevent accidental modification
Object.freeze(CONFIG);
Object.freeze(CONFIG.server);
Object.freeze(CONFIG.audio);
Object.freeze(CONFIG.animation);
Object.freeze(CONFIG.paths);
Object.freeze(CONFIG.modes);
Object.freeze(CONFIG.messageTypes);
"""
