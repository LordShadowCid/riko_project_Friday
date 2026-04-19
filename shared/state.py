"""
State management for Annabeth Desktop Companion.

This module provides thread-safe state management to replace
scattered global variables throughout the codebase.
"""
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Callable, Dict, Any

from .config import CompanionMode, Emotion


@dataclass
class CompanionState:
    """
    Thread-safe state container for the companion.
    
    Replaces scattered globals like _chat_silenced, _current_mode, etc.
    Uses a lock for thread safety since state is accessed from multiple threads.
    """
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    # Core state
    _mode: CompanionMode = CompanionMode.ACTIVE
    _silenced: bool = False
    _speaking: bool = False
    _emotion: Emotion = Emotion.NEUTRAL

    # Browser selection context (text highlighted in browser extension)
    _browser_selected_text: str = ""

    # Shutdown flag — set by frontend (Unity) when it closes
    shutdown_requested: bool = False
    
    # Callbacks for state change notifications
    _on_mode_change: Optional[Callable[[CompanionMode], None]] = field(default=None, repr=False)
    _on_silence_change: Optional[Callable[[bool], None]] = field(default=None, repr=False)
    
    @property
    def mode(self) -> CompanionMode:
        """Get current companion mode."""
        with self._lock:
            return self._mode
    
    @mode.setter
    def mode(self, value: CompanionMode) -> None:
        """Set companion mode with thread safety."""
        with self._lock:
            old_mode = self._mode
            self._mode = value
            callback = self._on_mode_change
        
        if callback and old_mode != value:
            callback(value)
        
        print(f"[State] Mode changed: {old_mode.value} -> {value.value}")
    
    @property
    def silenced(self) -> bool:
        """Check if chat is silenced."""
        with self._lock:
            return self._silenced
    
    @silenced.setter
    def silenced(self, value: bool) -> None:
        """Set silence state with thread safety."""
        with self._lock:
            old_value = self._silenced
            self._silenced = value
            callback = self._on_silence_change
        
        if callback and old_value != value:
            callback(value)
        
        status = "[MUTED] SILENCED" if value else "[UNMUTED] LISTENING"
        print(f"[State] Chat: {status}")
    
    def toggle_silence(self) -> bool:
        """Toggle silence state and return new value."""
        with self._lock:
            self._silenced = not self._silenced
            new_value = self._silenced
            callback = self._on_silence_change
        
        if callback:
            callback(new_value)
        
        status = "[MUTED] SILENCED" if new_value else "[UNMUTED] LISTENING"
        print(f"[State] Chat: {status}")
        return new_value
    
    @property
    def speaking(self) -> bool:
        """Check if avatar is currently speaking."""
        with self._lock:
            return self._speaking
    
    @speaking.setter
    def speaking(self, value: bool) -> None:
        """Set speaking state."""
        with self._lock:
            self._speaking = value
    
    @property
    def emotion(self) -> Emotion:
        """Get current emotion."""
        with self._lock:
            return self._emotion
    
    @emotion.setter
    def emotion(self, value: Emotion) -> None:
        """Set emotion state."""
        with self._lock:
            self._emotion = value
    
    def is_listening_paused(self) -> bool:
        """
        Check if listening should be paused.
        
        Listening is paused when:
        - Chat is silenced (S key)
        - Mode is not ACTIVE (idle or dance modes)
        """
        with self._lock:
            return self._silenced or self._mode != CompanionMode.ACTIVE
    
    def is_dancing(self) -> bool:
        """Check if in any dance mode."""
        with self._lock:
            return self._mode in (CompanionMode.DANCE_BEAT, CompanionMode.DANCE_FULL)
    
    def set_mode_change_callback(self, callback: Callable[[CompanionMode], None]) -> None:
        """Set callback for mode changes."""
        with self._lock:
            self._on_mode_change = callback
    
    def set_silence_change_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback for silence changes."""
        with self._lock:
            self._on_silence_change = callback

    @property
    def browser_selected_text(self) -> str:
        with self._lock:
            return self._browser_selected_text

    @browser_selected_text.setter
    def browser_selected_text(self, value: str) -> None:
        with self._lock:
            self._browser_selected_text = value
        if value:
            print(f"[State] Browser selection stored ({len(value)} chars)")

    def take_browser_selected_text(self) -> str:
        """Return stored browser selection and clear it (one-shot consume)."""
        with self._lock:
            text = self._browser_selected_text
            self._browser_selected_text = ""
            return text


@dataclass
class AudioState:
    """
    State for audio analysis.
    
    Updated by the audio analyzer, read by the animation system.
    """
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    # Frequency band energies (0.0 - 1.0+)
    _bass: float = 0.0
    _mid: float = 0.0
    _high: float = 0.0
    
    # Beat detection
    _is_beat: bool = False
    _last_beat_time: float = 0.0
    
    @property
    def bass(self) -> float:
        with self._lock:
            return self._bass
    
    @property
    def mid(self) -> float:
        with self._lock:
            return self._mid
    
    @property
    def high(self) -> float:
        with self._lock:
            return self._high
    
    @property
    def is_beat(self) -> bool:
        with self._lock:
            return self._is_beat
    
    @property
    def energy(self) -> float:
        """Total audio energy."""
        with self._lock:
            return (self._bass + self._mid + self._high) / 3.0
    
    def update(self, bass: float, mid: float, high: float, is_beat: bool) -> None:
        """Update all audio state atomically."""
        with self._lock:
            self._bass = bass
            self._mid = mid
            self._high = high
            self._is_beat = is_beat
    
    def has_audio(self, threshold: float = 0.05) -> bool:
        """Check if there's meaningful audio above threshold."""
        with self._lock:
            return (self._bass + self._mid) > threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary for WebSocket messages."""
        with self._lock:
            return {
                "bass": self._bass,
                "mid": self._mid,
                "high": self._high,
                "is_beat": self._is_beat,
                "energy": (self._bass + self._mid + self._high) / 3.0,
            }


# =============================================================================
# GLOBAL STATE INSTANCES
# =============================================================================

_companion_state: Optional[CompanionState] = None
_audio_state: Optional[AudioState] = None


def get_companion_state() -> CompanionState:
    """Get the global companion state instance."""
    global _companion_state
    if _companion_state is None:
        _companion_state = CompanionState()
    return _companion_state


def get_audio_state() -> AudioState:
    """Get the global audio state instance."""
    global _audio_state
    if _audio_state is None:
        _audio_state = AudioState()
    return _audio_state


def reset_state() -> None:
    """Reset all state to defaults (useful for testing)."""
    global _companion_state, _audio_state
    _companion_state = None
    _audio_state = None


# =============================================================================
# READ ALOUD STATE ACCESS
# =============================================================================

def get_read_aloud_manager():
    """
    Get the ReadAloudManager instance.
    
    Lazy import to avoid circular dependencies.
    """
    from server.process.read_aloud import get_read_aloud_manager as _get_manager
    return _get_manager()
