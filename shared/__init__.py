"""
Shared module for Annabeth Desktop Companion.

Provides centralized configuration, enums, and state management.
"""
from .config import (
    # Enums
    CompanionMode,
    MessageType,
    Emotion,
    
    # Config classes
    ServerConfig,
    AudioConfig,
    AnimationConfig,
    PathConfig,
    AnnabeConfig,
    
    # Functions
    get_config,
    reset_config,
    config_to_js,
)

from .state import (
    # State classes
    CompanionState,
    AudioState,
    
    # Functions
    get_companion_state,
    get_audio_state,
    reset_state,
    get_read_aloud_manager,
)

__all__ = [
    # Enums
    "CompanionMode",
    "MessageType", 
    "Emotion",
    
    # Config classes
    "ServerConfig",
    "AudioConfig",
    "AnimationConfig",
    "PathConfig",
    "AnnabeConfig",
    
    # State classes
    "CompanionState",
    "AudioState",
    
    # Functions
    "get_config",
    "reset_config",
    "config_to_js",
    "get_companion_state",
    "get_audio_state",
    "reset_state",
    "get_read_aloud_manager",
]
