"""
Runtime Settings Registry
=========================
A typed, runtime-editable registry of all tuneable Annabeth parameters.
Replaces scattered hardcoded constants across modules.

Usage::

    from server.settings_registry import registry

    # Read a value
    interval = registry.get("GRILLO_BEAT_INTERVAL")   # → 2700

    # Override at runtime (e.g. from WebSocket)
    registry.set("GRILLO_BEAT_INTERVAL", 1800)

    # Iterate all vars for a settings UI
    all_settings = registry.all_vars()

Call ``init_registry()`` once at server startup (done automatically via module import).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class VarDef:
    """Definition of a single registered variable."""
    key: str
    label: str
    default: Any
    value_type: type
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None


class SettingsRegistry:
    """
    Thread-safe (read-heavy) registry of runtime variables.

    All writes go through ``set()`` which validates the value before storing.
    """

    def __init__(self) -> None:
        self._vars: Dict[str, VarDef] = {}
        self._values: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, var: VarDef) -> None:
        """Register a variable definition and initialise it to its default."""
        self._vars[var.key] = var
        self._values[var.key] = var.default

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the current value, or *default* if not registered."""
        if key in self._values:
            return self._values[key]
        if key in self._vars:
            return self._vars[key].default
        return default

    def set(self, key: str, value: Any) -> bool:
        """
        Update a value. Runs the VarDef validator if present.
        Coerces *value* to the registered type.
        Returns True on success.
        """
        var = self._vars.get(key)
        if var is None:
            logger.warning("[Registry] Unknown key: %s", key)
            return False
        try:
            coerced = var.value_type(value)
        except (TypeError, ValueError) as exc:
            logger.warning("[Registry] Cannot coerce %r to %s for key %s: %s",
                           value, var.value_type.__name__, key, exc)
            return False
        if var.validator is not None and not var.validator(coerced):
            logger.warning("[Registry] Validation failed for %s = %r", key, coerced)
            return False
        self._values[key] = coerced
        logger.debug("[Registry] %s = %r", key, coerced)
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def all_vars(self) -> Dict[str, Any]:
        """Return a snapshot of all keys and their current values."""
        return {k: self._values.get(k, v.default) for k, v in self._vars.items()}

    def var_defs(self) -> Dict[str, VarDef]:
        """Return all VarDef definitions (useful for building a settings UI)."""
        return dict(self._vars)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = SettingsRegistry()


def init_registry() -> None:
    """
    Register all default variables.
    Safe to call multiple times — re-registration is a no-op (first write wins).
    Called automatically at the bottom of this module.
    """
    _vars = [
        # ── Grillo / reflection ────────────────────────────────────────
        VarDef(
            "GRILLO_BEAT_INTERVAL",
            "Grillo Beat Interval (s)",
            2700, int,
            "How often the Grillo beat loop fires (seconds). Default 45 min.",
            validator=lambda v: v >= 60,
        ),
        VarDef(
            "GRILLO_DREAM_ENABLED",
            "Grillo Dream Mode",
            True, bool,
            "When True, Grillo runs extra beats during system idle / sleep.",
        ),
        VarDef(
            "GRILLO_DREAM_TIME",
            "Grillo Dream Time (HH:MM)",
            "05:00", str,
            "Local time window when dream beats are preferred.",
        ),
        VarDef(
            "GRILLO_OBSERVER_ENABLED",
            "Grillo Observer Enabled",
            True, bool,
            "Grillo observes screen/window titles for context hints.",
        ),
        VarDef(
            "GRILLO_OBSERVER_INTERVAL",
            "Grillo Observer Interval (s)",
            3600, int,
            "How often the observer scans context.",
            validator=lambda v: v >= 60,
        ),

        # ── Emotion ────────────────────────────────────────────────────
        VarDef(
            "EMOTION_DECAY_TAU",
            "Emotion Decay Time Constant (s)",
            3600, int,
            "Exponential decay τ for emotion intensities (seconds).",
            validator=lambda v: v >= 60,
        ),
        VarDef(
            "EMOTION_MAX_DISPLAY",
            "Max Emotions Displayed",
            7, int,
            "Maximum number of top emotions shown in the UI at once.",
            validator=lambda v: 1 <= v <= 20,
        ),

        # ── Model routing ──────────────────────────────────────────────
        VarDef(
            "LATENCY_THRESHOLD_MS",
            "LLM Latency Threshold (ms)",
            5000, int,
            "If LLM response takes longer than this, switch to fast model.",
            validator=lambda v: v >= 500,
        ),
        VarDef(
            "MEMORY_THRESHOLD_PCT",
            "RAM Alert Threshold (%)",
            85, int,
            "If system RAM exceeds this %, use fast model.",
            validator=lambda v: 50 <= v <= 99,
        ),
        VarDef(
            "MODEL_SWITCH_COOLDOWN_S",
            "Model Switch Cooldown (s)",
            30, int,
            "Seconds before switching back to primary model after a forced fast-model period.",
            validator=lambda v: v >= 5,
        ),

        # ── Idle / screensaver ─────────────────────────────────────────
        VarDef(
            "IDLE_TIMEOUT_S",
            "Idle Timeout (s, 0 = disabled)",
            300, int,
            "Seconds of inactivity before entering idle pose. 0 = never.",
            validator=lambda v: v >= 0,
        ),

        # ── Idle speech bubbles ────────────────────────────────────────
        VarDef(
            "IDLE_BUBBLE_MIN_DELAY",
            "Idle Bubble Min Delay (s)",
            60, int,
            "Minimum seconds between idle speech bubble appearances.",
            validator=lambda v: v >= 10,
        ),
        VarDef(
            "IDLE_BUBBLE_MAX_DELAY",
            "Idle Bubble Max Delay (s)",
            300, int,
            "Maximum seconds between idle speech bubble appearances.",
            validator=lambda v: v >= 10,
        ),

        # ── Audio / VAD ────────────────────────────────────────────────
        VarDef(
            "VAD_AGGRESSIVENESS",
            "VAD Aggressiveness (0-3)",
            2, int,
            "WebRTC VAD aggressiveness. 0 = permissive, 3 = strict.",
            validator=lambda v: 0 <= v <= 3,
        ),
        VarDef(
            "VAD_SPEECH_RATIO",
            "VAD Speech Ratio Threshold",
            0.2, float,
            "Fraction of 30ms frames that must contain speech to pass VAD.",
            validator=lambda v: 0.0 < v <= 1.0,
        ),

        # ── RVC ────────────────────────────────────────────────────────
        VarDef(
            "RVC_ENABLED",
            "Enable RVC Post-Processing",
            False, bool,
            "Route TTS output through RVC voice conversion.",
        ),

        # ── Facial expressions ─────────────────────────────────────────
        VarDef(
            "FACIAL_EXPR_ENABLED",
            "Enable Facial Expression Tags",
            True, bool,
            "Parse [em_NAME:INTENSITY] tags from LLM output and drive VRM blend shapes.",
        ),
        VarDef(
            "FACIAL_EXPR_CHARS_PER_SEC",
            "Facial Expr Chars Per Second",
            12.0, float,
            "Assumed reading speed for timing expressions when audio duration is unknown.",
            validator=lambda v: v > 0,
        ),

        # ── Code self-improvement ──────────────────────────────────────
        VarDef(
            "MAX_DAILY_CODE_IMPROVES",
            "Max Daily Auto-Code Fixes",
            3, int,
            "Maximum automated code fixes applied per day.",
            validator=lambda v: 0 <= v <= 20,
        ),
        VarDef(
            "CODE_IMPROVE_INTERVAL_H",
            "Code Improvement Interval (hours)",
            168.0, float,
            "How often the self-improvement scanner runs (default 1 week).",
            validator=lambda v: v >= 1.0,
        ),
    ]

    for var in _vars:
        if var.key not in registry._vars:   # first-write-wins so live overrides survive reload
            registry.register(var)


# Auto-initialise on import
init_registry()
