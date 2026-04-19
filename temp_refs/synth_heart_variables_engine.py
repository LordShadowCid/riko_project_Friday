"""
SOURCE: https://github.com/XargonWan/Synthetic_Heart/blob/develop/core/variables_engine.py
REPO: Synthetic_Heart (XargonWan)
PURPOSE: Pattern reference for a typed config/settings registry with UI metadata.
         Annabeth will adapt this into server/annabeth_config.py or a new
         server/config/variables_engine.py for persisting settings with types.
"""

from typing import Any, Callable, Dict, Optional, Iterable
import re


class ValidationError(ValueError):
    pass


class ExposedVarDefinition:
    """
    Defines a configuration variable with type, UI hints, and validation.
    
    ui_type options:
    - string, password, number, bool, select, combobox, textarea, json, tags, file, color
    
    Annabeth: Store these in character_config.yaml or a settings.db SQLite table.
    """
    def __init__(
        self,
        key: str,
        label: str,
        default: Any = "",
        value_type: type | str = str,
        ui_type: str = "string",
        description: str = "",
        scope: str = "global",
        readonly: bool = False,
        dangerous: bool = False,
        advanced: bool = False,
        needs_component_reload: bool = False,
        hidden: bool = False,
        validator: Optional[Dict] | Optional[Callable[[Any], bool]] = None,
        tags: Optional[Iterable[str]] = None,
        options: Optional[list] = None,
        component: str = "",
    ):
        self.key = key
        self.label = label
        self.default = default
        self.value_type = value_type
        self.ui_type = ui_type
        self.description = description
        self.scope = scope
        self.readonly = readonly
        self.dangerous = dangerous
        self.advanced = advanced
        self.needs_component_reload = bool(needs_component_reload)
        self.hidden = bool(hidden)
        self.validator = validator
        self.tags = set(tags or [])
        self.options = options or []
        self.component = component

    def validate(self, value: Any) -> None:
        if value is None:
            return
        if self.ui_type == "file":
            return
        if self.value_type is not None and not callable(self.value_type):
            try:
                _ = self.value_type(value)
            except Exception as e:
                raise ValidationError(f"Value for {self.key} must be {self.value_type}: {e}")

        if self.validator is None:
            return
        if callable(self.validator):
            try:
                ok = self.validator(value)
            except Exception as e:
                raise ValidationError(f"Validator for {self.key} raised: {e}")
            if not ok:
                raise ValidationError(f"Validator refused value for {self.key}")
            return
        if isinstance(self.validator, dict):
            v = self.validator
            if "regex" in v:
                if not re.match(v["regex"], str(value)):
                    raise ValidationError(f"Value for {self.key} does not match pattern")
            if "min" in v:
                if float(value) < float(v["min"]):
                    raise ValidationError(f"Value for {self.key} below min {v['min']}")
            if "max" in v:
                if float(value) > float(v["max"]):
                    raise ValidationError(f"Value for {self.key} above max {v['max']}")
            if "choices" in v:
                if value not in v["choices"]:
                    raise ValidationError(f"Value for {self.key} not in allowed choices")


class ExposedVariableRegistry:
    """
    Singleton registry for all known configuration variables.
    Annabeth equivalent: extend annabeth_config.py with typed var registration.
    """
    def __init__(self):
        self._defs: Dict[str, ExposedVarDefinition] = {}

    def register(self, definition: ExposedVarDefinition) -> None:
        self._defs[definition.key] = definition

    def get_definition(self, key: str) -> Optional[ExposedVarDefinition]:
        return self._defs.get(key)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get current value from config system."""
        # Annabeth: read from character_config.yaml or SQLite settings table
        defn = self._defs.get(key)
        if defn is None:
            return default
        return defn.default  # placeholder — real impl reads from persistent store

    async def set_value(self, key: str, value: Any) -> None:
        """Validate and persist a value."""
        definition = self._defs.get(key)
        if not definition:
            raise KeyError(f"Unknown exposed variable: {key}")
        if definition.readonly:
            raise PermissionError(f"Exposed variable {key} is read-only")
        definition.validate(value)
        # Annabeth: write to character_config.yaml or SQLite settings table


# Singleton
exposed_vars = ExposedVariableRegistry()


def register_exposed_var(
    key: str,
    label: str,
    default: Any = "",
    value_type: type | str = str,
    ui_type: str = "string",
    description: str = "",
    scope: str = "global",
    readonly: bool = False,
    dangerous: bool = False,
    advanced: bool = False,
    needs_component_reload: bool = False,
    hidden: bool = False,
    validator: Optional[Dict] | Optional[Callable[[Any], bool]] = None,
    tags: Optional[Iterable[str]] = None,
    options: Optional[list] = None,
    component: str = "",
) -> ExposedVarDefinition:
    d = ExposedVarDefinition(
        key=key, label=label, default=default, value_type=value_type,
        ui_type=ui_type, description=description, scope=scope,
        readonly=readonly, dangerous=dangerous, advanced=advanced,
        needs_component_reload=needs_component_reload, hidden=hidden,
        validator=validator, tags=tags, options=options, component=component,
    )
    exposed_vars.register(d)
    return d


# ============================================================
# ANNABETH-SPECIFIC VARIABLE REGISTRATIONS
# (adapted from register_all() in Synthetic_Heart)
# ============================================================

def register_annabeth_vars():
    """Register all Annabeth settings as typed, labeled config vars."""
    
    # --- Grillo / Reflection beats ---
    register_exposed_var(
        "GRILLO_BEAT_INTERVAL",
        label="Reflection Beat Interval (s)",
        default=1800,
        value_type=int,
        ui_type="number",
        description="Seconds between autonomous reflection beats (default: 1800 = 30 min).",
        scope="grillo",
        component="reflection",
    )
    register_exposed_var(
        "GRILLO_DREAM_ENABLED",
        label="Enable Dream Beats",
        default=True,
        value_type=bool,
        ui_type="bool",
        description="Enable nightly dream generation during low-activity periods.",
        scope="grillo",
        component="reflection",
    )
    register_exposed_var(
        "GRILLO_DREAM_TIME",
        label="Dream Time (HH:MM)",
        default="02:00",
        value_type=str,
        ui_type="string",
        description="Local time when dream beat fires (default: 02:00 AM).",
        scope="grillo",
        component="reflection",
    )
    
    # --- Model switching ---
    register_exposed_var(
        "MODEL_SWITCH_ENABLED",
        label="Enable Auto Model Switching",
        default=True,
        value_type=bool,
        ui_type="bool",
        description="Allow Annabeth to switch Ollama models based on conversation complexity.",
        scope="llm",
        component="model_manager",
    )
    register_exposed_var(
        "MODEL_SWITCH_COOLDOWN_S",
        label="Model Switch Cooldown (s)",
        default=30,
        value_type=int,
        ui_type="number",
        description="Minimum seconds between model switches to prevent thrashing.",
        scope="llm",
        component="model_manager",
    )
    
    # --- Memory ---
    register_exposed_var(
        "MEMORY_COMPRESSION_ENABLED",
        label="Enable Memory Compression",
        default=True,
        value_type=bool,
        ui_type="bool",
        description="Automatically summarize old memories when count exceeds threshold.",
        scope="memory",
        component="memory_manager",
    )
    register_exposed_var(
        "MEMORY_COMPRESSION_THRESHOLD",
        label="Memory Compression Threshold",
        default=500,
        value_type=int,
        ui_type="number",
        description="Number of memory entries before auto-compression triggers.",
        scope="memory",
        component="memory_manager",
    )
    
    # --- Emotion ---
    register_exposed_var(
        "EMOTION_DECAY_TAU",
        label="Emotion Decay Half-Life (s)",
        default=3600,
        value_type=int,
        ui_type="number",
        description="How quickly emotions fade. Larger = slower decay.",
        scope="emotion",
        component="emotion_manager",
        advanced=True,
    )

    # --- VAD / Audio ---
    register_exposed_var(
        "VAD_AGGRESSIVENESS",
        label="Voice Activity Detection Level",
        default=3,
        value_type=int,
        ui_type="select",
        description="WebRTC VAD aggressiveness: 1=lenient, 2=balanced, 3=strict.",
        scope="audio",
        component="audio_input",
        options=[1, 2, 3],
    )
    
    # --- Screensaver (Unity) ---
    register_exposed_var(
        "SCREENSAVER_ENABLED",
        label="Enable Screensaver Mode",
        default=True,
        value_type=bool,
        ui_type="bool",
        description="Enter screensaver animation after idle timeout.",
        scope="unity",
        component="screensaver",
    )
    register_exposed_var(
        "SCREENSAVER_TIMEOUT_STEPS",
        label="Screensaver Timeout Index",
        default=2,
        value_type=int,
        ui_type="select",
        description="Idle time before screensaver: 0=30s, 1=1min, 2=5min, 3=15min, ...",
        scope="unity",
        component="screensaver",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    
    # --- Discord Rich Presence ---
    register_exposed_var(
        "DISCORD_RPC_ENABLED",
        label="Enable Discord Rich Presence",
        default=False,
        value_type=bool,
        ui_type="bool",
        description="Show Annabeth status on Discord profile.",
        scope="discord",
        component="discord_rpc",
    )
    register_exposed_var(
        "DISCORD_APP_ID",
        label="Discord Application ID",
        default="",
        value_type=str,
        ui_type="string",
        description="Your Discord Developer Portal Application ID.",
        scope="discord",
        component="discord_rpc",
    )
