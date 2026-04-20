"""
LLM Model Auto-Router

Selects the best Ollama model per intent category:
  - greeting / question_short  → fast_model  (e.g. qwen3:4b)
  - story / detailed / general → primary_model (e.g. qwen3:8b)
  - command / followup         → primary_model

Reads config from character_config.yaml:

    model_routing:
      enabled: true
      primary_model: qwen3:8b
      fast_model: qwen3:4b

If a requested model is not available in Ollama, falls back to primary_model
so startup never crashes.

Inspired by: MystiaTech/Mai/src/models/
"""
import time
import threading
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# Fast-intent categories (use the smaller/faster model)
# ---------------------------------------------------------------------------

_FAST_INTENTS = {"greeting", "question_short", "math", "factual"}


class ModelRouter:
    """Thread-safe model router."""

    def __init__(self, config: dict):
        cfg = (config.get("model_routing") or {})
        self._enabled: bool = bool(cfg.get("enabled", False))
        self._primary: str = cfg.get("primary_model") or config.get("model") or "llama3"
        self._fast: str = cfg.get("fast_model") or self._primary

        self._available_models: set[str] = set()
        self._probed = False
        self._lock = threading.Lock()

        if self._enabled and self._fast != self._primary:
            # Probe Ollama in background so startup isn't delayed
            threading.Thread(target=self._probe_models, daemon=True).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_model_for_intent(self, intent_category: str) -> str:
        """Return the recommended model name for the given intent category."""
        if not self._enabled:
            return self._primary

        if self._should_force_fast() or intent_category in _FAST_INTENTS:
            return self._resolve(self._fast)
        return self._primary

    def _should_force_fast(self) -> bool:
        """Return True if latency spike or memory pressure warrants fast model."""
        with _latency_lock:
            if time.time() < _forced_fast_until:
                return True
        try:
            import psutil
            from server.settings_registry import registry
            threshold = int(registry.get("MEMORY_THRESHOLD_PCT"))
            if psutil.virtual_memory().percent > threshold:
                print(f"[ModelRouter] High memory — using fast model")
                return True
        except Exception:
            pass
        return False

    def primary_model(self) -> str:
        return self._primary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, model: str) -> str:
        """Return model if available in Ollama, else fall back to primary."""
        with self._lock:
            if not self._probed:
                return model  # Haven't probed yet — optimistically trust config
            if model in self._available_models:
                return model
        print(f"[ModelRouter] '{model}' not available in Ollama — using primary model")
        return self._primary

    def _probe_models(self) -> None:
        """Fetch available model list from Ollama /api/tags and pre-warm fast model."""
        try:
            import requests
            from server.annabeth_config import load_config
            config = load_config()
            from server.process.llm_funcs.llm_scr import _get_ollama_settings  # noqa: PLC0415
            settings = _get_ollama_settings(config)
            host = settings.get("host", "http://localhost:11434")
            r = requests.get(f"{host}/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json()
            names = {m.get("name", "") for m in (tags.get("models") or [])}
            # Also add short names (strip :latest suffix)
            short_names = {n.split(":")[0] for n in names}
            with self._lock:
                self._available_models = names | short_names
                self._probed = True
            print(f"[ModelRouter] Available models: {sorted(names)}")

            # Pre-warm fast model so first request isn't 20s+
            if self._fast != self._primary and self._fast in (names | short_names):
                try:
                    r = requests.post(
                        f"{host}/api/generate",
                        json={"model": self._fast, "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
                        timeout=60,
                    )
                    print(f"[ModelRouter] Pre-warmed fast model '{self._fast}'")
                except Exception as e:
                    print(f"[ModelRouter] Fast model warm-up failed (non-fatal): {e}")
        except Exception as e:
            print(f"[ModelRouter] Model probe failed (non-fatal): {e}")
            with self._lock:
                self._probed = True  # Don't retry on every call


# ---------------------------------------------------------------------------
# Latency tracking + forced-fast window
# ---------------------------------------------------------------------------

_latency_lock = threading.Lock()
_latency_samples: deque = deque(maxlen=5)
_forced_fast_until: float = 0.0
_consecutive_violations: int = 0
_last_violation_time: float = 0.0
_total_samples: int = 0


def record_latency(ms: int) -> None:
    """Record an LLM response latency. Forces fast model if threshold is exceeded.
    
    Uses adaptive cooldown: repeated violations within 2 minutes escalate
    the lockout period (30s → 90s → 180s) to avoid thrashing on cold starts.
    Ignores the first 2 samples to avoid penalizing model loading.
    """
    global _forced_fast_until, _consecutive_violations, _last_violation_time, _total_samples
    if ms <= 0:
        return
    try:
        from server.settings_registry import registry
        threshold = int(registry.get("LATENCY_THRESHOLD_MS"))
        cooldown = int(registry.get("MODEL_SWITCH_COOLDOWN_S"))
    except Exception:
        threshold, cooldown = 5000, 30

    with _latency_lock:
        _latency_samples.append(ms)
        _total_samples += 1
        if ms > threshold:
            # Skip first 2 samples — model loading causes expected cold-start latency
            if _total_samples <= 2:
                print(f"[ModelRouter] Ignoring cold-start latency {ms}ms (sample {_total_samples})")
                return
            now = time.time()
            # Escalate if another violation within 2 minutes
            if now - _last_violation_time < 120:
                _consecutive_violations += 1
            else:
                _consecutive_violations = 1
            _last_violation_time = now
            # Adaptive: 30s, 90s, 180s (cap)
            effective_cooldown = min(cooldown * (3 ** (_consecutive_violations - 1)), 180)
            _forced_fast_until = now + effective_cooldown
            print(f"[ModelRouter] High latency {ms}ms > {threshold}ms — fast model forced for {effective_cooldown}s (violations={_consecutive_violations})")
        else:
            # Good latency — reset violation counter after 5 min of normalcy
            if time.time() - _last_violation_time > 300:
                _consecutive_violations = 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_router: Optional[ModelRouter] = None
_router_lock = threading.Lock()


def get_model_router() -> ModelRouter:
    """Return the global ModelRouter, creating it on first call."""
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                from server.annabeth_config import load_config
                _router = ModelRouter(load_config())
    return _router
