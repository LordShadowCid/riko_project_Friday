"""
Shared utilities for the Annabeth server.

Centralises helpers that were previously duplicated across modules:
- Audio device resolution (was in asr_vad.py AND sovits_ping.py)
- Ollama connection settings (was in llm_scr.py, conversation_summarizer.py, self_eval.py)
"""
import os
import sounddevice as sd
from typing import Optional, Dict, Any

from server.annabeth_config import load_config

# ---------------------------------------------------------------------------
# Audio device resolution (cached)
# ---------------------------------------------------------------------------
_device_cache: dict = {}


def resolve_device(device, kind: str = 'input') -> Optional[int]:
    """Resolve a sounddevice input/output device selector (cached).

    - None / "": use system default
    - int: treated as device index
    - str: case-insensitive substring match against device names
    - kind: 'input' or 'output' — filters to matching channel direction
    """
    if device is None or device == "":
        return None
    if isinstance(device, int):
        return device
    cache_key = (device, kind)
    if cache_key in _device_cache:
        return _device_cache[cache_key]
    if isinstance(device, str):
        devices = sd.query_devices()
        needle = device.lower().strip()
        for idx, d in enumerate(devices):
            name = str(d.get("name", "")).lower()
            if needle and needle in name:
                if kind == 'output' and d.get('max_output_channels', 0) > 0:
                    _device_cache[cache_key] = idx
                    return idx
                elif kind == 'input' and d.get('max_input_channels', 0) > 0:
                    _device_cache[cache_key] = idx
                    return idx
    _device_cache[cache_key] = None
    return None


# ---------------------------------------------------------------------------
# Ollama connection settings
# ---------------------------------------------------------------------------

def get_ollama_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return Ollama host / model / keep_alive / stream / num_ctx from config.

    Parameters
    ----------
    cfg : dict, optional
        Pre-loaded config dict.  If *None*, ``load_config()`` is called once.
    """
    if cfg is None:
        cfg = load_config()
    ollama_cfg = cfg.get("ollama", {}) or {}

    host = str(
        ollama_cfg.get("host")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).strip()
    if host.startswith("0.0.0.0"):
        host = "http://127.0.0.1:11434"
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host

    model = str(ollama_cfg.get("model") or cfg.get("model", "llama3.1:8b")).strip()
    keep_alive = ollama_cfg.get("keep_alive", 3600)
    stream = ollama_cfg.get("stream", True)
    num_ctx = ollama_cfg.get("num_ctx", 2048)

    return {
        "host": host.rstrip("/"),
        "model": model,
        "keep_alive": keep_alive,
        "stream": stream,
        "num_ctx": num_ctx,
    }
