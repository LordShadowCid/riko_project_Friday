"""
Shared utilities for the Annabeth server.

Centralises helpers that were previously duplicated across modules:
- Audio device resolution (was in asr_vad.py AND sovits_ping.py)
- Ollama connection settings (was in llm_scr.py, conversation_summarizer.py, self_eval.py)
"""
import os
import sounddevice as sd
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

from server.annabeth_config import load_config

# ---------------------------------------------------------------------------
# Audio device resolution (cached)
# ---------------------------------------------------------------------------
_device_cache: dict = {}


def configure_windows_cuda_runtime() -> None:
    """Expose venv-provided NVIDIA DLL directories for Windows Python runs.

    The PowerShell launcher already prepends these directories to PATH, but
    direct `python -m server.main_chat` runs do not. Adding them here keeps
    manual startup behavior aligned with the launcher.
    """
    if os.name != "nt":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    candidate_dirs = [
        site_packages / "cudnn" / "bin",
        site_packages / "cublas" / "bin",
    ]

    existing_path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized_entries = {os.path.normcase(entry) for entry in existing_path_entries if entry}

    prepend_entries = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue

        directory_str = str(directory)
        normalized_dir = os.path.normcase(directory_str)
        if normalized_dir not in normalized_entries:
            prepend_entries.append(directory_str)

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            try:
                add_dll_directory(directory_str)
            except OSError:
                pass

    if prepend_entries:
        os.environ["PATH"] = os.pathsep.join(prepend_entries + [os.environ.get("PATH", "")])


def describe_port_listener(port: int, expected_text: str = "") -> Optional[str]:
    """Return a short diagnostic for an already-bound localhost port.

    Used to turn duplicate-start socket errors into an actionable message.
    """
    health_note = ""
    if expected_text:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                body = response.read(4096).decode("utf-8", errors="ignore")
                if expected_text.lower() in body.lower():
                    health_note = f" Annabeth HTTP endpoint is already responding at http://127.0.0.1:{port}/."
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass

    if os.name != "nt":
        return f"Port {port} is already in use.{health_note}".strip()

    try:
        import ctypes
        from ctypes import wintypes

        MIB_TCP_STATE_LISTEN = 2
        AF_INET = 2
        TCP_TABLE_OWNER_PID_LISTENER = 3
        NO_ERROR = 0
        ERROR_INSUFFICIENT_BUFFER = 122

        class MIB_TCPROW_OWNER_PID(ctypes.Structure):
            _fields_ = [
                ("dwState", wintypes.DWORD),
                ("dwLocalAddr", wintypes.DWORD),
                ("dwLocalPort", wintypes.DWORD),
                ("dwRemoteAddr", wintypes.DWORD),
                ("dwRemotePort", wintypes.DWORD),
                ("dwOwningPid", wintypes.DWORD),
            ]

        size = wintypes.DWORD(0)
        iphlpapi = ctypes.WinDLL("iphlpapi.dll")
        get_extended_tcp_table = iphlpapi.GetExtendedTcpTable
        get_extended_tcp_table.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.BOOL,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
        ]
        get_extended_tcp_table.restype = wintypes.DWORD

        result = get_extended_tcp_table(None, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0)
        if result not in (NO_ERROR, ERROR_INSUFFICIENT_BUFFER):
            return f"Port {port} is already in use.{health_note}".strip()

        buffer = ctypes.create_string_buffer(size.value)
        result = get_extended_tcp_table(buffer, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0)
        if result != NO_ERROR:
            return f"Port {port} is already in use.{health_note}".strip()

        entry_count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD)).contents.value
        rows_offset = ctypes.sizeof(wintypes.DWORD)

        for index in range(entry_count):
            row = MIB_TCPROW_OWNER_PID.from_buffer_copy(buffer[rows_offset + index * ctypes.sizeof(MIB_TCPROW_OWNER_PID): rows_offset + (index + 1) * ctypes.sizeof(MIB_TCPROW_OWNER_PID)])
            local_port = int.from_bytes(row.dwLocalPort.to_bytes(4, byteorder="big"), byteorder="little") >> 16
            if row.dwState == MIB_TCP_STATE_LISTEN and local_port == port:
                return f"Port {port} is already in use by PID {row.dwOwningPid}.{health_note}".strip()
    except Exception:
        pass

    return f"Port {port} is already in use.{health_note}".strip()


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
