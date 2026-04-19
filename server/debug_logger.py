"""
Temporary Test-Session Debug Logger
====================================
Captures all Annabeth subsystem activity to ``d:\\Annabeth\\test_session.log``
during live testing so the developer can review a full activity trace.

USAGE — call once at server startup (main_chat.py does this automatically):

    from server.debug_logger import setup_test_logging
    setup_test_logging()

The log file has timestamped, tagged lines, e.g.:

    2024-01-01 12:00:00.123 [MAIN ] Server started
    2024-01-01 12:00:01.456 [LLM  ] User: "hello"
    2024-01-01 12:00:02.789 [LLM  ] Response: "Hey there! ..."
    2024-01-01 12:00:02.800 [FACE ] Expression: smile:0.8
    2024-01-01 12:00:05.000 [IDLE ] Idle state: True

All existing print() calls throughout the codebase are automatically captured
(stdout tee).  Structured event() calls provide clean JSON lines.

TO DISABLE: set env var ANNABETH_DEBUG_LOG=0 before starting the server.
TO REMOVE:  delete this file and remove the setup_test_logging() call from
            main_chat.py after testing is complete.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_PATH = Path("d:/Annabeth/test_session.log")
_ENABLED = os.environ.get("ANNABETH_DEBUG_LOG", "1") != "0"

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_setup_done = False
_lock = threading.RLock()  # Re-entrant so setup_test_logging can call _raw_write
_log_file: io.TextIOWrapper | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_test_logging() -> None:
    """
    Install stdout tee + file logger.  Safe to call multiple times (idempotent).
    No-op when ANNABETH_DEBUG_LOG=0.
    """
    global _setup_done, _log_file
    if not _ENABLED:
        return
    with _lock:
        if _setup_done:
            return
        _setup_done = True

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _log_file = open(str(LOG_PATH), "a", encoding="utf-8", buffering=1)

        # Reconfigure stdout to UTF-8 so Unicode characters (→, é, …) don't
        # crash the TeeStream.write() on Windows where the console defaults to
        # cp1252 / cp850.
        if hasattr(sys.__stdout__, "reconfigure"):
            try:
                sys.__stdout__.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass  # best-effort; TeeStream.write fallback handles the rest

        # --- Redirect stdout through tee ---
        sys.stdout = _TeeStream(sys.__stdout__, _log_file, tag="STDOUT")

        # --- Root logging → file ---
        fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(name)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(fh)
        logging.getLogger().setLevel(logging.DEBUG)

        _raw_write(
            "LOGGER",
            f"=== TEST SESSION STARTED — {datetime.now().isoformat()} ===",
        )
        print(f"[DebugLogger] Logging to {LOG_PATH}")


def event(tag: str, data: dict[str, Any] | str) -> None:
    """
    Write a structured event line to the log.

    Args:
        tag:  Short category label (≤ 8 chars), e.g. "LLM", "FACE", "IDLE".
        data: Dict or plain string describing the event.
    """
    if not _ENABLED or not _setup_done:
        return
    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    _raw_write(tag, payload)


def get_log_path() -> Path:
    """Return the path of the current test session log file."""
    return LOG_PATH


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raw_write(tag: str, msg: str) -> None:
    """Write a single timestamped line directly to the log file."""
    with _lock:
        if _log_file is None or _log_file.closed:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        _log_file.write(f"{ts} [{tag:<6}] {msg}\n")
        _log_file.flush()


class _TeeStream:
    """Wraps a stream and mirrors writes to a second stream (the log file)."""

    def __init__(
        self,
        primary: io.TextIOWrapper,
        secondary: io.TextIOWrapper,
        tag: str = "STDOUT",
    ):
        self._primary = primary
        self._secondary = secondary
        self._tag = tag
        self._buf = ""

    # Minimal stream interface -----------------------------------------------

    def write(self, text: str) -> int:
        try:
            self._primary.write(text)
        except UnicodeEncodeError:
            self._primary.write(text.encode(self._primary.encoding or "utf-8", errors="replace").decode(self._primary.encoding or "utf-8", errors="replace"))
        self._primary.flush()
        # Buffer until newline for clean log lines
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                _raw_write(self._tag, line)
        return len(text)

    def flush(self) -> None:
        self._primary.flush()
        if _log_file and not _log_file.closed:
            _log_file.flush()

    def fileno(self) -> int:
        return self._primary.fileno()

    def isatty(self) -> bool:
        return self._primary.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:
        return getattr(self._primary, "errors", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)
