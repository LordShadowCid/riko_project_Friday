"""
Centralised logging configuration for Annabeth.

Two modes
---------
TESTING  (ANNABETH_LOG_LEVEL=DEBUG  OR  "--debug-log" in sys.argv)
  • DEBUG+ to rotating file  C:\\annabeth_data\\logs\\annabeth_debug.log
    – 10 MB per file, 5 backups (50 MB total cap)
  • INFO+  to console with colour-coded prefix
  • Full tracebacks captured automatically by the root logger

PRODUCTION  (default)
  • WARNING+ to rotating file  C:\\annabeth_data\\logs\\annabeth.log
    – 1 MB per file, 3 backups (3 MB total cap)
  • No console handler
  • Minimal IO overhead

Usage
-----
Call configure_logging() once, as early as possible in main_chat.py::

    from server.logging_config import configure_logging
    configure_logging()

All modules that currently use bare print() will still work — their output
is unaffected.  Modules that use logging.getLogger(__name__) will have their
records routed automatically once this is called.

Post-testing cleanup
--------------------
To revert to production mode: remove the ANNABETH_LOG_LEVEL env var
(or stop passing --debug-log).  No code changes needed — the level
selection is entirely environment-driven.

Best practice note (for reference after testing)
-------------------------------------------------
• Keep PRODUCTION mode as default — WARNING+rotating is negligible overhead.
• Reserve DEBUG mode for active development / bug triaging only.
• RotatingFileHandler prevents unbounded log growth without cron jobs.
• Do NOT add a StreamHandler in production — it doubles console spam and
  slows down the main-thread print loop.
• If you add structured logging later (JSON lines), replace the Formatter
  here — everything else in the codebase stays the same.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# ── Path constants ─────────────────────────────────────────────────
_LOG_DIR = Path(os.environ.get("ANNABETH_DATA", r"C:\annabeth_data")) / "logs"

_PROD_LOG   = _LOG_DIR / "annabeth.log"
_DEBUG_LOG  = _LOG_DIR / "annabeth_debug.log"

# ── Format strings ────────────────────────────────────────────────
_FMT_DETAIL  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_FMT_PROD    = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATEFMT     = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes for console output (Windows 10+ VT supported)
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    """Console formatter that colour-codes the level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}" if colour else record.levelname
        return super().format(record)


def _is_test_mode() -> bool:
    return (
        os.environ.get("ANNABETH_LOG_LEVEL", "").upper() == "DEBUG"
        or "--debug-log" in sys.argv
    )


def configure_logging() -> None:
    """Configure the root logging system. Call once at process start."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()

    # Avoid double-configuration on re-import
    if root.handlers:
        return

    if _is_test_mode():
        _configure_test(root)
    else:
        _configure_production(root)

    # Silence noisy third-party loggers that flood at DEBUG
    for noisy in ("urllib3", "httpx", "httpcore", "websockets", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _configure_test(root: logging.Logger) -> None:
    root.setLevel(logging.DEBUG)

    # ── Rotating debug file ─────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        _DEBUG_LOG,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT_DETAIL, datefmt=_DATEFMT))
    root.addHandler(fh)

    # ── Console (INFO+) ─────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColouredFormatter(_FMT_DETAIL, datefmt=_DATEFMT))
    root.addHandler(ch)

    logging.info("[Logging] TEST mode — DEBUG → %s", _DEBUG_LOG)


def _configure_production(root: logging.Logger) -> None:
    root.setLevel(logging.WARNING)

    fh = logging.handlers.RotatingFileHandler(
        _PROD_LOG,
        maxBytes=1 * 1024 * 1024,    # 1 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter(_FMT_PROD, datefmt=_DATEFMT))
    root.addHandler(fh)
    # No console handler in production


def get_log_path() -> Path:
    """Return the active log file path (useful for real-time tail commands)."""
    return _DEBUG_LOG if _is_test_mode() else _PROD_LOG
