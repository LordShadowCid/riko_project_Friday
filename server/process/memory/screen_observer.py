"""
Grillo Screen Observer

Periodically reads the active window title and stores context hints
for use by the reflection loop and proactive speech system.

Controlled by GRILLO_OBSERVER_ENABLED in settings_registry.
"""
import ctypes
import ctypes.wintypes
import os
import threading
import time
from typing import Optional

_user32 = None
if os.name == "nt":
    _user32 = ctypes.windll.user32  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_current_context: str = ""
_last_title: str = ""
_title_start_time: float = 0.0
_history: list[tuple[float, str]] = []  # (timestamp, title) — last N changes
_MAX_HISTORY = 20


def get_screen_context() -> str:
    """Return a short string describing what the user is doing, for LLM injection."""
    with _lock:
        return _current_context


def get_recent_titles(n: int = 5) -> list[str]:
    """Return last N distinct window title changes."""
    with _lock:
        return [t for _, t in _history[-n:]]


# ---------------------------------------------------------------------------
# Win32 helpers
# ---------------------------------------------------------------------------

def _get_foreground_title() -> str:
    if _user32 is None:
        return ""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = _user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value.strip()


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

# Map of keywords to human-friendly activity descriptions
_ACTIVITY_HINTS = [
    # Browsers
    (["chrome", "firefox", "edge", "brave", "opera", "vivaldi"], "browsing the web"),
    # Games (common patterns)
    (["steam", "epic games"], "browsing a game launcher"),
    (["elden ring", "dark souls", "sekiro", "armored core"], "playing a FromSoftware game"),
    (["minecraft"], "playing Minecraft"),
    (["valorant", "league of legends", "overwatch"], "playing a competitive game"),
    (["genshin", "honkai", "wuthering waves", "zenless"], "playing a gacha game"),
    # Media
    (["youtube", "twitch", "crunchyroll", "netflix", "plex", "vlc", "mpv"], "watching something"),
    (["spotify", "foobar", "musicbee", "aimp"], "listening to music"),
    # Dev
    (["visual studio code", "vscode"], "coding in VS Code"),
    (["unity"], "working in Unity"),
    (["unreal"], "working in Unreal Engine"),
    (["blender"], "working in Blender"),
    # Communication
    (["discord"], "chatting on Discord"),
    (["telegram", "whatsapp", "signal"], "messaging someone"),
    # Files
    (["explorer", "file manager"], "browsing files"),
]


def _classify_title(title: str) -> str:
    """Turn a window title into a brief activity hint."""
    lower = title.lower()
    if not lower or lower in ("", "annabeth"):
        return ""
    for keywords, activity in _ACTIVITY_HINTS:
        if any(kw in lower for kw in keywords):
            return activity
    # Fallback: just report the window title
    if len(title) > 60:
        title = title[:57] + "..."
    return f"using: {title}"


def _update_context(title: str) -> None:
    """Thread-safe context update."""
    global _current_context, _last_title, _title_start_time

    activity = _classify_title(title)
    with _lock:
        if title != _last_title:
            if _last_title:
                _history.append((time.time(), _last_title))
                if len(_history) > _MAX_HISTORY:
                    _history.pop(0)
            _last_title = title
            _title_start_time = time.time()

        if activity:
            duration_min = int((time.time() - _title_start_time) / 60)
            if duration_min >= 2:
                _current_context = f"User appears to be {activity} (for ~{duration_min} min)"
            else:
                _current_context = f"User appears to be {activity}"
        else:
            _current_context = ""


# ---------------------------------------------------------------------------
# Observer loop
# ---------------------------------------------------------------------------

class ScreenObserver:
    """Periodically polls the foreground window title."""

    def __init__(self, interval: int = 30):
        self._interval = interval
        self._timer: Optional[threading.Timer] = None
        self._started = False

    def start(self) -> None:
        if self._started or os.name != "nt":
            return
        self._started = True
        self._schedule()
        print(f"[ScreenObserver] Started (interval={self._interval}s)")

    def stop(self) -> None:
        self._started = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _schedule(self) -> None:
        if not self._started:
            return
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        try:
            title = _get_foreground_title()
            if title:
                _update_context(title)
        except Exception as e:
            print(f"[ScreenObserver] Error: {e}")
        finally:
            self._schedule()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_observer: Optional[ScreenObserver] = None


def start_observer(interval: int = 30) -> None:
    """Start the singleton observer if not already running."""
    global _observer
    if _observer is None:
        _observer = ScreenObserver(interval=interval)
        _observer.start()


def stop_observer() -> None:
    global _observer
    if _observer:
        _observer.stop()
        _observer = None
