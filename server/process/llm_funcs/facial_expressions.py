"""
Facial Expression Timeline (Phase 2 — Synthetic_Heart)


Parses inline [em_NAME:INTENSITY] tags from LLM output and broadcasts
timed WebSocket messages to Unity to drive VRM BlendShapes.

Tag format:   [em_smile:0.8]
Valid names:  smile, grin, sad, blush, surprised, angry, wink, shy, neutral
INTENSITY:    0.0 – 1.0

Usage:
    from server.process.llm_funcs.facial_expressions import (
        parse_facial_expressions,
        play_expression_timeline,
    )

    clean, events = parse_facial_expressions(llm_text)
    # clean = text with tags removed (send to TTS)
    # events = [(char_pos, name, intensity), ...]

    # Schedule broadcasts concurrently with TTS (fire-and-forget coroutine)
    asyncio.create_task(
        play_expression_timeline(events, len(clean), ws_broadcast_fn)
    )
"""
import asyncio
import re
import threading
import time as _time
from typing import Callable, Awaitable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tag regex
# ---------------------------------------------------------------------------

# Matches [em_NAME:INTENSITY], [em_NAME] (default intensity), or bare [em] (reset)
_TAG_RE = re.compile(r'\[em(?:_(\w+))?(?::(\d+(?:\.\d+)?))?\]', re.IGNORECASE)

# Supported expression names (case-insensitive, stripped to lowercase in events)
VALID_EXPRESSIONS = frozenset({
    "smile", "grin", "sad", "blush", "surprised",
    "angry", "wink", "shy", "neutral", "thinking", "happy",
})

_DEFAULT_INTENSITY = 0.7  # Used when tag has name but no explicit intensity

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_facial_expressions(
    text: str,
) -> Tuple[str, List[Tuple[int, str, float]]]:
    """
    Strip ``[em_NAME:INTENSITY]`` tags from *text* and return:

    * ``clean_text`` – the text with all expression tags removed; safe to send to TTS.
    * ``events`` – list of ``(char_pos, name, intensity)`` tuples, where
      ``char_pos`` is the character index in the *original* text where the tag
      appeared (used to compute timing relative to TTS audio duration).

    Unknown expression names are silently ignored (tag still removed from text).
    Intensity is clamped to [0.0, 1.0].
    """
    events: List[Tuple[int, str, float]] = []

    def _replace(m: re.Match) -> str:
        raw_name = m.group(1)
        raw_intensity = m.group(2)

        # Bare [em] with no name = reset-to-base event
        if raw_name is None:
            events.append((m.start(), None, 0.0))
            return ""

        name = raw_name.lower()
        intensity = max(0.0, min(1.0, float(raw_intensity))) if raw_intensity else _DEFAULT_INTENSITY
        if name in VALID_EXPRESSIONS:
            events.append((m.start(), name, intensity))
        return ""

    clean = _TAG_RE.sub(_replace, text)
    return clean, events


# ---------------------------------------------------------------------------
# Timeline scheduler
# ---------------------------------------------------------------------------


async def _delayed_broadcast(
    delay: float,
    msg: dict,
    fn: Callable[[dict], Awaitable[None]],
) -> None:
    await asyncio.sleep(max(0.0, delay))
    await fn(msg)


async def play_expression_timeline(
    events: List[Tuple[int, str, float]],
    total_chars: int,
    broadcast_fn: Callable[[dict], Awaitable[None]],
    audio_duration_s: float = 0.0,
    chars_per_sec: float = 0.0,
) -> None:
    """
    Schedule a broadcast for each expression event proportional to audio timing.

    Args:
        events:           From :func:`parse_facial_expressions`.
        total_chars:      Character length of the clean (tag-stripped) text.
        broadcast_fn:     ``async (msg_dict) -> None``; sends JSON to Unity.
        audio_duration_s: Actual TTS audio length in seconds.  If 0, estimated
                          from ``chars_per_sec``.
        chars_per_sec:    Fallback speech rate.  If 0, reads from settings
                          registry (default 12.0 cps ≈ ~2-3 words/sec).
    """
    if not events:
        return

    # Resolve chars_per_sec from registry if not provided
    if chars_per_sec <= 0:
        try:
            from server.settings_registry import registry
            chars_per_sec = float(registry.get("FACIAL_EXPR_CHARS_PER_SEC"))
        except Exception:
            chars_per_sec = 12.0

    duration = audio_duration_s if audio_duration_s > 0 else (
        total_chars / chars_per_sec if total_chars > 0 else 1.0
    )

    tasks = []
    for char_pos, name, intensity in events:
        frac = char_pos / max(total_chars, 1)
        delay = frac * duration
        tasks.append(_delayed_broadcast(
            delay,
            {"type": "face_expression", "name": name, "intensity": intensity},
            broadcast_fn,
        ))

    # Reset all expressions after audio ends
    tasks.append(_delayed_broadcast(
        duration + 0.5,
        {"type": "face_expression", "name": None, "intensity": 0.0},
        broadcast_fn,
    ))

    await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# System prompt snippet (injected by annabeth_config when FACIAL_EXPR_ENABLED)
# ---------------------------------------------------------------------------

FACIAL_EXPR_INJECT = (
    " You may embed facial expression hints inline using [em_NAME:INTENSITY] where "
    "NAME is one of: smile, grin, sad, blush, surprised, angry, wink, shy, neutral, "
    "thinking, happy. INTENSITY is 0.0 to 1.0. Use bare [em] to reset expressions. "
    "Place them naturally, up to 3 per response. Example: "
    "'That's so exciting! [em_surprised:0.9] I can't wait to hear more.'"
)

# ---------------------------------------------------------------------------
# Sync / threaded API (for use from non-async llm_scr.py)
# ---------------------------------------------------------------------------

_face_loop: Optional[asyncio.AbstractEventLoop] = None
_face_broadcast: Optional[Callable] = None  # async (dict) -> None
_face_lock = threading.Lock()


def set_face_broadcast(loop: asyncio.AbstractEventLoop, fn: Callable) -> None:
    """Register the asyncio loop + broadcast callable.  Called from main_chat.py."""
    global _face_loop, _face_broadcast
    with _face_lock:
        _face_loop = loop
        _face_broadcast = fn


def _schedule_face_send(delay_s: float, name: Optional[str], intensity: float) -> None:
    """Schedule a single face_expression broadcast *delay_s* seconds from now."""
    with _face_lock:
        loop = _face_loop
        fn = _face_broadcast
    if loop is None or fn is None:
        return

    async def _send() -> None:
        await asyncio.sleep(max(0.0, delay_s))
        await fn({"type": "face_expression", "name": name, "intensity": intensity})

    asyncio.run_coroutine_threadsafe(_send(), loop)


def play_expression_timeline_sync(
    events: List[Tuple[int, str, float]],
    total_chars: int,
    audio_duration_s: float = 0.0,
    chars_per_sec: float = 0.0,
) -> None:
    """
    Thread-safe version of :func:`play_expression_timeline`.

    Fires scheduled broadcasts using the event loop registered via
    :func:`set_face_broadcast`.  Call from synchronous code (llm_scr.py).
    """
    if not events:
        return

    with _face_lock:
        if _face_loop is None or _face_broadcast is None:
            return

    if chars_per_sec <= 0:
        try:
            from server.settings_registry import registry
            chars_per_sec = float(registry.get("FACIAL_EXPR_CHARS_PER_SEC"))
        except Exception:
            chars_per_sec = 12.0

    duration = audio_duration_s if audio_duration_s > 0 else max(total_chars / chars_per_sec, 1.0)

    for char_pos, name, intensity in events:
        frac = char_pos / max(total_chars, 1)
        _schedule_face_send(frac * duration, name, intensity)

    # Reset after audio ends
    _schedule_face_send(duration + 0.5, None, 0.0)

