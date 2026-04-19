"""
Persistent Emotion State with Decay

Tracks Annabeth's emotional state using a Plutchik-wheel model.
Emotions decay toward a resting baseline over time (tau = 1 hour).
Emotion tags in LLM output look like: {happy 8.5, love 3.0}

Inspired by: XargonWan/Synthetic_Heart/develop/plugins/emotion_manager.py
"""
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_EMOTIONS = {
    "happy", "sad", "angry", "fear", "disgust", "surprised",
    "neutral", "relaxed", "love", "arousal", "devotion"
}

PLUTCHIK_OPPOSITES: Dict[str, str] = {
    "happy":     "sad",
    "sad":       "happy",
    "angry":     "fear",
    "fear":      "angry",
    "neutral":   "disgust",
    "surprised": "relaxed",
    "relaxed":   "surprised",
    "love":      "disgust",
    "arousal":   "disgust",
    "disgust":   "love",
    "devotion":  "angry",
}

# Resting intensity level each emotion decays toward
EMOTION_BASELINES: Dict[str, float] = {
    "neutral": 5.0,
    "relaxed": 1.0,
}
DEFAULT_BASELINE = 0.1

# Decay time-constant (seconds).  At tau=3600 an emotion at 10.0 decays to
# ~3.7 after one hour, reaching baseline asymptotically.
DECAY_TAU = 3600.0

# Intensity scale: 0.0 – 10.0
INTENSITY_MIN = 0.0
INTENSITY_MAX = 10.0

# Regex that matches LLM emotion tags in many forms the model produces:
#   Proper:     {happy 8.5, love 3.0}
#   Descriptive:{Frustration:sarcastic disappointment 8.2}
#   Orphaned:   amused 7.1}  or  anxious 4.2})
#   Old format: [em_surprised:0.9]
#   Empty:      {}
#   Garbled:    Inconsistent frustration}: ...
_EMOTION_TAG_RE = re.compile(
    r"|".join([
        # 1. Proper / descriptive brace blocks: { ... } or {{ ... }}
        r"\s*\{{1,2}[^}]{0,80}\}{1,2}",
        # 2. Old [em_xxx] or [em_xxx:number] format
        r"\[em_\w+(?::[^\]]{0,20})?\]",
        # 3. ANY word(s)/hyphens followed by number then }/) — catches all
        #    compound emotion labels like "Confusion-displeasure-blah 7.}}"
        r"\s*\b[\w][\w\s:,-]{0,60}?\s+-?\d+(?:\.\d*)?\s*\.?\s*[})]{1,3}",
        # 4. Emotion word followed by } with no number (garbled)
        r"\b(?:happy|sad|angry|fear|disgust|surprised|neutral|relaxed|love"
        r"|arousal|devotion|frustration|sarcas\w*|excitement|joy|gratitude"
        r"|disapproval|contempt|confusion|boredom|amusement|irritation"
        r"|playful\w*|affection\w*|tender\w*)"
        r"(?:[\w-])*\s*[})]+",
    ]),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

DB_PATH = Path(r"C:\annabeth_data\self_eval\feedback.db")
_db_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def _ensure_table() -> None:
    """Create emotion_state table if it doesn't exist."""
    with _db_lock:
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emotion_state (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion   TEXT    NOT NULL,
                intensity REAL    NOT NULL DEFAULT 0.0,
                timestamp REAL    NOT NULL,
                UNIQUE(emotion)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emotion_state_emotion ON emotion_state(emotion)")
        conn.commit()


_ensure_table()

# ---------------------------------------------------------------------------
# In-memory state (mirrors DB, faster reads)
# ---------------------------------------------------------------------------

@dataclass
class EmotionEntry:
    emotion: str
    intensity: float
    timestamp: float = field(default_factory=time.time)

    def get_decayed_intensity(self, now: Optional[float] = None) -> float:
        """Apply exponential decay toward baseline."""
        if now is None:
            now = time.time()
        baseline = EMOTION_BASELINES.get(self.emotion, DEFAULT_BASELINE)
        delta_t = max(0.0, now - self.timestamp)
        intensity = baseline + (self.intensity - baseline) * math.exp(-delta_t / DECAY_TAU)
        return max(INTENSITY_MIN, min(INTENSITY_MAX, intensity))


_state_lock = threading.Lock()
_emotions: Dict[str, EmotionEntry] = {}


def _load_from_db() -> None:
    """Populate in-memory state from DB on startup."""
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("SELECT emotion, intensity, timestamp FROM emotion_state").fetchall()
    with _state_lock:
        for emotion, intensity, ts in rows:
            _emotions[emotion] = EmotionEntry(emotion=emotion, intensity=intensity, timestamp=ts)


_load_from_db()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_emotion(emotion: str, intensity: float) -> None:
    """
    Set an emotion intensity.  Applies Plutchik dampening (raising one emotion
    lowers its opposite).  Persists to DB.
    """
    emotion = emotion.lower()
    if emotion not in CANONICAL_EMOTIONS:
        return  # Ignore unknown emotions gracefully

    intensity = max(INTENSITY_MIN, min(INTENSITY_MAX, float(intensity)))
    now = time.time()

    with _state_lock:
        _emotions[emotion] = EmotionEntry(emotion=emotion, intensity=intensity, timestamp=now)

        # Plutchik: dampen the opposite emotion
        opposite = PLUTCHIK_OPPOSITES.get(emotion)
        if opposite and opposite in _emotions:
            opp_entry = _emotions[opposite]
            current_opp = opp_entry.get_decayed_intensity(now)
            dampened = max(EMOTION_BASELINES.get(opposite, DEFAULT_BASELINE),
                           current_opp - intensity * 0.5)
            _emotions[opposite] = EmotionEntry(emotion=opposite, intensity=dampened, timestamp=now)

    # Persist asynchronously to avoid blocking TTS pipeline
    _save_to_db_async(emotion, intensity, now)
    if opposite and opposite in _emotions:
        opp_val = _emotions[opposite].intensity
        _save_to_db_async(opposite, opp_val, now)


def set_emotions_from_dict(tags: Dict[str, float]) -> None:
    """Apply a dict of {emotion: intensity} pairs in one call."""
    for emotion, intensity in tags.items():
        set_emotion(emotion, intensity)


def get_dominant_emotion() -> str:
    """Return the name of the highest-intensity decayed emotion."""
    now = time.time()
    with _state_lock:
        if not _emotions:
            return "neutral"
        best = max(_emotions.values(), key=lambda e: e.get_decayed_intensity(now))
        return best.emotion


def get_all_emotions() -> Dict[str, float]:
    """Return all current decayed intensities."""
    now = time.time()
    with _state_lock:
        return {name: entry.get_decayed_intensity(now) for name, entry in _emotions.items()}


def get_emotion_context() -> str:
    """Return a short string for LLM injection, e.g. 'Current mood: happy (8.2)'."""
    dominant = get_dominant_emotion()
    now = time.time()
    with _state_lock:
        entry = _emotions.get(dominant)
    if entry is None:
        return ""
    intensity = entry.get_decayed_intensity(now)
    return f"Current mood: {dominant} ({intensity:.1f}/10)"


# ---------------------------------------------------------------------------
# Tag parsing helpers
# ---------------------------------------------------------------------------

# Secondary cleanup for stragglers: orphaned "}" with optional preceding junk,
# and leftover colons after stripped words.
_STRAGGLER_RE = re.compile(
    r"|".join([
        # word(s) followed by orphaned }  (e.g. "Inconsistent frustration}")
        r"\b\w[\w\s-]{0,40}\}{1,2}",
        # word:descriptive-text }  (e.g. "gratitude:sarcastic acknowledgment }")
        r"\b\w+:[^}]{0,40}\}{1,2}",
        # Empty parentheses left after tag stripping
        r"\s*\(\s*\)",
        # lone orphaned } ) or }} after whitespace
        r"\s+[})]{1,3}(?=\s|$)",
        # orphaned ) at end of word (e.g. "anyway)")
        r"(?<=\w)\)(?=\s|$|[.,!?])",
        # dangling colon at end or before spaces
        r"\s*:\s*(?=\s*$)",
    ]),
    re.IGNORECASE,
)

def strip_emotion_tags(text: str) -> str:
    """Remove {emotion value, ...} tags from text before TTS synthesis."""
    result = _EMOTION_TAG_RE.sub("", text)
    result = _STRAGGLER_RE.sub("", result)
    # Collapse multiple spaces and trim
    result = re.sub(r"  +", " ", result)
    return result.strip()


def extract_emotion_tags(text: str) -> Dict[str, float]:
    """
    Extract emotion tags from text and return as {emotion: intensity} dict.

    Example:
        "Hello! {happy 8.5, love 3.0}" -> {"happy": 8.5, "love": 3.0}
    """
    tags: Dict[str, float] = {}
    # Also try a direct scan for canonical "emotion number" pairs anywhere
    _PAIR_RE = re.compile(
        r"\b(" + "|".join(re.escape(e) for e in CANONICAL_EMOTIONS) + r")\s+"
        r"(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for match in _EMOTION_TAG_RE.finditer(text):
        inner = match.group(0).strip().strip("{}[]")
        for pair in _PAIR_RE.finditer(inner):
            name = pair.group(1).lower()
            try:
                tags[name] = float(pair.group(2))
            except ValueError:
                pass
    return tags


# ---------------------------------------------------------------------------
# DB helpers (async write)
# ---------------------------------------------------------------------------

def _save_to_db_async(emotion: str, intensity: float, ts: float) -> None:
    """Write emotion row to DB in a background thread to avoid blocking."""
    def _write():
        try:
            with _db_lock:
                conn = _get_conn()
                conn.execute(
                    """INSERT INTO emotion_state (emotion, intensity, timestamp)
                       VALUES (?, ?, ?)
                       ON CONFLICT(emotion) DO UPDATE SET
                           intensity = excluded.intensity,
                           timestamp = excluded.timestamp""",
                    (emotion, intensity, ts),
                )
                conn.commit()
        except Exception as e:
            print(f"[EmotionState] DB write error: {e}")

    threading.Thread(target=_write, daemon=True).start()


# ---------------------------------------------------------------------------
# Decay loop
# ---------------------------------------------------------------------------

_decay_timer: Optional[threading.Timer] = None
_decay_started = False


def decay_once() -> None:
    """Apply decay to all tracked emotions (called by timer)."""
    now = time.time()
    with _state_lock:
        for name, entry in list(_emotions.items()):
            decayed = entry.get_decayed_intensity(now)
            baseline = EMOTION_BASELINES.get(name, DEFAULT_BASELINE)
            # If emotion has essentially reached baseline, mark timestamp
            # as now so future decays start from already-decayed value
            _emotions[name] = EmotionEntry(emotion=name, intensity=decayed, timestamp=now)
            _save_to_db_async(name, decayed, now)


def start_decay_loop(interval_seconds: int = 60) -> None:
    """Start the background decay timer (idempotent)."""
    global _decay_started
    if _decay_started:
        return
    _decay_started = True

    def _tick():
        global _decay_timer
        decay_once()
        _decay_timer = threading.Timer(interval_seconds, _tick)
        _decay_timer.daemon = True
        _decay_timer.start()

    _decay_timer = threading.Timer(interval_seconds, _tick)
    _decay_timer.daemon = True
    _decay_timer.start()
    print(f"[EmotionState] Decay loop started (interval={interval_seconds}s)")


def stop_decay_loop() -> None:
    """Cancel the decay timer."""
    global _decay_timer, _decay_started
    if _decay_timer:
        _decay_timer.cancel()
        _decay_timer = None
    _decay_started = False
