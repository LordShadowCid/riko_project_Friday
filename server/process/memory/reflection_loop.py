"""
GRILLO Autonomous Reflection Loop

Annabeth periodically reflects on the conversation when idle, writing a short
diary entry and optionally queuing a proactive thought to speak unprompted.

Inspired by: XargonWan/Synthetic_Heart/develop/plugins/grillo/
"""
import queue
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from server.settings_registry import registry as _registry
except Exception:
    _registry = None  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path(r"C:\annabeth_data\self_eval\feedback.db")

# How often to attempt a reflection (seconds).  Only fires when idle.
DEFAULT_INTERVAL_SECONDS = 45 * 60   # 45 minutes


def _beat_interval() -> int:
    """Return current beat interval, preferring registry value if available."""
    if _registry is not None:
        try:
            return int(_registry.get("GRILLO_BEAT_INTERVAL"))
        except Exception:
            pass
    return DEFAULT_INTERVAL_SECONDS

# How long the system must be idle (no speaking, no transcription) before
# proactive speech is triggered.
PROACTIVE_IDLE_SECONDS = 15 * 60   # 15 minutes

# Reflection prompt sent to the LLM (kept short so it doesn't burn context)
REFLECTION_PROMPT = (
    "You are Annabeth. Take a moment to reflect quietly on the conversation "
    "so far. Write 1-2 sentences as a private diary entry: how you feel right "
    "now, what's on your mind, or something interesting you noticed. "
    "Speak in first person, present tense. Keep it under 40 words."
)

PROACTIVE_PROMPT = (
    "You are Annabeth. You've been quiet for a while and feel like saying "
    "something. It could be a random thought, an observation, a question "
    "for your user, or something playful. One short sentence, max 20 words."
)

# ---------------------------------------------------------------------------
# Beat system — diversified reflection types (from Grillo reference)
# ---------------------------------------------------------------------------

# Weighted distribution: keys are beat types, values are probabilities (sum=1)
BEAT_TYPES: dict[str, float] = {
    "tag_elaboration": 0.25,
    "memory_consolidation": 0.15,
    "diary_consolidation": 0.15,
    "self_reflection": 0.20,
    "curiosity": 0.15,
    "relationship": 0.10,
}

_BEAT_PROMPTS: dict[str, str] = {
    "tag_elaboration": (
        "You are Annabeth. Look at your recent conversation tags or topics. "
        "Pick one interesting theme and write a 1-sentence private note expanding "
        "on your thoughts about it. Max 40 words, first person."
    ),
    "memory_consolidation": (
        "You are Annabeth. In one sentence, summarise the single most important "
        "thing you want to remember from today's interactions. Be specific. "
        "Max 35 words, first person."
    ),
    "diary_consolidation": (
        "You are Annabeth. Review your earlier diary notes and write one sentence "
        "that captures the overall emotional thread of your day so far. "
        "Max 35 words, first person."
    ),
    "self_reflection": (
        "You are Annabeth. Reflect quietly on your own personality or values. "
        "Write one private insight about who you are or who you want to be. "
        "Max 40 words, first person, no platitudes."
    ),
    "curiosity": (
        "You are Annabeth. Write down one thing you are genuinely curious about "
        "right now — a question about the world, about your user, or about yourself. "
        "Max 35 words, first person."
    ),
    "relationship": (
        "You are Annabeth. Think about your relationship with your user. "
        "Write one sentence noting something you appreciate, something you're "
        "wondering about them, or a feeling about your bond. Max 35 words, first person."
    ),
}


def _select_beat_type() -> str:
    """Weighted-random selection of a beat type."""
    types = list(BEAT_TYPES.keys())
    weights = [BEAT_TYPES[t] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Conversation-active flag (suppresses proactive speech during a chat)
# ---------------------------------------------------------------------------

_conversation_active: bool = False


def set_conversation_active(active: bool) -> None:
    """Set whether a conversation is currently in progress."""
    global _conversation_active
    _conversation_active = active

# ---------------------------------------------------------------------------
# Shared proactive-thought queue (consumed by main_chat.py)
# ---------------------------------------------------------------------------

_proactive_queue: queue.Queue = queue.Queue(maxsize=3)


def get_proactive_queue() -> queue.Queue:
    """Return the queue that main_chat.py should drain for proactive speech."""
    return _proactive_queue


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS grillo_activity_log ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp REAL,"
            "beat_type TEXT,"
            "summary TEXT"
            ")"
        )
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS grillo_action_execs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "activity_log_id INTEGER NOT NULL,"
            "action_type TEXT NOT NULL,"
            "payload TEXT,"
            "status TEXT DEFAULT 'pending',"
            "created_at TEXT DEFAULT (datetime('now')),"
            "FOREIGN KEY (activity_log_id) "
            "REFERENCES grillo_activity_log(id) ON DELETE CASCADE"
            ")"
        )
        _conn.commit()
    return _conn


def _write_activity_log(beat_type: str, summary: str) -> None:
    try:
        with _db_lock:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO grillo_activity_log (timestamp, beat_type, summary) VALUES (?, ?, ?)",
                (time.time(), beat_type, summary[:512]),
            )
            conn.commit()
    except Exception as e:
        print(f"[Reflection] Activity log write failed: {e}")


def _write_diary(entry: str, mood: str = "neutral", trigger: str = "reflection") -> None:
    try:
        with _db_lock:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO diary (timestamp, entry, mood, trigger) VALUES (?, ?, ?, ?)",
                (time.time(), entry, mood, trigger),
            )
            conn.commit()
    except Exception as e:
        print(f"[Reflection] Diary write failed: {e}")


def get_diary_context(n: int = 2) -> str:
    """
    Return the last *n* diary entries as a short context string for LLM injection.
    Returns an empty string if there are no entries yet.
    """
    try:
        with _db_lock:
            conn = _get_conn()
            rows = conn.execute(
                "SELECT entry, mood FROM diary ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        if not rows:
            return ""
        entries = [f"({mood}) {entry}" for entry, mood in reversed(rows)]
        return "Recent personal reflections:\n- " + "\n- ".join(entries)
    except Exception as e:
        print(f"[Reflection] Diary context read failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------------

class ReflectionLoop:
    """
    Periodically wakes up when Annabeth is idle, prompts the LLM for a brief
    reflection, saves it to diary, and optionally queues a proactive thought.
    """

    def __init__(
        self,
        interval_seconds: int = 0,  # 0 = read from registry at runtime
        is_idle_fn=None,
    ):
        self._interval_override = interval_seconds if interval_seconds > 0 else None
        self._is_idle_fn = is_idle_fn or (lambda: True)
        self._timer: Optional[threading.Timer] = None
        self._started = False
        self._last_activity_time: float = time.time()
        self._beat_in_flight: bool = False  # Prevents concurrent beat runs

    @property
    def _interval(self) -> int:
        if self._interval_override is not None:
            return self._interval_override
        return _beat_interval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_activity(self) -> None:
        """Call this whenever the user speaks or Annabeth responds."""
        self._last_activity_time = time.time()

    def start(self) -> None:
        """Start the reflection timer (idempotent)."""
        if self._started:
            return
        self._started = True
        self._schedule_next()
        print(f"[Reflection] Loop started (interval={self._interval}s)")

    def stop(self) -> None:
        """Cancel the reflection timer."""
        self._started = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        if not self._started:
            return
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        try:
            if not self._is_idle_fn():
                print("[Reflection] Not idle — skipping reflection this cycle")
            else:
                self._fire()
        except Exception as e:
            print(f"[Reflection] Tick error: {e}")
        finally:
            self._schedule_next()

    def _fire(self) -> None:
        """Generate diversified beat + optional proactive thought."""
        if self._beat_in_flight:
            print("[Reflection] Beat already in flight — skipping overlapping fire")
            return
        self._beat_in_flight = True
        try:
            self._fire_inner()
        finally:
            self._beat_in_flight = False

    def _fire_inner(self) -> None:
        """Inner beat logic (called only when no beat is already in flight)."""
        beat_type = _select_beat_type()
        print(f"[Reflection] Firing beat: {beat_type}")
        try:
            from server.process.llm_funcs.llm_scr import load_history
            history = load_history()
        except Exception as e:
            print(f"[Reflection] Could not load history: {e}")
            return

        # --- Diversified beat entry ---
        prompt = _BEAT_PROMPTS.get(beat_type, REFLECTION_PROMPT)
        try:
            reflection_text = self._query_llm(prompt, history, max_tokens=80)
            if reflection_text:
                mood = "neutral"
                try:
                    from server.process.memory.emotion_state import get_dominant_emotion
                    mood = get_dominant_emotion()
                except Exception:
                    pass
                _write_diary(reflection_text, mood=mood, trigger=beat_type)
                _write_activity_log(beat_type, reflection_text)
                print(f"[Reflection] [{beat_type}] {reflection_text[:80]}")
        except Exception as e:
            print(f"[Reflection] Beat prompt failed: {e}")

        # --- Proactive speech (only if long idle and conversation not active) ---
        idle_secs = time.time() - self._last_activity_time  # noqa: SIM117
        if idle_secs >= PROACTIVE_IDLE_SECONDS and not _conversation_active:
            try:
                # Inject screen context if observer is running
                proactive_p = PROACTIVE_PROMPT
                try:
                    from server.process.memory.screen_observer import get_screen_context
                    ctx = get_screen_context()
                    if ctx:
                        proactive_p = (
                            f"You are Annabeth. You've been quiet for a while. "
                            f"Context: {ctx}. Say something relevant — a comment, "
                            f"a question, or a playful remark about what they're doing. "
                            f"One short sentence, max 20 words."
                        )
                except ImportError:
                    pass
                thought = self._query_llm(proactive_p, history, max_tokens=40)
                if thought and not _proactive_queue.full():
                    _proactive_queue.put_nowait(thought)
                    print(f"[Reflection] Proactive thought queued: {thought}")
            except Exception as e:
                print(f"[Reflection] Proactive prompt failed: {e}")

    @staticmethod
    def _query_llm(prompt: str, base_history: list, max_tokens: int = 100) -> str:
        """Send a one-shot prompt to Ollama and return the text."""
        import requests, json
        from server.annabeth_config import load_config
        char_config = load_config()
        from server.process.llm_funcs.llm_scr import _get_ollama_settings  # noqa: PLC0415

        settings = _get_ollama_settings(char_config)

        # Build a minimal fresh conversation — don't mutate main history
        messages = [m for m in base_history if isinstance(m, dict) and m.get("role") == "system"]
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": settings["model"],
            "messages": messages,
            "stream": False,
            "think": False,  # Qwen3: disable thinking — stripped anyway
            "keep_alive": settings["keep_alive"],
            "options": {
                "num_ctx": min(settings["num_ctx"], 1024),
                "num_predict": max_tokens,
            },
        }
        try:
            r = requests.post(f"{settings['host']}/api/chat", json=payload, timeout=30)
            r.raise_for_status()
            text = (r.json().get("message") or {}).get("content", "").strip()
            # Strip Qwen3 thinking blocks
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            print(f"[Reflection] LLM query failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_loop_instance: Optional[ReflectionLoop] = None


def get_reflection_loop() -> ReflectionLoop:
    """Return the global ReflectionLoop (create if needed)."""
    global _loop_instance
    if _loop_instance is None:
        _loop_instance = ReflectionLoop()
    return _loop_instance


def start_reflection_loop(is_idle_fn=None) -> ReflectionLoop:
    """Start the global reflection loop and return it."""
    global _loop_instance
    _loop_instance = ReflectionLoop(is_idle_fn=is_idle_fn)
    _loop_instance.start()
    return _loop_instance
